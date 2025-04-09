import datetime
import typing
import os
import logging

import requests
import json
import xarray as xr
from multiformats import CID
from py_hamt import HAMT, IPFSStore, IPFSZarr3

from .dclimate_zarr_errors import (
    DatasetNotFoundError,
    NoMetadataFoundError,
    IpfsConnectionError,
    StacCatalogError,
)

DEFAULT_HOST = "http://127.0.0.1:5001/api/v0"
VALID_TIME_SPANS = ["daily", "hourly", "weekly", "quarterly"]
CID_ENDPOINT = "https://raw.githubusercontent.com/dClimate/dclimate-data-cids/refs/heads/main/cids.json"

# Configure logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# --- Caching ---
# Simple in-memory cache for STAC traversal results (dataset_id -> ipns_name)
# Cleared on each run. For persistent caching, a file-based approach could be used.
_stac_hamt_cid_cache: typing.Dict[str, str] = {}


# --- IPFSStore Configuration ---
def _get_ipfs_store(
    gateway_uri_stem: str | None = None, rpc_uri_stem: str | None = None
) -> IPFSStore:
    """Creates or retrieves a configured IPFSStore instance."""
    # Simple way to manage store instances per configuration.
    # More sophisticated caching/management might be needed in complex apps.
    store_kwargs = {}
    if gateway_uri_stem:
        store_kwargs["gateway_uri_stem"] = gateway_uri_stem
    if rpc_uri_stem:
        store_kwargs["rpc_uri_stem"] = rpc_uri_stem
    return IPFSStore(**store_kwargs)


# --- IPFS Interaction Helpers ---
def _fetch_json_from_ipfs(cid_str: str, ipfs_store: IPFSStore) -> dict:
    """Fetches and parses JSON data from an IPFS CID using IPFSStore."""
    try:
        logger.debug(f"Fetching JSON from CID: {cid_str}")
        # Ensure cid_str is just the CID, remove potential prefixes
        if cid_str.startswith("/ipfs/"):
            cid_str = cid_str[6:]
        cid = CID.decode(cid_str)
        json_bytes = ipfs_store.load(cid)
        if not json_bytes:
            raise StacCatalogError(f"No data returned for CID: {cid_str}")
        return json.loads(json_bytes)
    except json.JSONDecodeError as e:
        raise StacCatalogError(f"Failed to decode JSON from CID {cid_str}: {e}") from e
    except Exception as e:  # Catch potential py-hamt/connection errors
        # Check for common connection error patterns
        if "Connection refused" in str(e) or "Max retries exceeded" in str(e):
            raise IpfsConnectionError(
                f"Failed to connect to IPFS to fetch CID {cid_str}. "
                f"Is IPFS daemon running and accessible at {ipfs_store.rpc_uri_stem} / {ipfs_store.gateway_uri_stem}? Details: {e}"
            ) from e
        raise StacCatalogError(f"Error fetching data for CID {cid_str}: {e}") from e


def resolve_ipns_to_cid(ipns_name: str, rpc_uri_stem: str | None = None) -> str:
    """
    Resolves an IPNS name (e.g., /ipns/k51...) to its currently pinned IPFS CID.

    Args:
        ipns_name (str): The IPNS name (with or without /ipns/ prefix).
        rpc_uri_stem (str, optional): Custom IPFS RPC API URI stem. Defaults to IPFSStore default.

    Returns:
        str: The resolved IPFS CID string.

    Raises:
        IpfsConnectionError: If the IPFS daemon is unreachable.
        DatasetNotFoundError: If the IPNS name cannot be resolved.
    """
    if not ipns_name:
        raise ValueError("IPNS name cannot be empty.")

    # Use IPFSStore's configured RPC endpoint
    ipfs_store = _get_ipfs_store(rpc_uri_stem=rpc_uri_stem)
    rpc_base = (
        ipfs_store.rpc_uri_stem
    )  # Get the potentially defaulted/env-var configured stem

    # Ensure ipns_name starts with /ipns/ for the API call
    if not ipns_name.startswith("/ipns/"):
        if ipns_name.startswith("k51"):  # Basic check for common IPNS format
            ipns_name_for_api = f"/ipns/{ipns_name}"
        else:
            # Might be a DNSLink name or other format - try resolving as is
            logger.warning(
                f"Resolving potentially non-standard IPNS name format: {ipns_name}"
            )
            ipns_name_for_api = ipns_name  # Or maybe prefix with /ipns/? API behaviour varies. Let's assume prefix is needed.
            ipns_name_for_api = f"/ipns/{ipns_name}"

    resolve_url = f"{rpc_base}/name/resolve"
    params = {
        "arg": ipns_name_for_api,
        "stream": "false",
    }  # stream=false ensures single result
    logger.debug(f"Resolving IPNS: {ipns_name_for_api} via {resolve_url}")

    try:
        # Use POST as recommended by IPFS docs for /name/resolve
        response = requests.post(
            resolve_url, params=params, timeout=60
        )  # Increased timeout
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        resolved_data = response.json()
        resolved_path = resolved_data.get("Path")

        if not resolved_path or not resolved_path.startswith("/ipfs/"):
            raise DatasetNotFoundError(
                f"Could not resolve IPNS name '{ipns_name}' to an IPFS path. Response: {resolved_data}"
            )

        cid_str = resolved_path[6:]  # Remove /ipfs/
        logger.info(f"Resolved IPNS '{ipns_name}' to CID: {cid_str}")
        return cid_str

    except requests.exceptions.ConnectionError as e:
        raise IpfsConnectionError(
            f"Connection error resolving IPNS name '{ipns_name}' via {rpc_base}. Is IPFS daemon running? Details: {e}"
        ) from e
    except requests.exceptions.Timeout as e:
        raise IpfsConnectionError(
            f"Timeout resolving IPNS name '{ipns_name}' via {rpc_base}. IPFS network might be slow or name not resolvable."
        ) from e
    except requests.exceptions.RequestException as e:
        # Catch other request errors (like HTTPError from raise_for_status)
        err_msg = f"Error resolving IPNS name '{ipns_name}' via {rpc_base}: {e}"
        if response is not None:
            err_msg += f" Status Code: {response.status_code}, Response: {response.text[:200]}"  # Log response snippet
        raise DatasetNotFoundError(err_msg) from e


# --- STAC Traversal and HAMT CID Retrieval ---
def get_dataset_hamt_cid_from_stac(
    root_catalog_ipns: str,
    target_dataset_id: str,
    gateway_uri_stem: str | None = None,
    rpc_uri_stem: str | None = None,
) -> str:
    """
    Traverses the dClimate STAC catalog starting from a root IPNS name
    to find the HAMT root IPFS CID associated with the target dataset ID.

    Args:
        root_catalog_ipns (str): The IPNS name of the root STAC catalog (e.g., /ipns/k51...).
        target_dataset_id (str): The unique identifier of the dataset to find (e.g., 'cpc-precip-conus').
        gateway_uri_stem (str, optional): Custom IPFS HTTP Gateway URI stem.
        rpc_uri_stem (str, optional): Custom IPFS RPC API URI stem.

    Returns:
        str: The HAMT root IPFS CID string of the target dataset.

    Raises:
        DatasetNotFoundError: If the dataset ID is not found in the catalog or lacks the HAMT asset.
        IpfsConnectionError: If connection to IPFS fails.
        StacCatalogError: For issues during STAC parsing or traversal.
    """
    # Check cache first (using the renamed cache variable)
    if target_dataset_id in _stac_hamt_cid_cache:
        logger.info(f"Found dataset '{target_dataset_id}' HAMT CID in cache.")
        return _stac_hamt_cid_cache[target_dataset_id]

    logger.info(f"Searching STAC catalog for dataset: {target_dataset_id}")
    ipfs_store = _get_ipfs_store(gateway_uri_stem, rpc_uri_stem)

    # 1. Resolve root catalog IPNS to IPFS CID
    try:
        catalog_cid = resolve_ipns_to_cid(root_catalog_ipns, rpc_uri_stem)
        logger.info(f"Root catalog CID: {catalog_cid}")
    except (IpfsConnectionError, DatasetNotFoundError) as e:
        raise StacCatalogError(
            f"Failed to resolve root catalog IPNS '{root_catalog_ipns}': {e}"
        ) from e

    # 2. Fetch root catalog
    catalog = _fetch_json_from_ipfs(catalog_cid, ipfs_store)
    if not isinstance(catalog, dict) or catalog.get("type") != "Catalog":
        raise StacCatalogError(f"Invalid root catalog format for CID {catalog_cid}")

    # 3. Iterate through collections (same as before)
    collections_to_visit = []
    for link in catalog.get("links", []):
        if link.get("rel") == "child" and link.get("type") == "application/json":
            href = link.get("href")
            if href and href.startswith("/ipfs/"):
                collection_cid = href[6:]
                collections_to_visit.append(collection_cid)
            else:
                logger.warning(f"Skipping invalid child link in root catalog: {link}")

    if not collections_to_visit:
        raise StacCatalogError(
            f"No valid child collection links found in root catalog {catalog_cid}"
        )

    # 4. Iterate through items in each collection (Modified extraction)
    for collection_cid in collections_to_visit:
        logger.debug(f"Fetching collection: {collection_cid}")
        try:
            collection = _fetch_json_from_ipfs(collection_cid, ipfs_store)
            if (
                not isinstance(collection, dict)
                or collection.get("type") != "Collection"
            ):
                logger.warning(
                    f"Skipping invalid collection format for CID {collection_cid}"
                )
                continue

            for link in collection.get("links", []):
                if link.get("rel") == "item" and link.get("type") == "application/json":
                    href = link.get("href")
                    if href and href.startswith("/ipfs/"):
                        item_cid = href[6:]
                        logger.debug(f"Fetching item: {item_cid}")
                        try:
                            item = _fetch_json_from_ipfs(item_cid, ipfs_store)
                            if (
                                not isinstance(item, dict)
                                or item.get("type") != "Feature"
                            ):
                                logger.warning(
                                    f"Skipping invalid item format for CID {item_cid}"
                                )
                                continue

                            item_id = item.get("id")
                            if item_id == target_dataset_id:
                                logger.info(
                                    f"Found matching item for '{target_dataset_id}' with CID {item_cid}"
                                )

                                # *** MODIFIED EXTRACTION LOGIC ***
                                hamt_asset = item.get("assets", {}).get("hamt-zarr", {})
                                hamt_cid_href = hamt_asset.get("href")

                                if not hamt_cid_href or not hamt_cid_href.startswith(
                                    "/ipfs/"
                                ):
                                    raise StacCatalogError(
                                        f"STAC Item '{item_id}' (CID: {item_cid}) is missing a valid 'assets.hamt-zarr.href' "
                                        f"pointing to the HAMT CID. Found: '{hamt_cid_href}'"
                                    )

                                # Extract the CID string
                                hamt_cid_str = hamt_cid_href[6:]

                                logger.info(
                                    f"Found HAMT CID for '{target_dataset_id}': {hamt_cid_str}"
                                )
                                _stac_hamt_cid_cache[target_dataset_id] = (
                                    hamt_cid_str  # Update cache
                                )
                                return hamt_cid_str  # Return the HAMT CID string

                        # ... (rest of the try/except for item processing remains the same) ...
                        except (StacCatalogError, IpfsConnectionError) as item_err:
                            logger.error(
                                f"Error processing item {item_cid} in collection {collection_cid}: {item_err}"
                            )
                            raise item_err
                        except Exception as item_err:
                            logger.error(
                                f"Unexpected error processing item {item_cid}: {item_err}"
                            )
                            raise StacCatalogError(
                                f"Unexpected error processing item {item_cid}"
                            ) from item_err
                    # ... (Skipping invalid item link remains same) ...

        # ... (rest of the try/except for collection processing remains the same) ...
        except (StacCatalogError, IpfsConnectionError) as col_err:
            logger.error(f"Error processing collection {collection_cid}: {col_err}")
            raise col_err
        except Exception as col_err:
            logger.error(
                f"Unexpected error processing collection {collection_cid}: {col_err}"
            )
            raise StacCatalogError(
                f"Unexpected error processing collection {collection_cid}"
            ) from col_err

    # 5. If loop completes without finding the dataset
    raise DatasetNotFoundError(
        f"Dataset ID '{target_dataset_id}' not found in the STAC catalog rooted at '{root_catalog_ipns}'."
    )


def _get_host(uri: str = "/api/v0"):
    """Parse the ipfs api host address from `IPFS_HOST` environment variable.
    If not found, use localhost:5001/api/v0.

    Args:
        uri (str): the uri where ipfs gateway api listens

    Returns:
        str: ipfs gateway url

    """

    host_from_env = os.getenv("IPFS_HOST")
    return host_from_env + uri if host_from_env else DEFAULT_HOST


def _get_single_metadata(ipfs_hash: str) -> dict:
    """Get metadata for given ipfs hash over ipld

    Args:
        ipfs_hash (str): ipfs hash for which to get metadata

    Returns:
        dict: dict of metadata for hash
    """

    r = requests.get(f"{_get_host()}/ipfs/{ipfs_hash}")
    r.raise_for_status()
    return r.json()


def _get_previous_hash_from_metadata(metadata: dict) -> typing.Optional[str]:
    """Pull in last updated hash from STAC metadata

    Args:
        metadata (dict): STAC metadata

    Returns:
        str: Previous hash, or None if given root metadata
    """
    links = metadata["links"]
    try:
        link_to_previous = [
            link for link in links if link["rel"] in {"prev", "previous"}
        ][0]
    except IndexError:
        return None
    return link_to_previous["metadata href"]["/"]


def _resolve_ipns_name_hash(ipns_name_hash: str) -> str:
    """Find the latest IPFS hash corresponding to a stable ipns name hash

    Args:
        ipfs_name_hash (str): stable IPNS name hash

    Returns:
        str: ipfs hash corresponding to this ipns name hash
    """
    r = requests.get(f"{_get_host()}/ipns/{ipns_name_hash}", params={"offline": True})
    r.raise_for_status()
    return r.json()["Path"].split("/")[-1]


def update_cache_if_changed(new_data: dict) -> None:
    """Update the local cache file only if the new data differs from what is cached."""
    cache_file = os.path.join(os.path.dirname(__file__), "cids.json")
    try:
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cached_data = None

    if cached_data != new_data:
        with open(cache_file, "w") as f:
            json.dump(new_data, f)


def get_ipns_name_hash(ipns_key_str: str) -> str:
    """Find the latest IPNS name hash corresponding to a string (key)

    Args:
        ipfs_key_str (str): a string (key) identifying a dataset

    Raises:
        KeyError: raised if no IPNS key string is found in the IPNS keys list

    Returns:
        str: ipfsname hash corresponding to the provided string
    """

    try:
        # 1) Try to fetch from endpoint
        r = requests.get(CID_ENDPOINT, params={"decoder": "json"})
        r.raise_for_status()
        json_cid = r.json()  # raises JSONDecodeError if endpoint returns malformed JSON

        # Update cache only if there is a change
        update_cache_if_changed(json_cid)

        for entry in json_cid:
            if entry == ipns_key_str:
                return json_cid[entry]

    except (requests.RequestException, KeyError, json.JSONDecodeError):
        # 2) If remote fails or is malformed, try local fallback
        cache_file = os.path.join(os.path.dirname(__file__), "cids.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    json_cid = json.load(
                        f
                    )  # <-- can raise JSONDecodeError if file is empty/corrupt
                for entry in json_cid:
                    if entry == ipns_key_str:
                        return json_cid[entry]
            except (KeyError, json.JSONDecodeError) as err:
                # We tried local, but it’s also invalid (bad JSON or missing key)
                raise DatasetNotFoundError("Invalid dataset name") from err

    # 3) If we get here, local file either doesn't exist or didn't have the key
    raise DatasetNotFoundError("Invalid dataset name") from None


def _get_relevant_metadata(ipfs_head_hash: str, as_of: datetime.datetime) -> dict:
    """Iterates through STAC metadata until metadata generated before as_of is found

    Args:
        ipfs_head_hash (str): first hash in chain
        as_of (datetime.datetime): cutoff date for finding metadata

    Raises:
        NoMetadataFoundError: raised if no metadata older than cutoff date is found

    Returns:
        dict: relevant metadata
    """
    cur_metadata = _get_single_metadata(ipfs_head_hash)
    while True:
        time_generated = datetime.datetime.strptime(
            cur_metadata["properties"]["updated"], "%Y-%m-%dT%H:%M:%SZ"
        )
        if time_generated <= as_of:
            return cur_metadata
        prev_hash = _get_previous_hash_from_metadata(cur_metadata)
        if prev_hash is None:
            raise NoMetadataFoundError(f"No metadata found after as_of: {as_of}")
        cur_metadata = _get_single_metadata(prev_hash)


# def get_dataset_by_ipfs_hash(ipfs_hash: str) -> xr.Dataset:
#     """Gets xarray dataset using changing ipfs hash

#     Args:
#         ipfs_hash (str): ipfs hash that changes between updates

#     Returns:
#         xr.Dataset: dataset corresponding to hash
#     """
#     hamt_store = HAMT(store=IPFSStore(), root_node_id=ipfs_hash, read_only=True)
#     return xr.open_zarr(store=hamt_store, chunks=None)


# --- Zarr Dataset Loading ---
# Renamed from get_dataset_by_ipns_hash
def _get_dataset_by_ipfs_cid(
    ipfs_cid: str,
    gateway_uri_stem: str | None = None,
    rpc_uri_stem: str | None = None,
) -> xr.Dataset:
    """
    Gets an xarray dataset directly from its Zarr root IPFS CID using py-hamt.

    Args:
        ipfs_cid (str): The IPFS CID of the Zarr dataset's root node (e.g., HAMT root).
        gateway_uri_stem (str, optional): Custom IPFS HTTP Gateway URI stem.
        rpc_uri_stem (str, optional): Custom IPFS RPC API URI stem.

    Returns:
        xr.Dataset: The loaded dataset.

    Raises:
        IpfsConnectionError: If connection to IPFS fails during loading.
        Exception: Other errors during Zarr parsing or IPFS interaction.
    """
    if not ipfs_cid:
        raise ValueError("IPFS CID cannot be empty.")

    logger.info(f"Loading Zarr dataset from IPFS CID: {ipfs_cid}")
    ipfs_store = _get_ipfs_store(gateway_uri_stem, rpc_uri_stem)

    try:
        # root_node_id expects a CID object or string representation
        cid_obj = CID.decode(ipfs_cid)
        hamt_store = HAMT(store=ipfs_store, root_node_id=cid_obj, read_only=True)
        ipfszarr3_store = IPFSZarr3(hamt_store, read_only=True)

        # Ensure registered codecs (like encryption) are available if needed
        # zarr.registry.get_codec(...) or ensure they are registered beforehand

        # consolidated=False is typical for HAMT stores, but could be True if explicitly created that way
        ds = xr.open_zarr(store=ipfszarr3_store, chunks=None)
        logger.info(f"Successfully loaded dataset from CID: {ipfs_cid}")
        return ds
    except Exception as e:
        # More specific error catching could be added here (e.g., for Zarr format errors)
        if "Connection refused" in str(e) or "Max retries exceeded" in str(e):
            raise IpfsConnectionError(
                f"Failed to connect to IPFS while loading dataset from CID {ipfs_cid}. "
                f"Gateway: {ipfs_store.gateway_uri_stem}, RPC: {ipfs_store.rpc_uri_stem}. Details: {e}"
            ) from e
        logger.error(f"Failed to load dataset from IPFS CID {ipfs_cid}: {e}")
        # Re-raise a more specific or generic error
        raise RuntimeError(
            f"Failed to load Zarr dataset from IPFS CID {ipfs_cid}"
        ) from e


def get_metadata_by_key(key: str) -> dict:
    """Get STAC metadata for specific dataset

    Args:
        key (str): dataset key

    Returns:
        dict: STAC metadata corresponding to key
    """
    ipns_name = get_ipns_name_hash(key)
    ipfs_hash = _resolve_ipns_name_hash(ipns_name)
    return _get_single_metadata(ipfs_hash)


# def list_datasets() -> typing.List[str]:
#     """List datasets available on IPFS node

#     Returns:
#         typing.List[str]: List of available datasets' keys
#     """
#     # Try to fetch from endpoint first
#     try:
#         r = requests.get(CID_ENDPOINT, params={"decoder": "json"})
#         r.raise_for_status()
#         json_cid = r.json()  # may raise JSONDecodeError if malformed

#         # Update the local cache if remote data differs
#         update_cache_if_changed(json_cid)
#         return list(json_cid.keys())

#     except (requests.RequestException, json.JSONDecodeError):
#         # Fallback to local cache if endpoint is unreachable or JSON is malformed
#         cache_file = os.path.join(os.path.dirname(__file__), "cids.json")
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r") as f:
#                     json_cid = json.load(f)  # can raise JSONDecodeError
#                 return list(json_cid.keys())
#             except json.JSONDecodeError as err:
#                 # local file is corrupt or empty
#                 raise RuntimeError(
#                     "Failed to retrieve dataset list from endpoint or local cache."
#                 ) from err

#     # If both the endpoint and local file fail, raise an error
#     raise RuntimeError(
#         "Failed to retrieve dataset list from endpoint or local cache."
#     ) from None


# --- Metadata and Listing ---
def list_datasets(
    root_catalog_ipns: str | None = None,
    gateway_uri_stem: str | None = None,
    rpc_uri_stem: str | None = None,
) -> typing.List[str]:
    """
    Lists available dataset IDs by traversing the STAC catalog.
    Also populates the internal HAMT CID cache.
    """
    # ... (initial setup, resolve root catalog, fetch catalog - same as before) ...
    if root_catalog_ipns is None:
        from .client import DCLIMATE_STAC_CATALOG_IPNS  # Import locally

        root_catalog_ipns = DCLIMATE_STAC_CATALOG_IPNS

    logger.info(f"Listing datasets from STAC catalog: {root_catalog_ipns}")
    ipfs_store = _get_ipfs_store(gateway_uri_stem, rpc_uri_stem)
    dataset_ids = []

    try:
        catalog_cid = resolve_ipns_to_cid(root_catalog_ipns, rpc_uri_stem)
    except (IpfsConnectionError, DatasetNotFoundError) as e:
        raise StacCatalogError(
            f"Failed to resolve root catalog IPNS '{root_catalog_ipns}' for listing: {e}"
        ) from e

    catalog = _fetch_json_from_ipfs(catalog_cid, ipfs_store)
    # ... (validation of catalog type) ...
    if not isinstance(catalog, dict) or catalog.get("type") != "Catalog":
        raise StacCatalogError(
            f"Invalid root catalog format for listing (CID: {catalog_cid})"
        )

    collections_to_visit = []
    for link in catalog.get("links", []):
        if link.get("rel") == "child" and link.get("type") == "application/json":
            href = link.get("href")
            if href and href.startswith("/ipfs/"):
                collections_to_visit.append(href[6:])

    for collection_cid in collections_to_visit:
        try:
            collection = _fetch_json_from_ipfs(collection_cid, ipfs_store)
            # ... (validation of collection type) ...
            if (
                not isinstance(collection, dict)
                or collection.get("type") != "Collection"
            ):
                logger.warning(
                    f"Skipping invalid collection format during list (CID: {collection_cid})"
                )
                continue

            for link in collection.get("links", []):
                if link.get("rel") == "item" and link.get("type") == "application/json":
                    href = link.get("href")
                    if href and href.startswith("/ipfs/"):
                        item_cid = href[6:]
                        try:
                            item = _fetch_json_from_ipfs(item_cid, ipfs_store)
                            # ... (validation of item type) ...
                            if (
                                not isinstance(item, dict)
                                or item.get("type") != "Feature"
                            ):
                                logger.warning(
                                    f"Skipping invalid item format during list (CID: {item_cid})"
                                )
                                continue

                            item_id = item.get("id")
                            if item_id:
                                dataset_ids.append(item_id)
                                # *** UPDATE CACHE with HAMT CID if found ***
                                hamt_asset = item.get("assets", {}).get("hamt-zarr", {})
                                hamt_cid_href = hamt_asset.get("href")
                                if (
                                    item_id not in _stac_hamt_cid_cache
                                    and hamt_cid_href
                                    and hamt_cid_href.startswith("/ipfs/")
                                ):
                                    hamt_cid_str = hamt_cid_href[6:]
                                    _stac_hamt_cid_cache[item_id] = hamt_cid_str
                                    logger.debug(
                                        f"Cached HAMT CID {hamt_cid_str} for {item_id} during list."
                                    )

                        # ... (error handling for item processing remains same) ...
                        except (StacCatalogError, IpfsConnectionError) as item_err:
                            logger.error(
                                f"Skipping item {item_cid} during list due to error: {item_err}"
                            )
                        except Exception as item_err:
                            logger.error(
                                f"Skipping item {item_cid} during list due to unexpected error: {item_err}"
                            )
                    # ... (Skipping invalid item link remains same) ...

        # ... (error handling for collection processing remains same) ...
        except (StacCatalogError, IpfsConnectionError) as col_err:
            logger.error(
                f"Skipping collection {collection_cid} during list due to error: {col_err}"
            )
        except Exception as col_err:
            logger.error(
                f"Skipping collection {collection_cid} during list due to unexpected error: {col_err}"
            )

    logger.info(f"Found {len(set(dataset_ids))} unique datasets in STAC catalog.")
    return sorted(list(set(dataset_ids)))  # Return unique sorted list
