import datetime
import typing
import os
import logging

import requests
import json
import xarray as xr
from multiformats import CID
from py_hamt import HAMT, KuboCAS
from py_hamt.zarr_hamt_store import ZarrHAMTStore

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
# Simple in-memory cache for STAC traversal results (dataset_id -> hamt_cid)
# Cleared on each run. For persistent caching, a file-based approach could be used.
_stac_hamt_cid_cache: typing.Dict[str, str] = {}


async def get_dataset_stac(
    stac_root_cid: str,
    collection: str,
    dataset: str,
    gateway_uri_stem: str,
) -> dict:
    """
    Traverses a STAC catalog on IPFS starting from a root CID to find and
    return a specific dataset's STAC Item JSON.

    This function uses KuboCAS for all IPFS interactions.

    Args:
        stac_root_cid (str): The root CID of the STAC catalog.
        collection (str): The ID or title of the target collection.
        dataset (str): The ID of the target dataset (STAC Item).
        gateway_uri_stem (str): The base URL for the IPFS gateway.

    Returns:
        dict: The STAC Item JSON for the requested dataset.

    Raises:
        StacCatalogError: If the catalog structure is invalid or a network error occurs.
        DatasetNotFoundError: If the collection or dataset cannot be found.
    """
    logger.info(
        f"Searching STAC catalog for dataset '{dataset}' in collection '{collection}'"
    )
    async with KuboCAS(gateway_base_url=gateway_uri_stem) as kubo_cas:
        try:
            # 1. Fetch the root catalog
            logger.debug(f"Fetching root catalog from CID: {stac_root_cid}")
            root_catalog_bytes = await kubo_cas.load(CID.decode(stac_root_cid))
            root_catalog = json.loads(root_catalog_bytes)

            if not isinstance(root_catalog, dict) or root_catalog.get("type") != "Catalog":
                raise StacCatalogError("Invalid root catalog format.")

            # 2. Find the target collection's CID
            collection_cid_str = None
            for link in root_catalog.get("links", []):
                # Match by collection ID or title for flexibility
                if link.get("rel") == "child" and (
                    link.get("title") == collection or link.get("id") == collection
                ):
                    href_obj = link.get("href")
                    if isinstance(href_obj, dict) and "/" in href_obj:
                        collection_cid_str = href_obj["/"]
                        break
            
            if not collection_cid_str:
                raise DatasetNotFoundError(f"Collection '{collection}' not found in root catalog.")

            # 3. Fetch the collection
            logger.debug(f"Fetching collection '{collection}' from CID: {collection_cid_str}")
            collection_bytes = await kubo_cas.load(CID.decode(collection_cid_str))
            collection_json = json.loads(collection_bytes)

            if not isinstance(collection_json, dict) or collection_json.get("type") != "Collection":
                raise StacCatalogError(f"Invalid format for collection '{collection}'.")

            # 4. Find and return the target dataset (item) or collection
            for item_link in collection_json.get("links", []):
                if item_link.get("rel") == "item" or item_link.get("rel") == "child":
                    item_href_obj = item_link.get("href")
                    item_cid_str = None
                    if isinstance(item_href_obj, dict) and "/" in item_href_obj:
                        item_cid_str = item_href_obj["/"]
                    
                    if not item_cid_str:
                        continue

                    # Fetch the item to check its ID
                    item_bytes = await kubo_cas.load(CID.decode(item_cid_str))
                    item_json = json.loads(item_bytes)
                    
                    if item_json.get("id") == dataset:
                        logger.info(f"Found matching STAC Item for dataset '{dataset}'.")
                        return item_json # Return the full STAC item

            # If the loop completes, the dataset was not found in the collection
            raise DatasetNotFoundError(f"Dataset '{dataset}' not found in collection '{collection}'.")

        except Exception as e:
            # Catch potential errors from KuboCAS (e.g., network issues) or JSON parsing
            # and wrap them in a more specific error.
            logger.error(f"Failed during STAC traversal: {e}", exc_info=True)
            raise StacCatalogError(f"An error occurred during STAC traversal: {e}") from e

# --- IPFSStore Configuration ---
# async def _get_ipfs_store(
#     gateway_uri_stem: str | None = None, rpc_uri_stem: str | None = None
# ) -> KuboCAS:
#     """Creates or retrieves a configured IPFSStore instance."""
#     # Use environment variables as defaults if not provided
#     gateway = gateway_uri_stem or os.environ.get("IPFS_GATEWAY_URI_STEM")
#     rpc = rpc_uri_stem or os.environ.get("IPFS_RPC_URI_STEM")

#     async with KuboCAS(gateway_base_url=gateway, rpc_base_url=rpc) as kubo_cas:
#         return kubo_cas



# --- IPFS Interaction Helpers ---
# async def fetch_json_from_cid(cid_str: str, ipfs_store: KuboCAS) -> dict:
#     """Fetches and parses JSON data from an IPFS CID using KuboCAS (gateway or RPC)."""
#     try:
#         logger.debug(f"Fetching JSON from CID: {cid_str}")
#         if cid_str.startswith("/ipfs/"):
#             cid_str = cid_str[6:]

#         # Validate CID format before attempting fetch
#         try:
#             cid = CID.decode(cid_str)
#         except Exception as decode_err:
#             raise StacCatalogError(
#                 f"Failed to decode CID string '{cid_str}': {decode_err}"
#             ) from decode_err

#         # Use the store's timeout
#         json_bytes = await ipfs_store.load(cid)

#         if not json_bytes:
#             raise StacCatalogError(f"No data returned for CID: {cid_str}")
#         return json.loads(json_bytes)
#     except json.JSONDecodeError as e:
#         raise StacCatalogError(f"Failed to decode JSON from CID {cid_str}: {e}") from e
#     except requests.exceptions.Timeout as e:
#         raise IpfsConnectionError(
#             f"Timeout fetching CID {cid_str} via KuboCAS (using {ipfs_store.gateway_uri_stem or ipfs_store.rpc_uri_stem}). Details: {e}"
#         ) from e
#     except Exception as e:
#         if (
#             "Connection refused" in str(e)
#             or "Max retries exceeded" in str(e)
#             or "Failed to establish a new connection" in str(e)
#         ):
#             raise IpfsConnectionError(
#                 f"Failed to connect via KuboCAS (using {ipfs_store.gateway_uri_stem or ipfs_store.rpc_uri_stem}) to fetch CID {cid_str}. "
#                 f"Is IPFS daemon/gateway running/accessible? Details: {e}"
#             ) from e
#         raise StacCatalogError(
#             f"Error fetching data for CID {cid_str} via KuboCAS: {e}"
#         ) from e


# --- STAC Traversal and HAMT CID Retrieval ---
# async def get_dataset_hamt_cid_from_stac(
#     root_catalog_ipns: str,
#     target_dataset_id: str,
#     gateway_uri_stem: str | None = None,
#     rpc_uri_stem: (
#         str | None
#     ) = None,  # Keep rpc_uri_stem for IPFSStore config if needed by fetch_json_from_cid
# ) -> str:
#     """
#     Traverses the dClimate STAC catalog starting from a root IPNS name
#     to find the HAMT root IPFS CID associated with the target dataset ID.
#     Handles IPLD link format `{"/": "cid_string"}`.
#     """
#     if target_dataset_id in _stac_hamt_cid_cache:
#         logger.info(f"Found dataset '{target_dataset_id}' HAMT CID in cache.")
#         return _stac_hamt_cid_cache[target_dataset_id]

#     logger.info(f"Searching STAC catalog for dataset: {target_dataset_id}")
#     # Still need ipfs_store for subsequent CID fetches if rpc differs from gateway
#     ipfs_store = _get_ipfs_store(gateway_uri_stem, rpc_uri_stem)

#     # *** Use fetch_json_from_ipns for the root catalog ***
#     try:
#         logger.debug(f"Fetching root catalog content from IPNS: {root_catalog_ipns}")
#         catalog = fetch_json_from_ipns(
#             root_catalog_ipns, gateway_uri_stem=gateway_uri_stem
#         )
#     except (IpfsConnectionError, StacCatalogError, ValueError) as e:
#         raise StacCatalogError(
#             f"Failed to fetch root catalog from IPNS '{root_catalog_ipns}': {e}"
#         ) from e

#     # Rest of the logic remains similar, using fetch_json_from_cid for /ipfs/ links
#     if not isinstance(catalog, dict) or catalog.get("type") != "Catalog":
#         raise StacCatalogError(
#             f"Invalid root catalog format fetched from IPNS {root_catalog_ipns}"
#         )

#     collections_to_visit = []
#     for link in catalog.get("links", []):
#         if link.get("rel") == "child" and link.get("type") == "application/json":
#             href_obj = link.get("href")
#             if isinstance(href_obj, dict):
#                 collection_cid_str = href_obj.get("/")
#                 if isinstance(collection_cid_str, str):
#                     collections_to_visit.append(collection_cid_str)
#                     logger.debug(
#                         f"Found child collection link (CID): {collection_cid_str}"
#                     )
#                 else:
#                     logger.warning(
#                         f"Skipping child link with invalid href dict content in root catalog: {link}"
#                     )
#             elif isinstance(href_obj, str) and href_obj.startswith("/ipfs/"):
#                 # Allow legacy /ipfs/ string format for backward compatibility? Risky.
#                 # For now, strictly expect dict for IPLD links as per example.
#                 logger.warning(
#                     f"Skipping child link with unexpected string href format (expected dict): {link}"
#                 )
#                 collections_to_visit.append(href_obj[6:])
#             else:
#                 logger.warning(
#                     f"Skipping invalid child link format in root catalog: {link}"
#                 )

#     if not collections_to_visit:
#         raise StacCatalogError(
#             f"No valid child collection links found in root catalog fetched from IPNS {root_catalog_ipns}"
#         )

#     logger.info(f"Found {len(collections_to_visit)} collections to search.")

#     for collection_cid in collections_to_visit:
#         logger.debug(f"Fetching collection content for CID: {collection_cid}")
#         try:
#             # --- collection JSON ---
#             collection = await fetch_json_from_cid(collection_cid, ipfs_store)
#             if (
#                 not isinstance(collection, dict)
#                 or collection.get("type") != "Collection"
#             ):
#                 logger.warning(
#                     f"Skipping invalid collection format for CID {collection_cid}. "
#                     f"Type: {collection.get('type')}"
#                 )
#                 continue

#             items_found_in_collection = 0
#             for link in collection.get("links", []):
#                 if link.get("rel") == "item" and link.get("type") == "application/json":
#                     item_href_obj = link.get("href")
#                     item_cid = None  # reset each link

#                     # --- handle IPLD or legacy /ipfs/ links ---
#                     if isinstance(item_href_obj, dict):
#                         item_cid = item_href_obj.get("/")  # IPLD dict
#                     elif isinstance(item_href_obj, str) and item_href_obj.startswith(
#                         "/ipfs/"
#                     ):
#                         logger.warning(
#                             f"Found legacy string href in {collection_cid}: {link}"
#                         )
#                         item_cid = item_href_obj[6:]  # strip "/ipfs/"
#                     else:
#                         logger.warning(
#                             f"Skipping invalid item link in {collection_cid}: {link}"
#                         )
#                         continue

#                     if not isinstance(item_cid, str):
#                         # already logged warning
#                         continue

#                     items_found_in_collection += 1
#                     item_id: str | None = None  # keep in scope for except
#                     try:
#                         # --- item JSON ---
#                         item = await fetch_json_from_cid(item_cid, ipfs_store)

#                         if not isinstance(item, dict) or item.get("type") != "Feature":
#                             logger.warning(
#                                 f"Skipping invalid item format for CID {item_cid}. "
#                                 f"Type: {item.get('type')}"
#                             )
#                             continue

#                         item_id = item.get("id")
#                         if item_id != target_dataset_id:
#                             # not the dataset we're looking for
#                             continue

#                         logger.info(
#                             f"Found matching item for '{target_dataset_id}' "
#                             f"(CID {item_cid}) in collection {collection_cid}"
#                         )
#                         hamt_asset = item.get("assets", {}).get("hamt-zarr", {})
#                         hamt_cid_href = hamt_asset.get("href")  # expected "/ipfs/<cid>"

#                         if not isinstance(
#                             hamt_cid_href, str
#                         ) or not hamt_cid_href.startswith("/ipfs/"):
#                             raise StacCatalogError(
#                                 f"STAC Item '{item_id}' (CID: {item_cid}) is missing a "
#                                 f"valid string 'assets.hamt-zarr.href' starting with "
#                                 f"/ipfs/. Found: '{hamt_cid_href}' "
#                                 f"(type: {type(hamt_cid_href).__name__})"
#                             )

#                         hamt_cid_str = hamt_cid_href[6:]  # drop "/ipfs/"
#                         logger.info(
#                             f"Successfully extracted HAMT CID for '{target_dataset_id}': "
#                             f"{hamt_cid_str}"
#                         )
#                         _stac_hamt_cid_cache[target_dataset_id] = hamt_cid_str
#                         return hamt_cid_str

#                     # ── error handling ──────────────────────────────────────────
#                     except StacCatalogError as item_err:
#                         # If the failing item *is* the target dataset, bubble it up.
#                         if item_id == target_dataset_id:
#                             raise item_err
#                         logger.error(
#                             f"Error processing non-target item {item_cid} in collection "
#                             f"{collection_cid}: {item_err}"
#                         )
#                     except IpfsConnectionError as item_err:
#                         logger.error(
#                             f"IPFS error processing item {item_cid} in collection "
#                             f"{collection_cid}: {item_err}"
#                         )
#                     except Exception as item_err:
#                         logger.error(
#                             f"Unexpected error processing item {item_cid} in collection "
#                             f"{collection_cid}: {type(item_err).__name__}: {item_err}"
#                         )

#             logger.debug(
#                 f"Finished searching {items_found_in_collection} items in collection "
#                 f"{collection.get('id', collection_cid)}."
#             )

#         except StacCatalogError as col_err:
#             # ← this is the error that means “the target dataset is malformed”
#             #    → let it propagate to the caller so tests (and callers) can see it.
#             raise col_err
#         except IpfsConnectionError as col_err:
#             # ← still swallow network errors so other collections can be tried
#             logger.error(
#                 f"IPFS error processing collection {collection_cid}, continuing search: "
#                 f"{col_err}"
#             )
#         except Exception as col_err:
#             logger.error(
#                 f"Unexpected error processing collection {collection_cid}, continuing "
#                 f"search: {type(col_err).__name__}: {col_err}"
#             )

#     # If the loop completes without returning
#     raise DatasetNotFoundError(
#         f"Dataset ID '{target_dataset_id}' not found after searching all collections "
#         f"in the STAC catalog rooted at IPNS '{root_catalog_ipns}'."
#     )


# def _get_host(uri: str = "/api/v0"):
#     """Parse the ipfs api host address from `IPFS_HOST` environment variable.
#     If not found, use localhost:5001/api/v0.

#     Args:
#         uri (str): the uri where ipfs gateway api listens

#     Returns:
#         str: ipfs gateway url

#     """

#     host_from_env = os.getenv("IPFS_HOST")
#     return host_from_env + uri if host_from_env else DEFAULT_HOST


# def _get_single_metadata(ipfs_hash: str) -> dict:
#     """Get metadata for given ipfs hash over ipld

#     Args:
#         ipfs_hash (str): ipfs hash for which to get metadata

#     Returns:
#         dict: dict of metadata for hash
#     """

#     r = requests.get(f"{_get_host()}/ipfs/{ipfs_hash}")
#     r.raise_for_status()
#     return r.json()


# def _get_previous_hash_from_metadata(metadata: dict) -> typing.Optional[str]:
#     """Pull in last updated hash from STAC metadata

#     Args:
#         metadata (dict): STAC metadata

#     Returns:
#         str: Previous hash, or None if given root metadata
#     """
#     links = metadata["links"]
#     try:
#         link_to_previous = [
#             link for link in links if link["rel"] in {"prev", "previous"}
#         ][0]
#     except IndexError:
#         return None
#     return link_to_previous["metadata href"]["/"]


# def _resolve_ipns_name_hash(ipns_name_hash: str) -> str:
#     """Find the latest IPFS hash corresponding to a stable ipns name hash

#     Args:
#         ipfs_name_hash (str): stable IPNS name hash

#     Returns:
#         str: ipfs hash corresponding to this ipns name hash
#     """
#     r = requests.get(f"{_get_host()}/ipns/{ipns_name_hash}", params={"offline": True})
#     r.raise_for_status()
#     return r.json()["Path"].split("/")[-1]


# def update_cache_if_changed(new_data: dict) -> None:
#     """Update the local cache file only if the new data differs from what is cached."""
#     cache_file = os.path.join(os.path.dirname(__file__), "cids.json")
#     try:
#         with open(cache_file, "r") as f:
#             cached_data = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         cached_data = None

#     if cached_data != new_data:
#         with open(cache_file, "w") as f:
#             json.dump(new_data, f)


# def get_ipns_name_hash(ipns_key_str: str) -> str:
#     """Find the latest IPNS name hash corresponding to a string (key)

#     Args:
#         ipfs_key_str (str): a string (key) identifying a dataset

#     Raises:
#         KeyError: raised if no IPNS key string is found in the IPNS keys list

#     Returns:
#         str: ipfsname hash corresponding to the provided string
#     """

#     try:
#         # 1) Try to fetch from endpoint
#         r = requests.get(CID_ENDPOINT, params={"decoder": "json"})
#         r.raise_for_status()
#         json_cid = r.json()  # raises JSONDecodeError if endpoint returns malformed JSON

#         # Update cache only if there is a change
#         update_cache_if_changed(json_cid)

#         for entry in json_cid:
#             if entry == ipns_key_str:
#                 return json_cid[entry]

#     except (requests.RequestException, KeyError, json.JSONDecodeError):
#         # 2) If remote fails or is malformed, try local fallback
#         cache_file = os.path.join(os.path.dirname(__file__), "cids.json")
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r") as f:
#                     json_cid = json.load(
#                         f
#                     )  # <-- can raise JSONDecodeError if file is empty/corrupt
#                 for entry in json_cid:
#                     if entry == ipns_key_str:
#                         return json_cid[entry]
#             except (KeyError, json.JSONDecodeError) as err:
#                 # We tried local, but it’s also invalid (bad JSON or missing key)
#                 raise DatasetNotFoundError("Invalid dataset name") from err

#     # 3) If we get here, local file either doesn't exist or didn't have the key
#     raise DatasetNotFoundError("Invalid dataset name") from None


# def _get_relevant_metadata(ipfs_head_hash: str, as_of: datetime.datetime) -> dict:
#     """Iterates through STAC metadata until metadata generated before as_of is found

#     Args:
#         ipfs_head_hash (str): first hash in chain
#         as_of (datetime.datetime): cutoff date for finding metadata

#     Raises:
#         NoMetadataFoundError: raised if no metadata older than cutoff date is found

#     Returns:
#         dict: relevant metadata
#     """
#     cur_metadata = _get_single_metadata(ipfs_head_hash)
#     while True:
#         time_generated = datetime.datetime.strptime(
#             cur_metadata["properties"]["updated"], "%Y-%m-%dT%H:%M:%SZ"
#         )
#         if time_generated <= as_of:
#             return cur_metadata
#         prev_hash = _get_previous_hash_from_metadata(cur_metadata)
#         if prev_hash is None:
#             raise NoMetadataFoundError(f"No metadata found after as_of: {as_of}")
#         cur_metadata = _get_single_metadata(prev_hash)


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
# def _get_dataset_by_ipfs_cid(
#     ipfs_cid: str,
#     gateway_uri_stem: str | None = None,
#     rpc_uri_stem: str | None = None,
# ) -> xr.Dataset:
#     """
#     Gets an xarray dataset directly from its Zarr root IPFS CID using py-hamt.

#     Args:
#         ipfs_cid (str): The IPFS CID of the Zarr dataset's root node (e.g., HAMT root).
#         gateway_uri_stem (str, optional): Custom IPFS HTTP Gateway URI stem.
#         rpc_uri_stem (str, optional): Custom IPFS RPC API URI stem.

#     Returns:
#         xr.Dataset: The loaded dataset.

#     Raises:
#         IpfsConnectionError: If connection to IPFS fails during loading.
#         Exception: Other errors during Zarr parsing or IPFS interaction.
#     """
#     if not ipfs_cid:
#         raise ValueError("IPFS CID cannot be empty.")

#     logger.info(f"Loading Zarr dataset from IPFS CID: {ipfs_cid}")
#     ipfs_store = _get_ipfs_store(gateway_uri_stem, rpc_uri_stem)

#     try:
#         # root_node_id expects a CID object or string representation
#         cid_obj = None
#         try:
#             cid_obj = CID.decode(ipfs_cid)
#         except Exception as decode_err:
#             raise ValueError(
#                 f"Invalid IPFS CID format: {ipfs_cid}. Error: {decode_err}"
#             ) from decode_err

#         # Initialize HAMT store
#         hamt_store = HAMT(store=ipfs_store, root_node_id=cid_obj, read_only=True)

#         # Wrap with ZarrHAMTStore adapter
#         zarr_hamt__store = ZarrHAMTStore(hamt_store, read_only=True)

#         # Ensure registered codecs (like encryption) are available if needed
#         # zarr.registry.get_codec(...) or ensure they are registered beforehand

#         # consolidated=False is typical for HAMT stores, but could be True if explicitly created that way
#         ds = xr.open_zarr(store=zarr_hamt__store, chunks=None)
#         logger.info(f"Successfully loaded dataset from CID: {ipfs_cid}")
#         return ds
#     except (requests.exceptions.RequestException, IpfsConnectionError) as e:
#         # Catch connection/network errors during loading
#         if (
#             "Connection refused" in str(e)
#             or "Max retries exceeded" in str(e)
#             or "Timeout" in str(e)
#         ):
#             raise IpfsConnectionError(
#                 f"IPFS connection failed while loading dataset from CID {ipfs_cid}. "
#                 f"Gateway: {ipfs_store.gateway_uri_stem}, RPC: {ipfs_store.rpc_uri_stem}. Details: {e}"
#             ) from e
#         else:
#             # Other network errors
#             raise RuntimeError(
#                 f"Network error loading Zarr dataset from IPFS CID {ipfs_cid}: {e}"
#             ) from e
#     except FileNotFoundError as e:
#         # xarray/zarr raises FileNotFoundError if root metadata (.zgroup, .zarray) is missing
#         raise StacCatalogError(
#             f"Zarr metadata not found at CID {ipfs_cid}. Is it a valid Zarr root? Error: {e}"
#         ) from e
#     except ValueError:
#         # Let ValueErrors propagate, e.g. from invalid CID format
#         raise
#     except Exception as e:
#         # Catch other potential errors (e.g., Zarr format errors, py-hamt errors)
#         logger.error(
#             f"Failed to load Zarr dataset from IPFS CID {ipfs_cid}: {type(e).__name__}: {e}",
#             exc_info=True,
#         )
#         raise RuntimeError(
#             f"Failed to load Zarr dataset from IPFS CID {ipfs_cid}"
#         ) from e


# def get_metadata_by_key(key: str) -> dict:
#     """Get STAC metadata for specific dataset

#     Args:
#         key (str): dataset key

#     Returns:
#         dict: STAC metadata corresponding to key
#     """
#     ipns_name = get_ipns_name_hash(key)
#     ipfs_hash = _resolve_ipns_name_hash(ipns_name)
#     return _get_single_metadata(ipfs_hash)


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
# async def list_datasets(
#     root_catalog_ipns: str | None = None,
#     gateway_uri_stem: str | None = None,
#     rpc_uri_stem: str | None = None,
# ) -> typing.List[str]:
#     """
#     Lists available dataset IDs by traversing the STAC catalog starting from IPNS.
#     Handles IPLD link format `{"/": "cid_string"}`.
#     Also populates the internal HAMT CID cache as a side effect.
#     """
#     if root_catalog_ipns is None:
#         # Avoid circular import, get constant dynamically if needed
#         from .client import DCLIMATE_STAC_CATALOG_IPNS

#         root_catalog_ipns = DCLIMATE_STAC_CATALOG_IPNS
#         if not root_catalog_ipns:
#             raise ValueError("Root catalog IPNS name is not defined.")

#     logger.info(
#         f"Listing datasets by traversing STAC catalog from IPNS: {root_catalog_ipns}"
#     )
#     ipfs_store = _get_ipfs_store(gateway_uri_stem, rpc_uri_stem)
#     dataset_ids = set()

#     # *** Use fetch_json_from_ipns for the root catalog ***
#     try:
#         catalog = fetch_json_from_ipns(
#             root_catalog_ipns, gateway_uri_stem=gateway_uri_stem
#         )
#     except (IpfsConnectionError, StacCatalogError, ValueError) as e:
#         # Make error message slightly more specific for listing context
#         raise StacCatalogError(
#             f"Failed to fetch or parse root catalog from IPNS '{root_catalog_ipns}' for listing datasets: {e}"
#         ) from e

#     if not isinstance(catalog, dict) or catalog.get("type") != "Catalog":
#         raise StacCatalogError(
#             f"Invalid root catalog format fetched from IPNS {root_catalog_ipns} for listing. Type: {catalog.get('type')}"
#         )

#     collections_to_visit = []
#     for link in catalog.get("links", []):
#         if link.get("rel") == "child" and link.get("type") == "application/json":
#             href_obj = link.get("href")
#             # *** MODIFIED: Handle dict href for IPLD links ***
#             if isinstance(href_obj, dict):
#                 collection_cid_str = href_obj.get("/")
#                 if isinstance(collection_cid_str, str):
#                     collections_to_visit.append(collection_cid_str)
#                 else:
#                     logger.warning(
#                         f"Skipping child link with invalid href dict content in root catalog during list: {link}"
#                     )
#             # Add handling for legacy string format if necessary/desired
#             # elif isinstance(href_obj, str) and href_obj.startswith("/ipfs/"): ...
#             else:
#                 logger.warning(
#                     f"Skipping invalid child link format in root catalog during list: {link}"
#                 )

#     if not collections_to_visit:
#         logger.warning(
#             f"No child collection links found in root catalog {root_catalog_ipns} during list."
#         )
#         return []  # Return empty list if no collections to check

#     logger.info(f"Found {len(collections_to_visit)} collections to scan for items.")

#     for collection_cid in collections_to_visit:
#         try:
#             # *** Use fetch_json_from_cid with the extracted CID string ***
#             collection = await fetch_json_from_cid(collection_cid, ipfs_store)

#             if (
#                 not isinstance(collection, dict)
#                 or collection.get("type") != "Collection"
#             ):
#                 logger.warning(
#                     f"Skipping invalid collection format during list (CID: {collection_cid}). Type: {collection.get('type')}"
#                 )
#                 continue

#             for link in collection.get("links", []):
#                 if link.get("rel") == "item" and link.get("type") == "application/json":
#                     item_href_obj = link.get("href")
#                     item_cid = None

#                     # *** MODIFIED: Handle dict href for item links ***
#                     if isinstance(item_href_obj, dict):
#                         item_cid = item_href_obj.get("/")
#                     # Add handling for legacy string format if necessary/desired
#                     # elif isinstance(item_href_obj, str) and item_href_obj.startswith("/ipfs/"): ...
#                     else:
#                         logger.warning(
#                             f"Skipping invalid item link format in collection {collection_cid} during list: {link}"
#                         )
#                         continue

#                     if isinstance(item_cid, str):
#                         try:
#                             # Fetch item JSON *only* to get the ID and cache the HAMT CID
#                             # Avoids deeper processing if only listing
#                             item = await fetch_json_from_cid(item_cid, ipfs_store)

#                             if (
#                                 not isinstance(item, dict)
#                                 or item.get("type") != "Feature"
#                             ):
#                                 logger.warning(
#                                     f"Skipping invalid item format during list (CID: {item_cid}). Type: {item.get('type')}"
#                                 )
#                                 continue

#                             item_id = item.get("id")
#                             if isinstance(item_id, str) and item_id:
#                                 dataset_ids.add(item_id)  # Add valid ID to our set

#                                 # Populate cache as a side effect
#                                 hamt_asset = item.get("assets", {}).get("hamt-zarr", {})
#                                 hamt_cid_href = hamt_asset.get("href")
#                                 if (
#                                     item_id not in _stac_hamt_cid_cache
#                                     and isinstance(hamt_cid_href, str)
#                                     and hamt_cid_href.startswith("/ipfs/")
#                                 ):
#                                     hamt_cid_str = hamt_cid_href[6:]
#                                     _stac_hamt_cid_cache[item_id] = hamt_cid_str
#                                     logger.debug(
#                                         f"Cached HAMT CID {hamt_cid_str} for {item_id} during list."
#                                     )
#                             else:
#                                 logger.warning(
#                                     f"Item {item_cid} has missing or invalid 'id'. Skipping."
#                                 )

#                         except (StacCatalogError, IpfsConnectionError) as item_err:
#                             # Log and skip this specific item if fetching/parsing fails
#                             logger.warning(
#                                 f"Skipping item {item_cid} during list due to error: {item_err}"
#                             )
#                         except Exception as item_err:  # Catch unexpected errors
#                             logger.warning(
#                                 f"Skipping item {item_cid} during list due to unexpected error: {type(item_err).__name__}: {item_err}"
#                             )
#                     # else: Invalid item CID extracted, already logged warning

#         # 1️⃣  propagate a StacCatalogError that bubbled up from the **target item**
#         except StacCatalogError as col_err:
#             raise col_err

#         # 2️⃣  still swallow IPFS/network problems so that other collections can be tried
#         except IpfsConnectionError as col_err:
#             logger.error(
#                 f"IPFS error processing collection {collection_cid}, continuing search: "
#                 f"{col_err}"
#             )
#         except Exception as col_err:  # Catch unexpected errors
#             logger.warning(
#                 f"Skipping collection {collection_cid} during list due to unexpected error: {type(col_err).__name__}: {col_err}"
#             )

#     found_datasets = sorted(list(dataset_ids))
#     logger.info(
#         f"Finished listing datasets. Found {len(found_datasets)} unique dataset IDs."
#     )
#     return found_datasets
