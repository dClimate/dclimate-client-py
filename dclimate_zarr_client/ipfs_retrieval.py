import datetime
import typing
import os

import requests
import json
import xarray as xr
from py_hamt import HAMT, IPFSStore, IPFSZarr3

from .dclimate_zarr_errors import DatasetNotFoundError, NoMetadataFoundError

DEFAULT_HOST = "http://127.0.0.1:5001/api/v0"
VALID_TIME_SPANS = ["daily", "hourly", "weekly", "quarterly"]
CID_ENDPOINT = "https://raw.githubusercontent.com/dClimate/dclimate-data-cids/refs/heads/main/cids.json"


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


def get_dataset_by_ipns_hash(
    ipns_name_hash: str,
    as_of: typing.Optional[datetime.datetime] = None,
    gateway_uri: str | None = None,
) -> xr.Dataset:
    """Gets xarray dataset using fixed ipns name hash

    Args:
        ipns_name_hash (str): ipns hash that will remain fixed between updates
        as_of (typing.Optional[datetime.datetime], optional): cutoff date for finding metadata. Defaults to None.
            if None, function will return most recent dataset

    Returns:
        xr.Dataset: dataset corresponding to hash and as_of date
    """
    store_kwargs = {}
    if gateway_uri:
        store_kwargs["gateway_uri_stem"] = gateway_uri

    hamt_store = HAMT(
        store=IPFSStore(**store_kwargs), root_node_id=ipns_name_hash, read_only=True
    )

    # Load the HAMT as a IPFSZarr3
    ipfszarr3_store = IPFSZarr3(hamt_store, read_only=True)
    return xr.open_zarr(store=ipfszarr3_store, chunks=None)


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


def list_datasets() -> typing.List[str]:
    """List datasets available on IPFS node

    Returns:
        typing.List[str]: List of available datasets' keys
    """
    # Try to fetch from endpoint first
    try:
        r = requests.get(CID_ENDPOINT, params={"decoder": "json"})
        r.raise_for_status()
        json_cid = r.json()  # may raise JSONDecodeError if malformed

        # Update the local cache if remote data differs
        update_cache_if_changed(json_cid)
        return list(json_cid.keys())

    except (requests.RequestException, json.JSONDecodeError):
        # Fallback to local cache if endpoint is unreachable or JSON is malformed
        cache_file = os.path.join(os.path.dirname(__file__), "cids.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    json_cid = json.load(f)  # can raise JSONDecodeError
                return list(json_cid.keys())
            except json.JSONDecodeError as err:
                # local file is corrupt or empty
                raise RuntimeError(
                    "Failed to retrieve dataset list from endpoint or local cache."
                ) from err

    # If both the endpoint and local file fail, raise an error
    raise RuntimeError(
        "Failed to retrieve dataset list from endpoint or local cache."
    ) from None
