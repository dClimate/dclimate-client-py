"""
Functions that will map to endpoints in the flask app
"""
from .loaders.era5 import ERA5Loader
from .geotemporal_data import GeotemporalData
import xarray as xr
# from .registry import get_cid_from_registry
from py_hamt import KuboCAS
from multiformats import CID
import json


# Mapping of collections to their specific loader classes
LOADER_STRATEGIES = {
    "era5": ERA5Loader,
    # Add other collections here, e.g., "cpc": CPCLoader
}

DCLIMATE_GATEWAY_URL = "https://ipfs-gateway.dclimate.net/"


async def dclimate_dataset_loader(collection: str, dataset: str, options: dict = None, gateway_url: str = DCLIMATE_GATEWAY_URL) -> xr.Dataset:
    """
    Loads a dClimate dataset by dynamically fetching the STAC catalog CID
    from an on-chain registry.

    Args:
        collection (str): The name of the data collection (e.g., "era5").
        dataset (str): The name of the dataset (e.g., "2m_temperature").
        options (dict, optional): Options for loading, e.g., {'finalized_only': True}.

    Returns:
        xr.Dataset: The loaded Xarray dataset.
    """
    options = options or {}

    # Select the appropriate loading strategy
    loader_class = LOADER_STRATEGIES.get(collection)
    loader = loader_class(
        gateway_url=gateway_url,
    )

    return await loader.load(dataset=dataset, options=options)

async def get_geo_temporal_dataset(
    collection: str,
    dataset: str,
    options: dict = None,
    gateway_url: str = DCLIMATE_GATEWAY_URL
) -> GeotemporalData:
    """
    Loads a geo-temporal dataset from dClimate by dynamically fetching the STAC catalog CID
    from an on-chain registry.
    """
    options = options or {}

    # Select the appropriate loading strategy
    loader_class = LOADER_STRATEGIES.get(collection)
    loader = loader_class(
        gateway_url=gateway_url,
    )

    ds = await loader.load(dataset=dataset, options=options)
    return GeotemporalData(ds, dataset_name=dataset)



async def _load_stac_json_from_cid(cid: str) -> dict:
    """Loads a STAC JSON from a given CID."""
    async with KuboCAS(gateway_base_url=DCLIMATE_GATEWAY_URL) as kubo_cas:
        try:
            item_bytes = await kubo_cas.load(CID.decode(cid))
            item = json.loads(item_bytes)
            return item
        except Exception as e:
            print(f"Error loading STAC JSON from CID {cid}: {e}")
            raise

# Get all datasets in the catalog
# async def get_all_datasets(collection: str, gateway_url: str = DCLIMATE_GATEWAY_URL) -> list:
#     """
#     Fetches all datasets available in a specific collection from dClimate.

#     Args:
#         collection (str): The name of the data collection (e.g., "era5").
#         gateway_url (str): The IPFS gateway URL to use for fetching data.

#     Returns:
#         list: A list of dataset names available in the specified collection.
#     """
#     stac_root_cid = get_cid_from_registry()

#     stac_json = await _load_stac_json_from_cid(stac_root_cid)
#     if not stac_json:
#         raise ValueError(f"No STAC JSON found for collection: {collection}")
    
#     collection_cid = None

#     # Capitalize all letters in collection
#     collection = collection.upper()

#     # The first links that are of rel child are all the collections
#     for link in stac_json.get("links", []):
#         if link.get("rel") == "child" and link.get("title") == collection:
#             collection_cid = link.get("href")["/"]
#             break

#     if not collection_cid:
#         raise ValueError(f"No datasets found for collection: {collection}")

#     # Get the stac for the collection
#     collection_stac = await _load_stac_json_from_cid(collection_cid)
#     if not collection_stac:
#         raise ValueError(f"No STAC JSON found for collection CID: {collection_cid}")
#     # Extract dataset names from the collection STAC
#     datasets = []
#     for item in collection_stac.get("links", []):
#         if item.get("rel") == "child":
#             datasets.append(item.get("title"))
#         if item.get("rel") == "item":
#             datasets.append(item.get("title"))
#     return datasets

# async def get_all_collections(gateway_url: str = DCLIMATE_GATEWAY_URL) -> list:
#     """
#     Fetches all collections available in the dClimate catalog.
#     """
#     stac_root_cid = get_cid_from_registry()

#     stac_json = await _load_stac_json_from_cid(stac_root_cid)
#     if not stac_json:
#         raise ValueError(f"No STAC JSON found for collection: {collection}")

#     collections = []
#     for link in stac_json.get("links", []):
#         if link.get("rel") == "child":
#             collections.append(link.get("title"))

#     return collections