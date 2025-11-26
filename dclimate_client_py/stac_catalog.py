"""
STAC Catalog Integration for dClimate

This module provides integration with STAC (SpatioTemporal Asset Catalog) format
for discovering and accessing dClimate datasets stored on IPFS.
"""

from typing import Optional, Dict, List
import requests
import pystac
import time


def get_root_catalog_cid() -> str:
    """
    Get the root STAC catalog CID.

    Fetches the latest catalog CID from the dClimate IPFS gateway API.

    Returns:
        str: The IPFS CID of the root STAC catalog

    Raises:
        requests.HTTPError: If the API request fails
        KeyError: If the response doesn't contain the expected 'cid' field
    """
    url = "https://ipfs-gateway.dclimate.net/stac"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data["cid"]


class IPFSStacIO(pystac.StacIO):
    """
    Custom StacIO implementation that resolves ipfs:// URIs via HTTP gateway.

    This allows pystac to transparently load STAC catalogs, collections, and items
    that are stored on IPFS and referenced using ipfs:// protocol URIs.
    """

    def __init__(self, gateway_url: str):
        """
        Initialize the IPFS STAC I/O handler.

        Args:
            gateway_url: Base URL of the IPFS HTTP gateway (e.g., 'https://ipfs-gateway.dclimate.net')
        """
        self.gateway_url = gateway_url.rstrip('/')

    def read_text(self, source: str, *args, **kwargs) -> str:
        """
        Read text content from a source URI.

        If the source starts with 'ipfs://', resolves it via the HTTP gateway.
        Otherwise, delegates to the default StacIO implementation.

        Args:
            source: URI to read from (e.g., 'ipfs://bafkrei...' or 'https://...')

        Returns:
            str: The text content

        Raises:
            requests.HTTPError: If the HTTP request fails
        """
        if source.startswith("ipfs://"):
            cid = source.replace("ipfs://", "")
            url = f"{self.gateway_url}/ipfs/{cid}"
            response = requests.get(url)
            response.raise_for_status()
            return response.text

        # Fall back to default behavior for HTTP/HTTPS URLs
        return super().read_text(source, *args, **kwargs)

    def write_text(self, dest: str, txt: str, *args, **kwargs) -> None:
        """
        Write text content is not supported for IPFS.

        Raises:
            NotImplementedError: Always, as IPFS is read-only in this context
        """
        raise NotImplementedError("Writing to IPFS is not supported via StacIO")


def load_stac_catalog(
    gateway_url: str,
    root_cid: Optional[str] = None
) -> pystac.Catalog:
    """
    Load the dClimate STAC catalog from IPFS.

    Args:
        gateway_url: Base URL of the IPFS HTTP gateway
        root_cid: Optional IPFS CID of the root catalog. If None, fetches via get_root_catalog_cid()

    Returns:
        pystac.Catalog: The loaded STAC catalog with all links and references

    Raises:
        requests.HTTPError: If fetching from IPFS fails
        pystac.STACError: If the catalog structure is invalid
    """
    if root_cid is None:
        root_cid = get_root_catalog_cid()

    # Set up custom IPFS I/O handler
    stac_io = IPFSStacIO(gateway_url)
    pystac.StacIO.set_default(lambda: stac_io)

    # Load the root catalog
    catalog_uri = f"ipfs://{root_cid}"
    catalog = pystac.Catalog.from_file(catalog_uri)

    return catalog


def resolve_dataset_cid_from_stac(
    catalog: pystac.Catalog,
    collection: str,
    dataset: str,
    variant: Optional[str] = None
) -> str:
    """
    Resolve a dataset to its IPFS CID by querying the STAC catalog.

    This function navigates the STAC catalog structure to find the specific dataset variant
    and extracts the Zarr data CID from the STAC Item's assets.

    Args:
        catalog: The loaded STAC catalog
        collection: Collection ID (e.g., 'ifs', 'era5', 'aifs')
        dataset: Dataset name (e.g., 'temperature', 'precipitation')
        variant: Optional variant name (e.g., 'single', 'ensemble'). Required for multi-variant datasets

    Returns:
        str: The IPFS CID of the Zarr dataset (without 'ipfs://' prefix)

    Raises:
        ValueError: If collection, dataset, or variant is not found in the catalog
    """
    # Find the collection by dclimate:id
    collection_obj = None
    for link in catalog.get_child_links():
        if link.extra_fields.get("dclimate:id") == collection:
            collection_obj = link.resolve_stac_object(root=catalog).target
            break

    if collection_obj is None:
        raise ValueError(f"Collection '{collection}' not found in STAC catalog")

    # Find the item matching dataset and variant
    # STAC item IDs follow pattern: "{collection}-{dataset}-{variant}"
    # e.g., "ifs-temperature-single"
    for item in collection_obj.get_items():
        # Parse the item ID
        parts = item.id.split("-")

        if len(parts) < 2:
            continue

        # Extract dataset and variant from item ID
        # Format: collection-dataset or collection-dataset-variant
        item_collection = parts[0]
        item_dataset = parts[1]
        item_variant = parts[2] if len(parts) > 2 else None

        # Match dataset
        if item_dataset != dataset:
            continue

        # Match variant (if specified)
        if variant is not None:
            if item_variant != variant:
                continue

        # Found the matching item - extract CID from data asset
        if "data" in item.assets:
            href = item.assets["data"].href
            # Remove ipfs:// prefix if present
            if href.startswith("ipfs://"):
                return href.replace("ipfs://", "")
            return href
        else:
            raise ValueError(f"Item '{item.id}' does not have a 'data' asset")

    # If we get here, no matching item was found
    if variant:
        raise ValueError(
            f"Dataset '{dataset}' with variant '{variant}' not found in collection '{collection}'"
        )
    else:
        raise ValueError(
            f"Dataset '{dataset}' not found in collection '{collection}'"
        )


def list_available_datasets(catalog: pystac.Catalog) -> Dict[str, Dict[str, any]]:
    """
    List all available datasets from the STAC catalog.

    Returns a dictionary mapping collection IDs to their metadata, including
    the dataset types available in each collection.

    Args:
        catalog: The loaded STAC catalog

    Returns:
        dict: Dictionary with structure:
            {
                "collection_id": {
                    "id": "collection_id",
                    "title": "Collection Title",
                    "types": ["dataset_type1", "dataset_type2", ...]
                },
                ...
            }

    Example:
        {
            "ifs": {
                "id": "ifs",
                "title": "Integrated Forecasting System (IFS)",
                "types": ["temperature", "precipitation", "wind_u", ...]
            },
            "era5": {
                "id": "era5",
                "title": "ERA5 Reanalysis",
                "types": ["2m_temperature", "total_precipitation", ...]
            }
        }
    """
    result = {}

    for link in catalog.get_child_links():
        collection_id = link.extra_fields.get("dclimate:id")

        if not collection_id:
            continue

        # Get dataset types from the collection link's extra fields
        types = link.extra_fields.get("dclimate:types", [])

        result[collection_id] = {
            "id": collection_id,
            "title": link.title or collection_id,
            "types": types
        }

    return result
