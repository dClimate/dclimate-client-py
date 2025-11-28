"""
Dataset catalog and resolution logic for dClimate datasets.

This module provides a structured catalog of available datasets and functions
to resolve dataset requests to IPFS CIDs. Similar to dclimate-client-js datasets.ts.
"""

import typing
from typing import TypedDict, List, Optional
import logging
import requests

from .dclimate_zarr_errors import (
    DatasetNotFoundError,
    InvalidSelectionError,
    VariantNotFoundError,
    CollectionNotFoundError,
    IpfsConnectionError,
)

logger = logging.getLogger(__name__)


# --- Type Definitions ---

hydrogen_endpoint = "https://dclimate-ceramic.duckdns.org/api/datasets";


class DatasetVariantConfig(TypedDict, total=False):
    """Configuration for a single dataset variant."""
    variant: str
    cid: Optional[str]  # Direct IPFS CID
    url: Optional[str]  # API endpoint that returns CID (for future use)
    concat_priority: Optional[int]  # Lower number = higher priority for concatenation
    concat_dimension: Optional[str]  # Dimension to concatenate along (default: "time")


class CatalogDataset(TypedDict):
    """A dataset with its variants."""
    dataset: str
    variants: List[DatasetVariantConfig]


class CatalogCollection(TypedDict):
    """A collection of related datasets."""
    collection: str
    datasets: List[CatalogDataset]


# Type alias for the entire catalog
DatasetCatalog = List[CatalogCollection]


class ResolvedDatasetSource(TypedDict):
    """Resolved dataset information."""
    collection: str
    dataset: str
    variant: str
    slug: str  # Full dataset identifier (e.g., "era5/temp2m/finalized")
    cid: Optional[str]
    url: Optional[str]


class UrlFetchResult(TypedDict, total=False):
    """Result from fetching CID from URL endpoint."""
    cid: str
    dataset: Optional[str]
    timestamp: Optional[int]  # Unix timestamp in milliseconds


class DatasetMetadata(TypedDict, total=False):
    """Metadata about a loaded dataset."""
    collection: str
    dataset: str
    variant: str
    slug: str  # Full dataset identifier (e.g., "era5/temp2m/finalized")
    cid: str  # The actual CID used to load the dataset
    url: Optional[str]  # URL if one was used in the resolution
    timestamp: Optional[int]  # Unix timestamp in milliseconds when dataset was last updated
    source: typing.Literal["catalog", "stac", "direct_cid"]  # How the dataset was loaded
    organization: Optional[str]

# --- Helper Functions ---
