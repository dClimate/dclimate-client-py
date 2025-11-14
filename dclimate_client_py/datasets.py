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
    source: typing.Literal["catalog", "direct_cid"]  # How the dataset was loaded


# --- Internal Dataset Catalog ---

DATASET_CATALOG_INTERNAL: DatasetCatalog = [
    {
        "collection": "era5",
        "datasets": [
            {
                "dataset": "2m_temperature",
                "variants": [
                    {
                        "variant": "finalized",
                        "cid": "bafyr4iacuutc5bgmirkfyzn4igi2wys7e42kkn674hx3c4dv4wrgjp2k2u",
                        # "concat_priority": 1,
                        # "concat_dimension": "time",
                    },
                    {
                        "variant": "non-finalized",
                        "cid": "bafyr4ihicmzx4uw4pefk7idba3mz5r5g27au3l7d62yj4gguxx6neaa5ti",
                        # "concat_priority": 2,
                        # "concat_dimension": "time",
                    },
                    ],
            },
            {
                "dataset": "total_precipitation",
                "variants": [
                {
                    "variant": "finalized",
                    "cid": "bafyr4icium3zr6dyewfzkcwpnsb77nmxeblaomdk3kz3f2wz2rqq3i2yfi",
                    # "concatPriority": 1,
                    # "concatDimension": "time",
                },
                {
                    "variant": "non-finalized",
                    "cid": "bafyr4ifh3khz7f2mj6subudsbri7wfbna7s2iw5inn2wvbhkgms7k6n6ly",
                    # "concatPriority": 2,
                    # "concatDimension": "time",
                },
                ],
            },
            {
                "dataset": "10m_u_wind",
                "variants": [
                {
                    "variant": "finalized",
                    "cid": "bafyr4ih6kgfr2pgucs6cgxbyboayqrejv7wbsbcv23ldr7zqbtfhdhniwa",
                    # "concatPriority": 1,
                    # "concatDimension": "time",
                },
                {
                    "variant": "non-finalized",
                    "cid": "bafyr4ihaevzkwj6ozhbwcpg6h3cacfa2voa4ezhsdlcfnshu7wccutup24",
                    # "concatPriority": 2,
                    # "concatDimension": "time",
                },
                ],
            },
            {
            "dataset": "10m_v_wind",
            "variants": [
                {
                    "variant": "finalized",
                    "cid": "bafyr4igqxykzgn7ueyuxnyupb42bgav3o2v6ikarwyhlxisknypeyfjz5q",
                    "concatPriority": 1,
                    "concatDimension": "time",
                },
                {
                    "variant": "non-finalized",
                    "cid": "bafyr4ih5y3nkxdycxjzqhapynjdzbuj56fo4n3apdlcvqhgnggojk22ca4",
                    "concatPriority": 2,
                    "concatDimension": "time",
                },
                ],
            },
            {
            "dataset": "surface_solar_radiation",
            "variants": [
                {
                    "variant": "finalized",
                    "cid": "bafyr4ico6t4t2ztxbniigqmiy2rfbmhxpoge56oae3afqwxwwdw3ou4qya",
                    "concatPriority": 1,
                    "concatDimension": "time",
                },
                {
                    "variant": "non-finalized",
                    "cid": "bafyr4iaqdlk2ircn72rlaigrb6hufgavcxsqrjvoywokgz25ctel3btqzu",
                    "concatPriority": 2,
                    "concatDimension": "time",
                },
                ],
            },
            {
            "dataset": "land_total_precipitation",
            "variants": [
                {
                    "variant": "finalized",
                    "cid": "bafyr4ifqx5pq4zwv6tvusndvwm5h3ic2l3wewjroilfeeor55yvzriah5a",
                },
                ],
            },
        ],
    },
    {
        "collection": "aifs",
        "datasets": [
        {
            "dataset": "precipitation",
            "variants": [
            { "variant": "single", "url": f"{hydrogen_endpoint}/aifs-single-precip" },
            { "variant": "ensemble", "url": f"{hydrogen_endpoint}/aifs-ensemble-precip" },
            ],
        },
        {
            "dataset": "temperature",
            "variants": [
            { "variant": "single", "url": f"{hydrogen_endpoint}/aifs-single-temperature" },
            { "variant": "ensemble", "url": f"{hydrogen_endpoint}/aifs-ensemble-temperature" },
            ],
        },
        {
            "dataset": "wind_u",
            "variants": [
            { "variant": "single", "url": f"{hydrogen_endpoint}/aifs-single-wind-u" },
            { "variant": "ensemble", "url": f"{hydrogen_endpoint}/aifs-ensemble-wind-u" },
            ],
        },
        {
            "dataset": "wind_v",
            "variants": [
            { "variant": "single", "url": f"{hydrogen_endpoint}/aifs-single-wind-v" },
            { "variant": "ensemble", "url": f"{hydrogen_endpoint}/aifs-ensemble-wind-v" },
            ],
        },
        {
            "dataset": "solar_radiation",
            "variants": [
            { "variant": "single", "url": f"{hydrogen_endpoint}/aifs-single-solar-radiation" },
            { "variant": "ensemble", "url": f"{hydrogen_endpoint}/aifs-ensemble-solar-radiation" },
            ],
        },
        ],
    },
    {
        "collection": "copernicus",
        "datasets": [
        {
            "dataset": "fpar",
            "variants": [
            {
                "variant": "default",
                "cid": "bafyr4iatibj6bk3mvjec5be6ffnxsxde63yekxfhgym4yxgrxoifll6eda",
            },
            ],
        },
        ],
    },
    {
        "collection": "ifs",
        "datasets": [
        {
            "dataset": "precipitation",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/ifs-precip" }],
        },
        {
            "dataset": "temperature",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/ifs-temperature" }],
        },
        {
            "dataset": "wind_u",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/ifs-wind-u" }],
        },
        {
            "dataset": "wind_v",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/ifs-wind-v" }],
        },
        {
            "dataset": "soil_moisture_l3",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/ifs-soil-moisture-l3" }],
        },
        {
            "dataset": "solar_radiation",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/ifs-solar-radiation" }],
        },
        ],
    },
    {
        "collection": "prism",
        "datasets": [
        {
            "dataset": "precipitation_800m",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/prism-precip-800m" }],
        },
        {
            "dataset": "tmax_800m",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/prism-tmax-800m" }],
        },
        ],
    },
    {
        "collection": "gfs",
        "datasets": [
        {
            "dataset": "precipitation_rate",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/gfs-precipitation-rate" }],
        },
        {
            "dataset": "precipitation_total",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/gfs-precipitation-total" }],
        },
        {
            "dataset": "temperature_max",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/gfs-max-temperature" }],
        },
        {
            "dataset": "temperature_min",
            "variants": [{ "variant": "default", "url": f"{hydrogen_endpoint}/gfs-min-temperature" }],
        },
        ],
    },
]


# --- Helper Functions ---


def normalize_key(key: str) -> str:
    """
    Normalize a key for case-insensitive, whitespace-tolerant matching.

    Args:
        key: The key to normalize

    Returns:
        Normalized key (lowercase, trimmed, spaces/hyphens replaced with underscores)
    """
    if not isinstance(key, str):
        return ""
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def find_collection_by_name(
    catalog: DatasetCatalog, collection_name: str
) -> Optional[CatalogCollection]:
    """
    Find a collection in the catalog by name (case-insensitive).

    Args:
        catalog: The dataset catalog to search
        collection_name: Name of the collection to find

    Returns:
        The matching collection or None if not found
    """
    normalized = normalize_key(collection_name)
    for collection in catalog:
        if normalize_key(collection["collection"]) == normalized:
            return collection
    return None


def find_dataset_by_name(
    collection: CatalogCollection, dataset_name: str
) -> Optional[CatalogDataset]:
    """
    Find a dataset within a collection by name (case-insensitive).

    Args:
        collection: The collection to search
        dataset_name: Name of the dataset to find

    Returns:
        The matching dataset or None if not found
    """
    normalized = normalize_key(dataset_name)
    for dataset in collection["datasets"]:
        if normalize_key(dataset["dataset"]) == normalized:
            return dataset
    return None


def find_variant_by_name(
    dataset: CatalogDataset, variant_name: str
) -> Optional[DatasetVariantConfig]:
    """
    Find a variant within a dataset by name (case-insensitive).

    Args:
        dataset: The dataset to search
        variant_name: Name of the variant to find

    Returns:
        The matching variant or None if not found
    """
    normalized = normalize_key(variant_name)
    for variant in dataset["variants"]:
        if normalize_key(variant["variant"]) == normalized:
            return variant
    return None


def get_concatenable_variants(
    dataset: CatalogDataset,
) -> List[DatasetVariantConfig]:
    """
    Get variants that can be auto-concatenated, sorted by priority.

    Note: This function is kept for future use when xarray's lazy concatenation
    is fully supported. Currently, auto-concatenation is disabled.

    Args:
        dataset: The dataset to check

    Returns:
        List of variants with concat_priority defined, sorted by priority (ascending)
    """
    concatenable = [
        variant for variant in dataset["variants"]
        if "concat_priority" in variant and variant["concat_priority"] is not None
    ]
    # Sort by concat_priority (lower number = higher priority)
    concatenable.sort(key=lambda v: v["concat_priority"])
    return concatenable


def find_collection_for_dataset(
    catalog: DatasetCatalog, dataset_name: str
) -> Optional[CatalogCollection]:
    """
    Auto-detect which collection contains a dataset.

    Args:
        catalog: The dataset catalog to search
        dataset_name: Name of the dataset to find

    Returns:
        The collection containing the dataset, or None if not found or ambiguous
    """
    normalized = normalize_key(dataset_name)
    matches = []

    for collection in catalog:
        for dataset in collection["datasets"]:
            if normalize_key(dataset["dataset"]) == normalized:
                matches.append(collection)
                break  # Don't add same collection twice

    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches[0]
    else:
        # Ambiguous - dataset exists in multiple collections
        logger.warning(
            f"Dataset '{dataset_name}' found in multiple collections. "
            f"Please specify collection explicitly."
        )
        return None


def resolve_dataset_source(
    dataset_name: str,
    collection_name: Optional[str] = None,
    variant_name: Optional[str] = None,
    catalog: Optional[DatasetCatalog] = None,
) -> ResolvedDatasetSource:
    """
    Resolve a dataset request to a specific variant and its source (CID or URL).

    This function mimics the resolution logic from dclimate-client-js.

    Args:
        dataset_name: Name of the dataset (required)
        collection_name: Name of the collection (optional, will auto-detect if not provided)
        variant_name: Name of the variant (optional, will use default or single variant)
        catalog: Dataset catalog to use (defaults to DATASET_CATALOG_INTERNAL)

    Returns:
        ResolvedDatasetSource with collection, dataset, variant, slug, and source info

    Raises:
        DatasetNotFoundError: If dataset cannot be found
        CollectionNotFoundError: If specified collection doesn't exist
        VariantNotFoundError: If specified variant doesn't exist
        InvalidSelectionError: If selection is ambiguous
    """
    if catalog is None:
        catalog = DATASET_CATALOG_INTERNAL

    # 1. Find the collection
    if collection_name:
        collection = find_collection_by_name(catalog, collection_name)
        if not collection:
            available = [c["collection"] for c in catalog]
            raise CollectionNotFoundError(
                f"Collection '{collection_name}' not found. "
                f"Available collections: {available}"
            )
    else:
        # Auto-detect collection
        collection = find_collection_for_dataset(catalog, dataset_name)
        if not collection:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_name}' not found in any collection. "
                f"Please check available datasets using list_dataset_catalog()."
            )

    # 2. Find the dataset within the collection
    dataset = find_dataset_by_name(collection, dataset_name)
    if not dataset:
        available = [d["dataset"] for d in collection["datasets"]]
        raise DatasetNotFoundError(
            f"Dataset '{dataset_name}' not found in collection '{collection['collection']}'. "
            f"Available datasets: {available}"
        )

    # 3. Find or select the variant
    variant: Optional[DatasetVariantConfig] = None

    if variant_name:
        # Explicit variant requested
        variant = find_variant_by_name(dataset, variant_name)
        if not variant:
            available = [v["variant"] for v in dataset["variants"]]
            raise VariantNotFoundError(
                f"Variant '{variant_name}' not found for dataset '{dataset_name}'. "
                f"Available variants: {available}"
            )
    else:
        # No explicit variant - use selection logic
        if len(dataset["variants"]) == 1:
            # Single variant - use it
            variant = dataset["variants"][0]
        else:
            # Multiple variants - require explicit variant selection
            # Note: Auto-concatenation is disabled due to xarray lazy concat limitations
            available = [v["variant"] for v in dataset["variants"]]
            raise InvalidSelectionError(
                f"Dataset '{dataset_name}' has multiple variants. "
                f"Please specify one: {available}. "
                f"Note: Auto-concatenation is currently disabled due to xarray's "
                f"lazy concatenation not being fully supported."
            )

    # 4. Build the slug
    slug = f"{collection['collection']}/{dataset['dataset']}/{variant['variant']}"

    # 5. Return resolved source
    return ResolvedDatasetSource(
        collection=collection["collection"],
        dataset=dataset["dataset"],
        variant=variant["variant"],
        slug=slug,
        cid=variant.get("cid"),
        url=variant.get("url"),
    )


def fetch_cid_from_url(url: str, timeout: int = 30) -> UrlFetchResult:
    """
    Fetch IPFS CID and metadata from a URL endpoint.

    The endpoint should return JSON with a 'cid' field and optionally 'timestamp' and 'dataset' fields,
    or just the CID as plain text.

    Args:
        url: The URL endpoint to fetch the CID from
        timeout: Request timeout in seconds (default: 30)

    Returns:
        UrlFetchResult with at minimum the CID, and optionally timestamp and dataset name

    Raises:
        IpfsConnectionError: If the request fails or CID cannot be extracted
    """
    try:
        logger.info(f"Fetching CID from URL: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # Try to parse as JSON first
        try:
            data = response.json()
            # Look for 'cid' field in various common formats
            if isinstance(data, dict):
                cid = data.get("cid") or data.get("CID") or data.get("ipfs_cid")
                if cid:
                    logger.info(f"Extracted CID from JSON response: {cid}")
                    result: UrlFetchResult = {"cid": str(cid)}

                    # Extract optional fields
                    if "timestamp" in data:
                        result["timestamp"] = int(data["timestamp"])
                    if "dataset" in data:
                        result["dataset"] = str(data["dataset"])

                    return result
                else:
                    raise IpfsConnectionError(
                        f"No 'cid' field found in JSON response from {url}. "
                        f"Response keys: {list(data.keys())}"
                    )
            elif isinstance(data, str):
                # JSON string containing just the CID
                logger.info(f"Extracted CID from JSON string: {data}")
                return {"cid": data.strip()}
            else:
                raise IpfsConnectionError(
                    f"Unexpected JSON response type from {url}: {type(data)}"
                )
        except (ValueError, requests.exceptions.JSONDecodeError):
            # Not JSON, treat as plain text CID
            cid = response.text.strip()
            if cid:
                logger.info(f"Extracted CID from plain text response: {cid}")
                return {"cid": cid}
            else:
                raise IpfsConnectionError(f"Empty response from URL: {url}")

    except requests.exceptions.Timeout as e:
        raise IpfsConnectionError(
            f"Timeout ({timeout}s) fetching CID from URL: {url}"
        ) from e
    except requests.exceptions.RequestException as e:
        raise IpfsConnectionError(
            f"Failed to fetch CID from URL {url}: {e}"
        ) from e
    except Exception as e:
        raise IpfsConnectionError(
            f"Unexpected error fetching CID from URL {url}: {e}"
        ) from e


def list_dataset_catalog(
    catalog: Optional[DatasetCatalog] = None,
    include_sources: bool = False,
    format: Optional[str] = None,
) -> typing.Union[DatasetCatalog, str]:
    """
    List all available datasets in the catalog.

    Returns a deep copy of the catalog with CIDs and URLs stripped out by default
    for security and cleaner output. Use include_sources=True to get the full catalog.

    Args:
        catalog: Dataset catalog to list (defaults to DATASET_CATALOG_INTERNAL)
        include_sources: If True, include CID and URL information. If False (default),
                        strip out sensitive source information.
        format: Optional output format. Options:
                - None (default): Return as Python dict/list structure
                - "json": Return as formatted JSON string
                - "pretty": Return as human-readable formatted string

    Returns:
        Deep copy of the dataset catalog (dict/list), or formatted string if format is specified
    """
    import copy
    import json

    if catalog is None:
        catalog = DATASET_CATALOG_INTERNAL

    # Create a deep copy to prevent modifications
    catalog_copy = copy.deepcopy(catalog)

    # Strip out CIDs and URLs if requested
    if not include_sources:
        for collection in catalog_copy:
            for dataset in collection["datasets"]:
                for variant in dataset["variants"]:
                    # Remove CID and URL fields
                    variant.pop("cid", None)
                    variant.pop("url", None)

    # Format output if requested
    if format == "json":
        return json.dumps(catalog_copy, indent=2)
    elif format == "pretty":
        # Create a human-readable formatted string
        lines = []
        lines.append("=" * 80)
        lines.append("dClimate Dataset Catalog")
        lines.append("=" * 80)

        for collection in catalog_copy:
            lines.append(f"\n📦 Collection: {collection['collection']}")
            lines.append("-" * 80)

            for dataset in collection["datasets"]:
                lines.append(f"  📊 Dataset: {dataset['dataset']}")

                for variant in dataset["variants"]:
                    variant_name = variant["variant"]
                    lines.append(f"    ├─ Variant: {variant_name}")

                    # Show concatenation metadata if present
                    if "concat_priority" in variant:
                        lines.append(f"    │  ├─ Concat Priority: {variant['concat_priority']}")
                    if "concat_dimension" in variant:
                        lines.append(f"    │  └─ Concat Dimension: {variant['concat_dimension']}")

                    # Show sources if included
                    if include_sources:
                        if "cid" in variant:
                            lines.append(f"    │  └─ CID: {variant['cid']}")
                        if "url" in variant:
                            lines.append(f"    │  └─ URL: {variant['url']}")

                lines.append("")  # Blank line between datasets

        lines.append("=" * 80)
        return "\n".join(lines)

    return catalog_copy
