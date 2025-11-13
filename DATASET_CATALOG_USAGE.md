# Dataset Catalog Usage Guide

This document explains how to use the new dataset catalog functionality in dclimate-zarr-client, which provides similar features to dclimate-client-js.

## Overview

The new functionality includes:

1. **`load_dataset()`** - Main entry point for loading datasets from the catalog
2. **`list_dataset_catalog()`** - List all available datasets in the catalog
3. **Dataset catalog structure** - Internal catalog of datasets with collections, variants, and CIDs

**Note:** Auto-concatenation of variants is currently disabled due to xarray's lazy concatenation not being fully supported. Users must explicitly specify a variant for datasets with multiple variants. The catalog maintains `concat_priority` and `concat_dimension` metadata for future use when lazy concatenation becomes available.

## Key Features

### 1. Load Datasets by Name

Instead of manually managing CIDs, you can now load datasets by their logical names:

```python
from dclimate_client_py import load_dataset

# Load a specific variant
ds = load_dataset(
    dataset="temp2m",
    collection="era5",
    variant="finalized"
)
```

### 2. Explicit Variant Selection

For datasets with multiple variants, you must explicitly specify which variant to load:

```python
# Load the finalized variant
ds = load_dataset(
    dataset="temp2m",
    collection="era5",
    variant="finalized"  # Required for multi-variant datasets
)

# Or load the non-finalized variant
ds = load_dataset(
    dataset="temp2m",
    collection="era5",
    variant="non-finalized"
)

# Note: Auto-concatenation is currently disabled due to xarray's
# lazy concatenation not being fully supported
```

### 3. Get Raw xarray.Dataset

If you don't want the GeotemporalData wrapper, you can get the raw xarray Dataset:

```python
# Get raw xarray.Dataset instead of GeotemporalData
xr_ds = load_dataset(
    dataset="temp2m",
    collection="era5",
    variant="finalized",
    return_xarray=True
)
```

### 4. List Available Datasets

Discover what datasets are available in the catalog with multiple output formats:

```python
from dclimate_client_py import list_dataset_catalog

catalog = list_dataset_catalog()

# Catalog structure (CIDs and URLs are stripped by default):
# [
#     {
#         "collection": "era5",
#         "datasets": [
#             {
#                 "dataset": "temp2m",
#                 "variants": [
#                     {
#                         "variant": "finalized",
#                         "concat_priority": 1,
#                         "concat_dimension": "time"
#                     },
#                     {
#                         "variant": "non-finalized",
#                         "concat_priority": 2,
#                         "concat_dimension": "time"
#                     }
#                 ]
#             }
#         ]
#     }
# ]

# To get full catalog with CIDs and URLs:
# catalog_with_sources = list_dataset_catalog(include_sources=True)

# Get catalog as formatted JSON string:
# json_output = list_dataset_catalog(format="json")
# print(json_output)

# Get catalog as pretty-printed string (recommended for display):
pretty_output = list_dataset_catalog(format="pretty")
print(pretty_output)
# Output:
# ================================================================================
# dClimate Dataset Catalog
# ================================================================================
#
# 📦 Collection: era5
# --------------------------------------------------------------------------------
#   📊 Dataset: 2m_temperature
#     ├─ Variant: finalized
#     │  ├─ Concat Priority: 1
#     │  └─ Concat Dimension: time
#     ├─ Variant: non-finalized
#     │  ├─ Concat Priority: 2
#     │  └─ Concat Dimension: time
#
# ...

# Or iterate through the catalog structure:
for collection in catalog:
    print(f"Collection: {collection['collection']}")
    for dataset in collection['datasets']:
        print(f"  Dataset: {dataset['dataset']}")
        for variant in dataset['variants']:
            print(f"    Variant: {variant['variant']}")
```

### 5. Direct CID Loading

You can still load datasets directly by CID, bypassing catalog resolution:

```python
ds = load_dataset(
    dataset="temp2m",  # Used for metadata only
    cid="bafybeibg5o7c3hzj4eyhwvqq4fkzp6rw7gm5vu5f5qvj2p7v5zq2w2y3x4"
)
```

## Complete API Reference

### `load_dataset()`

```python
load_dataset(
    dataset: str,                           # Required: dataset name
    collection: Optional[str] = None,       # Collection name (auto-detected if omitted)
    variant: Optional[str] = None,          # Variant name (or use auto_concatenate)
    cid: Optional[str] = None,              # Direct CID override
    gateway_uri_stem: Optional[str] = None, # Custom IPFS gateway
    rpc_uri_stem: Optional[str] = None,     # Custom IPFS RPC
    return_xarray: bool = False,            # Return xarray.Dataset instead of GeotemporalData
    catalog: Optional[DatasetCatalog] = None # Custom catalog
    # Note: auto_concatenate parameter has been removed
) -> Union[GeotemporalData, xr.Dataset]
```

### `list_dataset_catalog()`

```python
list_dataset_catalog(
    catalog: Optional[DatasetCatalog] = None,
    include_sources: bool = False,
    format: Optional[str] = None  # Options: None, "json", "pretty"
) -> Union[DatasetCatalog, str]
```

Returns a deep copy of the dataset catalog with CIDs and URLs stripped out by default for security and cleaner output.

**Parameters:**
- `include_sources`: If `True`, include CID and URL information (default: `False`)
- `format`: Output format options:
  - `None` (default): Return as Python dict/list structure
  - `"json"`: Return as formatted JSON string
  - `"pretty"`: Return as human-readable formatted string with visual hierarchy

## Concatenation Metadata (For Future Use)

The catalog maintains concatenation metadata for future use when xarray's lazy concatenation is fully supported:

1. **`concat_priority`**: Variants are ordered by priority (lower number = higher priority)
2. **`concat_dimension`**: The dimension along which variants should be concatenated (typically "time")

### Example: ERA5 Temperature Data

```python
# ERA5 temp2m has two variants with concatenation metadata:
# - "finalized": Historical data up to ~5 days ago (concat_priority: 1)
# - "non-finalized": Recent data including last 5 days (concat_priority: 2)

# Currently, you must load each variant separately:
finalized = load_dataset(
    dataset="temp2m",
    collection="era5",
    variant="finalized"
)

non_finalized = load_dataset(
    dataset="temp2m",
    collection="era5",
    variant="non-finalized"
)

# Manual concatenation is possible but may trigger eager loading for large datasets
# Auto-concatenation will be re-enabled when xarray supports lazy concat
```

## Dataset Catalog Structure

The internal catalog (`DATASET_CATALOG_INTERNAL`) contains:

### Current Datasets (Example)

```python
{
    "collection": "era5",
    "datasets": [
        {
            "dataset": "temp2m",
            "variants": [
                {
                    "variant": "finalized",
                    "cid": "bafybei...",
                    "concat_priority": 1,
                    "concat_dimension": "time"
                },
                {
                    "variant": "non-finalized",
                    "cid": "bafybei...",
                    "concat_priority": 2,
                    "concat_dimension": "time"
                }
            ]
        },
        {
            "dataset": "precip",
            "variants": [...]
        }
    ]
}
```

**Note**: The example CIDs in the current catalog are placeholders. You'll need to update [datasets.py](dclimate_client_py/datasets.py) with real CIDs for your datasets.

## Error Handling

The new functions provide helpful error messages:

```python
from dclimate_client_py import load_dataset
from dclimate_client_py.dclimate_zarr_errors import (
    DatasetNotFoundError,
    CollectionNotFoundError,
    VariantNotFoundError,
    InvalidSelectionError
)

try:
    ds = load_dataset(dataset="nonexistent")
except DatasetNotFoundError as e:
    print(f"Dataset not found: {e}")

try:
    ds = load_dataset(
        dataset="temp2m",
        collection="nonexistent"
    )
except CollectionNotFoundError as e:
    print(f"Collection not found: {e}")

try:
    ds = load_dataset(
        dataset="temp2m",
        collection="era5",
        variant="nonexistent"
    )
except VariantNotFoundError as e:
    print(f"Variant not found: {e}")

try:
    # Multi-variant dataset without specifying variant (will raise error)
    ds = load_dataset(
        dataset="temp2m",
        collection="era5"
        # Missing: variant="finalized" or variant="non-finalized"
    )
except InvalidSelectionError as e:
    print(f"Invalid selection: {e}")
    # Error will mention: "Please specify one: ['finalized', 'non-finalized']"
```

### To New API

```python
# NEW: Catalog-based loading with explicit variant selection
from dclimate_client_py import load_dataset

ds = load_dataset(
    dataset="temp2m",
    collection="era5",
    variant="finalized",  # Must specify variant for multi-variant datasets
    gateway_uri_stem="http://localhost:8080"
)
```

## Future Enhancements

The catalog structure supports features that can be added in the future:

1. **Lazy auto-concatenation**: Re-enable auto-concatenation when xarray's lazy concat is fully supported
2. **URL-based variants**: Support for `url` field to fetch CIDs from API endpoints
3. **STAC integration**: Keep STAC support for dynamic dataset discovery
4. **Metadata enrichment**: Add more metadata fields (descriptions, spatial/temporal extent, etc.)
5. **Custom catalogs**: Users can provide their own catalog definitions

## Updating the Catalog

To add new datasets to the catalog, edit [datasets.py](dclimate_client_py/datasets.py):

```python
DATASET_CATALOG_INTERNAL: DatasetCatalog = [
    {
        "collection": "your-collection",
        "datasets": [
            {
                "dataset": "your-dataset",
                "variants": [
                    {
                        "variant": "your-variant",
                        "cid": "bafybei...",  # Your IPFS CID
                        "concat_priority": 1,  # Optional: for auto-concat
                        "concat_dimension": "time"  # Optional: concat dimension
                    }
                ]
            }
        ]
    },
    # ... existing collections
]
```

## Comparison with dclimate-client-js

| Feature | dclimate-client-js | dclimate-zarr-client (Python) |
|---------|-------------------|-------------------------------|
| Load by dataset name | ✅ `loadDataset()` | ✅ `load_dataset()` |
| Auto-concatenation | ✅ Smart concatenation | ⏸️ Disabled (pending lazy concat support) |
| List catalog | ✅ `listDatasetCatalog()` | ✅ `list_dataset_catalog()` |
| Return type options | ✅ Jaxray or GeoTemporal | ✅ xarray or GeotemporalData |
| Direct CID loading | ✅ `cid` option | ✅ `cid` parameter |
| Collection auto-detect | ✅ Yes | ✅ Yes |
| URL-based CID fetch | ✅ Yes | 🔜 Not yet implemented |
| Catalog structure | TypeScript types | Python TypedDicts |

## Examples

See the [examples directory](examples/) for complete working examples:

