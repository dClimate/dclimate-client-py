# Dataset Catalog Usage Guide

This document explains how to use the STAC-based dataset catalog functionality in dclimate-client-py.

## Overview

dClimate uses the [STAC (SpatioTemporal Asset Catalog)](https://stacspec.org/) standard for organizing and discovering datasets stored on IPFS. The client provides a high-level interface that makes it easy to discover and load datasets without managing CIDs directly.

### Key Features

1. **STAC Catalog Integration** - Uses industry-standard STAC format for dataset discovery
2. **Automatic CID Resolution** - Datasets are resolved from logical names to IPFS CIDs automatically
3. **Dataset Discovery** - List and browse available datasets and collections
4. **Variant Support** - Handle multiple variants of datasets (e.g., forecasts with different ensemble members)
5. **IPFS Transparency** - The client handles all IPFS gateway interactions automatically

## Quick Start

### Basic Usage with dClimateClient

The recommended way to use the catalog is through the `dClimateClient` async context manager:

```python
from dclimate_client_py import dClimateClient
import asyncio

async def main():
    async with dClimateClient() as client:
        # List available datasets
        datasets = client.list_datasets()

        # Print collections and their dataset types
        for collection_id, info in datasets.items():
            org = f" [{info['organization']}]" if info.get("organization") else ""
            print(f"{info['title']} ({collection_id}){org}")
            print(f"  Types: {', '.join(info['types'])}")

        # Load a specific dataset
        data, metadata = await client.load_dataset(
            collection="era5",  # or "ecmwf_era5"
            organization="ecmwf",
            dataset="temperature_2m",
            variant="finalized"  # Specify variant if multiple exist
        )

        # Work with the data
        print(f"Loaded: {metadata['slug']}")
        print(f"CID: {metadata['cid']}")
        print(f"Source: {metadata['source']}")  # Will be 'stac'

asyncio.run(main())
```

### Low-Level STAC Catalog Access

For more control, you can work with the STAC catalog directly:

```python
from dclimate_client_py import (
    load_stac_catalog,
    list_available_datasets,
    resolve_dataset_cid_from_stac,
)

# Load the STAC catalog from IPFS
catalog = load_stac_catalog("https://ipfs-gateway.dclimate.net")

# List all available datasets
datasets = list_available_datasets(catalog)

# Example output structure:
# {
#     "ecmwf_era5": {
#         "id": "ecmwf_era5",
#         "organization": "ecmwf",
#         "title": "ERA5 Reanalysis",
#         "types": ["temperature_2m", "precipitation_total"]
#     },
#     "ecmwf_ifs": {
#         "id": "ecmwf_ifs",
#         "organization": "ecmwf",
#         "title": "Integrated Forecasting System (IFS)",
#         "types": ["temperature_forecast", "wind_u_forecast", ...]
#     }
# }

# Resolve a specific dataset to its CID
resolved = resolve_dataset_cid_from_stac(
    catalog=catalog,
    collection="ecmwf_era5",
    dataset="temperature_2m",
    variant="finalized",  # Optional, only needed if dataset has multiple variants
    organization="ecmwf",
)

print(f"Dataset CID: {resolved.cid} (variant: {resolved.variant})")
```

## STAC Catalog Structure

The dClimate STAC catalog follows this hierarchy:

```
Root Catalog (ipfs://...)
├── Organization: ecmwf
│   ├── Collection: ecmwf_era5
│   │   ├── Item: ecmwf_era5-temperature_2m-finalized
│   │   │   └── Asset: data (ipfs://... -> Zarr dataset)
│   │   └── Item: ecmwf_era5-precipitation_total-finalized
│   └── Collection: ecmwf_ifs
│       └── Item: ecmwf_ifs-temperature_forecast-single
├── Organization: copernicus
│   └── Collection: copernicus_clms
│       └── Item: copernicus_clms-fpar-default
└── ...
```

### Catalog Components

- **Root Catalog**: Entry point containing links to all organizations
- **Organizations**: Catalogs grouping collections (e.g., "ecmwf", "copernicus")
- **Collections**: Groups of related datasets (e.g., "ecmwf_era5", "ecmwf_ifs")
  - Each collection has a `dclimate:id` field for identification
  - Collections have a `dclimate:types` field listing available dataset types
- **Items**: Individual dataset variants following the pattern `{collection}-{dataset}-{variant}`
- **Assets**: Links to the actual Zarr data on IPFS (usually under the "data" asset key)

## API Reference

### `dClimateClient.load_dataset()`

Load a dataset using the managed STAC catalog.

```python
async def load_dataset(
    dataset: str,                    # Dataset name (e.g., "temperature")
    collection: str,                 # Collection ID (e.g., "ecmwf_ifs")
    variant: Optional[str] = None,   # Variant name (e.g., "single", "ensemble")
    organization: Optional[str] = None,  # Organization/agency that owns the collection
    cid: Optional[str] = None,       # Direct CID override (bypasses STAC)
    return_xarray: bool = False,     # Return raw xarray.Dataset instead of GeotemporalData
    zarr_group: Optional[str] = None, # Explicit Zarr group for grouped/pyramid stores
) -> Union[
    Tuple[GeotemporalData, DatasetMetadata],
    Tuple[xr.Dataset, DatasetMetadata]
]
```

**Parameters:**
- `dataset`: The dataset type name (see `list_datasets()` for available types)
- `collection`: The collection ID containing the dataset. Can be prefixed (e.g., `ecmwf_era5`)
  or the unprefixed short name when `organization` is provided.
- `variant`: Optional variant name. Required if the dataset has multiple variants
- `organization`: Optional organization/agency id (e.g., `ecmwf`, `copernicus`). When provided,
  the client resolves the collection within that organization's catalog. When omitted, it is
  inferred from the root catalog metadata.
- `cid`: Optional direct CID to bypass STAC catalog resolution
- `return_xarray`: If True, return raw `xarray.Dataset`, otherwise return `GeotemporalData` wrapper
- `zarr_group`: Optional Zarr group to open. If omitted, multi-group py-hamt v2 stores default to group `"0"` when available.

**Returns:**
- Tuple of (dataset, metadata)
  - `dataset`: Either `GeotemporalData` or `xarray.Dataset` depending on `return_xarray`
  - `metadata`: Dictionary with keys:
    - `collection`: Collection ID
    - `dataset`: Dataset name
    - `variant`: Variant name
    - `slug`: Full dataset identifier (org/collection/dataset/variant when organization is known)
    - `cid`: IPFS CID that was loaded
    - `source`: Either "stac" or "direct_cid"
    - `url`: Always None for STAC-based loading
    - `timestamp`: Always None for STAC-based loading
    - `organization`: The resolved organization id when available
    - `zarr_group`: Present when an explicit or inferred Zarr group was opened

**Raises:**
- `RuntimeError`: If client is not used as async context manager
- `InvalidSelectionError`: If collection is not provided when using STAC resolution
- `ValueError`: If dataset, collection, or variant is not found in STAC catalog

**Examples:**

```python
# Basic usage
async with dClimateClient() as client:
    data, metadata = await client.load_dataset(
        collection="era5",  # or "ecmwf_era5"
        organization="ecmwf",
        dataset="temperature_2m",
        variant="finalized"
    )

# Get raw xarray.Dataset
async with dClimateClient() as client:
    xr_ds, metadata = await client.load_dataset(
        collection="era5",
        organization="ecmwf",
        dataset="temperature_2m",
        variant="finalized",
        return_xarray=True
    )

# Load with direct CID (bypasses STAC)
async with dClimateClient() as client:
    data, metadata = await client.load_dataset(
        dataset="temperature",  # Used for metadata only
        collection="ecmwf_ifs",
        organization="ecmwf",
        cid="bafybeiabc123...",
        zarr_group="0"
    )
    # metadata['source'] will be 'direct_cid'
```

### `dClimateClient.list_datasets()`

List all available datasets from the STAC catalog.

```python
def list_datasets() -> Dict[str, Dict[str, Any]]
```

**Returns:**
- Dictionary mapping collection IDs to collection metadata:
  ```python
  {
      "collection_id": {
          "id": "collection_id",
          "title": "Collection Title",
          "types": ["dataset_type1", "dataset_type2", ...]
      },
      ...
  }
  ```

**Example:**

```python
async with dClimateClient() as client:
    datasets = client.list_datasets()

    # Iterate through collections
    for collection_id, info in datasets.items():
        print(f"{info['title']} ({collection_id})")
        for dataset_type in info['types']:
            print(f"  - {dataset_type}")
```

### `load_stac_catalog()`

Low-level function to load the STAC catalog from IPFS.

```python
def load_stac_catalog(
    gateway_url: str,
    root_cid: Optional[str] = None
) -> pystac.Catalog
```

**Parameters:**
- `gateway_url`: IPFS HTTP gateway URL (e.g., "https://ipfs-gateway.dclimate.net")
- `root_cid`: Optional specific catalog CID. If None, fetches latest from API

**Returns:**
- `pystac.Catalog`: Loaded STAC catalog object

**Example:**

```python
from dclimate_client_py import load_stac_catalog

# Load latest catalog
catalog = load_stac_catalog("https://ipfs-gateway.dclimate.net")

# Load specific catalog version
catalog = load_stac_catalog(
    "https://ipfs-gateway.dclimate.net",
    root_cid="bafybeiabc123..."
)
```

### `list_available_datasets()`

Low-level function to list datasets from a loaded STAC catalog.

```python
def list_available_datasets(
    catalog: pystac.Catalog
) -> Dict[str, Dict[str, Any]]
```

**Parameters:**
- `catalog`: A loaded pystac.Catalog object

**Returns:**
- Dictionary of collections and their dataset types (same structure as `dClimateClient.list_datasets()`)

**Example:**

```python
from dclimate_client_py import load_stac_catalog, list_available_datasets

catalog = load_stac_catalog("https://ipfs-gateway.dclimate.net")
datasets = list_available_datasets(catalog)

for collection_id, info in datasets.items():
    print(f"{info['title']}: {len(info['types'])} dataset types")
```

### `resolve_dataset_cid_from_stac()`

Low-level function to resolve a dataset name to its IPFS CID.

```python
def resolve_dataset_cid_from_stac(
    catalog: pystac.Catalog,
    collection: str,
    dataset: str,
    variant: Optional[str] = None
) -> ResolvedDataset
```

**Parameters:**
- `catalog`: Loaded STAC catalog
- `collection`: Collection ID
- `dataset`: Dataset type name
- `variant`: Optional variant name

**Returns:**
- `ResolvedDataset`: IPFS CID (without "ipfs://" prefix) and selected variant

**Raises:**
- `ValueError`: If collection, dataset, or variant is not found

**Example:**

```python
from dclimate_client_py import load_stac_catalog, resolve_dataset_cid_from_stac

catalog = load_stac_catalog("https://ipfs-gateway.dclimate.net")

# Resolve dataset to CID
resolved = resolve_dataset_cid_from_stac(
    catalog=catalog,
    collection="ifs",
    dataset="temperature",
    variant="single"
)

print(f"Dataset CID: {resolved.cid} (variant: {resolved.variant})")
```

### `get_root_catalog_cid()`

Fetch the latest STAC catalog CID from the dClimate API.

```python
def get_root_catalog_cid() -> str
```

**Returns:**
- `str`: The IPFS CID of the latest root STAC catalog

**Raises:**
- `requests.HTTPError`: If the API request fails
- `KeyError`: If response doesn't contain expected 'cid' field

**Example:**

```python
from dclimate_client_py.stac_catalog import get_root_catalog_cid

cid = get_root_catalog_cid()
print(f"Latest catalog CID: {cid}")
```

## Working with Variants

Many datasets have multiple variants (e.g., different forecast ensemble members, finalized vs non-finalized data). The STAC catalog uses item IDs following the pattern: `{collection}-{dataset}-{variant}`

### Finding Variants

```python
async with dClimateClient() as client:
    # Load catalog and navigate to a collection
    catalog = load_stac_catalog("https://ipfs-gateway.dclimate.net")

    # Get child links
    for link in catalog.get_child_links():
        if link.extra_fields.get("dclimate:id") == "ifs":
            collection = link.resolve_stac_object(root=catalog).target

            # List all items (dataset variants)
            for item in collection.get_items():
                print(f"Item ID: {item.id}")
                # Example output: ifs-temperature-single
                #                 ifs-temperature-ensemble
                #                 ifs-precipitation-single
```

### Loading Specific Variants

```python
async with dClimateClient() as client:
    # Load single-member variant
    single, meta = await client.load_dataset(
        collection="ifs",
        dataset="temperature",
        variant="single"
    )

    # Load ensemble variant
    ensemble, meta = await client.load_dataset(
        collection="ifs",
        dataset="temperature",
        variant="ensemble"
    )
```

## Custom IPFS Endpoints

You can use custom IPFS gateways and RPC endpoints:

```python
async with dClimateClient(
    gateway_base_url="http://localhost:8080",
    rpc_base_url="http://localhost:5001"
) as client:
    data, metadata = await client.load_dataset(
        collection="ifs",
        dataset="temperature",
        variant="single"
    )
```

## Error Handling

The STAC catalog integration provides clear error messages:

```python
from dclimate_client_py import dClimateClient
from dclimate_client_py.dclimate_zarr_errors import InvalidSelectionError

async with dClimateClient() as client:
    try:
        # Missing collection parameter
        data, meta = await client.load_dataset(dataset="temperature")
    except InvalidSelectionError as e:
        print(f"Error: {e}")
        # "collection parameter is required. Use client.list_datasets()..."

    try:
        # Non-existent collection
        data, meta = await client.load_dataset(
            collection="nonexistent",
            dataset="temperature"
        )
    except ValueError as e:
        print(f"Error: {e}")
        # "Collection 'nonexistent' not found in STAC catalog"

    try:
        # Non-existent dataset
        data, meta = await client.load_dataset(
            collection="ifs",
            dataset="nonexistent"
        )
    except ValueError as e:
        print(f"Error: {e}")
        # "Dataset 'nonexistent' not found in collection 'ifs'"

    try:
        # Non-existent variant
        data, meta = await client.load_dataset(
            collection="ifs",
            dataset="temperature",
            variant="nonexistent"
        )
    except ValueError as e:
        print(f"Error: {e}")
        # "Dataset 'temperature' with variant 'nonexistent' not found..."
```

## Advanced Usage

### Navigating the STAC Catalog

```python
import pystac
from dclimate_client_py import load_stac_catalog

# Load catalog
catalog = load_stac_catalog("https://ipfs-gateway.dclimate.net")

# Navigate through collections
for child_link in catalog.get_child_links():
    collection_id = child_link.extra_fields.get("dclimate:id")
    if not collection_id:
        continue

    print(f"Collection: {collection_id}")

    # Resolve the collection
    collection = child_link.resolve_stac_object(root=catalog).target

    # List all items in the collection
    for item in collection.get_items():
        print(f"  Item: {item.id}")

        # Access item metadata
        if "dclimate:dataset" in item.properties:
            print(f"    Dataset: {item.properties['dclimate:dataset']}")

        # Access assets
        for asset_key, asset in item.assets.items():
            print(f"    Asset '{asset_key}': {asset.href}")
```

### Using Custom StacIO

The client uses a custom `IPFSStacIO` class to handle `ipfs://` URIs:

```python
from dclimate_client_py.stac_catalog import IPFSStacIO
import pystac

# Create custom IPFS StacIO handler
stac_io = IPFSStacIO("https://ipfs-gateway.dclimate.net")
pystac.StacIO.set_default(lambda: stac_io)

# Now you can load STAC objects from ipfs:// URIs
catalog = pystac.Catalog.from_file("ipfs://bafybeiabc123...")
```

## Comparison with Legacy Approach

### Old: Manual CID Management
```python
# Had to know and manage CIDs manually
cid = "bafybeiabc123..."
ds = load_dataset_from_cid(cid, gateway_url="https://...")
```

### New: STAC-Based Discovery
```python
# Discover and load by logical name
async with dClimateClient() as client:
    # Browse what's available
    datasets = client.list_datasets()

    # Load by name
    data, metadata = await client.load_dataset(
        collection="ifs",
        dataset="temperature",
        variant="single"
    )
    # CID is resolved automatically: metadata['cid']
```

## Benefits of STAC Integration

1. **Industry Standard**: STAC is a widely-adopted standard for geospatial data
2. **Discoverable**: Datasets can be browsed and discovered programmatically
3. **Metadata-Rich**: STAC supports extensive metadata about datasets
4. **Extensible**: Custom fields like `dclimate:id` and `dclimate:types` extend STAC for dClimate needs
5. **Interoperable**: Works with standard STAC tools and libraries
6. **Version Control**: Each STAC catalog has its own CID, enabling versioning
7. **Human-Readable**: Dataset names are more intuitive than raw CIDs

## Future Enhancements

Potential future additions to the STAC catalog integration:

1. **Temporal/Spatial Extent**: Add STAC extent information for datasets
2. **Search Capabilities**: Filter datasets by time range, spatial bounds, or metadata
3. **Multiple Catalog Sources**: Support for additional STAC catalogs beyond dClimate
4. **Caching**: Cache catalog metadata for faster repeated access
5. **Automatic Updates**: Detect and refresh when new catalog versions are available

## Resources

- [STAC Specification](https://stacspec.org/)
- [PySTAC Documentation](https://pystac.readthedocs.io/)
- [dClimate Documentation](https://docs.dclimate.net/)
- [IPFS Documentation](https://docs.ipfs.tech/)
