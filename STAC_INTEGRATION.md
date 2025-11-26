# STAC Integration for dClimate Client

This document describes the STAC (SpatioTemporal Asset Catalog) integration added to dclimate-client-py.

## Overview

The STAC integration allows the dClimate Python client to discover and load datasets from a STAC catalog stored on IPFS, replacing the need for hardcoded catalog entries.

## Files Added/Modified

### New Files

1. **`dclimate_client_py/stac_catalog.py`** - STAC catalog utilities
   - `get_root_catalog_cid()` - Returns root STAC catalog CID (currently mock)
   - `IPFSStacIO` - Custom pystac.StacIO for resolving `ipfs://` URIs
   - `load_stac_catalog()` - Loads STAC catalog from IPFS
   - `resolve_dataset_cid_from_stac()` - Resolves dataset to CID via STAC
   - `list_available_datasets()` - Lists all available datasets

2. **`test_stac_integration.py`** - Test script for STAC integration

### Modified Files

1. **`pyproject.toml`**
   - Added `pystac>=1.10.0` dependency

2. **`dclimate_client_py/dclimate_client.py`**
   - Added `use_stac` parameter to `__init__()` (default: `True`)
   - Added `_stac_catalog` instance variable
   - Modified `load_dataset()` to support STAC catalog resolution
   - Added `list_datasets()` method for listing available datasets

3. **`dclimate_client_py/__init__.py`**
   - Exported `load_stac_catalog` and `list_available_datasets`

## Usage

### Basic Usage with STAC

```python
import asyncio
from dclimate_client_py import dClimateClient

async def main():
    # Create client with STAC enabled (default)
    async with dClimateClient(use_stac=True) as client:
        # List available datasets
        datasets = client.list_datasets()
        print(datasets["ifs"]["types"])
        # Output: ['temperature', 'precipitation', 'wind_u', ...]

        # Load a dataset
        data, metadata = await client.load_dataset(
            collection="ifs",
            dataset="temperature",
            variant="single"
        )

        print(f"Loaded from: {metadata['source']}")  # 'stac'
        print(f"CID: {metadata['cid']}")

asyncio.run(main())
```

### Fallback to Legacy Catalog

```python
# Use legacy DATASET_CATALOG_INTERNAL
async with dClimateClient(use_stac=False) as client:
    data, metadata = await client.load_dataset(
        dataset="2m_temperature",
        collection="era5",
        variant="finalized"
    )
```

### Direct STAC Catalog Access

```python
from dclimate_client_py import load_stac_catalog, list_available_datasets

# Load STAC catalog directly
catalog = load_stac_catalog(gateway_url="https://ipfs-gateway.dclimate.net")

# List datasets
datasets = list_available_datasets(catalog)
print(datasets)
```

## How It Works

### 1. IPFS Resolution

The `IPFSStacIO` class extends pystac's StacIO to resolve `ipfs://` URIs:

```
ipfs://bafkrei... → https://ipfs-gateway.dclimate.net/ipfs/bafkrei...
```

This happens transparently when pystac loads linked STAC objects.

### 2. STAC Structure

```
Root Catalog (ipfs://bafkrei...)
├── Collection: "ifs" (dclimate:id = "ifs")
│   ├── dclimate:types = ["temperature", "precipitation", ...]
│   └── Item: "ifs-temperature-single"
│       └── Asset "data": ipfs://bafyr... (Zarr CID)
├── Collection: "era5" (dclimate:id = "era5")
│   └── ...
└── Collection: "aifs" (dclimate:id = "aifs")
    └── ...
```

### 3. Dataset Resolution Flow

1. Load root STAC catalog from IPFS (lazy, cached)
2. Find collection by `dclimate:id` attribute
3. Iterate collection items to find matching dataset/variant
4. Parse item ID (format: `{collection}-{dataset}-{variant}`)
5. Extract CID from item's `assets.data.href`
6. Return CID (without `ipfs://` prefix)
7. Load Zarr data from IPFS using existing client code

### 4. Listing Datasets

The `list_datasets()` method reads `dclimate:types` from collection links in the root catalog, avoiding the need to load all items.

## Configuration

### Root Catalog CID

Currently hardcoded in `get_root_catalog_cid()`:
```python
return "bafkreiamnbh76x7njoh7zu7ct6uzzozv4kyb6wecefnref7hmr454rkkiu"
```

**TODO**: Replace with dynamic lookup (API endpoint or smart contract).

### IPFS Gateway

Configured via `dClimateClient`:
```python
async with dClimateClient(
    gateway_base_url="https://ipfs-gateway.dclimate.net",
    use_stac=True
) as client:
    # ...
```

## Testing

Run the test script:
```bash
python test_stac_integration.py
```

Tests:
1. ✓ List available datasets from STAC
2. Load dataset from STAC (requires IPFS with actual data)
3. ✓ Fallback to legacy catalog

## Benefits

- ✅ Single source of truth (STAC catalog on IPFS)
- ✅ Automatic discovery of new datasets
- ✅ Standard STAC 1.0.0 format
- ✅ Rich metadata from STAC
- ✅ Backward compatible (legacy catalog still works)
- ✅ Uses standard pystac library

## Future Improvements

1. **Dynamic Root CID Lookup**
   - Fetch latest catalog CID from API or smart contract
   - Support versioned catalogs

2. **Caching**
   - Cache STAC catalog locally
   - Implement TTL or update mechanism

3. **Enhanced Metadata**
   - Extract spatial/temporal extents from STAC
   - Expose additional STAC properties

4. **Async STAC Loading**
   - Use async HTTP client for IPFS fetches
   - Parallel loading of STAC objects

5. **STAC Browser Integration**
   - Generate URLs for STAC Browser visualization
   - Deep linking to specific items
