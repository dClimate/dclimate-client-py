<p align="center">
<a href="https://dclimate.net/" target="_blank" rel="noopener noreferrer">
<img width="50%" src="https://user-images.githubusercontent.com/41392423/173133333-79ef15d0-6671-4be3-ac97-457344e9e958.svg" alt="dClimate logo">
</a>
</p>

# dClimate-Client-Py
[![codecov](https://codecov.io/gh/dClimate/dClimate-Zarr-Client/graph/badge.svg?token=AovaMO6DX5)](https://codecov.io/gh/dClimate/dClimate-Zarr-Client)

Retrieve dClimate GIS zarr datasets stored on IPFS

Uses [py-hamt](https://github.com/dClimate/py-hamt) to access Zarr data structures stored efficiently on IPFS.

Filtering and aggregation are packaged into convenience functions optimized for flexibility and performance.

> **Looking for JavaScript?** Check out our [JavaScript client](https://www.npmjs.com/package/@dclimate/dclimate-client-js) for Node.js and browser environments.

## Usage

```python
from datetime import datetime
import dclimate_client_py as client
from dclimate_client_py import dClimateClient

# --- Recommended: Using dClimateClient (async context manager) ---

async def main():
    # The client manages IPFS connections automatically
    # No need to import or configure KuboCAS directly!
    async with dClimateClient() as dclimate:
        # Load datasets by name from the internal catalog
        # For datasets with multiple variants, you must specify which variant
        # Returns a tuple: (dataset, metadata)
        dataset, metadata = await dclimate.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized",  # Required for multi-variant datasets
            return_xarray=False   # Returns GeotemporalData wrapper (default)
        )

        # Check metadata about what was loaded
        print(f"Loaded: {metadata['slug']}")
        print(f"CID: {metadata['cid']}")
        print(f"Timestamp: {metadata.get('timestamp')}")  # If available from URL fetch
        print(f"Source: {metadata['source']}")  # 'catalog' or 'direct_cid'

        # Apply queries using the GeotemporalData interface
        dataset_filtered = dataset.point(latitude=40.875, longitude=-104.875)
        dataset_filtered = dataset_filtered.time_range(
            datetime(2023, 1, 1),
            datetime(2023, 1, 5)
        )
        data_dict = dataset_filtered.as_dict()
        print(data_dict['data'])

# Custom IPFS endpoints (optional)
async def main_custom_ipfs():
    async with dClimateClient(
        gateway_base_url="https://ipfs.io",
        rpc_base_url="http://localhost:5001"
    ) as dclimate:
        dataset, metadata = await dclimate.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized"
        )
        # Query dataset...

# Get raw xarray.Dataset directly
async def main_xarray():
    async with dClimateClient() as dclimate:
        xr_dataset, metadata = await dclimate.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized",
            return_xarray=True  # Returns xarray.Dataset
        )
        print(xr_dataset)
        print(f"Dataset CID: {metadata['cid']}")

# List available datasets in the catalog (synchronous helper)
catalog = client.list_dataset_catalog()
for collection in catalog:
    print(f"Collection: {collection['collection']}")
    for dataset in collection['datasets']:
        print(f"  Dataset: {dataset['dataset']}")
        for variant in dataset['variants']:
            print(f"    Variant: {variant['variant']}")

```

> More examples can be found at [dClimate Jupyter Notebooks](https://github.com/dClimate/jupyter-notebooks/tree/main/notebooks). To run your own IPFS gateway follow the instructions for [installing ipfs](https://docs.ipfs.tech/install/command-line/#install-official-binary-distributions). For additional assistance find us on [Discord](https://discord.com/invite/bYWVdNDMpe ), if you are an organization or business reach out to us at community at dclimate dot net.

## Create and activate a virtual environment:

``` shell
uv venv .venv
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\activate   # Windows
```

## Install Dependencies

```shell
uv sync --extra dev --extra testing
```

## Run tests for your local environment
```shell
uv run pytest tests/
```

## Use Coverage

```shell
uv run pytest --cov=dclimate_client_py tests/ --cov-report=xml
```

## Environment requirements

- Optionally you can run your own IPFS Server to host your own datasets or connect to others.


## File breakdown:

### client.py

Entrypoint to code, contains `geo_temporal_query`, which combines all possible subsetting
and aggregation logic in a single function. Can output the data as either a `dict`
or `bytes` representing an `xarray` dataset.

---

### dclimate_zarr_errors.py

Various exceptions to be raised for bad or invalid user input.

---

### geo_utils.py

Functions to manipulate `xarray` datasets. Contains polygon, rectangle, circle and point spatial
subsetting options, as well as temporal subsetting. Also allows for both spatial and temporal
aggregations.

---

### ipfs_retrieval.py

Functions for resolving IPNS names, traversing the dClimate STAC catalog stored on IPFS, and loading Zarr datasets using `py-hamt`. Handles interaction with IPFS gateways and RPC endpoints.
