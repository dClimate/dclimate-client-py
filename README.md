[![codecov](https://codecov.io/gh/dClimate/dClimate-Zarr-Client/graph/badge.svg?token=AovaMO6DX5)](https://codecov.io/gh/dClimate/dClimate-Zarr-Client)
# dClimate-Zarr-Client
Retrieve dClimate GIS zarr datasets stored on IPFS, discoverable via a STAC catalog linked by IPNS.

Uses [py-hamt](https://github.com/dClimate/py-hamt) to access Zarr data structures stored efficiently on IPFS. The client navigates dClimate's STAC (SpatioTemporal Asset Catalog) to find the appropriate dataset identifier (IPNS name), resolves this identifier to the current IPFS content address (CID), and then loads the data. It provides filtering and aggregation functionality using `xarray` native methods wherever possible.

Filtering and aggregation are packaged into convenience functions optimized for flexibility and performance.

The main entrypoint for IPFS data is `dclimate_zarr_client.client.load_ipns` or `dclimate_zarr_client.client.geo_temporal_query` with `source='ipfs'`.

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


## Usage

```python
from datetime import datetime
import xarray as xr
import dclimate_zarr_client as client
from dclimate_zarr_client import dclimate_zarr_errors # For specific error catching

# --- IPFS/IPNS via STAC Catalog ---

# Option 1: Load the dataset first, then query (Pythonic Interface)
ds_name_ipfs = "cpc-precip-conus" # Example dataset ID from STAC
try:
   # Load the dataset structure using the STAC catalog to find the HAMT CID
    # Optionally provide custom gateway/rpc URIs if not using defaults/env vars
    # gateway = "http://127.0.0.1:8080"
    # rpc = "http://127.0.0.1:5001"
    # dataset = client.load_ipfs_via_stac(ds_name_ipfs, gateway_uri_stem=gateway, rpc_uri_stem=rpc)
    dataset = client.load_ipfs_via_stac(ds_name_ipfs)

    # Apply queries
    dataset_filtered = dataset.point(latitude=40.875, longitude=-104.875)
    dataset_filtered = dataset_filtered.time_range(datetime(2023, 1, 1), datetime(2023, 1, 5))

    # Get data as dictionary or NetCDF bytes
    data_dict = dataset_filtered.as_dict()
    # netcdf_bytes = dataset_filtered.to_netcdf()
    # ds = xr.open_dataset(netcdf_bytes) # Example if using NetCDF

    print(data_dict['data'])

# Catch specific errors if needed
except dclimate_zarr_errors.DatasetNotFoundError as e:
    print(f"Error finding dataset '{ds_name_ipfs}' in STAC or loading: {e}")
except dclimate_zarr_errors.IpfsConnectionError as e:
    print(f"IPFS connection error: {e}")
except dclimate_zarr_errors.StacCatalogError as e:
    print(f"STAC Catalog traversal error: {e}")
except Exception as e: # Catch any other unexpected errors
    print(f"An unexpected error occurred: {e}")



# Option 2: Use the all-in-one query function
try:
    # Returns dict by default
    result_dict = client.geo_temporal_query(
        dataset_name=ds_name_ipfs,
        source="ipfs", # Explicitly state source
        point_kwargs={"latitude": 40.875, "longitude": -104.875},
        time_range=[datetime(2023, 1, 1), datetime(2023, 1, 5)],
        # gateway_uri_stem=gateway, # Optional custom URIs
        # rpc_uri_stem=rpc,
        # output_format="netcdf" # Optionally get NetCDF bytes
    )
    print(result_dict['data'])

    # Example: Get NetCDF output
    # netcdf_bytes = client.geo_temporal_query(
    #     dataset_name=ds_name_ipfs,
    #     source="ipfs",
    #     point_kwargs={"latitude": 40.875, "longitude": -104.875},
    #     time_range=[datetime(2023, 1, 1), datetime(2023, 1, 5)],
    #     output_format="netcdf"
    # )
    # ds_from_nc = xr.open_dataset(netcdf_bytes)
    # print(ds_from_nc)

except dclimate_zarr_errors.DatasetNotFoundError as e:
    print(f"Error querying dataset '{ds_name_ipfs}': {e}")
except dclimate_zarr_errors.IpfsConnectionError as e:
    print(f"IPFS connection error during query: {e}")
except dclimate_zarr_errors.StacCatalogError as e:
    print(f"STAC Catalog traversal error during query: {e}")
except Exception as e:
    print(f"An unexpected error occurred during query: {e}")


# --- S3 Access ---
ds_name_s3 = "era5_wind_100m_u-hourly" # Example dataset name on S3
s3_bucket = "your-s3-bucket-name" # Replace with actual bucket

try:
    # Using all-in-one query
    result_s3 = client.geo_temporal_query(
        dataset_name=ds_name_s3,
        source="s3",
        bucket_name=s3_bucket,
        point_kwargs={"latitude": 40, "longitude": -120},
        time_range=[datetime(2021, 1, 1), datetime(2022, 12, 31)],
    )
    print(result_s3['data'])

    # Using Pythonic interface
    # dataset_s3 = client.load_s3(ds_name_s3, s3_bucket)
    # ... apply filters ...
    # data_dict_s3 = dataset_s3_filtered.as_dict()

except client.dclimate_zarr_errors.DatasetNotFoundError as e:
     print(f"Error loading S3 dataset '{ds_name_s3}' from bucket '{s3_bucket}': {e}")
except Exception as e: # Catch other potential errors like S3 access issues
     print(f"An error occurred with S3: {e}")
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
uv run pytest --cov=dclimate_zarr_client tests/ --cov-report=xml
```

## Environment requirements

- Running IPFS daemon
- Dataset parsed with [etl-scripts](https://github.com/dClimate/etl-scripts) with name `ds_name`
- Up-to-date IPNS table (IPNS key for `ds_name` can't be expired).
  If `ipfs name resolve <ipns key>` stalls out, the IPNS key is expired and `publish_metadata` step of ETL must be rerun.
