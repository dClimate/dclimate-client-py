[![codecov](https://codecov.io/gh/dClimate/dClimate-Zarr-Client/graph/badge.svg?token=AovaMO6DX5)](https://codecov.io/gh/dClimate/dClimate-Zarr-Client)
# dClimate-Zarr-Client
Retrieve dclimate GIS zarrs stored on IPLD.

Uses [py-hamt](https://github.com/dClimate/py-hamt) to actually access zarrs, then provides
filtering and aggregation functionality to these zarrs using `xarray` native methods wherever possible.
Filtering and aggregation are packaged into a minimal number of convenience functions optimized for flexbility
and performance.

A limit is imposed on the total number of points users can request to prevent overwhelming the API. This limit
can be manually overridden in exceptional cases.

The main entrypoint to the repo's code is `dclimate_zarr_client.client.geo_temporal_query`


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

Functions for accessing zarrs over IPFS/IPNS. Functionality includes resolving IPNS keys to IPFS hashes
based on key names, as well as using `py-hamt` to open the zarrs that those IPFS hashes point to.


## Usage

```python
from datetime import datetime
import xarray as xr
import dclimate_zarr_client as client

# Singleton Function Interface
ds_name = "era5_wind_100m_u-hourly"
ds_bytes = client.geo_temporal_query(
    ds_name,
    point_kwargs={"lat": 40, "lon": -120},
    time_range=[datetime(2021, 1, 1), datetime(2022, 12, 31)],
    output_format="netcdf",
    # gateway_uri="http://<IP>:<PORT>" # Optionally pass a custom gateway URI (default: http://127.0.0.1:8080) Note: IPFS must be running locally in the default case
)
ds = xr.open_dataset(ds_bytes)

# Pythonic Interface
dataset = client.load_ipns(ds_name)
# Optionally custom gateway
# dataset = client.load_ipns(ds_name, gateway_uri="http://<IP>:<PORT>")
dataset = dataset.point(lat=40, lon=-120)
dataset = dataset.time_range(datetime(2021, 1, 1), datetime(2022, 12, 31))
ds_bytes = dataset.to_netcdf()

ds = xr.open_dataset(ds_bytes)
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
