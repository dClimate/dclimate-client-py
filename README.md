<p align="center">
<a href="https://dclimate.net/" target="_blank" rel="noopener noreferrer">
<img width="50%" src="https://user-images.githubusercontent.com/41392423/173133333-79ef15d0-6671-4be3-ac97-457344e9e958.svg" alt="dClimate logo">
</a>
</p>

# dClimate-Client-Py
[![codecov](https://codecov.io/gh/dClimate/dclimate-client-py/graph/badge.svg)](https://codecov.io/gh/dClimate/dclimate-client-py)

Retrieve dClimate GIS zarr datasets stored on IPFS

Uses [STAC (SpatioTemporal Asset Catalog)](https://stacspec.org/) for dataset discovery and [py-hamt](https://github.com/dClimate/py-hamt) to access Zarr data structures stored efficiently on IPFS.

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
            dataset="temperature_2m",
            collection="era5",  # Can also pass "ecmwf_era5"
            organization="ecmwf",
            variant="finalized",  # Required for multi-variant datasets
            return_xarray=False   # Returns GeotemporalData wrapper (default)
        )

        # Check metadata about what was loaded
        print(f"Loaded: {metadata['slug']}")
        print(f"CID: {metadata['cid']}")
        print(f"Timestamp: {metadata.get('timestamp')}")  # If available from URL fetch
        print(f"Source: {metadata['source']}")  # 'stac' or 'direct_cid'

        # Apply queries using the GeotemporalData interface
        dataset_filtered = dataset.point(latitude=40.875, longitude=-104.875)
        dataset_filtered = dataset_filtered.time_range(
            datetime(2023, 1, 1),
            datetime(2023, 1, 5)
        )
        data_dict = dataset_filtered.as_dict()
        print(data_dict['data'])

        # Or load and select in one call.
        western_europe, metadata = await dclimate.select_dataset(
            request={
                "dataset": "temperature_2m",
                "collection": "era5",
                "organization": "ecmwf",
                "variant": "finalized",
            },
            selection={
                # Bounds are [west, south, east, north].
                "bounds": [-12, 35, 16, 60],
                "time_range": {
                    "start": datetime(2024, 1, 1),
                    "end": datetime(2024, 1, 7, 23),
                },
            },
        )

# ERA5 land datasets
#
# ERA5 and ERA5-Land datasets are separate dataset IDs within the ECMWF ERA5
# collection. Use list_datasets() or list_available_datasets() to inspect the
# exact names before loading.
async def main_era5_land():
    async with dClimateClient() as dclimate:
        # Non-land ERA5 total precipitation
        precip, precip_metadata = await dclimate.load_dataset(
            dataset="precipitation_total",
            collection="era5",
            organization="ecmwf",
            variant="finalized",
        )

        # ERA5-Land total precipitation
        land_precip, land_metadata = await dclimate.load_dataset(
            dataset="precipitation_total_land",
            collection="era5",
            organization="ecmwf",
            variant="finalized",
        )

        # ERA5-Land wind datasets follow the same pattern:
        # dataset="wind_u_10m_land" or dataset="wind_v_10m_land"

# Custom IPFS endpoints (optional)
async def main_custom_ipfs():
    async with dClimateClient(
        gateway_base_url="https://ipfs.io",
        rpc_base_url="http://localhost:5001"
    ) as dclimate:
        dataset, metadata = await dclimate.load_dataset(
            dataset="temperature_2m",
            collection="era5",
            organization="ecmwf",
            variant="finalized"
        )
        # Query dataset...

# Get raw xarray.Dataset directly
async def main_xarray():
    async with dClimateClient() as dclimate:
        xr_dataset, metadata = await dclimate.load_dataset(
            dataset="temperature_2m",
            collection="era5",
            organization="ecmwf",
            variant="finalized",
            zarr_group="0",  # Optional for grouped/pyramid sharded stores
            return_xarray=True  # Returns xarray.Dataset
        )
        print(xr_dataset)
        print(f"Dataset CID: {metadata['cid']}")

# List available datasets from the STAC catalog
from dclimate_client_py import list_available_datasets, load_stac_catalog

# Load the STAC catalog
stac_catalog = load_stac_catalog("https://ipfs-gateway.dclimate.net")

# List all available datasets
datasets = list_available_datasets(stac_catalog)
for collection_id, info in datasets.items():
    print(
        f"Collection: {info['title']} ({collection_id})"
        + (f" | org: {info['organization']}" if info.get('organization') else "")
    )
    print(f"  Dataset types: {', '.join(info['types'])}")

# Resolve a CID directly without blocking the event loop. Calls made without
# an injected client reuse a pooled httpx.AsyncClient for the current loop.
from dclimate_client_py import (
    aclose_stac_server_client,
    aresolve_cid_from_stac_server,
)

async def resolve_cid():
    try:
        resolved = await aresolve_cid_from_stac_server(
            collection="ecmwf_aifs",
            dataset="temperature_forecast",
            variant="single",
        )
        print(resolved.cid)
    finally:
        # Call once when an application event loop shuts down.
        await aclose_stac_server_client()

```

### Dataset version history

For datasets that advertise version history in STAC, the client follows the
item's `dclimate:versions_api` URL. This automatically selects Hydrogen,
Tritium, or another future version service without a client-side routing map.

```python
async def list_aigfs_versions():
    client = dClimateClient()
    versions = await client.list_dataset_versions(
        collection="noaa_aigfs",
        dataset="wind_u_forecast",
        variant="operational",
        anchored=True,
    )
    for release in versions.versions:
        print(release.version_label, release.cid)

    exact_version = await client.get_dataset_version(
        collection="noaa_aigfs",
        dataset="wind_u_forecast",
        variant="operational",
        commit_id="commit-id",
    )
    print(exact_version.cid)
```

The lower-level functions in `dclimate_client_py.ceramic_api` retain their
existing names and explicit `base_url` support. STAC-aware applications should
prefer `list_dataset_versions()` and `get_dataset_version()` so they do not need
to know which service owns a dataset.

### Multiresolution datasets

Pyramidal datasets require an explicit resolution (recommended) or raw Zarr
group. The client reports the available resolutions instead of silently
choosing between different precision, chunking, and fetching strategies.

```python
data, metadata = await client.load_dataset(
    collection="copernicus_clms",
    dataset="fpar",
    resolution="2km",
)
print(metadata["resolution"], metadata["zarr_group"])
```

FPAR's advertised mappings are `500m` → group `"0"`, `2km` → group `"1"`,
and `8km` → group `"2"`. For example, replace `resolution="2km"` above with
`"500m"` or `"8km"` to select those levels. Raw `zarr_group="1"` is supported
when a caller intentionally works at the storage level, but do not pass it
together with `resolution`.

STAC may temporarily include a legacy `assets.data` alias for 500 m alongside
the three named assets. The client ignores that alias when enumerating choices,
so it is not a default or a fourth resolution. Consumers that previously
relied on `assets.data` or implicit group `"0"` should migrate to an explicit
resolution before the alias is removed in a future breaking release.

Callers loading a direct CID have no STAC resolution mapping and must pass
`zarr_group` when the store contains multiple groups; human-readable
`resolution` is rejected because the mapping exists only in STAC. The STAC
generator's `metadataGroup` is internal catalog-generation configuration and
does not influence client selection.

## Station data usage

Gridded Zarr datasets come from `load_dataset`. Point-observation **station**
datasets (GHCND and friends) live under `client.stations`, and read the same way:
degrees, ISO timestamps, chained selections.

Station support needs [`tabular-py`](https://github.com/dClimate/tabular-py),
which is not on PyPI yet, so it is not installed by default:

``` shell
uv pip install git+https://github.com/dClimate/tabular-py
```

Until it is installed, `client.stations` raises `TabularNotInstalledError` with
those instructions. Once `tabular-py` is published it moves into the ordinary
dependency list, matching `dclimate-client-js`.

```python
async with dClimateClient() as client:
    stations = await client.stations.load("bafyr4i...")

    # Every station, with position and coverage window.
    for s in await stations.list_stations():
        print(s.station_id, s.latitude, s.longitude, s.start, s.end)

    # Stations within 50 km of a point, over one week.
    records = await (
        stations
        .circle(40.75, -73.99, 50)
        .time_range("2023-01-01", "2023-01-07")
        .to_records("TMAX")
    )
```

Reads go through the client's own IPFS transport, so pinning, retries, and
configured endpoints apply to station reads too. Resolution is by CID for now;
STAC catalog support will follow.

Selections return new instances, so a partial selection can be branched:

```python
week = stations.time_range("2023-01-01", "2023-01-07")
nyc = await week.select("USW00094728").rows()
lax = await week.select("USW00023174").rows()
```

Two things differ from `GeotemporalData`, because the data model differs:

- **`nearest(lat, lon, max_km=...)` instead of a point selection.** A grid always
  has a cell under any coordinate; stations are irregular, so the nearest one may
  be far away. Pass `max_km` to make that a hard bound rather than a surprise.
- **`where(...)` has no gridded counterpart.** Row-level predicates are pushed
  down to fragment statistics, so most fragments are skipped without being read:

```python
from tabular_py import gt

# `nearest` reads the station index to find the match, so it is awaited --
# unlike the synchronous selections above.
hot_days = await (
    (await stations.nearest(29.98, -95.36))
    .time_range("2025-01-01", "2025-12-31")
    .where(gt("TMAX", 350))  # tenths of °C, so 35 °C
    .rows()
)
```

`nearest` alone means the nearest *station*, which is not always the nearest
*usable data*: near downtown Los Angeles the closest station is 0.63 km away and
has never recorded `TMAX`, while the closest that has is 5.4 km away. Ask for the
columns you need, and narrow them to a time range when "has ever reported" is not
good enough:

```python
# Resolves the dataset and the station in one call.
station = await client.stations.nearest(
    "bafyr4i...",
    34.0522, -118.2437,
    columns=["TMAX"],
    within=("2024-01-01", "2024-12-31"),
    max_km=50,
)
print(station.station_id, station.km)
```

Station failures arrive as this library's own errors, for the whole chain rather
than just `load`: an unknown column is an `InvalidSelectionError`, a well-formed
query that matched nothing is a `NoDataFoundError`, and bytes that do not
describe a readable dataset are a `DatasetCorruptError`. All descend from
`ZarrClientError`, so one `except` still covers the client.

## Siren API usage

The Python client also exposes a Siren REST client for metrics and regions.

```python
from dclimate_client_py import (
    dClimateClient,
    SirenApiKeyAuth,
    SirenMetricQuery,
    SirenOptions,
)

async def main():
    client = dClimateClient(
        siren=SirenOptions(
            auth=SirenApiKeyAuth()  # reads SIREN_API_KEY and SIREN_ACCOUNT_ID from env
        )
    )

    regions = await client.list_regions()
    print(f"Loaded {len(regions)} regions")

    data = await client.get_metric_data(
        SirenMetricQuery(
            region_id=regions[0].id,
            metric="average_precip",
            start_date="2025-01-01",
            end_date="2025-01-31",
        )
    )
    print(data[:3])
```

`x402` is included in the default install.

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

### geotemporal_data.py

`GeotemporalData`, a wrapper around `xarray` datasets. Contains polygon, rectangle, circle and
point spatial subsetting options, as well as temporal subsetting. Also allows for both spatial
and temporal aggregations.

---

### stac_catalog.py

STAC (SpatioTemporal Asset Catalog) integration for dClimate datasets. Provides functions to:
- Fetch the latest STAC catalog CID from the dClimate IPFS gateway
- Load and navigate STAC catalogs stored on IPFS
- Resolve dataset names to IPFS CIDs using the STAC catalog structure
- List all available datasets and collections

Uses a custom `IPFSStacIO` implementation to transparently resolve `ipfs://` URIs via HTTP gateways, allowing pystac to work seamlessly with IPFS-hosted catalogs.

---

### ipfs_retrieval.py

Functions for loading Zarr datasets from IPFS using `py-hamt`. Handles interaction with IPFS gateways and RPC endpoints through the KuboCAS interface.
