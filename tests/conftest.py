import asyncio
import datetime
import itertools
import pathlib

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import httpx
import zarr
import zarr.storage

from tests.ipfs_config import IPFS_GATEWAY_URL, IPFS_RPC_URL, STAC_CATALOG_URL


@pytest.fixture
def install_httpx_mock(monkeypatch):
    """Inject pooled sync and async clients through a module's accessors."""
    clients: list[httpx.Client] = []
    async_clients: list[httpx.AsyncClient] = []

    def install(module, handler):
        client = httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=30,
            follow_redirects=True,
        )

        async def async_handler(request: httpx.Request) -> httpx.Response:
            return await asyncio.to_thread(handler, request)

        async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(async_handler),
            timeout=30,
            follow_redirects=True,
        )
        clients.append(client)
        async_clients.append(async_client)
        monkeypatch.setattr(module, "_client", lambda: client)
        monkeypatch.setattr(module, "_async_client", lambda: async_client)
        return client

    yield install

    for client in clients:
        client.close()
    for async_client in async_clients:
        asyncio.run(async_client.aclose())


def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring external services",
    )
    config.addinivalue_line(
        "markers", "ipfs_rpc: mark test as requiring a writable IPFS RPC endpoint"
    )
    config.addinivalue_line(
        "markers", "stac_pointer: mark test as requiring the STAC CID endpoint"
    )


def pytest_collection_modifyitems(config, items):
    """Gate tests that require external integration services."""
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    ipfs_items = [item for item in items if "ipfs" in item.keywords]
    skip_ipfs = None
    if ipfs_items:
        if not is_ipfs_running(IPFS_GATEWAY_URL):
            skip_ipfs = pytest.mark.skip(
                reason=f"IPFS gateway not responding at {IPFS_GATEWAY_URL}"
            )
    ipfs_rpc_items = [item for item in items if "ipfs_rpc" in item.keywords]
    skip_ipfs_rpc = None
    if ipfs_rpc_items and not is_ipfs_rpc_running(IPFS_RPC_URL):
        skip_ipfs_rpc = pytest.mark.skip(
            reason=f"IPFS RPC endpoint not responding at {IPFS_RPC_URL}"
        )
    stac_pointer_items = [item for item in items if "stac_pointer" in item.keywords]
    skip_stac_pointer = None
    if stac_pointer_items and not is_stac_pointer_running(STAC_CATALOG_URL):
        skip_stac_pointer = pytest.mark.skip(
            reason=f"STAC catalog pointer not responding at {STAC_CATALOG_URL}"
        )

    for item in items:
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
        if "ipfs" in item.keywords and skip_ipfs is not None:
            item.add_marker(skip_ipfs)
        if "ipfs_rpc" in item.keywords and skip_ipfs_rpc is not None:
            item.add_marker(skip_ipfs_rpc)
        if "stac_pointer" in item.keywords and skip_stac_pointer is not None:
            item.add_marker(skip_stac_pointer)


HERE = pathlib.Path(__file__).parent
ETC = HERE / "etc"
SAMPLE_ZARRS = ETC / "sample_zarrs"


@pytest.fixture
def input_ds():
    # Keeping local fixtures for tests that don't need IPFS loading (like test_geotemporal_data)
    with zarr.storage.ZipStore(ETC / "retrieval_test.zip", mode="r") as in_zarr:
        return xr.open_zarr(in_zarr, chunks=None, decode_timedelta=True).compute()


@pytest.fixture
def forecast_ds():
    # Keeping local fixtures for tests that don't need IPFS loading
    with zarr.storage.ZipStore(
        ETC / "forecast_retrieval_test.zip", mode="r"
    ) as in_zarr:
        return xr.open_zarr(in_zarr, chunks=None, decode_timedelta=True).compute()


@pytest.fixture
def oversized_polygons_mask():
    shp = gpd.read_file(ETC / "northern_ca_counties.geojson")
    return shp.geometry.values


@pytest.fixture
def undersized_polygons_mask():
    shp = gpd.read_file(ETC / "central_ca_farm.geojson")
    return shp.geometry.values


@pytest.fixture
def polygons_mask():
    shp = gpd.read_file(ETC / "central_northern_ca_counties.geojson")
    return shp.geometry.values


@pytest.fixture
def points_mask():
    points = gpd.read_file(ETC / "northern_ca_points.geojson")
    return points.geometry.values


def date_sequence(start, delta):
    date = start
    while True:
        yield date
        date += delta


def make_dataset(vars=3, shape=[20, 20, 20]):
    start = datetime.date(2000, 1, 1)
    times = date_sequence(start, datetime.timedelta(days=1))
    time = np.fromiter(itertools.islice(times, shape[0]), dtype="datetime64[ns]")
    time = xr.DataArray(time, dims="time", coords={"time": np.arange(len(time))})
    latitude = np.arange(0, 10 * shape[1], 10)
    latitude = xr.DataArray(latitude, dims="latitude", coords={"latitude": latitude})
    longitude = np.arange(180, 180 + 5 * shape[2], 5)
    longitude = xr.DataArray(
        longitude, dims="longitude", coords={"longitude": longitude}
    )

    points = shape[0] * shape[1] * shape[2]
    data_vars = {}
    for i in range(vars):
        var_name = f"var_{i + 1}"
        data = [10000 * i + j for j in range(points)]
        data = np.array(data).reshape(shape)
        data_vars[var_name] = xr.DataArray(
            data,
            dims=("time", "latitude", "longitude"),
            coords=(time, latitude, longitude),
            attrs={"units": "K"},
        )

    return xr.Dataset(data_vars)


@pytest.fixture
def dataset():
    return make_dataset()


@pytest.fixture
def single_var_dataset():
    return make_dataset(vars=1)


def is_ipfs_running(gateway_url: str) -> bool:
    """Check if IPFS daemon Gateway is responsive."""

    try:
        # Check gateway root or /ipfs/ path - depends on gateway config
        # A lightweight check: try to access the root or a known path
        # Use a known immutable CID (e.g., the empty directory CID)
        # Let's try a known immutable path: "Hello from IPFS Gateway Checker"
        known_cid = "bafybeifx7yeb55armcsxwwitkymga5xf53dxiarykms3ygqic223w5sk3m"  # Example file
        response = httpx.head(
            f"{gateway_url.rstrip('/')}/ipfs/{known_cid}",
            timeout=5,
            follow_redirects=True,
        )
        # Allow 200 OK or 404 Not Found (if CID isn't locally available but gateway is up)
        # Avoid checking strict 200 as CID might not be pinned locally but gateway is running
        if response.status_code < 500:
            print(
                f"IPFS Gateway check successful (Status: {response.status_code}) at {gateway_url}"
            )
            return True
        else:
            print(
                f"IPFS Gateway check failed (Status: {response.status_code}) at {gateway_url}"
            )
            return False
    except httpx.ConnectError:
        print(f"IPFS Gateway connection failed at {gateway_url}")
        return False
    except httpx.TimeoutException:
        print(f"IPFS Gateway check timed out at {gateway_url}")
        return False
    except httpx.HTTPError as e:
        print(f"IPFS Gateway check failed with unexpected error: {e}")
        return False


def is_ipfs_rpc_running(rpc_url: str) -> bool:
    """Check whether the writable Kubo RPC API is responsive."""
    try:
        response = httpx.post(f"{rpc_url}/api/v0/id", timeout=5, follow_redirects=True)
        response.raise_for_status()
        payload = response.json()
        return isinstance(payload, dict) and bool(payload.get("ID"))
    except (httpx.HTTPError, ValueError):
        return False


def is_stac_pointer_running(catalog_url: str) -> bool:
    """Check whether the STAC pointer returns a non-empty root CID."""
    try:
        response = httpx.get(catalog_url, timeout=5, follow_redirects=True)
        response.raise_for_status()
        payload = response.json()
        return isinstance(payload, dict) and bool(payload.get("cid"))
    except (httpx.HTTPError, ValueError):
        return False


# Define known dataset IDs accessible via STAC for tests
KNOWN_STAC_DATASET_ID = "cpc-precip-conus"
KNOWN_STAC_DATASET_ID_2 = "chirps-final-p05"
KNOWN_STAC_DATASET_VAR = "precip"
KNOWN_STAC_COORD_LAT = 40.875
KNOWN_STAC_COORD_LON = -104.875
KNOWN_STAC_DATE = datetime.datetime(2023, 1, 1)
KNOWN_STAC_DATE_END = datetime.datetime(2023, 1, 5)

# Known dataset for forecast tests (if applicable and available via STAC)
# If no suitable forecast dataset is guaranteed via STAC, skip forecast tests or mock them
# KNOWN_STAC_FORECAST_ID = "gfs-temperature-forecast" # Fictional example
KNOWN_STAC_FORECAST_ID = None  # Set to None if no reliable forecast dataset via STAC
