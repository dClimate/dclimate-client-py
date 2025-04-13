import datetime
import itertools
import pathlib

import geopandas as gpd
import os
import numpy as np
import pytest
import xarray as xr
import requests  # Import requests here for the check
import zarr
import zarr.storage

HERE = pathlib.Path(__file__).parent
ETC = HERE / "etc"
SAMPLE_ZARRS = ETC / "sample_zarrs"


@pytest.fixture
def input_ds():
    # Keeping local fixtures for tests that don't need IPFS loading (like test_geotemporal_data)
    with zarr.storage.ZipStore(ETC / "retrieval_test.zip", mode="r") as in_zarr:
        return xr.open_zarr(in_zarr, chunks=None).compute()


@pytest.fixture
def forecast_ds():
    # Keeping local fixtures for tests that don't need IPFS loading
    with zarr.storage.ZipStore(
        ETC / "forecast_retrieval_test.zip", mode="r"
    ) as in_zarr:
        return xr.open_zarr(in_zarr, chunks=None).compute()


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


# --- Add IPFS Connection Check Fixture ---
# We need the check_ipfs_connection fixture available for multiple test files
# Let's reuse the one from test_integration_ipfs.py


@pytest.fixture(scope="session")  # Changed scope to session for efficiency
def ipfs_gateway_url():
    """Returns the IPFS gateway URL to check."""
    # Prioritize environment variable, then default from py-hamt's IPFSStore
    return os.environ.get("IPFS_GATEWAY_URI_STEM", "http://127.0.0.1:8080")


def is_ipfs_running(gateway_url: str) -> bool:
    """Check if IPFS daemon Gateway is responsive."""

    try:
        # Check gateway root or /ipfs/ path - depends on gateway config
        # A lightweight check: try to access the root or a known path
        # Use a known immutable CID (e.g., the empty directory CID)
        # Let's try a known immutable path: "Hello from IPFS Gateway Checker"
        known_cid = "bafybeifx7yeb55armcsxwwitkymga5xf53dxiarykms3ygqic223w5sk3m"  # Example file
        response = requests.head(
            f"{gateway_url.rstrip('/')}/ipfs/{known_cid}", timeout=5
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
    except requests.exceptions.ConnectionError:
        print(f"IPFS Gateway connection failed at {gateway_url}")
        return False
    except requests.exceptions.Timeout:
        print(f"IPFS Gateway check timed out at {gateway_url}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"IPFS Gateway check failed with unexpected error: {e}")
        return False


# Apply autouse=True to run this check once for the session for all tests
@pytest.fixture(scope="session", autouse=True)
def check_ipfs_connection(ipfs_gateway_url):
    """Skips tests if IPFS daemon Gateway is not accessible."""
    if not is_ipfs_running(ipfs_gateway_url):
        pytest.skip(
            f"IPFS daemon Gateway not responding at {ipfs_gateway_url}. Skipping integration tests."
        )
    else:
        print(
            f"IPFS daemon Gateway responding at {ipfs_gateway_url}. Proceeding with integration tests."
        )


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
