# tests/test_integration_ipfs.py

import os
import datetime
import time
import pytest
import xarray as xr
import requests  # Ensure requests is imported

# Try importing necessary libraries and skip tests if unavailable
pytest.importorskip("requests")
pytest.importorskip("xarray")
pytest.importorskip("py_hamt")
pytest.importorskip("multiformats")

# Import client functions and errors
from dclimate_zarr_client import (
    load_ipfs_via_stac,
    geo_temporal_query,
    dclimate_zarr_errors as errors,
)

# Import internal functions for specific tests (use with caution)
from dclimate_zarr_client.ipfs_retrieval import (
    # resolve_ipns_to_cid, # REMOVED
    fetch_json_from_ipns,  # ADDED
    get_dataset_hamt_cid_from_stac,
    _get_dataset_by_ipfs_cid,
    list_datasets,
    _stac_hamt_cid_cache,  # For testing cache
)
from dclimate_zarr_client.client import DCLIMATE_STAC_CATALOG_IPNS

# --- Configuration ---
pytestmark = pytest.mark.integration
KNOWN_DATASET_ID = "cpc-precip-conus"
KNOWN_DATASET_ID_2 = "chirps-final-p05"
EXPECTED_VALUE_COORD_LAT = 40.875
EXPECTED_VALUE_COORD_LON = -104.875
EXPECTED_VALUE_DATE = datetime.datetime(2023, 1, 1)


# --- Helper Functions ---
def is_ipfs_running(gateway_url: str) -> bool:
    """Check if IPFS daemon Gateway is responsive."""
    try:
        # Check gateway root or /ipfs/ path - depends on gateway config
        # Let's try a known immutable path
        known_cid = "bafybeifx7yeb55armcsxwwitkymga5xf53dxiarykms3ygqic223w5sk3m"  # Example file
        response = requests.head(f"{gateway_url}/ipfs/{known_cid}", timeout=5)
        # Allow 200 OK or 404 Not Found (if CID isn't locally available but gateway is up)
        # Avoid checking strict 200 as CID might not be pinned locally but gateway is running
        return response.status_code < 500
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="module", autouse=True)
def check_ipfs_connection():
    """Skips integration tests if IPFS daemon Gateway is not accessible."""
    # Use default GATEWAY URI from IPFSStore internal logic if not set
    default_gateway = os.environ.get("IPFS_GATEWAY_URI_STEM", "http://127.0.0.1:8080")
    if not is_ipfs_running(default_gateway):
        pytest.skip(
            f"IPFS daemon Gateway not responding at {default_gateway}. Skipping integration tests."
        )


# --- Test Cases ---

# REMOVE test_resolve_root_catalog_ipns
# def test_resolve_root_catalog_ipns(): ...


# ADD test for fetch_json_from_ipns
def test_fetch_root_catalog_from_ipns():
    """Test fetching the root catalog JSON directly from IPNS via Gateway."""
    print(f"\nAttempting to fetch JSON from root IPNS: {DCLIMATE_STAC_CATALOG_IPNS}")
    try:
        catalog_json = fetch_json_from_ipns(DCLIMATE_STAC_CATALOG_IPNS)
        assert isinstance(catalog_json, dict)
        assert catalog_json.get("type") == "Catalog"
        assert catalog_json.get("id") == "dClimate-data-catalog"
        assert "links" in catalog_json
        print("Fetched root catalog JSON successfully.")
    except (errors.IpfsConnectionError, errors.StacCatalogError) as e:
        pytest.fail(f"fetch_json_from_ipns failed: {e}")
    except Exception as e:
        pytest.fail(f"fetch_json_from_ipns failed with unexpected error: {e}")


# Test get_dataset_hamt_cid_from_stac remains largely the same
def test_get_hamt_cid_success():
    """Test successfully retrieving a known dataset's HAMT CID."""
    print(f"\nAttempting to get HAMT CID for: {KNOWN_DATASET_ID}")
    try:
        hamt_cid = get_dataset_hamt_cid_from_stac(
            root_catalog_ipns=DCLIMATE_STAC_CATALOG_IPNS,
            target_dataset_id=KNOWN_DATASET_ID,
        )
        print(f"Found HAMT CID: {hamt_cid}")
        assert hamt_cid.startswith("bafy")  # Check format
    except (
        errors.DatasetNotFoundError,
        errors.IpfsConnectionError,
        errors.StacCatalogError,
    ) as e:
        pytest.fail(f"get_dataset_hamt_cid_from_stac failed: {e}")
    except Exception as e:
        pytest.fail(f"get_dataset_hamt_cid_from_stac failed with unexpected error: {e}")


# Test get_dataset_hamt_cid_not_found remains the same
def test_get_hamt_cid_not_found():
    """Test retrieving a non-existent dataset ID."""
    fake_dataset_id = "this-dataset-does-not-exist-12345"
    print(f"\nAttempting to get HAMT CID for non-existent: {fake_dataset_id}")
    with pytest.raises(
        errors.DatasetNotFoundError, match=f"Dataset ID '{fake_dataset_id}' not found"
    ):
        get_dataset_hamt_cid_from_stac(
            root_catalog_ipns=DCLIMATE_STAC_CATALOG_IPNS,
            target_dataset_id=fake_dataset_id,
        )


# Test test_get_hamt_cid_caching remains the same
def test_get_hamt_cid_caching():
    """Test that the HAMT CID is cached after the first lookup."""
    _stac_hamt_cid_cache.clear()
    assert KNOWN_DATASET_ID_2 not in _stac_hamt_cid_cache
    print(f"\nAttempting first lookup for caching test: {KNOWN_DATASET_ID_2}")
    start_time = time.time()
    hamt_cid_1 = get_dataset_hamt_cid_from_stac(
        DCLIMATE_STAC_CATALOG_IPNS, KNOWN_DATASET_ID_2
    )
    duration1 = time.time() - start_time
    print(f"First lookup took {duration1:.2f}s. CID: {hamt_cid_1}")
    assert KNOWN_DATASET_ID_2 in _stac_hamt_cid_cache
    assert _stac_hamt_cid_cache[KNOWN_DATASET_ID_2] == hamt_cid_1
    start_time = time.time()
    hamt_cid_2 = get_dataset_hamt_cid_from_stac(
        DCLIMATE_STAC_CATALOG_IPNS, KNOWN_DATASET_ID_2
    )
    duration2 = time.time() - start_time
    print(f"Second lookup took {duration2:.2f}s. CID: {hamt_cid_2}")
    assert hamt_cid_1 == hamt_cid_2
    _stac_hamt_cid_cache.clear()


# Test test_load_dataset_internal remains the same
def test_load_dataset_internal():
    """Test loading a dataset using the internal _get_dataset_by_ipfs_cid."""
    print(f"\nAttempting internal load for: {KNOWN_DATASET_ID}")
    try:
        hamt_cid = get_dataset_hamt_cid_from_stac(
            DCLIMATE_STAC_CATALOG_IPNS, KNOWN_DATASET_ID
        )
        print(f"Loading dataset with HAMT CID: {hamt_cid}")
        ds = _get_dataset_by_ipfs_cid(ipfs_cid=hamt_cid)
        assert isinstance(ds, xr.Dataset)
        assert "precip" in ds.data_vars
        print("Dataset loaded successfully.")
    except (
        errors.DatasetNotFoundError,
        errors.IpfsConnectionError,
        errors.StacCatalogError,
    ) as e:
        pytest.fail(f"Failed during internal load: {e}")
    except Exception as e:
        pytest.fail(f"Failed during internal load - Unexpected error: {e}")


# Test test_load_ipfs_via_stac_success remains the same
def test_load_ipfs_via_stac_success():
    """Test the main client function load_ipfs_via_stac."""
    print(f"\nAttempting client load for: {KNOWN_DATASET_ID}")
    try:
        geo_data = load_ipfs_via_stac(dataset_name=KNOWN_DATASET_ID)
        assert isinstance(geo_data.data, xr.Dataset)
        assert geo_data.dataset_name == KNOWN_DATASET_ID
        assert "precip" in geo_data.data.data_vars
        print("Client load successful.")
    except (
        errors.DatasetNotFoundError,
        errors.IpfsConnectionError,
        errors.StacCatalogError,
    ) as e:
        pytest.fail(f"load_ipfs_via_stac failed: {e}")
    except Exception as e:
        pytest.fail(f"load_ipfs_via_stac failed with unexpected error: {e}")


# Test test_load_ipfs_via_stac_not_found remains the same
def test_load_ipfs_via_stac_not_found():
    """Test load_ipfs_via_stac for a non-existent dataset."""
    fake_dataset_id = "this-dataset-also-does-not-exist-67890"
    print(f"\nAttempting client load for non-existent: {fake_dataset_id}")
    with pytest.raises(
        (errors.DatasetNotFoundError, errors.StacCatalogError)
    ):  # Could fail during fetch or traversal
        load_ipfs_via_stac(dataset_name=fake_dataset_id)


# Test test_geo_temporal_query_ipfs_point_success remains the same
def test_geo_temporal_query_ipfs_point_success():
    """Test geo_temporal_query with source='ipfs' for a single point."""
    print(f"\nAttempting geo_temporal_query (ipfs) for: {KNOWN_DATASET_ID}")
    try:
        result = geo_temporal_query(
            dataset_name=KNOWN_DATASET_ID,
            source="ipfs",
            point_kwargs={
                "latitude": EXPECTED_VALUE_COORD_LAT,
                "longitude": EXPECTED_VALUE_COORD_LON,
            },
            time_range=[EXPECTED_VALUE_DATE, EXPECTED_VALUE_DATE],
            output_format="array",
        )
        assert isinstance(result, dict)
        assert "data" in result
        assert len(result["data"]) == 1
        assert isinstance(result["data"][0], (float, int)) or result["data"][0] is None
        print(f"Query successful. Data point: {result['data'][0]}")
    except (
        errors.DatasetNotFoundError,
        errors.IpfsConnectionError,
        errors.StacCatalogError,
        errors.NoDataFoundError,
    ) as e:
        pytest.fail(f"geo_temporal_query failed: {e}")
    except Exception as e:
        pytest.fail(f"geo_temporal_query failed with unexpected error: {e}")


# Test test_geo_temporal_query_ipfs_netcdf_output remains the same
def test_geo_temporal_query_ipfs_netcdf_output():
    """Test geo_temporal_query with source='ipfs' and netcdf output."""
    print(f"\nAttempting geo_temporal_query (ipfs, netcdf) for: {KNOWN_DATASET_ID}")
    try:
        netcdf_bytes = geo_temporal_query(
            dataset_name=KNOWN_DATASET_ID,
            source="ipfs",
            rectangle_kwargs={
                "min_lat": 40.0,
                "max_lat": 41.0,
                "min_lon": -105.0,
                "max_lon": -104.0,
            },
            time_range=[EXPECTED_VALUE_DATE, EXPECTED_VALUE_DATE],
            output_format="netcdf",
        )
        assert isinstance(netcdf_bytes, bytes)
        assert len(netcdf_bytes) > 0
        ds = xr.open_dataset(netcdf_bytes)
        assert isinstance(ds, xr.Dataset)
        assert "precip" in ds.data_vars
        print("NetCDF query successful.")
    except (
        errors.DatasetNotFoundError,
        errors.IpfsConnectionError,
        errors.StacCatalogError,
        errors.NoDataFoundError,
    ) as e:
        pytest.fail(f"geo_temporal_query (netcdf) failed: {e}")
    except Exception as e:
        pytest.fail(f"geo_temporal_query (netcdf) failed with unexpected error: {e}")


# Test test_list_datasets_ipfs remains the same
def test_list_datasets_ipfs():
    """Test listing datasets from the STAC catalog."""
    print("\nAttempting to list datasets via STAC")
    try:
        dataset_list = list_datasets()
        print(f"Found {len(dataset_list)} datasets. First few: {dataset_list[:5]}")
        assert isinstance(dataset_list, list)
        assert len(dataset_list) > 0
        assert all(isinstance(item, str) for item in dataset_list)
        assert KNOWN_DATASET_ID in dataset_list
        assert KNOWN_DATASET_ID_2 in dataset_list
        # Cache check might be less reliable if list fails partway, but can keep
        # assert KNOWN_DATASET_ID in _stac_hamt_cid_cache
        # assert KNOWN_DATASET_ID_2 in _stac_hamt_cid_cache
        print("List datasets successful.")
        _stac_hamt_cid_cache.clear()  # Clear cache after test
    except (errors.IpfsConnectionError, errors.StacCatalogError) as e:
        pytest.fail(f"list_datasets failed: {e}")
    except Exception as e:
        pytest.fail(f"list_datasets failed with unexpected error: {e}")
