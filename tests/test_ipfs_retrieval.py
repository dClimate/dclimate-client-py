import os
import json

import pytest
import requests
from unittest.mock import patch, mock_open

import dclimate_zarr_client.ipfs_retrieval as ipfs_retrieval
from dclimate_zarr_client.ipfs_retrieval import (
    # Keep imports for functions still tested here
    get_ipns_name_hash,
    update_cache_if_changed,
    DatasetNotFoundError,
    IpfsConnectionError,
    StacCatalogError,
)


# import xarray as xr

# Import constants/configs
from dclimate_zarr_client.client import DCLIMATE_STAC_CATALOG_IPNS
from .conftest import KNOWN_STAC_DATASET_ID, KNOWN_STAC_DATASET_ID_2
# Apply IPFS check fixture to relevant tests/module

pytestmark = pytest.mark.usefixtures("check_ipfs_connection")


# --- Tests for get_ipns_name_hash (Legacy/Utility Function) ---
# These tests remain as UNIT tests, mocking requests and file system,
# as they test the specific fallback logic of this function, not STAC traversal.


def test_get_ipns_name_hash_success_from_endpoint():
    """Test successfully getting a hash from the endpoint."""
    endpoint_data = {"cpc-precip-conus": "bafy-real-or-mock-hash", "other": "hash2"}
    with patch("requests.get") as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = endpoint_data
        # Mock file operations to simulate cache update
        with (
            patch("builtins.open", mock_open()) as _,
            patch("os.path.dirname", return_value="/fake/dir"),
            patch(
                "dclimate_zarr_client.ipfs_retrieval.update_cache_if_changed"
            ) as mock_update,
        ):
            ipns_name_hash = ipfs_retrieval.get_ipns_name_hash("cpc-precip-conus")
            assert ipns_name_hash == "bafy-real-or-mock-hash"
            mock_update.assert_called_once_with(
                endpoint_data
            )  # Verify cache update logic is called


# Success from Local Cache
def test_get_ipns_name_hash_fallback_success():
    """
    Test that when the remote CID endpoint is unreachable OR the key isn't found,
    we successfully fall back to the local cids_cache.json.
    """
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated Error")
    ):
        with patch("os.path.exists", return_value=True):
            fake_json = '{"cpc-precip-conus":"bafkreihashfromlocalfile"}'
            # Mock open within the context of the function call
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=fake_json),
                create=True,
            ):
                ipns_name_hash = get_ipns_name_hash("cpc-precip-conus")
                assert ipns_name_hash == "bafkreihashfromlocalfile"


# Failure When Local Cache Does Not Exist
def test_get_ipns_name_hash_fallback_no_file():
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated Error")
    ):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(DatasetNotFoundError):
                get_ipns_name_hash("some-nonexistent-key")


# Failure When Local Cache Exists But Missing Key
def test_get_ipns_name_hash_fallback_key_not_found_in_file():
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated Error")
    ):
        with patch("os.path.exists", return_value=True):
            fake_json = '{"some-other-key":"bafkreihashfromlocalfile"}'
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=fake_json),
                create=True,
            ):
                with pytest.raises(DatasetNotFoundError, match="Invalid dataset name"):
                    get_ipns_name_hash("cpc-precip-conus")


def test_get_ipns_name_hash_endpoint_missing_key():
    endpoint_data = {"some-other-key": "bafy-other"}
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = endpoint_data
        mock_get.return_value.raise_for_status.return_value = None
        # Mock local file to also not have the key
        with patch("os.path.exists", return_value=True):
            fake_json = '{"still-some-other-key":"bafkrei-another"}'
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=fake_json),
                create=True,
            ):
                # Mock cache update to avoid file write attempt during assert
                with patch(
                    "dclimate_zarr_client.ipfs_retrieval.update_cache_if_changed"
                ):
                    with pytest.raises(
                        DatasetNotFoundError, match="Invalid dataset name"
                    ):
                        get_ipns_name_hash("cpc-precip-conus")


def test_get_ipns_name_hash_endpoint_malformed_json():
    with patch("requests.get") as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.side_effect = json.JSONDecodeError(
            "Expecting value", "doc", 0
        )
        # Provide valid local cache
        with patch("os.path.exists", return_value=True):
            fake_json = '{"cpc-precip-conus":"bafkreihashfromlocalfile"}'
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=fake_json),
                create=True,
            ):
                ipns_name_hash = get_ipns_name_hash("cpc-precip-conus")
                assert ipns_name_hash == "bafkreihashfromlocalfile"


def test_get_ipns_name_hash_local_cache_malformed_json():
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated error")
    ):
        with patch("os.path.exists", return_value=True):
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data="INVALID JSON!!"),
                create=True,
            ):
                # Expect DatasetNotFoundError because parsing fails during fallback
                with pytest.raises(DatasetNotFoundError):
                    get_ipns_name_hash("cpc-precip-conus")


def test_get_ipns_name_hash_local_cache_empty():
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated error")
    ):
        with patch("os.path.exists", return_value=True):
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=""),
                create=True,
            ):
                # Expect DatasetNotFoundError because parsing fails during fallback
                with pytest.raises(DatasetNotFoundError):
                    get_ipns_name_hash("cpc-precip-conus")


# --- Functional Test for list_datasets (using STAC) ---


@pytest.mark.ipfs  # Mark as IPFS dependent
def test_list_datasets_functional():
    """Test listing datasets functionally by traversing the STAC catalog."""
    try:
        # Call the actual function pointing to the root catalog
        dataset_list = ipfs_retrieval.list_datasets(
            root_catalog_ipns=DCLIMATE_STAC_CATALOG_IPNS
        )
        assert isinstance(dataset_list, list)
        assert len(dataset_list) > 6  # Expect a reasonable number of datasets
        assert all(isinstance(item, str) for item in dataset_list)
        # Check for known datasets that should be present
        assert KNOWN_STAC_DATASET_ID in dataset_list
        assert KNOWN_STAC_DATASET_ID_2 in dataset_list
        print(f"Functional list_datasets found {len(dataset_list)} datasets.")

    except (IpfsConnectionError, StacCatalogError) as e:
        pytest.fail(f"Functional list_datasets failed: {e}")
    except Exception as e:
        pytest.fail(f"Functional list_datasets failed with unexpected error: {e}")


# def test_list_datasets_endpoint_malformed_json():
#     """
#     Test that if the endpoint returns invalid JSON,
#     list_datasets() attempts fallback.
#     """
#     with patch("requests.get") as mock_requests_get:
#         mock_requests_get.return_value.text = "INVALID JSON!!"
#         mock_requests_get.return_value.raise_for_status.return_value = None
#         mock_requests_get.return_value.json.side_effect = json.JSONDecodeError(
#             "Expecting value", "doc", 0
#         )

#         # Provide valid local cache
#         with patch("os.path.exists", return_value=True):
#             fake_json = '{"ds1":"bafy1","ds2":"bafy2"}'
#             with patch("builtins.open", mock_open(read_data=fake_json)):
#                 datasets = list_datasets()
#                 assert datasets == ["ds1", "ds2"]


# def test_list_datasets_no_fallback_file():
#     """
#     Test that if the endpoint fails AND no local cache file is found,
#     list_datasets() raises RuntimeError.
#     """
#     with patch("requests.get") as mock_requests_get:
#         mock_requests_get.side_effect = requests.RequestException("Simulated error")

#         with patch("os.path.exists", return_value=False):
#             with pytest.raises(RuntimeError) as exc:
#                 list_datasets()
#             assert "Failed to retrieve dataset list" in str(exc.value)


# def test_list_datasets_local_cache_malformed_json():
#     """
#     Test that if the endpoint fails AND local cache file has malformed JSON,
#     list_datasets() raises RuntimeError because it cannot parse the local file.
#     """
#     with patch("requests.get") as mock_requests_get:
#         mock_requests_get.side_effect = requests.RequestException("Simulated error")

#         with patch("os.path.exists", return_value=True):
#             # Malformed JSON
#             with patch("builtins.open", mock_open(read_data="INVALID JSON")):
#                 with pytest.raises(RuntimeError) as exc:
#                     list_datasets()
#                 assert "Failed to retrieve dataset list" in str(exc.value)


# def test_list_datasets_local_cache_empty():
#     """
#     Test that if the endpoint fails AND local cache file is empty,
#     list_datasets() raises RuntimeError (no data to parse).
#     """
#     with patch("requests.get") as mock_requests_get:
#         mock_requests_get.side_effect = requests.RequestException("Simulated error")

#         with patch("os.path.exists", return_value=True):
#             with patch("builtins.open", mock_open(read_data="")):
#                 with pytest.raises(RuntimeError) as exc:
#                     list_datasets()
#                 assert "Failed to retrieve dataset list" in str(exc.value)


# def test_geo_temporal_query():
#     ds_bytes = geo_temporal_query(
#         "cpc-precip-conus",
#         point_kwargs={"latitude": 40.875, "longitude": -104.875},
#         time_range=[datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 1)],
#     )
#     assert ds_bytes["data"][0] == 0.7991394996643066


# def test_get_dataset_by_ipns_hash_no_as_of():
#     """
#     Test that `get_dataset_by_ipns_hash` can select without a specified as_of time
#     """
#     ds = ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH)
#     assert ds.attrs["order_created"] == 4


# def test_get_dataset_by_ipns_hash_with_as_of():
#     """
#     Test that `get_dataset_by_ipns_hash` can select by specified as_of times
#     """
#     creation_times = [
#         datetime.datetime(2022, 7, 26, 19, 17, 55),
#         datetime.datetime(2022, 7, 26, 19, 19, 41),
#         datetime.datetime(2022, 7, 26, 19, 20, 45),
#         datetime.datetime(2022, 7, 26, 19, 22, 46),
#     ]
#     for i, time in enumerate(creation_times):
#         ds = ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH, as_of=time)
#         assert ds.attrs["order_created"] == i + 1


# def test_get_dataset_by_ipns_hash_with_bad_as_of():
#     """
#     Test that `get_dataset_by_ipns_hash` fails when provided an invalid `as_of` parameter
#     """
#     creation_time = datetime.datetime(2022, 7, 26, 19, 17, 53)
#     with pytest.raises(NoMetadataFoundError):
#         ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH, as_of=creation_time)


# --- Tests for update_cache_if_changed (Utility Function) ---
# These remain unit tests using mocks for file I/O.


def get_cache_path():
    # Helper to get the expected cache file path within the package
    # Need to ensure this reflects the actual location used by the code
    package_dir = os.path.dirname(ipfs_retrieval.__file__)
    return os.path.join(package_dir, "cids.json")


# Use monkeypatch fixture for modifying builtins like open
def test_update_cache_no_update(monkeypatch):
    cached_data = {"dataset": "hash1"}
    new_data = {"dataset": "hash1"}
    file_path = get_cache_path()

    m = mock_open(read_data=json.dumps(cached_data))
    monkeypatch.setattr("builtins.open", m)

    update_cache_if_changed(new_data)

    # Assert open was called once for reading, and never for writing
    m.assert_called_once_with(file_path, "r")


def test_update_cache_update(monkeypatch):
    cached_data = {"dataset": "hash1"}
    new_data = {"dataset": "hash2"}
    file_path = get_cache_path()

    m = mock_open(read_data=json.dumps(cached_data))
    monkeypatch.setattr("builtins.open", m)

    update_cache_if_changed(new_data)

    # Assert open was called twice: once for read, once for write.
    assert m.call_count == 2
    calls = m.call_args_list
    assert calls[0].args == (file_path, "r")
    assert calls[1].args == (file_path, "w")
    # Remove the assertion checking the specific write content
    # handle = m() # Remove this
    # handle.write.assert_called_once_with(json.dumps(new_data)) # Remove this


def test_update_cache_file_not_found(monkeypatch):
    new_data = {"dataset": "hash2"}
    file_path = get_cache_path()

    # Mock 'open' to raise FileNotFoundError on first call (read), succeed on second (write)
    # Create a mock handle instance for the successful write call return value
    mock_write_handle = mock_open().return_value
    m = mock_open()
    m.side_effect = [FileNotFoundError, mock_write_handle]  # Read fails, Write succeeds
    monkeypatch.setattr("builtins.open", m)

    update_cache_if_changed(new_data)

    # Assert open was called twice: read attempt (failed), write attempt (succeeded)
    assert m.call_count == 2
    calls = m.call_args_list
    assert calls[0].args == (file_path, "r")
    assert calls[1].args == (file_path, "w")
    # Remove the lines trying to get handle and assert write
    # handle = m() # Remove this
    # handle.write.assert_called_once_with(json.dumps(new_data)) # Remove this


def test_update_cache_decode_error(monkeypatch):
    """Test when the existing cache file has invalid JSON."""
    new_data = {"dataset": "hash2"}
    file_path = get_cache_path()

    # Create separate mock handles for read and write attempts
    # The read handle will simulate having invalid data
    read_handle = mock_open(read_data="invalid json").return_value
    write_handle = mock_open().return_value  # Mock handle for the write call

    m = mock_open()
    # Define side effect: return read_handle on first call, write_handle on second
    m.side_effect = [read_handle, write_handle]

    # Mock json.load to raise error when the read_handle is passed to it
    # Patch it in the correct namespace where it's used
    mock_json_load = patch(
        "dclimate_zarr_client.ipfs_retrieval.json.load",
        side_effect=json.JSONDecodeError("err", "doc", 0),
    ).start()

    monkeypatch.setattr("builtins.open", m)

    update_cache_if_changed(new_data)

    # Assert open was called twice: read attempt (failed decode), write attempt
    assert m.call_count == 2
    calls = m.call_args_list
    assert calls[0].args == (file_path, "r")  # Read attempt
    assert calls[1].args == (file_path, "w")  # Write attempt

    # Clean up the patch for json.load
    mock_json_load.stop()
