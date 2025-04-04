import os
import datetime
import json
import pathlib

import pytest
import requests
from unittest.mock import patch, mock_open
from dclimate_zarr_client.ipfs_retrieval import (
    get_ipns_name_hash,
    DatasetNotFoundError,
    list_datasets,
    update_cache_if_changed,
)

import dclimate_zarr_client.ipfs_retrieval as ipfs_retrieval

# import xarray as xr
# import zarr
from dclimate_zarr_client.client import geo_temporal_query

IPNS_NAME_HASH = "k2k4r8niyotlqqqvqoh7jr4gp6zp0b0975k88zmak151chv87w2p11qz"


def patched_get_single_metadata(ipfs_hash):
    with open(
        pathlib.Path(__file__).parent / "etc" / "stac_metadata" / f"{ipfs_hash}.json"
    ) as f:
        return json.load(f)


def patched_resolve_ipns_name_hash(ipns_name_hash):
    return "bafyr4iff6vuvpkb4zexkx7exoafsjvfnno4ofidyc37gsto2aaxixg5zia"


# def patched_get_dataset_by_ipfs_hash(ipfs_hash):
#     with zarr.ZipStore(
#         pathlib.Path(__file__).parent / "etc" / "sample_zarrs" / f"{ipfs_hash}.zip",
#         mode="r",
#     ) as in_zarr:
#         return xr.open_zarr(in_zarr).compute()


@pytest.fixture(scope="module", autouse=True)
def default_session_fixture(module_mocker):
    """
    Patch metadata and Zarr retrieval functions in this test
    """
    module_mocker.patch(
        "dclimate_zarr_client.ipfs_retrieval._get_single_metadata",
        patched_get_single_metadata,
    )
    module_mocker.patch(
        "dclimate_zarr_client.ipfs_retrieval._resolve_ipns_name_hash",
        patched_resolve_ipns_name_hash,
    )
    # module_mocker.patch(
    #     "dclimate_zarr_client.ipfs_retrieval.get_dataset_by_ipfs_hash",
    #     patched_get_dataset_by_ipfs_hash,
    # )


def test_get_ipns_name_hash():
    """
    Test that `get_ipns_name_hash`  returns json of dataset names and hashes
    """
    ipns_name_hash = ipfs_retrieval.get_ipns_name_hash("cpc-precip-conus")
    # Assert that it starts with ba
    assert ipns_name_hash.startswith("ba")


# Success from Local Cache
def test_get_ipns_name_hash_fallback_success():
    """
    Test that when the remote CID endpoint is unreachable OR the key isn't found,
    we successfully fall back to the local cids_cache.json.
    """
    # We'll simulate "requests.get" raising a RequestException,
    # so we go straight to the fallback code.
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException(
            "Simulated RequestException"
        )

        # Next, we mock "os.path.exists" to return True, as if cids_cache.json does exist
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            # Provide a cids_cache.json that DOES have the desired entry
            fake_json = '{"cpc-precip-conus":"bafkreihashfromlocalfile"}'
            with patch("builtins.open", mock_open(read_data=fake_json)):
                ipns_name_hash = get_ipns_name_hash("cpc-precip-conus")
                assert ipns_name_hash == "bafkreihashfromlocalfile"


# Failure When Local Cache Does Not Exist
def test_get_ipns_name_hash_fallback_no_file():
    """
    Test that if requests.get fails and the local fallback file does NOT exist,
    we raise DatasetNotFoundError.
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException(
            "Simulated RequestException"
        )

        # cids_cache.json does NOT exist
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            with pytest.raises(DatasetNotFoundError):
                get_ipns_name_hash("some-nonexistent-key")


# Failure When Local Cache Exists But Missing Key
def test_get_ipns_name_hash_fallback_key_not_found_in_file():
    """
    Even if cids_cache.json exists, if the key is not found,
    raise DatasetNotFoundError.
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException(
            "Simulated RequestException"
        )

        with patch("os.path.exists", return_value=True):
            fake_json = '{"some-other-key":"bafkreihashfromlocalfile"}'
            with patch("builtins.open", mock_open(read_data=fake_json)):
                with pytest.raises(DatasetNotFoundError) as exc:
                    get_ipns_name_hash("cpc-precip-conus")

                assert "Invalid dataset name" in str(exc.value)


def test_get_ipns_name_hash_endpoint_missing_key():
    """
    Test the case where the endpoint is reachable, but the JSON
    does NOT contain the requested key. Ensure it falls back to local cache.
    If the local cache also doesn't have the key, raise DatasetNotFoundError.
    """
    # Mock a valid endpoint JSON that doesn't contain the desired key
    endpoint_data = {"some-other-key": "bafy-other"}
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.return_value.json.return_value = endpoint_data
        mock_requests_get.return_value.raise_for_status.return_value = None

        # Also mock local file to not have the key
        with patch("os.path.exists", return_value=True):
            fake_json = '{"still-some-other-key":"bafkrei-another"}'
            with patch("builtins.open", mock_open(read_data=fake_json)):
                with pytest.raises(DatasetNotFoundError) as exc:
                    get_ipns_name_hash("cpc-precip-conus")
                assert "Invalid dataset name" in str(exc.value)


def test_get_ipns_name_hash_endpoint_malformed_json():
    """
    Test the case where the endpoint returns malformed JSON,
    triggering a JSONDecodeError. We should then fall back to local cache.
    """
    with patch("requests.get") as mock_requests_get:
        # Simulate valid status but invalid JSON content
        mock_requests_get.return_value.text = "INVALID JSON!!"
        mock_requests_get.return_value.raise_for_status.return_value = None
        mock_requests_get.return_value.json.side_effect = json.JSONDecodeError(
            "Expecting value", "doc", 0
        )

        # Provide a valid local cache to ensure fallback works
        with patch("os.path.exists", return_value=True):
            fake_json = '{"cpc-precip-conus":"bafkreihashfromlocalfile"}'
            with patch("builtins.open", mock_open(read_data=fake_json)):
                ipns_name_hash = get_ipns_name_hash("cpc-precip-conus")
                assert ipns_name_hash == "bafkreihashfromlocalfile"


def test_get_ipns_name_hash_local_cache_malformed_json():
    """
    Test that if requests.get fails AND the local cache is malformed JSON,
    we raise DatasetNotFoundError.
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException("Simulated error")

        with patch("os.path.exists", return_value=True):
            # Local file is present but has invalid JSON
            with patch("builtins.open", mock_open(read_data="INVALID JSON!!")):
                with pytest.raises(DatasetNotFoundError):
                    get_ipns_name_hash("cpc-precip-conus")


def test_get_ipns_name_hash_local_cache_empty():
    """
    Test that if requests.get fails AND the local cache file is empty,
    we raise DatasetNotFoundError (because there's no valid data).
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException("Simulated error")

        with patch("os.path.exists", return_value=True):
            # Local file is present but empty
            with patch("builtins.open", mock_open(read_data="")):
                with pytest.raises(DatasetNotFoundError):
                    get_ipns_name_hash("cpc-precip-conus")


def test_list_datasets():
    """
    Test that `list_datasets` returns a list of dataset keys
    """
    datasets = ipfs_retrieval.list_datasets()
    assert len(datasets) == 3
    assert "cpc-precip-conus" in datasets


def test_list_datasets_fallback_success():
    """
    Test that if the endpoint is unreachable or fails,
    list_datasets() returns keys from the local cache.
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException("Simulated error")

        with patch("os.path.exists", return_value=True):
            fake_json = '{"cpc-precip-conus":"bafy-hash","other-dataset":"bafy-other"}'
            with patch("builtins.open", mock_open(read_data=fake_json)):
                datasets = list_datasets()
                assert "cpc-precip-conus" in datasets
                assert "other-dataset" in datasets


def test_list_datasets_endpoint_malformed_json():
    """
    Test that if the endpoint returns invalid JSON,
    list_datasets() attempts fallback.
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.return_value.text = "INVALID JSON!!"
        mock_requests_get.return_value.raise_for_status.return_value = None
        mock_requests_get.return_value.json.side_effect = json.JSONDecodeError(
            "Expecting value", "doc", 0
        )

        # Provide valid local cache
        with patch("os.path.exists", return_value=True):
            fake_json = '{"ds1":"bafy1","ds2":"bafy2"}'
            with patch("builtins.open", mock_open(read_data=fake_json)):
                datasets = list_datasets()
                assert datasets == ["ds1", "ds2"]


def test_list_datasets_no_fallback_file():
    """
    Test that if the endpoint fails AND no local cache file is found,
    list_datasets() raises RuntimeError.
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException("Simulated error")

        with patch("os.path.exists", return_value=False):
            with pytest.raises(RuntimeError) as exc:
                list_datasets()
            assert "Failed to retrieve dataset list" in str(exc.value)


def test_list_datasets_local_cache_malformed_json():
    """
    Test that if the endpoint fails AND local cache file has malformed JSON,
    list_datasets() raises RuntimeError because it cannot parse the local file.
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException("Simulated error")

        with patch("os.path.exists", return_value=True):
            # Malformed JSON
            with patch("builtins.open", mock_open(read_data="INVALID JSON")):
                with pytest.raises(RuntimeError) as exc:
                    list_datasets()
                assert "Failed to retrieve dataset list" in str(exc.value)


def test_list_datasets_local_cache_empty():
    """
    Test that if the endpoint fails AND local cache file is empty,
    list_datasets() raises RuntimeError (no data to parse).
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException("Simulated error")

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="")):
                with pytest.raises(RuntimeError) as exc:
                    list_datasets()
                assert "Failed to retrieve dataset list" in str(exc.value)


def test_geo_temporal_query():
    ds_bytes = geo_temporal_query(
        "cpc-precip-conus",
        point_kwargs={"latitude": 40.875, "longitude": -104.875},
        time_range=[datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 1)],
    )
    print(ds_bytes)
    assert ds_bytes["data"][0] == 0.7991394996643066

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


def get_cache_path():
    import dclimate_zarr_client.ipfs_retrieval as ipfs_retrieval

    return os.path.join(os.path.dirname(ipfs_retrieval.__file__), "cids.json")


def test_update_cache_no_update(monkeypatch):
    """
    If the cached data is identical to new data,
    update_cache_if_changed should not write to the file.
    """
    cached_data = {"dataset": "hash1"}
    new_data = {"dataset": "hash1"}
    file_path = get_cache_path()

    # Prepare a mock_open that returns the cached data for reading.
    m = mock_open(read_data=json.dumps(cached_data))
    monkeypatch.setattr("builtins.open", m)

    update_cache_if_changed(new_data)

    # Since the data hasn't changed, only a read call should occur.
    # (i.e. open() should be called once in "r" mode.)
    assert m.call_count == 1
    m.assert_called_with(file_path, "r")


def test_update_cache_update(monkeypatch):
    """
    If the cached data differs from new data,
    update_cache_if_changed should update (write) the cache file.
    """
    cached_data = {"dataset": "hash1"}
    new_data = {"dataset": "hash2"}
    file_path = get_cache_path()

    # Prepare a mock_open with the initial cached data.
    m = mock_open(read_data=json.dumps(cached_data))
    monkeypatch.setattr("builtins.open", m)

    update_cache_if_changed(new_data)

    # There should be two calls: one to read and one to write.
    assert m.call_count == 2

    calls = m.call_args_list
    # First call: reading the file
    assert calls[0][0] == (file_path, "r")
    # Second call: writing the new data
    assert calls[1][0] == (file_path, "w")


def test_update_cache_file_not_found(monkeypatch):
    """
    If the cache file doesn't exist (i.e. FileNotFoundError is raised during reading),
    update_cache_if_changed should write the new data.
    """
    new_data = {"dataset": "hash2"}
    file_path = get_cache_path()

    # Create a mock_open whose first call (read) raises FileNotFoundError,
    # and whose second call (write) succeeds.
    m = mock_open()
    m.side_effect = [FileNotFoundError, m.return_value]
    monkeypatch.setattr("builtins.open", m)

    update_cache_if_changed(new_data)

    # There should be two calls: the first attempt (reading) fails, and then writing occurs.
    assert m.call_count == 2

    calls = m.call_args_list
    assert calls[0][0] == (file_path, "r")
    assert calls[1][0] == (file_path, "w")
