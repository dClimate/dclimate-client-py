import pytest
import xarray as xr

import dclimate_client_py.s3_retrieval as s3_retrieval


def test_get_dataset_from_s3_accepts_missing_update_in_progress(monkeypatch):
    dataset = xr.Dataset({"temperature": ("time", [1.0, 2.0])})

    monkeypatch.setattr(s3_retrieval, "get_s3_fs", lambda: object())
    monkeypatch.setattr(s3_retrieval, "S3Map", lambda *args, **kwargs: object())
    monkeypatch.setattr(s3_retrieval.xr, "open_zarr", lambda *args, **kwargs: dataset)

    result = s3_retrieval.get_dataset_from_s3("temperature", "test-bucket")

    assert result is dataset


def _patch_s3(monkeypatch, dataset):
    monkeypatch.setattr(s3_retrieval, "get_s3_fs", lambda: object())
    monkeypatch.setattr(s3_retrieval, "S3Map", lambda *args, **kwargs: object())
    monkeypatch.setattr(s3_retrieval.xr, "open_zarr", lambda *args, **kwargs: dataset)


def test_get_dataset_from_s3_initial_parse_raises(monkeypatch):
    dataset = xr.Dataset(
        {"temperature": ("time", [1.0, 2.0])},
        attrs={"update_in_progress": True, "initial_parse": True},
    )
    _patch_s3(monkeypatch, dataset)

    with pytest.raises(s3_retrieval.DatasetNotFoundError, match="initial parse"):
        s3_retrieval.get_dataset_from_s3("temperature", "test-bucket")


def test_get_dataset_from_s3_append_only_update_slices_to_previous_end(monkeypatch):
    times = xr.date_range("2020-01-01", periods=4, freq="h")
    dataset = xr.Dataset(
        {"temperature": ("time", [1.0, 2.0, 3.0, 4.0])},
        coords={"time": times},
        attrs={
            "update_in_progress": True,
            "update_is_append_only": True,
            "date range": ["2020010100", "2020010103"],
            "update_previous_end_date": "2020010102",
        },
    )
    _patch_s3(monkeypatch, dataset)

    result = s3_retrieval.get_dataset_from_s3("temperature", "test-bucket")

    assert result.sizes["time"] == 3


def test_get_dataset_from_s3_full_update_uses_date_range(monkeypatch):
    times = xr.date_range("2020-01-01", periods=4, freq="h")
    dataset = xr.Dataset(
        {"temperature": ("time", [1.0, 2.0, 3.0, 4.0])},
        coords={"time": times},
        attrs={
            "update_in_progress": True,
            "update_is_append_only": False,
            "date range": ["2020010100", "2020010101"],
        },
    )
    _patch_s3(monkeypatch, dataset)

    result = s3_retrieval.get_dataset_from_s3("temperature", "test-bucket")

    assert result.sizes["time"] == 2
