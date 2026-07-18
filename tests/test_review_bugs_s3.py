import pytest
import xarray as xr

import dclimate_client_py.s3_retrieval as s3_retrieval


@pytest.fixture(autouse=True)
def check_ipfs_connection():
    """Keep these regression tests independent of the local IPFS gateway."""
    pass


def test_get_dataset_from_s3_accepts_missing_update_in_progress(monkeypatch):
    dataset = xr.Dataset({"temperature": ("time", [1.0, 2.0])})

    monkeypatch.setattr(s3_retrieval, "get_s3_fs", lambda: object())
    monkeypatch.setattr(s3_retrieval, "S3Map", lambda *args, **kwargs: object())
    monkeypatch.setattr(s3_retrieval.xr, "open_zarr", lambda *args, **kwargs: dataset)

    result = s3_retrieval.get_dataset_from_s3("temperature", "test-bucket")

    assert result is dataset
