from __future__ import annotations

from aiobotocore import session
from functools import lru_cache
import datetime
import os
import typing
import json
import xarray as xr

from dclimate_client_py.dclimate_zarr_errors import DatasetNotFoundError

if typing.TYPE_CHECKING:
    from s3fs import S3FileSystem


def __getattr__(name: str):
    if name in {"S3FileSystem", "S3Map"}:
        import s3fs

        value = getattr(s3fs, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@lru_cache(maxsize=1)
def get_aio_session():
    return session.AioSession(profile=os.environ["ZARR_AWS_PROFILE_NAME"])


def get_s3_fs() -> S3FileSystem:
    """Gets an S3 filesystem based on provided credentials

    Returns:
        S3FileSystem:
    """
    s3_file_system = globals().get("S3FileSystem")
    if s3_file_system is None:
        s3_file_system = __getattr__("S3FileSystem")

    if "ZARR_AWS_PROFILE_NAME" in os.environ:
        return s3_file_system(session=get_aio_session())
    elif "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        return s3_file_system(
            key=os.environ["AWS_ACCESS_KEY_ID"],
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
    else:
        return s3_file_system(anon=False)


def get_dataset_from_s3(dataset_name: str, bucket_name: str) -> xr.Dataset:
    """Get a dataset from s3 from its name

    Args:
        dataset_name (str): key for datasets
        bucket_name (str): bucket name from where the datasets are fetched

    Returns:
        xr.Dataset: dataset corresponding to key
    """
    try:
        s3_map_type = globals().get("S3Map")
        if s3_map_type is None:
            s3_map_type = __getattr__("S3Map")
        s3_map = s3_map_type(
            f"s3://{bucket_name}/datasets/{dataset_name}.zarr",
            s3=get_s3_fs(),
        )
        ds = xr.open_zarr(s3_map, chunks=None)
    except FileNotFoundError:
        raise DatasetNotFoundError(f"Invalid dataset name {dataset_name}")

    attrs = getattr(ds, "attrs", {})
    if attrs.get(
        "update_in_progress", getattr(ds, "update_in_progress", False)
    ):
        if attrs.get("initial_parse", getattr(ds, "initial_parse", False)):
            raise DatasetNotFoundError(
                f"Dataset {dataset_name} is undergoing initial parse, retry request later"
            )
        if attrs.get(
            "update_is_append_only", getattr(ds, "update_is_append_only", False)
        ):
            start, end = ds.attrs["date range"][0], ds.attrs["update_previous_end_date"]
        else:
            start, end = ds.attrs["date range"]
        date_range = slice(
            *[datetime.datetime.strptime(t, "%Y%m%d%H") for t in (start, end)]
        )
        if "time" in ds:
            ds = ds.sel(time=date_range)
        elif "forecast_reference_time" in ds:
            ds = ds.sel(forecast_reference_time=date_range)

    return ds


def list_s3_datasets(bucket_name: str) -> typing.List[str]:
    """List all datasets available over s3

     Args:
        bucket_name (str): bucket name from where the datasets are fetched

    Returns:
        list[str]: available datasets
    """
    s3 = get_s3_fs()
    root_keys = s3.ls(f"s3://{bucket_name}/datasets")
    file_names = [key.split("/")[-1] for key in root_keys]
    zarr_names = [name[:-5] for name in file_names if name.endswith(".zarr")]
    return zarr_names


def get_metadata_by_s3_key(key: str, bucket_name: str) -> dict:
    """Get metadata for specific dataset

    Args:
        key (str): dataset key
        bucket_name (str): bucket name from where the datasets are fetched

    Returns:
        dict: metadata corresponding to key
    """
    s3 = get_s3_fs()
    try:
        attr_text = s3.cat(f"s3://{bucket_name}/datasets/{key}.zarr/.zattrs")
    except FileNotFoundError:
        raise DatasetNotFoundError("Invalid dataset name")
    return json.loads(attr_text)
