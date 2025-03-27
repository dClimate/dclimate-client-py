import datetime
import itertools
import pathlib

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import zarr
import zarr.storage

HERE = pathlib.Path(__file__).parent
ETC = HERE / "etc"
SAMPLE_ZARRS = ETC / "sample_zarrs"


@pytest.fixture
def input_ds():
    with zarr.storage.ZipStore(ETC / "retrieval_test.zip", mode="r") as in_zarr:
        return xr.open_zarr(in_zarr, chunks=None).compute()


@pytest.fixture
def forecast_ds():
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
