import geopandas as gpd
import pytest
from shapely.geometry import Point

from dclimate_client_py import dclimate_zarr_errors as errors
from dclimate_client_py.geotemporal_data import GeotemporalData


def test_point_without_snap_uses_one_e_minus_five_tolerance(dataset):
    data = GeotemporalData(dataset, dataset_name="tolerance test")
    latitude = float(dataset.latitude.values[0])
    longitude = float(dataset.longitude.values[0])

    with pytest.raises(errors.NoDataFoundError):
        data.point(latitude + 5e-5, longitude, snap_to_grid=False)

    selected = data.point(latitude + 5e-6, longitude, snap_to_grid=False)
    assert selected.data.latitude.item() == latitude
    assert selected.data.longitude.item() == longitude


def test_points_without_snap_uses_one_e_minus_five_tolerance(dataset):
    data = GeotemporalData(dataset, dataset_name="tolerance test")
    latitude = float(dataset.latitude.values[0])
    longitude = float(dataset.longitude.values[0])

    off_grid_mask = gpd.GeoSeries(
        [Point(longitude, latitude + 5e-5)], crs=4326
    ).geometry.values
    with pytest.raises(errors.NoDataFoundError):
        data.points(off_grid_mask, epsg_crs=4326, snap_to_grid=False)

    within_tolerance_mask = gpd.GeoSeries(
        [Point(longitude, latitude + 5e-6)], crs=4326
    ).geometry.values
    selected = data.points(
        within_tolerance_mask, epsg_crs=4326, snap_to_grid=False
    )
    assert selected.data.latitude.item() == latitude
    assert selected.data.longitude.item() == longitude
