"""Regression tests for query keyword validation at the public boundary."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from dclimate_client_py import dclimate_zarr_errors as errors
from dclimate_client_py.geotemporal_data import GeotemporalData


@pytest.fixture
def geotemporal_data() -> GeotemporalData:
    dataset = xr.Dataset(
        {
            "temperature": (
                ("latitude", "longitude"),
                np.arange(9, dtype=float).reshape(3, 3),
            )
        },
        coords={
            "latitude": [39.0, 40.0, 41.0],
            "longitude": [-75.0, -74.0, -73.0],
        },
    )
    return GeotemporalData(dataset, dataset_name="synthetic")


def test_query_rejects_circle_kwargs_missing_lon(
    geotemporal_data: GeotemporalData,
) -> None:
    with pytest.raises(errors.InvalidSelectionError, match="lon"):
        geotemporal_data.query(circle_kwargs={"lat": 40.0})


def test_query_rejects_none_required_circle_value(
    geotemporal_data: GeotemporalData,
) -> None:
    with pytest.raises(errors.InvalidSelectionError, match="lat"):
        geotemporal_data.query(
            circle_kwargs={"lat": None, "lon": -74.0, "radius": 10.0}
        )


def test_query_rejects_empty_point_kwargs(
    geotemporal_data: GeotemporalData,
) -> None:
    with pytest.raises(errors.InvalidSelectionError, match="latitude"):
        geotemporal_data.query(point_kwargs={})


def test_query_rejects_rectangle_kwargs_missing_max_lon(
    geotemporal_data: GeotemporalData,
) -> None:
    with pytest.raises(errors.InvalidSelectionError, match="max_lon"):
        geotemporal_data.query(
            rectangle_kwargs={
                "min_lat": 39.0,
                "min_lon": -75.0,
                "max_lat": 41.0,
            }
        )


@pytest.mark.parametrize("keyword", ["polygon_kwargs", "multiple_points_kwargs"])
def test_query_rejects_empty_selection_kwargs(
    geotemporal_data: GeotemporalData,
    keyword: str,
) -> None:
    with pytest.raises(errors.InvalidSelectionError, match=keyword):
        geotemporal_data.query(**{keyword: {}})


def test_query_accepts_complete_circle_kwargs(
    geotemporal_data: GeotemporalData,
) -> None:
    selected = geotemporal_data.query(
        circle_kwargs={"lat": 40.0, "lon": -74.0, "radius": 10.0}
    )

    assert selected.data.sizes == {"latitude": 1, "longitude": 1}
    assert selected.data["temperature"].item() == 4.0
