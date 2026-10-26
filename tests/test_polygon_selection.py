"""Polygon selection and spatial aggregation.

These mirror tests/polygon.test.ts and tests/spatial-aggregation.test.ts in
dclimate-client-js, so both clients select and reduce the same cells.
"""

import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, box

from dclimate_client_py import dclimate_zarr_errors as errors
from dclimate_client_py.geotemporal_data import GeotemporalData

TIMES = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")


def _grid(**attrs):
    """Three days on a 3x3 grid at latitudes 0..2 and longitudes 10..12.

    A cell's value is 100 * day + 10 * row + column, so any value names its
    cell.
    """
    day, row, column = np.meshgrid(
        np.arange(3), np.arange(3), np.arange(3), indexing="ij"
    )
    return GeotemporalData(
        xr.Dataset(
            {
                "value": (
                    ("time", "latitude", "longitude"),
                    (100 * day + 10 * row + column).astype(float),
                    {"units": "K"},
                )
            },
            coords={
                "time": TIMES,
                "latitude": [0.0, 1.0, 2.0],
                "longitude": [10.0, 11.0, 12.0],
            },
            attrs={"title": "grid", **attrs},
        ),
        "grid",
    )


def _mask(*shapes):
    return gpd.GeoSeries(list(shapes), crs=4326).array


# Inside cell centres: (0, 10), (0, 11), (1, 10).
TRIANGLE = Polygon([(9.5, -0.5), (11.8, -0.5), (9.5, 1.8)])


def test_keeps_cells_inside_and_masks_the_rest_of_the_box():
    selected = _grid().polygons(_mask(TRIANGLE)).data

    assert list(selected["latitude"].values) == [0.0, 1.0]
    assert list(selected["longitude"].values) == [10.0, 11.0]
    np.testing.assert_array_equal(
        selected["value"].isel(time=0).values, [[0.0, 1.0], [10.0, np.nan]]
    )
    assert selected.attrs["title"] == "grid"
    assert selected.attrs["polygon_cells_inside"] == 3
    assert selected.attrs["polygon_cells_in_bounds"] == 4
    assert "polygon_fallback" not in selected.attrs


def test_counts_cells_once_not_once_per_timestep():
    selected = _grid().polygons(_mask(box(9.5, -0.5, 11.5, 1.5))).data

    assert selected.sizes["time"] == 3
    assert selected.attrs["polygon_cells_inside"] == 4
    assert selected.attrs["polygon_cells_in_bounds"] == 4


def test_keeps_every_cell_of_a_shape_one_row_high():
    # rio.clip cannot clip a one-row raster and raised NoDataInBounds, which
    # used to collapse this three-cell strip to a single point.
    selected = _grid().polygons(_mask(box(9.6, 0.8, 12.4, 1.2))).data

    assert selected.attrs["polygon_cells_inside"] == 3
    np.testing.assert_array_equal(
        selected["value"].isel(time=0).values, [[10.0, 11.0, 12.0]]
    )


def test_does_not_need_a_spatial_resolution_attribute():
    selected = _grid().polygons(_mask(box(9.5, -0.5, 11.5, 1.5)))

    assert selected.data.attrs["polygon_cells_inside"] == 4


def test_falls_back_to_the_nearest_cell_for_a_shape_smaller_than_a_cell():
    # Between the centres at latitude 1..2 and longitude 11..12, nearer (2, 12).
    selected = _grid().polygons(_mask(box(11.6, 1.6, 11.9, 1.9))).data

    assert selected["value"].dims == ("time",)
    assert list(selected["value"].values) == [22.0, 122.0, 222.0]
    assert selected.attrs["polygon_cells_inside"] == 0
    assert selected.attrs["polygon_cells_in_bounds"] == 0
    assert selected.attrs["polygon_fallback"] == "nearest_cell"


def test_does_not_fall_back_while_a_cell_centre_is_inside():
    # Far smaller than one cell, but it holds the centre of (1, 11).
    selected = _grid().polygons(_mask(box(10.9, 0.9, 11.1, 1.1))).data

    assert selected.attrs["polygon_cells_inside"] == 1
    assert "polygon_fallback" not in selected.attrs
    assert list(selected["value"].values.ravel()) == [11.0, 111.0, 211.0]


def test_fallback_uses_a_point_inside_the_shape_not_its_centroid():
    # A U around cell (1, 11) holding no cell centre. Its centroid falls in the
    # hollow, nearest (1, 11); a point inside it lies in the left arm, nearest
    # (1, 10).
    u_shape = Polygon(
        [
            (10.4, 0.45),
            (11.55, 0.45),
            (11.55, 1.45),
            (11.5, 1.45),
            (11.5, 0.5),
            (10.5, 0.5),
            (10.5, 1.45),
            (10.4, 1.45),
        ]
    )
    selected = _grid().polygons(_mask(u_shape)).data

    assert selected.attrs["polygon_fallback"] == "nearest_cell"
    assert list(selected["value"].values) == [10.0, 110.0, 210.0]


def test_refuses_a_bounding_box_over_the_point_limit():
    with pytest.raises(errors.SelectionTooLargeError):
        _grid().polygons(_mask(box(9.5, -0.5, 12.5, 2.5)), point_limit=26)


def test_spatial_aggregation_averages_only_the_cells_a_polygon_kept():
    triangle = Polygon([(9.5, -0.5), (10.8, -0.5), (9.5, 2.5)])
    reduced = _grid().polygons(_mask(triangle)).spatial_aggregation("mean").data

    # Inside: (0, 10) and (1, 10), holding 0 and 10 on day 0.
    assert list(reduced["value"].values) == [5.0, 105.0, 205.0]
    assert reduced.attrs["polygon_cells_inside"] == 2


def test_spatial_aggregation_computes_std_over_all_cells_at_once():
    data = xr.Dataset(
        {"t": (("time", "latitude", "longitude"), [[[1.0, 2.0], [3.0, 4.0]]])},
        coords={"time": TIMES[:1], "latitude": [0.0, 1.0], "longitude": [10.0, 11.0]},
    )
    reduced = GeotemporalData(data, "t").spatial_aggregation("std").data

    assert float(reduced["t"].values[0]) == pytest.approx(np.sqrt(1.25))


def test_spatial_aggregation_keeps_non_spatial_dimensions():
    values = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    data = xr.Dataset(
        {"t": (("latitude", "time", "longitude", "number"), values)},
        coords={
            "latitude": [0.0, 1.0],
            "time": TIMES[:2],
            "longitude": [10.0, 11.0],
            "number": [0, 1],
        },
    )
    reduced = GeotemporalData(data, "t").spatial_aggregation("mean").data

    assert set(reduced["t"].dims) == {"time", "number"}
    expected = values.mean(axis=(0, 2))
    np.testing.assert_allclose(
        reduced["t"].transpose("time", "number").values, expected
    )


def test_spatial_aggregation_returns_a_point_selection_unchanged():
    point = _grid().point(latitude=0, longitude=10)
    reduced = point.spatial_aggregation("mean").data

    assert list(reduced["value"].values) == [0.0, 100.0, 200.0]


def test_polygon_boundaries_are_excluded():
    selected = _grid().polygons(_mask(box(10, 0, 12, 2))).data
    assert selected.attrs["polygon_cells_inside"] == 1
    assert selected.value.values[:, 0, 0].tolist() == [11, 111, 211]


def test_polygon_crossing_antimeridian_is_rejected():
    with pytest.raises(errors.InvalidSelectionError, match="antimeridian"):
        _grid().polygons(_mask(Polygon([(179, 0), (-179, 0), (-179, 2), (179, 0)])))


def test_spatial_scalar_and_missing_sum():
    grid = _grid()
    grid.data["elevation"] = (("latitude", "longitude"), np.arange(9).reshape(3, 3))
    reduced = grid.spatial_aggregation("mean").data
    assert reduced.elevation.dims == ()
    assert reduced.elevation.item() == 4
    grid.data["value"][:] = np.nan
    assert grid.spatial_aggregation("sum").data.value.isnull().all()


def test_spatial_variable_subset_and_point_count():
    grid = _grid()
    grid.data["other"] = grid.data.value.copy()
    reduced = grid.spatial_aggregation("mean", variables=["value"]).data
    assert reduced.value.dims == ("time",)
    assert reduced.other.dims == ("time", "latitude", "longitude")
    assert grid.point_count == 27


def test_fallback_chooses_widest_polygon_like_js():
    selected = _grid().polygons(_mask(box(10.1, 0.1, 10.2, 0.2), box(11.2, 1.1, 11.9, 1.2))).data
    assert selected.attrs["polygon_fallback"] == "nearest_cell"
    assert selected.value.values.tolist() == [12, 112, 212]
