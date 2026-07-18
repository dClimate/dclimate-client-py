import tracemalloc

import numpy as np
import xarray as xr

from dclimate_client_py.geotemporal_data import GeotemporalData


def _rectangle_dataset():
    values = np.arange(84, dtype=float).reshape(2, 6, 7)
    return xr.Dataset(
        {"value": (("time", "latitude", "longitude"), values)},
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "latitude": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "longitude": [180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0],
        },
    )


def test_rectangle_ascending_coordinates_selects_expected_cells():
    result = GeotemporalData(_rectangle_dataset(), "synthetic").rectangle(
        10.0, 185.0, 30.0, 195.0
    )

    np.testing.assert_array_equal(result.data.latitude, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(result.data.longitude, [185.0, 190.0, 195.0])
    np.testing.assert_array_equal(
        result.data["value"],
        [
            [[8.0, 9.0, 10.0], [15.0, 16.0, 17.0], [22.0, 23.0, 24.0]],
            [[50.0, 51.0, 52.0], [57.0, 58.0, 59.0], [64.0, 65.0, 66.0]],
        ],
    )


def test_rectangle_descending_latitude_selects_same_cells():
    descending = _rectangle_dataset().isel(latitude=slice(None, None, -1))

    result = GeotemporalData(descending, "synthetic").rectangle(
        10.0, 185.0, 30.0, 195.0
    )

    np.testing.assert_array_equal(result.data.latitude, [30.0, 20.0, 10.0])
    np.testing.assert_array_equal(result.data.longitude, [185.0, 190.0, 195.0])
    np.testing.assert_array_equal(
        result.data["value"],
        [
            [[22.0, 23.0, 24.0], [15.0, 16.0, 17.0], [8.0, 9.0, 10.0]],
            [[64.0, 65.0, 66.0], [57.0, 58.0, 59.0], [50.0, 51.0, 52.0]],
        ],
    )


def test_rectangle_bounds_between_grid_points_use_cells_inside_bounds():
    result = GeotemporalData(_rectangle_dataset(), "synthetic").rectangle(
        5.0, 181.0, 34.0, 199.0
    )

    np.testing.assert_array_equal(result.data.latitude, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(result.data.longitude, [185.0, 190.0, 195.0])
    np.testing.assert_array_equal(
        result.data["value"],
        [
            [[8.0, 9.0, 10.0], [15.0, 16.0, 17.0], [22.0, 23.0, 24.0]],
            [[50.0, 51.0, 52.0], [57.0, 58.0, 59.0], [64.0, 65.0, 66.0]],
        ],
    )


def test_rectangle_bounds_outside_grid_are_clipped_to_available_cells():
    result = GeotemporalData(_rectangle_dataset(), "synthetic").rectangle(
        -50.0, 175.0, 15.0, 500.0
    )

    np.testing.assert_array_equal(result.data.latitude, [0.0, 10.0])
    np.testing.assert_array_equal(
        result.data.longitude, [180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0]
    )
    np.testing.assert_array_equal(
        result.data["value"],
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            ],
            [
                [42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
                [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            ],
        ],
    )


def test_rectangle_empty_selection_drops_both_spatial_dimensions():
    result = GeotemporalData(_rectangle_dataset(), "synthetic").rectangle(
        61.0, 185.0, 69.0, 195.0
    )

    assert result.data.sizes == {"time": 2, "latitude": 0, "longitude": 0}
    assert result.data["value"].shape == (2, 0, 0)


def test_circle_matches_current_tight_crop_and_nan_placement():
    dataset = xr.Dataset(
        {
            "value": (
                ("time", "latitude", "longitude"),
                np.arange(25, dtype=float).reshape(1, 5, 5),
            )
        },
        coords={
            "time": [0],
            "latitude": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "longitude": [-2.0, -1.0, 0.0, 1.0, 2.0],
        },
    )

    result = GeotemporalData(dataset, "synthetic").circle(0.0, 0.0, 125.0)

    np.testing.assert_array_equal(result.data.latitude, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(result.data.longitude, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(
        result.data["value"],
        [[[np.nan, 7.0, np.nan], [11.0, 12.0, 13.0], [np.nan, 17.0, np.nan]]],
    )


def _large_dataset():
    return xr.Dataset(
        {
            "value": (
                ("time", "latitude", "longitude"),
                np.zeros((50, 320, 320), dtype=np.float64),
            )
        },
        coords={
            "time": np.arange(50),
            "latitude": np.linspace(-20.0, 20.0, 320),
            "longitude": np.linspace(-20.0, 20.0, 320),
        },
    )


def test_rectangle_tiny_selection_allocates_less_than_10_mb():
    data = GeotemporalData(_large_dataset(), "synthetic")

    tracemalloc.start()
    try:
        result = data.rectangle(-0.2, -0.2, 0.2, 0.2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.data.sizes["latitude"] > 0
    assert result.data.sizes["longitude"] > 0
    assert peak < 10 * 1024 * 1024, f"peak allocation was {peak / 1024**2:.1f} MB"


def test_circle_small_radius_allocates_less_than_10_mb():
    data = GeotemporalData(_large_dataset(), "synthetic")

    tracemalloc.start()
    try:
        result = data.circle(0.0, 0.0, 50.0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.data.sizes["latitude"] > 0
    assert result.data.sizes["longitude"] > 0
    assert peak < 10 * 1024 * 1024, f"peak allocation was {peak / 1024**2:.1f} MB"


def test_rectangle_descending_longitude_selects_same_cells():
    descending = _rectangle_dataset().isel(longitude=slice(None, None, -1))

    result = GeotemporalData(descending, "synthetic").rectangle(
        10.0, 185.0, 30.0, 195.0
    )

    np.testing.assert_array_equal(result.data.latitude, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(result.data.longitude, [195.0, 190.0, 185.0])
    np.testing.assert_array_equal(
        result.data["value"],
        [
            [[10.0, 9.0, 8.0], [17.0, 16.0, 15.0], [24.0, 23.0, 22.0]],
            [[52.0, 51.0, 50.0], [59.0, 58.0, 57.0], [66.0, 65.0, 64.0]],
        ],
    )


def _circle_reference(data, lat, lon, radius):
    # Implementation-independent reference: full-grid haversine mask.
    from dclimate_client_py.geotemporal_data import _haversine

    distances = _haversine(lat, lon, data["latitude"], data["longitude"])
    return data.where(distances < radius, drop=True)


def test_circle_high_latitude_widens_longitude_bounding_box():
    # At 60N a 300 km radius spans ~5.4 degrees of longitude but only
    # ~2.7 degrees of latitude; a bbox that forgets the cos(latitude)
    # widening would silently drop the outer longitude columns.
    latitudes = np.arange(58.0, 63.0)
    longitudes = np.arange(-10.0, 11.0)
    values = np.arange(len(latitudes) * len(longitudes), dtype=float).reshape(
        len(latitudes), len(longitudes)
    )
    dataset = xr.Dataset(
        {"value": (("latitude", "longitude"), values)},
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    result = GeotemporalData(dataset, "synthetic").circle(60.0, 0.0, 300.0)

    expected = _circle_reference(dataset, 60.0, 0.0, 300.0)
    xr.testing.assert_identical(result.data, expected)
    assert result.data.sizes["longitude"] == 11


def test_rectangle_non_monotonic_coordinates_fall_back_to_mask_path():
    latitudes = np.array([0.0, 20.0, 10.0, 30.0, 40.0, 50.0])
    longitudes = np.arange(180.0, 210.0, 5.0)
    values = np.arange(len(latitudes) * len(longitudes), dtype=float).reshape(
        len(latitudes), len(longitudes)
    )
    dataset = xr.Dataset(
        {"value": (("latitude", "longitude"), values)},
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    result = GeotemporalData(dataset, "synthetic").rectangle(
        10.0, 185.0, 30.0, 195.0
    )

    np.testing.assert_array_equal(result.data.latitude, [20.0, 10.0, 30.0])
    np.testing.assert_array_equal(result.data.longitude, [185.0, 190.0, 195.0])


def test_rectangle_nan_coordinate_falls_back_to_mask_path():
    latitudes = np.array([0.0, 10.0, np.nan, 30.0, 40.0, 50.0])
    longitudes = np.arange(180.0, 210.0, 5.0)
    values = np.arange(len(latitudes) * len(longitudes), dtype=float).reshape(
        len(latitudes), len(longitudes)
    )
    dataset = xr.Dataset(
        {"value": (("latitude", "longitude"), values)},
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    result = GeotemporalData(dataset, "synthetic").rectangle(
        10.0, 185.0, 30.0, 195.0
    )

    # The NaN latitude row must be excluded, exactly as the old mask did.
    np.testing.assert_array_equal(result.data.latitude, [10.0, 30.0])
