import subprocess
import textwrap
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from dclimate_client_py.geotemporal_data import GeotemporalData


def _polygon_dataset():
    values = np.arange(8, dtype=float).reshape(2, 2, 2)
    return xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), values)},
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
        attrs={"spatial resolution": 0.5},
    )


def _polygon_mask():
    return gpd.GeoSeries([box(-0.4, -0.4, 1.4, 1.4)], crs=4326).array


def _force_clip_no_data(monkeypatch):
    import rioxarray  # noqa: F401 -- registers the xarray ``rio`` accessor
    from rioxarray.exceptions import NoDataInBounds
    from rioxarray.raster_dataset import RasterDataset

    def raise_no_data_in_bounds(self, *args, **kwargs):
        raise NoDataInBounds("polygon contains no grid-cell centers")

    monkeypatch.setattr(RasterDataset, "clip", raise_no_data_in_bounds)
    return NoDataInBounds


def test_polygons_registers_rioxarray_accessor_in_fresh_interpreter():
    script = textwrap.dedent(
        """
        import sys

        assert "rioxarray" not in sys.modules

        import geopandas as gpd
        import numpy as np
        import xarray as xr
        from shapely.geometry import box

        from dclimate_client_py.geotemporal_data import GeotemporalData

        dataset = xr.Dataset(
            {"temperature": (("time", "latitude", "longitude"), np.ones((1, 2, 2)))},
            coords={
                "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
            },
            attrs={"spatial resolution": 0.5},
        )
        mask = gpd.GeoSeries([box(-0.4, -0.4, 1.4, 1.4)], crs=4326).array

        result = GeotemporalData(dataset, "the-dataset-name").polygons(mask)

        assert result.data.sizes["latitude"] == 2
        assert result.data.sizes["longitude"] == 2
        """
    )
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_polygons_handles_rioxarray_no_data_in_bounds(monkeypatch):
    _force_clip_no_data(monkeypatch)
    geotemporal = GeotemporalData(_polygon_dataset(), "the-dataset-name")
    mask = _polygon_mask()
    expected = geotemporal.reduce_polygon_to_point(mask)

    result = geotemporal.polygons(mask)

    xr.testing.assert_identical(result.data, expected.data)


def test_polygons_does_not_mutate_caller_dataset():
    # The pre-fix implementation attached rio spatial dims / CRS to
    # self.data with inplace=True, polluting the caller's dataset.
    dataset = _polygon_dataset()
    original = dataset.copy(deep=True)
    geotemporal = GeotemporalData(dataset, "the-dataset-name")

    geotemporal.polygons(_polygon_mask())

    assert "spatial_ref" not in dataset.coords
    assert dataset.attrs == original.attrs
    xr.testing.assert_identical(dataset, original)


def test_rolling_aggregation_preserves_time_with_all_nan_spatial_cell():
    times = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-07"),
        dtype="datetime64[D]",
    )
    values = np.broadcast_to(
        np.arange(1.0, 7.0)[:, np.newaxis, np.newaxis], (6, 2, 2)
    ).copy()
    values[:, 0, 0] = np.nan
    dataset = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), values)},
        coords={"time": times, "latitude": [10.0, 11.0], "longitude": [20.0, 21.0]},
    )

    result = GeotemporalData(dataset, "the-dataset-name").rolling_aggregation(
        window_size=3, agg_method="mean"
    )

    assert result.data.sizes["time"] == dataset.sizes["time"] - 2
    np.testing.assert_array_equal(result.data.time.values, dataset.time.values[2:])
    np.testing.assert_allclose(
        result.data["temperature"].isel(latitude=1, longitude=1).values,
        [2.0, 3.0, 4.0, 5.0],
    )
    assert result.data["temperature"].isel(latitude=0, longitude=0).isnull().all()
