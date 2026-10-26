import numpy as np
import pytest
import xarray as xr

from dclimate_client_py.geotemporal_data import GeotemporalData
from dclimate_client_py.dclimate_zarr_errors import InvalidSelectionError


def cube():
    return GeotemporalData(xr.Dataset({"value": (("forecast_reference_time", "step", "number"),
        np.arange(6).reshape(2, 3, 1))}, coords={
        "forecast_reference_time": np.array(["2024-01-01T00", "2024-01-01T12"], dtype="datetime64[ns]"),
        "step": ("step", [0, 6, 12], {"units": "hours"}), "number": [0]}), "forecast")


def test_forecast_selects_exact_midnight_and_preserves_singleton_axes():
    result = cube().forecast("2024-01-01").data
    assert result.value.dims == ("time", "number")
    assert result.value.values[:, 0].tolist() == [0, 1, 2]
    np.testing.assert_array_equal(result.time.values,
        np.array(["2024-01-01T00", "2024-01-01T06", "2024-01-01T12"], dtype="datetime64[ns]"))


def test_missing_run_has_client_selection_error():
    with pytest.raises(InvalidSelectionError):
        cube().forecast("2024-01-01T06:00:00Z")


def test_forecast_timezone_is_normalized_to_utc():
    assert cube().forecast("2024-01-01T01:00:00+01:00").data.value.values[0, 0] == 0


def test_reindex_forecast_fills_missing_hours():
    result = cube().forecast("2024-01-01").reindex_forecast().data
    assert result.sizes["time"] == 13
    assert np.isnan(result.value.values[1, 0])
    assert result.value.values[6, 0] == 1
