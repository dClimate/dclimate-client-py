import numpy as np
import pytest

from dclimate_client_py import dclimate_zarr_errors as errors
from dclimate_client_py.geotemporal_data import _check_input_parameters, _haversine


def test_haversine_single_points():
    dist = _haversine(0, 0, 0, 1)
    assert dist == pytest.approx(111.195, rel=1e-3)

    dist = _haversine(36.12, -86.67, 33.94, -118.40)
    assert dist == pytest.approx(2886.44, rel=1e-2)


def test_haversine_arrays():
    lats1 = np.array([0, 10])
    lons1 = np.array([0, 0])
    lats2 = np.array([0, 20])
    lons2 = np.array([1, 0])
    dists = _haversine(lats1, lons1, lats2, lons2)
    assert np.allclose(dists, [111.195, 1111.95], rtol=1e-3)


def test_check_input_parameters_invalid_period():
    with pytest.raises(errors.InvalidTimePeriodError):
        _check_input_parameters(time_period="decade")


def test_check_input_parameters_invalid_method():
    with pytest.raises(errors.InvalidAggregationMethodError):
        _check_input_parameters(agg_method="average")
