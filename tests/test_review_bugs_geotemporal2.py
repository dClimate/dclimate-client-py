import copy
import warnings

import numpy as np
import pytest
import xarray as xr

from dclimate_client_py import dclimate_zarr_errors as errors
from dclimate_client_py.geotemporal_data import GeotemporalData


def test_check_dataset_size_accepts_scalar_dataset():
    dataset = xr.Dataset({"temperature": xr.DataArray(12.5)})
    data = GeotemporalData(dataset, "static-temperature")

    data.check_dataset_size(point_limit=1)


def test_to_netcdf_preserves_original_dataset_attributes():
    original_attrs = {
        "bbox": [-180.0, -90.0, 180.0, 90.0],
        "date range": ["2020010100", "2020010200"],
        "tags": ["temperature", "static"],
        "finalization date": None,
        "update_date_range": ["2020010100", "2020010200"],
    }
    dataset = xr.Dataset(
        {"temperature": ("location", [12.5])}, attrs=copy.deepcopy(original_attrs)
    )
    data = GeotemporalData(dataset, "static-temperature")

    serialized = data.to_netcdf()

    assert isinstance(serialized, bytes)
    assert dataset.attrs == original_attrs


def test_temporal_aggregation_quarter_uses_supported_alias_without_warning():
    times = np.array(
        [
            "2024-01-15",
            "2024-02-15",
            "2024-04-15",
            "2024-06-15",
            "2024-07-15",
        ],
        dtype="datetime64[ns]",
    )
    dataset = xr.Dataset(
        {"temperature": ("time", [1.0, 3.0, 5.0, 7.0, 9.0])},
        coords={"time": times},
    )
    data = GeotemporalData(dataset, "quarterly-temperature")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = data.temporal_aggregation(time_period="quarter", agg_method="mean")

    xr.testing.assert_allclose(
        result.data["temperature"],
        xr.DataArray(
            [2.0, 6.0, 9.0],
            dims="time",
            coords={
                "time": np.array(
                    ["2024-03-31", "2024-06-30", "2024-09-30"],
                    dtype="datetime64[ns]",
                )
            },
            name="temperature",
        ),
    )
    deprecated_q_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, (FutureWarning, DeprecationWarning))
        and "'Q'" in str(warning.message)
    ]
    assert deprecated_q_warnings == []


def test_check_dataset_size_counts_scalar_dataset_as_one_point():
    dataset = xr.Dataset({"temperature": xr.DataArray(12.5)})
    data = GeotemporalData(dataset, "static-temperature")

    with pytest.raises(errors.SelectionTooLargeError):
        data.check_dataset_size(point_limit=0)
