import importlib

import numpy as np
import pytest
import xarray as xr


concatenate = importlib.import_module("dclimate_client_py.concatenate")


@pytest.fixture(autouse=True)
def check_ipfs_connection():
    """Keep these performance tests independent of a local IPFS gateway."""


async def test_concatenate_datasets_uses_single_xarray_concat(monkeypatch):
    datasets = []
    expected_times = [0, 1, 2]
    expected_values = [0, 1, 2]

    for variant in range(8):
        start = variant * 2
        times = np.arange(start, start + 3)
        values = variant * 100 + np.arange(3)
        datasets.append(
            xr.Dataset({"value": ("time", values)}, coords={"time": times})
        )
        if variant:
            expected_times.extend(times[1:].tolist())
            expected_values.extend(values[1:].tolist())

    real_concat = xr.concat
    concat_calls = []

    def counting_concat(*args, **kwargs):
        concat_calls.append((args, kwargs))
        return real_concat(*args, **kwargs)

    monkeypatch.setattr(concatenate.xr, "concat", counting_concat)

    result = await concatenate.concatenate_datasets(datasets, dimension="time")

    assert result["time"].values.tolist() == expected_times
    assert result["value"].values.tolist() == expected_values
    assert len(concat_calls) == 1, f"xr.concat was called {len(concat_calls)} times"
