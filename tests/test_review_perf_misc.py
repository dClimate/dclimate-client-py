import asyncio
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Point
from zarr.core.buffer import default_buffer_prototype

from dclimate_client_py import encryption_codec as encryption_codec_module
from dclimate_client_py.encryption_codec import EncryptionCodec
from dclimate_client_py.geotemporal_data import GeotemporalData


def test_points_uses_vectorized_coordinate_access(monkeypatch):
    latitudes = np.arange(10, dtype=float)
    longitudes = np.arange(20, 30, dtype=float)
    expected_values = np.arange(100).reshape(10, 10)
    dataset = xr.Dataset(
        {"temperature": (("latitude", "longitude"), expected_values)},
        coords={"latitude": latitudes, "longitude": longitudes},
    )
    coordinates = [
        (longitude, latitude) for latitude in latitudes for longitude in longitudes
    ]
    points_mask = gpd.GeoSeries(
        [Point(longitude, latitude) for longitude, latitude in coordinates],
        crs=4326,
    ).array

    original_y = Point.y
    property_accesses = 0

    def counting_y(point):
        nonlocal property_accesses
        property_accesses += 1
        return original_y.__get__(point, type(point))

    monkeypatch.setattr(Point, "y", property(counting_y))

    selected = GeotemporalData(dataset, "vectorization-test").points(
        points_mask,
        epsg_crs=4326,
    )

    assert property_accesses == 0
    np.testing.assert_array_equal(selected.data.latitude.values, latitudes.repeat(10))
    np.testing.assert_array_equal(
        selected.data.longitude.values, np.tile(longitudes, 10)
    )
    np.testing.assert_array_equal(
        selected.data["temperature"].values,
        expected_values.reshape(-1),
    )


def _codec_inputs(payload: bytes):
    prototype = default_buffer_prototype()
    chunk_spec = SimpleNamespace(prototype=prototype)
    chunk_bytes = prototype.buffer.from_bytes(payload)
    return chunk_bytes, chunk_spec


@pytest.fixture
def configured_encryption_key():
    previous_key = EncryptionCodec._encryption_key
    EncryptionCodec.set_encryption_key(b"k" * 32)
    yield
    EncryptionCodec._encryption_key = previous_key


async def _roundtrip(codec: EncryptionCodec, payload: bytes) -> bytes:
    chunk_bytes, chunk_spec = _codec_inputs(payload)
    encoded = await codec._encode_single(chunk_bytes, chunk_spec)
    decoded = await codec._decode_single(encoded, chunk_spec)
    return decoded.to_bytes()


@pytest.mark.parametrize(
    ("size", "fill"),
    [(16 * 1024, b"s"), (1024 * 1024, b"L")],
    ids=["small", "large"],
)
async def test_encryption_codec_roundtrip(size, fill, configured_encryption_key):
    codec = EncryptionCodec(header="offline-roundtrip")
    payload = fill * size

    assert await _roundtrip(codec, payload) == payload


async def test_encryption_codec_only_offloads_large_chunks(
    monkeypatch, configured_encryption_key
):
    codec = EncryptionCodec(header="dispatch-test")
    real_to_thread = asyncio.to_thread
    calls = []

    async def recording_to_thread(function, /, *args, **kwargs):
        calls.append(function)
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(
        encryption_codec_module.asyncio,
        "to_thread",
        recording_to_thread,
    )

    small_payload = b"s" * (16 * 1024)
    large_payload = b"L" * (1024 * 1024)
    assert await _roundtrip(codec, small_payload) == small_payload
    small_chunk_thread_calls = len(calls)

    assert await _roundtrip(codec, large_payload) == large_payload
    large_chunk_thread_calls = len(calls) - small_chunk_thread_calls

    assert small_chunk_thread_calls == 0
    # Encode AND decode must each offload for large chunks.
    assert large_chunk_thread_calls == 2


async def test_encryption_codec_threshold_boundary(
    monkeypatch, configured_encryption_key
):
    # Pin the 128 KiB dispatch boundary itself: a plaintext one byte under
    # the threshold encodes inline; at the threshold it offloads. (Decode
    # sizes shift by the 40-byte nonce+tag overhead, so decode of the
    # just-under payload may legitimately offload — only encode is pinned.)
    threshold = encryption_codec_module._THREAD_OFFLOAD_THRESHOLD
    codec = EncryptionCodec(header="boundary-test")
    real_to_thread = asyncio.to_thread
    calls = []

    async def recording_to_thread(function, /, *args, **kwargs):
        calls.append(function.__name__)
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(
        encryption_codec_module.asyncio,
        "to_thread",
        recording_to_thread,
    )

    class _Spec:
        class prototype:
            class buffer:
                @staticmethod
                def from_bytes(data):
                    return _Bytes(data)

    class _Bytes:
        def __init__(self, data):
            self._data = data

        def to_bytes(self):
            return self._data

    under = await codec._encode_single(_Bytes(b"u" * (threshold - 1)), _Spec)
    assert calls == []
    assert len(under.to_bytes()) == threshold - 1 + 40

    await codec._encode_single(_Bytes(b"a" * threshold), _Spec)
    assert calls == ["encrypt"]


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_in_fresh_interpreter(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_package_import_does_not_eagerly_load_heavy_dependencies():
    completed = _run_in_fresh_interpreter(
        "import dclimate_client_py, sys; "
        "print(','.join(m for m in ('s3fs','geopandas','pystac') if m in sys.modules))"
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == ""


def test_lazy_public_api_still_supports_geospatial_selection():
    script = textwrap.dedent(
        """
        from dclimate_client_py import GeotemporalData, dClimateClient

        import geopandas as gpd
        import numpy as np
        import xarray as xr
        from shapely.geometry import Point

        assert dClimateClient.__name__ == "dClimateClient"
        dataset = xr.Dataset(
            {"temperature": (("latitude", "longitude"), [[1, 2], [3, 4]])},
            coords={"latitude": [0.0, 1.0], "longitude": [10.0, 11.0]},
        )
        mask = gpd.GeoSeries([Point(11.0, 1.0)], crs=4326).array
        selected = GeotemporalData(dataset, "lazy-api-test").points(mask, 4326)
        assert selected.data["temperature"].item() == 4
        """
    )

    completed = _run_in_fresh_interpreter(script)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_points_rejects_missing_geometries(dataset):
    import geopandas as gpd
    from shapely.geometry import Point

    from dclimate_client_py import dclimate_zarr_errors as errors
    from dclimate_client_py.geotemporal_data import GeotemporalData

    mask = gpd.GeoSeries([Point(180.0, 0.0), None]).array
    data = GeotemporalData(dataset, "missing-geometry")

    with pytest.raises(errors.InvalidSelectionError, match="missing geometries"):
        data.points(mask, epsg_crs=4326)
