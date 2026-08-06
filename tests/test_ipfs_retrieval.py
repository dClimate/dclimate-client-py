import warnings

import httpx
import pytest
import xarray as xr

import dclimate_client_py.dclimate_client as dclimate_client_module
from dclimate_client_py import ipfs_retrieval
from dclimate_client_py.dclimate_client import dClimateClient
from dclimate_client_py.dclimate_zarr_errors import IpfsConnectionError


VALID_CID = "bafkreigh2akiscaildc6snya3u5ox6jz5p3xxrrbf2znsnz2j3twg2ucqi"


class DummyKuboCAS:
    gateway_base_url = "http://example.test"


class DummyStore:
    pass


class DummyGroupedStore:
    def _v2_requires_explicit_group_for_root_read(self):
        return True

    def _v2_top_level_groups(self):
        return {"1", "0"}


def _chained(wrapper: Exception, cause: Exception) -> Exception:
    wrapper.__cause__ = cause
    return wrapper


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection refused"),
        TimeoutError("timed out opening sharded store"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("gateway timed out"),
        httpx.ConnectError("connection refused"),
        httpx.ConnectError("max retries exceeded"),
        _chained(RuntimeError("wrapped"), httpx.ReadTimeout("gateway timed out")),
    ],
)
def test_is_connection_error_classifies_gateway_failures(error):
    assert ipfs_retrieval._is_connection_error(error)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("no such shard file"),
        PermissionError("permission denied"),
        IsADirectoryError("is a directory"),
        ValueError("not a sharded zarr store"),
        httpx.HTTPStatusError(
            "500 Server Error: Internal Server Error",
            request=httpx.Request("GET", "https://gateway.example"),
            response=httpx.Response(500),
        ),
        _chained(
            RuntimeError("retries exhausted on 500s"),
            httpx.HTTPStatusError(
                "too many 500 responses",
                request=httpx.Request("GET", "https://gateway.example"),
                response=httpx.Response(500),
            ),
        ),
        _chained(RuntimeError("wrapped"), FileNotFoundError("missing metadata")),
    ],
)
def test_is_connection_error_rejects_non_network_failures(error):
    # Filesystem/parse errors must not classify as gateway failures, or the
    # caller would skip the HAMT fallback for them.
    assert not ipfs_retrieval._is_connection_error(error)


@pytest.mark.asyncio
async def test_sharded_connection_error_skips_hamt_fallback(monkeypatch):
    async def sharded_open(**kwargs):
        raise TimeoutError("timed out opening sharded store")

    async def hamt_build(**kwargs):
        raise AssertionError("HAMT fallback should not be attempted")

    monkeypatch.setattr(ipfs_retrieval.ShardedZarrStore, "open", sharded_open)
    monkeypatch.setattr(ipfs_retrieval.HAMT, "build", hamt_build)

    with pytest.raises(IpfsConnectionError, match="IPFS connection failed"):
        await ipfs_retrieval._load_dataset_from_ipfs_cid(
            VALID_CID,
            DummyKuboCAS(),
        )


@pytest.mark.asyncio
async def test_multigroup_sharded_store_requires_explicit_group(monkeypatch):
    open_kwargs = []

    async def sharded_open(**kwargs):
        open_kwargs.append(kwargs)
        return DummyGroupedStore()

    monkeypatch.setattr(ipfs_retrieval.ShardedZarrStore, "open", sharded_open)

    with pytest.raises(ipfs_retrieval.MultiresolutionSelectionRequiredError) as raised:
        await ipfs_retrieval._load_dataset_from_ipfs_cid(
            VALID_CID,
            DummyKuboCAS(),
        )

    assert raised.value.available_groups == ("0", "1")
    assert open_kwargs[0]["shard_read_mode"] == "sparse"


@pytest.mark.asyncio
async def test_explicit_zarr_group_is_passed_to_open_zarr(monkeypatch):
    async def sharded_open(**kwargs):
        return DummyStore()

    opened_groups = []

    def open_zarr(*, store, group=None, decode_timedelta=False):
        assert decode_timedelta is True
        opened_groups.append(group)
        return xr.Dataset()

    monkeypatch.setattr(ipfs_retrieval.ShardedZarrStore, "open", sharded_open)
    monkeypatch.setattr(ipfs_retrieval.xr, "open_zarr", open_zarr)

    ds = await ipfs_retrieval._load_dataset_from_ipfs_cid(
        VALID_CID,
        DummyKuboCAS(),
        zarr_group="2",
    )

    assert opened_groups == ["2"]
    assert ds.attrs["_ipfs_zarr_group"] == "2"


@pytest.mark.asyncio
async def test_zarr_group_error_after_sharded_open_does_not_fallback(monkeypatch):
    async def sharded_open(**kwargs):
        return DummyStore()

    async def hamt_build(**kwargs):
        raise AssertionError("HAMT fallback should not be attempted")

    def open_zarr(*, store, group=None, decode_timedelta=False):
        assert decode_timedelta is True
        raise ValueError("explicit Zarr group required")

    monkeypatch.setattr(ipfs_retrieval.ShardedZarrStore, "open", sharded_open)
    monkeypatch.setattr(ipfs_retrieval.HAMT, "build", hamt_build)
    monkeypatch.setattr(ipfs_retrieval.xr, "open_zarr", open_zarr)

    with pytest.raises(
        ipfs_retrieval.MultiresolutionSelectionRequiredError,
        match="explicit zarr_group",
    ):
        await ipfs_retrieval._load_dataset_from_ipfs_cid(
            VALID_CID,
            DummyKuboCAS(),
        )


@pytest.mark.asyncio
async def test_sharded_v1_warning_is_suppressed_in_client_loader(monkeypatch):
    async def sharded_open(**kwargs):
        warnings.warn(
            "sharded_zarr_v1 is deprecated",
            ipfs_retrieval.ShardedZarrV1DeprecationWarning,
            stacklevel=2,
        )
        return DummyStore()

    monkeypatch.setattr(ipfs_retrieval.ShardedZarrStore, "open", sharded_open)
    monkeypatch.setattr(
        ipfs_retrieval.xr,
        "open_zarr",
        lambda *, store, group=None, decode_timedelta=False: xr.Dataset(),
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        await ipfs_retrieval._load_dataset_from_ipfs_cid(
            VALID_CID,
            DummyKuboCAS(),
        )

    assert not any(
        issubclass(
            warning.category,
            ipfs_retrieval.ShardedZarrV1DeprecationWarning,
        )
        for warning in caught_warnings
    )


@pytest.mark.asyncio
async def test_hamt_fallback_preserves_explicit_zarr_group(monkeypatch):
    async def sharded_open(**kwargs):
        raise ValueError("not a sharded zarr store")

    async def hamt_build(**kwargs):
        return DummyStore()

    opened_groups = []

    def open_zarr(*, store, group=None, decode_timedelta=False):
        assert decode_timedelta is True
        opened_groups.append(group)
        return xr.Dataset()

    monkeypatch.setattr(ipfs_retrieval.ShardedZarrStore, "open", sharded_open)
    monkeypatch.setattr(ipfs_retrieval.HAMT, "build", hamt_build)
    monkeypatch.setattr(
        ipfs_retrieval,
        "ZarrHAMTStore",
        lambda hamt_store, read_only: hamt_store,
    )
    monkeypatch.setattr(ipfs_retrieval.xr, "open_zarr", open_zarr)

    ds = await ipfs_retrieval._load_dataset_from_ipfs_cid(
        VALID_CID,
        DummyKuboCAS(),
        zarr_group="/2/",
    )

    assert opened_groups == ["2"]
    assert ds.attrs["_ipfs_store_type"] == "ZarrHAMTStore"
    assert ds.attrs["_ipfs_zarr_group"] == "2"


@pytest.mark.asyncio
async def test_client_direct_cid_passes_zarr_group_and_records_metadata(monkeypatch):
    async def load_dataset_from_ipfs_cid(
        *, ipfs_cid, kubo_cas, zarr_group, shard_read_mode
    ):
        assert ipfs_cid == VALID_CID
        assert zarr_group == "2"
        assert shard_read_mode == "sparse"
        return xr.Dataset(attrs={"_ipfs_zarr_group": "2"})

    monkeypatch.setattr(
        dclimate_client_module,
        "_load_dataset_from_ipfs_cid",
        load_dataset_from_ipfs_cid,
    )

    dclimate = dClimateClient()
    dclimate._kubo_cas = DummyKuboCAS()

    ds, metadata = await dclimate.load_dataset(
        dataset="pyramid",
        cid=VALID_CID,
        return_xarray=True,
        zarr_group="2",
        shard_read_mode="sparse",
    )

    assert isinstance(ds, xr.Dataset)
    assert metadata["zarr_group"] == "2"
