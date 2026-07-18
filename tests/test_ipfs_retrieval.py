import warnings

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


@pytest.mark.parametrize(
    "message",
    [
        "Connection refused",
        "Max retries exceeded",
        "Name or service not known",
        "network is unreachable",
        "nodename nor servname provided",
        "temporary failure in name resolution",
        "timed out opening sharded store",
    ],
)
def test_is_connection_error_classifies_gateway_failures(message):
    assert ipfs_retrieval._is_connection_error(RuntimeError(message))


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
async def test_multigroup_sharded_store_defaults_to_group_zero(monkeypatch):
    open_kwargs = []

    async def sharded_open(**kwargs):
        open_kwargs.append(kwargs)
        return DummyGroupedStore()

    opened_groups = []

    def open_zarr(*, store, group=None):
        opened_groups.append(group)
        return xr.Dataset()

    monkeypatch.setattr(ipfs_retrieval.ShardedZarrStore, "open", sharded_open)
    monkeypatch.setattr(ipfs_retrieval.xr, "open_zarr", open_zarr)

    ds = await ipfs_retrieval._load_dataset_from_ipfs_cid(
        VALID_CID,
        DummyKuboCAS(),
    )

    assert opened_groups == ["0"]
    assert open_kwargs[0]["shard_read_mode"] == "sparse"
    assert ds.attrs["_ipfs_store_type"] == "ShardedZarrStore"
    assert ds.attrs["_ipfs_zarr_group"] == "0"


@pytest.mark.asyncio
async def test_explicit_zarr_group_is_passed_to_open_zarr(monkeypatch):
    async def sharded_open(**kwargs):
        return DummyStore()

    opened_groups = []

    def open_zarr(*, store, group=None):
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

    def open_zarr(*, store, group=None):
        raise ValueError("explicit Zarr group required")

    monkeypatch.setattr(ipfs_retrieval.ShardedZarrStore, "open", sharded_open)
    monkeypatch.setattr(ipfs_retrieval.HAMT, "build", hamt_build)
    monkeypatch.setattr(ipfs_retrieval.xr, "open_zarr", open_zarr)

    with pytest.raises(ValueError, match="explicit Zarr group"):
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
        lambda *, store, group=None: xr.Dataset(),
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

    def open_zarr(*, store, group=None):
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
