import pytest

from dclimate_client_py import ipfs_retrieval
from dclimate_client_py.dclimate_zarr_errors import IpfsConnectionError


VALID_CID = "bafkreigh2akiscaildc6snya3u5ox6jz5p3xxrrbf2znsnz2j3twg2ucqi"


class DummyKuboCAS:
    gateway_base_url = "http://example.test"


@pytest.fixture(autouse=True)
def check_ipfs_connection():
    return None


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
    async def sharded_open(*, root_cid, cas, read_only):
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
