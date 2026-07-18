from unittest.mock import AsyncMock

import pytest

from dclimate_client_py.dclimate_client import dClimateClient


@pytest.fixture(autouse=True)
def check_ipfs_connection():
    """Keep these unit tests independent of a local IPFS gateway."""


@pytest.mark.asyncio
async def test_aexit_closes_kubo_when_siren_close_raises():
    client = dClimateClient()
    kubo_cas = AsyncMock()
    siren_client = AsyncMock()
    siren_client.aclose.side_effect = RuntimeError("Siren close failed")
    client._kubo_cas = kubo_cas
    client._siren_client = siren_client

    with pytest.raises(RuntimeError, match="Siren close failed"):
        await client.__aexit__(None, None, None)

    kubo_cas.__aexit__.assert_awaited_once_with(None, None, None)
    assert client._kubo_cas is None
