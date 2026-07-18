import asyncio
from unittest.mock import AsyncMock

import pytest

from dclimate_client_py.dclimate_client import dClimateClient


def _client_with_mocks(
    siren_error: BaseException | None = None,
    kubo_error: BaseException | None = None,
) -> tuple[dClimateClient, AsyncMock, AsyncMock]:
    client = dClimateClient()
    kubo_cas = AsyncMock()
    siren_client = AsyncMock()
    if siren_error is not None:
        siren_client.aclose.side_effect = siren_error
    if kubo_error is not None:
        kubo_cas.__aexit__.side_effect = kubo_error
    client._kubo_cas = kubo_cas
    client._siren_client = siren_client
    return client, kubo_cas, siren_client


@pytest.mark.asyncio
async def test_aexit_forwards_with_block_exception_to_kubo():
    client, kubo_cas, _ = _client_with_mocks()
    exc = ValueError("boom")

    await client.__aexit__(ValueError, exc, None)

    kubo_cas.__aexit__.assert_awaited_once_with(ValueError, exc, None)
    assert client._kubo_cas is None


@pytest.mark.asyncio
async def test_aexit_dual_failure_propagates_later_error_with_context():
    siren_error = RuntimeError("Siren close failed")
    kubo_error = ValueError("Kubo close failed")
    client, _, _ = _client_with_mocks(siren_error=siren_error, kubo_error=kubo_error)

    with pytest.raises(ValueError, match="Kubo close failed") as excinfo:
        await client.__aexit__(None, None, None)

    assert excinfo.value.__context__ is siren_error
    assert client._kubo_cas is None


@pytest.mark.asyncio
async def test_aexit_cancellation_outranks_ordinary_error():
    # Cancellation from either cleanup must propagate, never be demoted
    # to the __cause__/__context__ of an ordinary error.
    client, _, _ = _client_with_mocks(
        siren_error=RuntimeError("Siren close failed"),
        kubo_error=asyncio.CancelledError(),
    )
    with pytest.raises(asyncio.CancelledError):
        await client.__aexit__(None, None, None)
    assert client._kubo_cas is None

    client, _, _ = _client_with_mocks(
        siren_error=asyncio.CancelledError(),
        kubo_error=RuntimeError("Kubo close failed"),
    )
    with pytest.raises(asyncio.CancelledError):
        await client.__aexit__(None, None, None)
    assert client._kubo_cas is None


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
