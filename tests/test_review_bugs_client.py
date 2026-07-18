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


@pytest.mark.asyncio
async def test_aexit_body_cancellation_outranks_ordinary_cleanup_error():
    siren_error = RuntimeError("Siren close failed")
    client, kubo_cas, siren_client = _client_with_mocks(siren_error=siren_error)

    # __aenter__ replaces the mock, so exercise __aexit__ through a minimal
    # context wrapper that keeps the prepared cleanup clients.
    class Context:
        async def __aenter__(self):
            return client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return await client.__aexit__(exc_type, exc_val, exc_tb)

    async def run_context():
        async with Context():
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await run_context()

    assert excinfo.value.__context__ is siren_error
    siren_client.aclose.assert_awaited_once()
    kubo_cas.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_aexit_body_cancellation_keeps_all_cleanup_failures_as_context():
    siren_error = RuntimeError("Siren close failed")
    kubo_error = RuntimeError("Kubo close failed")
    client, _, _ = _client_with_mocks(siren_error=siren_error, kubo_error=kubo_error)
    incoming = asyncio.CancelledError()

    suppress = await client.__aexit__(asyncio.CancelledError, incoming, None)

    assert suppress is False
    assert incoming.__context__ is kubo_error
    assert kubo_error.__context__ is siren_error
