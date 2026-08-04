from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
import xarray as xr

from dclimate_client_py import dclimate_client, stac_server
from dclimate_client_py.stac_server import ResolvedDataset


COLLECTION = "example_collection"
DATASET = "temperature_mean"


def _feature(dataset: str, variant: str, cid: str) -> dict:
    return {
        "id": f"{COLLECTION}-{dataset}-{variant}",
        "collection": COLLECTION,
        "properties": {
            "dclimate:dataset_id": dataset,
            "dclimate:variant": variant,
        },
        "assets": {"data": {"href": f"ipfs://{cid}"}},
    }


async def _close_all_pooled_clients() -> None:
    """Close registry clients before removing the test-owned references."""
    errors: list[BaseException] = []
    for client in list(stac_server._ASYNC_HTTP_CLIENTS.values()):
        try:
            await client.aclose()
        except BaseException as error:
            errors.append(error)
    stac_server._ASYNC_HTTP_CLIENTS.clear()
    if errors:
        raise errors[0]


@pytest.fixture(autouse=True)
async def close_pooled_clients_between_tests():
    await _close_all_pooled_clients()
    try:
        yield
    finally:
        await _close_all_pooled_clients()


@pytest.mark.asyncio
async def test_async_resolver_follows_pagination_with_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        if request.url.params.get("page") == "2":
            payload = {"features": [_feature(DATASET, "default", "bafy-target")]}
        else:
            payload = {
                "features": [_feature("other_dataset", "default", "bafy-other")],
                "links": [
                    {
                        "rel": "next",
                        "href": "/search?page=2",
                        "method": "GET",
                    }
                ],
            }
        return httpx.Response(200, json=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            stac_server,
            "_async_client",
            lambda: (_ for _ in ()).throw(AssertionError("injected client ignored")),
        )

        resolved = await stac_server.aresolve_cid_from_stac_server(
            collection=COLLECTION,
            dataset=DATASET,
            variant="default",
            server_url="https://stac.example",
            client=client,
        )

    assert resolved == ResolvedDataset(cid="bafy-target", variant="default")
    assert requests == [
        ("POST", "https://stac.example/search"),
        ("GET", "https://stac.example/search?page=2"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "next_href",
    [
        "https://attacker.example/collect",
        "http://stac.example/search?page=2",
    ],
)
async def test_async_resolver_rejects_untrusted_pagination_links(
    next_href: str,
) -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "features": [_feature("other_dataset", "default", "bafy-other")],
                "links": [{"rel": "next", "href": next_href}],
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="configured server origin"):
            await stac_server.aresolve_cid_from_stac_server(
                collection=COLLECTION,
                dataset=DATASET,
                server_url="https://stac.example",
                client=client,
            )

    assert requests == ["https://stac.example/search"]


@pytest.mark.asyncio
async def test_async_resolver_drops_linked_headers_on_plaintext_pagination() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            payload = {
                "features": [_feature("other_dataset", "default", "bafy-other")],
                "links": [
                    {
                        "rel": "next",
                        "href": "/search?page=2",
                        "headers": {"Authorization": "Bearer continuation"},
                    }
                ],
            }
        else:
            payload = {"features": [_feature(DATASET, "default", "bafy-target")]}
        return httpx.Response(200, json=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolved = await stac_server.aresolve_cid_from_stac_server(
            collection=COLLECTION,
            dataset=DATASET,
            server_url="http://stac.example",
            client=client,
        )

    assert resolved.cid == "bafy-target"
    assert len(requests) == 2
    assert "authorization" not in requests[1].headers


def test_sync_resolver_rejects_untrusted_pagination_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "features": [_feature("other_dataset", "default", "bafy-other")],
                "links": [{"rel": "next", "href": "https://attacker.example/collect"}],
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(stac_server, "_client", lambda: client)
        with pytest.raises(ValueError, match="configured server origin"):
            stac_server.resolve_cid_from_stac_server(
                collection=COLLECTION,
                dataset=DATASET,
                server_url="https://stac.example",
            )

    assert requests == ["https://stac.example/search"]


@pytest.mark.asyncio
async def test_async_resolver_reuses_and_closes_loop_local_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_async_client = httpx.AsyncClient
    clients: list[httpx.AsyncClient] = []
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={"features": [_feature(DATASET, "default", "bafy-pooled")]},
            request=request,
        )

    def client_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        client = original_async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(handler),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(stac_server.httpx, "AsyncClient", client_factory)

    first = await stac_server.aresolve_cid_from_stac_server(
        collection=COLLECTION,
        dataset=DATASET,
        server_url="https://stac.example",
    )
    second = await stac_server.aresolve_cid_from_stac_server(
        collection=COLLECTION,
        dataset=DATASET,
        server_url="https://stac.example",
    )

    assert first == second == ResolvedDataset("bafy-pooled", "default")
    assert len(clients) == 1
    assert request_count == 2
    assert clients[0].is_closed is False

    await stac_server.aclose_stac_server_client()

    assert clients[0].is_closed is True
    assert not stac_server._ASYNC_HTTP_CLIENTS


@pytest.mark.asyncio
async def test_async_resolver_matches_sync_selection_rules() -> None:
    features = [
        _feature(DATASET, "latest", "bafy-latest"),
        _feature(DATASET, "final", "bafy-final"),
        _feature(DATASET, "default", "bafy-default"),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": features}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolved = await stac_server.aresolve_cid_from_stac_server(
            collection=COLLECTION,
            dataset=DATASET,
            server_url="https://stac.example",
            client=client,
        )

    assert resolved == ResolvedDataset("bafy-default", "default")


@pytest.mark.asyncio
async def test_high_level_client_uses_native_async_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = AsyncMock(return_value=ResolvedDataset("bafy-native", "default"))

    async def load_from_ipfs(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["ipfs_cid"] == "bafy-native"
        return xr.Dataset({"temperature": ("time", [21.0])}, coords={"time": [0]})

    monkeypatch.setattr(dclimate_client, "aresolve_cid_from_stac_server", resolver)
    monkeypatch.setattr(dclimate_client, "_load_dataset_from_ipfs_cid", load_from_ipfs)
    client = dclimate_client.dClimateClient(stac_server_url="https://stac.example")
    client._kubo_cas = object()
    stac_http_client = AsyncMock(spec=httpx.AsyncClient)
    stac_http_client.is_closed = False
    client._stac_http_client = stac_http_client

    dataset, metadata = await client.load_dataset(
        collection=COLLECTION,
        dataset=DATASET,
        variant="default",
        return_xarray=True,
    )

    resolver.assert_awaited_once_with(
        collection=COLLECTION,
        dataset=DATASET,
        variant="default",
        server_url="https://stac.example",
        client=stac_http_client,
    )
    assert dataset["temperature"].values.tolist() == [21.0]
    assert metadata["cid"] == "bafy-native"
