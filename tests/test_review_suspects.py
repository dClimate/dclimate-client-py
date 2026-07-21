from __future__ import annotations

import gc
from datetime import datetime, timezone
import json
from typing import Any
from unittest.mock import Mock

import httpx
import pystac
import pytest

from dclimate_client_py import ipfs_retrieval, stac_catalog, stac_server


_install_mock_client = None


@pytest.fixture(autouse=True)
def _use_managed_httpx_clients(install_httpx_mock):
    global _install_mock_client
    _install_mock_client = install_httpx_mock
    yield
    _install_mock_client = None


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def _install_post(monkeypatch, post) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        kwargs = {
            "json": json.loads(request.content),
            "timeout": request.extensions["timeout"]["read"],
        }
        if "authorization" in request.headers:
            kwargs["headers"] = {
                "Authorization": request.headers["authorization"],
            }
        response = post(str(request.url), **kwargs)
        return httpx.Response(200, json=response._payload, request=request)

    assert _install_mock_client is not None
    _install_mock_client(stac_server, handler)


def _catalog_with_item(item: pystac.Item) -> pystac.Catalog:
    root = pystac.Catalog(id="root", description="Root")
    organization = pystac.Catalog(id="org", description="Organization")
    collection = pystac.Collection(
        id="chirps",
        description="CHIRPS",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            pystac.TemporalExtent([[item.datetime, None]]),
        ),
    )
    collection.add_item(item)
    organization.add_child(collection)
    root.add_child(organization)

    organization_link = root.get_child_links()[0]
    organization_link.extra_fields.update(
        {
            "dclimate:id": "org",
            "dclimate:type": "organization",
            "dclimate:collections:historical": ["chirps"],
            "dclimate:datasets": ["chirps/precip-daily"],
        }
    )
    collection_link = organization.get_child_links()[0]
    collection_link.extra_fields["dclimate:id"] = "chirps"
    return root


def test_stac_resolvers_honor_hyphenated_dataset_and_variant(monkeypatch):
    cid = "bafy-hyphenated-identifiers"
    properties = {
        "dclimate:dataset_id": "precip-daily",
        "dclimate:variant": "final-p05",
    }
    feature = {
        "id": "chirps-precip-daily-final-p05",
        "collection": "chirps",
        "properties": properties,
        "assets": {"data": {"href": f"ipfs://{cid}"}},
    }
    _install_post(
        monkeypatch,
        lambda *args, **kwargs: _Response({"features": [feature]}),
    )

    item = pystac.Item(
        id=feature["id"],
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties=properties,
    )
    item.add_asset("data", pystac.Asset(href=f"ipfs://{cid}"))
    catalog = _catalog_with_item(item)

    assert (
        stac_server.resolve_cid_from_stac_server(
            collection="chirps",
            dataset="precip-daily",
            variant="final-p05",
            server_url="https://example.test",
        ).cid
        == cid
    )
    assert (
        stac_catalog.resolve_dataset_cid_from_stac(
            catalog,
            collection="chirps",
            dataset="precip-daily",
            variant="final-p05",
            organization="org",
        ).cid
        == cid
    )


def test_stac_server_follows_next_link_to_resolve_later_item(monkeypatch):
    collection = "test_collection"
    first_page_features = [
        {
            "id": f"{collection}-other_{index}-default",
            "collection": collection,
            "properties": {
                "dclimate:dataset_id": f"other_{index}",
                "dclimate:variant": "default",
            },
            "assets": {"data": {"href": f"ipfs://bafy-other-{index}"}},
        }
        for index in range(100)
    ]
    target = {
        "id": f"{collection}-target-finalized",
        "collection": collection,
        "properties": {
            "dclimate:dataset_id": "target",
            "dclimate:variant": "finalized",
        },
        "assets": {"data": {"href": "ipfs://bafy-page-two-target"}},
    }
    pages = [
        {
            "features": first_page_features,
            "links": [
                {
                    "rel": "next",
                    "href": "https://example.test/search?token=page-2",
                    "method": "POST",
                    "body": {"token": "page-2"},
                }
            ],
        },
        {"features": [target], "links": []},
    ]
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def post(url, json=None, **kwargs):
        calls.append((url, json))
        return _Response(pages[len(calls) - 1])

    _install_post(monkeypatch, post)

    resolved = stac_server.resolve_cid_from_stac_server(
        collection=collection,
        dataset="target",
        variant="finalized",
        server_url="https://example.test",
    )

    assert resolved.cid == "bafy-page-two-target"
    assert len(calls) == 2


def test_non_network_timeout_text_is_not_a_connection_error():
    error = RuntimeError(
        "invalid Zarr metadata: variable 'timeout' has an unsupported dtype"
    )

    assert not ipfs_retrieval._is_connection_error(error)


def test_stac_resolvers_agree_on_default_variant_for_bare_items(monkeypatch):
    # A bare item (no variant segment, no properties) is reported by the
    # listing APIs as variant "default"; BOTH resolvers must accept that
    # name so server->catalog fallback returns the same result.
    cid = "bafy-bare-default"
    feature = {
        "id": "chirps-temp",
        "collection": "chirps",
        "assets": {"data": {"href": f"ipfs://{cid}"}},
    }
    _install_post(
        monkeypatch,
        lambda *args, **kwargs: _Response({"features": [feature]}),
    )

    item = pystac.Item(
        id="chirps-temp",
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={},
    )
    item.add_asset("data", pystac.Asset(href=f"ipfs://{cid}"))
    catalog = _catalog_with_item(item)

    assert (
        stac_server.resolve_cid_from_stac_server(
            collection="chirps",
            dataset="temp",
            variant="default",
            server_url="https://example.test",
        ).cid
        == cid
    )
    assert (
        stac_catalog.resolve_dataset_cid_from_stac(
            catalog,
            collection="chirps",
            dataset="temp",
            variant="default",
            organization="org",
        ).cid
        == cid
    )


def test_stac_server_resolves_hyphenated_variant_without_properties(monkeypatch):
    # Same hyphenated grammar, but with no dclimate:* properties at all —
    # resolution must work from the item id alone given the dataset hint.
    cid = "bafy-id-only-hyphens"
    feature = {
        "id": "chirps-precip-daily-final-p05",
        "collection": "chirps",
        "assets": {"data": {"href": f"ipfs://{cid}"}},
    }
    _install_post(
        monkeypatch,
        lambda *args, **kwargs: _Response({"features": [feature]}),
    )

    assert (
        stac_server.resolve_cid_from_stac_server(
            collection="chirps",
            dataset="precip-daily",
            variant="final-p05",
            server_url="https://example.test",
        ).cid
        == cid
    )


def test_stac_server_merge_next_link_keeps_collections_filter(monkeypatch):
    # STAC API next-link contract: "merge": true extends the original body.
    # Dropping the collections filter on page 2 could poison resolution with
    # foreign-collection items.
    cid = "bafy-merged-page-two"
    bodies = []

    def post(url, json=None, timeout=None):
        bodies.append(json)
        if len(bodies) == 1:
            return _Response(
                {
                    "features": [
                        {
                            "id": "chirps-other",
                            "collection": "chirps",
                            "assets": {"data": {"href": "ipfs://bafy-other"}},
                        }
                    ],
                    "links": [
                        {
                            "rel": "next",
                            "href": "https://example.test/search",
                            "method": "POST",
                            "merge": True,
                            "body": {"token": "page-2"},
                        }
                    ],
                }
            )
        return _Response(
            {
                "features": [
                    {
                        "id": "chirps-temp-final",
                        "collection": "chirps",
                        "assets": {"data": {"href": f"ipfs://{cid}"}},
                    }
                ]
            }
        )

    _install_post(monkeypatch, post)

    assert (
        stac_server.resolve_cid_from_stac_server(
            collection="chirps",
            dataset="temp",
            variant="final",
            server_url="https://example.test",
        ).cid
        == cid
    )
    assert bodies[1]["token"] == "page-2"
    assert bodies[1]["collections"] == ["chirps"]


def test_stac_server_merge_next_link_without_body_keeps_original_body_and_headers(
    monkeypatch,
):
    calls = []

    def post(url, json=None, timeout=None, headers=None):
        calls.append({"url": url, "json": json, "headers": headers})
        if len(calls) == 1:
            return _Response(
                {
                    "features": [{"id": "chirps-other"}],
                    "links": [
                        {
                            "rel": "next",
                            "href": "/search",
                            "method": "POST",
                            "merge": True,
                            "headers": {"Authorization": "Bearer page-two"},
                        }
                    ],
                }
            )
        return _Response({"features": []})

    _install_post(monkeypatch, post)

    list(
        stac_server._search_pages(
            "https://example.test", {"limit": 100, "collections": ["chirps"]}, 10
        )
    )

    assert calls[1]["json"] == {"limit": 100, "collections": ["chirps"]}
    assert calls[1]["headers"] == {"Authorization": "Bearer page-two"}


def test_stac_server_raises_when_page_limit_would_truncate_results(monkeypatch):
    monkeypatch.setattr(stac_server, "_MAX_SEARCH_PAGES", 2)
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(
            {
                "features": [{"id": f"chirps-other-{calls}"}],
                "links": [
                    {
                        "rel": "next",
                        "href": f"/search?page={calls + 1}",
                        "method": "POST",
                        "body": {"page": calls + 1},
                    }
                ],
            }
        )

    monkeypatch.setattr(stac_server.requests, "post", post)

    with pytest.raises(ValueError, match="page limit of 2"):
        list(stac_server._search_pages("https://example.test", {"limit": 100}, 10))

    assert calls == 2


def test_stac_server_default_variant_can_be_on_a_later_page(monkeypatch):
    pages = [
        {
            "features": [
                {
                    "id": "chirps-temp-latest",
                    "collection": "chirps",
                    "assets": {"data": {"href": "ipfs://bafy-latest"}},
                }
            ],
            "links": [
                {
                    "rel": "next",
                    "href": "/search?page=2",
                    "method": "POST",
                    "body": {"page": 2},
                }
            ],
        },
        {
            "features": [
                {
                    "id": "chirps-temp",
                    "collection": "chirps",
                    "assets": {"data": {"href": "ipfs://bafy-default"}},
                }
            ]
        },
    ]
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        response = _Response(pages[calls])
        calls += 1
        return response

    _install_post(monkeypatch, post)

    resolved = stac_server.resolve_cid_from_stac_server(
        "chirps", "temp", server_url="https://example.test"
    )

    assert resolved.cid == "bafy-default"
    assert calls == 2


def test_known_hyphenated_dataset_does_not_match_shorter_prefix(monkeypatch):
    legacy_feature = {
        "id": "chirps-precip-daily-final-p05",
        "collection": "chirps",
        "assets": {"data": {"href": "ipfs://bafy-legacy-hyphenated"}},
    }
    explicit_sibling = {
        "id": "chirps-precip-daily-prelim-p05",
        "collection": "chirps",
        "properties": {
            "dclimate:dataset_id": "precip-daily",
            "dclimate:variant": "prelim-p05",
        },
        "assets": {"data": {"href": "ipfs://bafy-explicit-sibling"}},
    }
    _install_post(
        monkeypatch,
        lambda *args, **kwargs: _Response(
            {"features": [legacy_feature, explicit_sibling]}
        ),
    )

    with pytest.raises(ValueError, match="No items found"):
        stac_server.resolve_cid_from_stac_server(
            "chirps", "precip", server_url="https://example.test"
        )

    item = pystac.Item(
        id=legacy_feature["id"],
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={},
    )
    item.add_asset("data", pystac.Asset(href="ipfs://bafy-legacy-hyphenated"))
    catalog = _catalog_with_item(item)

    with pytest.raises(ValueError, match="Dataset 'precip' not found"):
        stac_catalog.resolve_dataset_cid_from_stac(
            catalog,
            collection="chirps",
            dataset="precip",
            organization="org",
        )


def test_requested_hyphenated_dataset_is_a_disambiguation_candidate(monkeypatch):
    features = [
        {
            "id": "chirps-precip-daily-final",
            "collection": "chirps",
            "properties": {},
            "assets": {"data": {"href": "ipfs://bafy-precip-daily-final"}},
        },
        {
            "id": "chirps-precip-default",
            "collection": "chirps",
            "properties": {
                "dclimate:dataset_id": "precip",
                "dclimate:variant": "default",
            },
            "assets": {"data": {"href": "ipfs://bafy-precip-default"}},
        },
    ]
    _install_post(
        monkeypatch,
        lambda *args, **kwargs: _Response({"features": features}),
    )

    resolved = stac_server.resolve_cid_from_stac_server(
        "chirps",
        "precip-daily",
        variant="final",
        server_url="https://example.test",
    )

    assert resolved.cid == "bafy-precip-daily-final"


def test_catalog_requested_dataset_is_a_disambiguation_candidate():
    target = pystac.Item(
        id="chirps-precip-daily-final",
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={},
    )
    target.add_asset(
        "data", pystac.Asset(href="ipfs://bafy-catalog-precip-daily-final")
    )
    catalog = _catalog_with_item(target)

    # Simulate incomplete metadata that declares only the shorter sibling.
    catalog.get_child_links()[0].extra_fields["dclimate:datasets"] = ["chirps/precip"]
    collection = catalog.get_child("org").get_child("chirps")
    sibling = pystac.Item(
        id="chirps-precip-default",
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={
            "dclimate:dataset_id": "precip",
            "dclimate:variant": "default",
        },
    )
    sibling.add_asset("data", pystac.Asset(href="ipfs://bafy-catalog-precip"))
    collection.add_item(sibling)

    cid = stac_catalog.resolve_dataset_cid_from_stac(
        catalog,
        collection="chirps",
        dataset="precip-daily",
        variant="final",
        organization="org",
    )

    assert cid == "bafy-catalog-precip-daily-final"


def test_load_stac_catalog_binds_io_without_mutating_pystac_default(monkeypatch):
    observed = {}

    def from_file(cls, href, stac_io=None):
        observed["href"] = href
        observed["stac_io"] = stac_io
        return pystac.Catalog(id="root", description="Root")

    monkeypatch.setattr(pystac.Catalog, "from_file", classmethod(from_file))
    monkeypatch.setattr(
        pystac.StacIO,
        "set_default",
        lambda *args, **kwargs: pytest.fail("global pystac default was mutated"),
    )

    stac_catalog.load_stac_catalog("https://gateway-a.test", root_cid="bafy-root")

    assert observed["href"] == "ipfs://bafy-root"
    assert isinstance(observed["stac_io"], stac_catalog.IPFSStacIO)
    assert observed["stac_io"].gateway_url == "https://gateway-a.test"


def test_load_stac_catalog_closes_client_when_parsing_fails(monkeypatch):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: None))
    monkeypatch.setattr(stac_catalog.httpx, "Client", lambda *args, **kwargs: client)

    def fail_from_file(cls, href, stac_io=None):
        raise RuntimeError("invalid catalog")

    monkeypatch.setattr(pystac.Catalog, "from_file", classmethod(fail_from_file))

    with pytest.raises(RuntimeError, match="invalid catalog"):
        stac_catalog.load_stac_catalog("https://gateway.test", root_cid="bafy-invalid")

    assert client.is_closed


def test_load_stac_catalog_closes_client_when_catalog_is_released(monkeypatch):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: None))
    monkeypatch.setattr(stac_catalog.httpx, "Client", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        pystac.Catalog,
        "from_file",
        classmethod(
            lambda cls, href, stac_io=None: pystac.Catalog(
                id="root", description="Root"
            )
        ),
    )

    catalog = stac_catalog.load_stac_catalog(
        "https://gateway.test", root_cid="bafy-root"
    )
    assert not client.is_closed

    del catalog
    gc.collect()

    assert client.is_closed


def test_load_stac_catalog_uses_configured_pointer_endpoint(monkeypatch):
    response = Mock()
    response.json.return_value = {"cid": "bafy-configured-root"}
    pointer_client = Mock()
    pointer_client.get.return_value = response
    monkeypatch.setattr(stac_catalog, "_client", lambda: pointer_client)
    observed = {}

    def from_file(cls, href, stac_io=None):
        observed["href"] = href
        return pystac.Catalog(id="root", description="Root")

    monkeypatch.setattr(pystac.Catalog, "from_file", classmethod(from_file))

    stac_catalog.load_stac_catalog(
        "https://gateway.test", catalog_url="https://control.test/catalog-root"
    )

    pointer_client.get.assert_called_once_with(
        "https://control.test/catalog-root", timeout=30, headers=None, auth=None
    )
    assert observed["href"] == "ipfs://bafy-configured-root"


def test_load_stac_catalog_threads_gateway_credentials(monkeypatch):
    """Auth/headers must reach both the pointer fetch and gateway I/O.

    Regression guard: authenticated gateways previously 401'd on catalog
    fallback because credentials were dropped between the KuboCAS data path
    and the STAC catalog reads.
    """
    headers = {"Authorization": "Bearer token"}
    auth = ("user", "secret")

    response = Mock()
    response.json.return_value = {"cid": "bafy-auth-root"}
    pointer_client = Mock()
    pointer_client.get.return_value = response
    monkeypatch.setattr(stac_catalog, "_client", lambda: pointer_client)

    captured: dict[str, Any] = {}

    def recording_init(self, gateway_url, *, headers=None, auth=None):
        captured["gateway_url"] = gateway_url
        captured["headers"] = headers
        captured["auth"] = auth
        self.gateway_url = gateway_url.rstrip("/")
        self.client = Mock()

    monkeypatch.setattr(stac_catalog.IPFSStacIO, "__init__", recording_init)

    def from_file(cls, href, stac_io=None):
        return pystac.Catalog(id="root", description="Root")

    monkeypatch.setattr(pystac.Catalog, "from_file", classmethod(from_file))

    stac_catalog.load_stac_catalog(
        "https://gateway.test",
        catalog_url="https://control.test/catalog-root",
        headers=headers,
        auth=auth,
    )

    # Pointer endpoint (same host as the gateway) carries the credentials.
    pointer_client.get.assert_called_once_with(
        "https://control.test/catalog-root",
        timeout=30,
        headers=headers,
        auth=auth,
    )
    # Gateway I/O handler is constructed with the credentials too.
    assert captured["headers"] == headers
    assert captured["auth"] == auth


def test_catalog_lister_uses_dataset_metadata_for_partial_item_properties():
    item = pystac.Item(
        id="chirps-precip-daily-final-p05",
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={"dclimate:dataset_id": "precip-daily"},
    )
    item.add_asset("data", pystac.Asset(href="ipfs://bafy-partial-catalog"))

    listing = stac_catalog.list_available_datasets(_catalog_with_item(item))
    variant = listing["chirps"]["variants"][0]

    assert variant["dataset"] == "precip-daily"
    assert variant["variant"] == "final-p05"
    assert variant["cid"] == "bafy-partial-catalog"


def test_catalog_lister_keeps_bare_item_with_default_variant_property():
    item = pystac.Item(
        id="chirps-precip-daily",
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={"dclimate:variant": "default"},
    )
    item.add_asset("data", pystac.Asset(href="ipfs://bafy-bare-default-catalog"))

    listing = stac_catalog.list_available_datasets(_catalog_with_item(item))
    variant = listing["chirps"]["variants"][0]

    assert variant["dataset"] == "precip-daily"
    assert variant["variant"] == "default"
