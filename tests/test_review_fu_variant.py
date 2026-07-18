from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pystac
import xarray as xr

import dclimate_client_py.dclimate_client as dclimate_client_module
from dclimate_client_py import stac_catalog, stac_server
from dclimate_client_py.dclimate_client import dClimateClient


COLLECTION = "example_collection"
DATASET = "temperature"


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def _feature(variant: str, cid: str) -> dict[str, Any]:
    return {
        "id": f"{COLLECTION}-{DATASET}-{variant}",
        "collection": COLLECTION,
        "properties": {
            "dclimate:dataset_id": DATASET,
            "dclimate:variant": variant,
        },
        "assets": {"data": {"href": f"ipfs://{cid}"}},
    }


def _item(variant: str, cid: str) -> pystac.Item:
    item = pystac.Item(
        id=f"{COLLECTION}-{DATASET}-{variant}",
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={
            "dclimate:dataset_id": DATASET,
            "dclimate:variant": variant,
        },
    )
    item.add_asset("data", pystac.Asset(href=f"ipfs://{cid}"))
    return item


def _catalog_with_items(*items: pystac.Item) -> pystac.Catalog:
    root = pystac.Catalog(id="root", description="Root")
    organization = pystac.Catalog(id="example", description="Organization")
    collection = pystac.Collection(
        id=COLLECTION,
        description="Example collection",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            pystac.TemporalExtent([[items[0].datetime, None]]),
        ),
    )
    for item in items:
        collection.add_item(item)
    organization.add_child(collection)
    root.add_child(organization)

    root.get_child_links()[0].extra_fields["dclimate:id"] = "example"
    organization.get_child_links()[0].extra_fields["dclimate:id"] = COLLECTION
    return root


async def _stub_dataset_loader(**kwargs: Any) -> xr.Dataset:
    return xr.Dataset({"temperature": ("x", [1.0])})


def _cid(result: Any) -> str:
    """Accept today's str and the post-fix ResolvedDataset in behavior tests."""
    return getattr(result, "cid", result)


async def test_load_dataset_reports_variant_selected_by_stac_server(monkeypatch):
    monkeypatch.setattr(
        stac_server.requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "features": [
                    _feature("latest", "bafy-latest"),
                    _feature("final", "bafy-final"),
                ]
            }
        ),
    )
    monkeypatch.setattr(
        dclimate_client_module,
        "_load_dataset_from_ipfs_cid",
        _stub_dataset_loader,
    )
    client = dClimateClient(stac_server_url="https://stac.example")
    client._kubo_cas = object()

    _, metadata = await client.load_dataset(
        dataset=DATASET,
        collection=COLLECTION,
        variant=None,
        return_xarray=True,
    )

    assert metadata["variant"] == "final"
    assert metadata["slug"].endswith("/final")


async def test_load_dataset_reports_variant_selected_by_stac_catalog(monkeypatch):
    monkeypatch.setattr(
        dclimate_client_module,
        "_load_dataset_from_ipfs_cid",
        _stub_dataset_loader,
    )
    client = dClimateClient(stac_server_url=None)
    client._kubo_cas = object()
    client._stac_catalog = _catalog_with_items(
        _item("latest", "bafy-latest"),
        _item("final", "bafy-final"),
    )

    _, metadata = await client.load_dataset(
        dataset=DATASET,
        collection=COLLECTION,
        organization="example",
        variant=None,
        return_xarray=True,
    )

    assert metadata["variant"] == "final"
    assert metadata["slug"].endswith("/final")


async def test_direct_cid_without_variant_uses_unknown_consistently(monkeypatch):
    monkeypatch.setattr(
        dclimate_client_module,
        "_load_dataset_from_ipfs_cid",
        _stub_dataset_loader,
    )
    client = dClimateClient(stac_server_url=None)
    client._kubo_cas = object()

    _, metadata = await client.load_dataset(
        dataset=DATASET,
        collection=COLLECTION,
        cid="bafy-direct",
        variant=None,
        return_xarray=True,
    )

    assert metadata["variant"] == "unknown"
    assert metadata["slug"].endswith("/unknown")


async def test_explicit_variant_is_preserved_in_loaded_metadata(monkeypatch):
    monkeypatch.setattr(
        stac_server.requests,
        "post",
        lambda *args, **kwargs: _Response(
            {"features": [_feature("latest", "bafy-latest")]}
        ),
    )
    monkeypatch.setattr(
        dclimate_client_module,
        "_load_dataset_from_ipfs_cid",
        _stub_dataset_loader,
    )
    client = dClimateClient(stac_server_url="https://stac.example")
    client._kubo_cas = object()

    _, metadata = await client.load_dataset(
        dataset=DATASET,
        collection=COLLECTION,
        variant="latest",
        return_xarray=True,
    )

    assert metadata["variant"] == "latest"
    assert metadata["slug"].endswith("/latest")


def test_no_variant_search_paginates_to_preferred_default(monkeypatch):
    first_page_with_default = {
        "features": [
            {
                "id": f"{COLLECTION}-{DATASET}",
                "collection": COLLECTION,
                "properties": {"dclimate:dataset_id": DATASET},
                "assets": {"data": {"href": "ipfs://bafy-default"}},
            }
        ],
        "links": [
            {
                "rel": "next",
                "href": "https://stac.example/search",
                "method": "POST",
                "body": {"token": "page-2"},
            }
        ],
    }
    default_page_calls = 0

    def post_default_page(url, *, json, timeout):
        nonlocal default_page_calls
        default_page_calls += 1
        if default_page_calls > 1:
            raise AssertionError("default on page one should stop pagination")
        return _Response(first_page_with_default)

    monkeypatch.setattr(stac_server.requests, "post", post_default_page)
    page_one_default = stac_server.resolve_cid_from_stac_server(
        COLLECTION,
        DATASET,
        server_url="https://stac.example",
    )
    assert _cid(page_one_default) == "bafy-default"
    assert default_page_calls == 1

    pages = [
        {
            "features": [_feature("latest", "bafy-latest")],
            "links": [
                {
                    "rel": "next",
                    "href": "https://stac.example/search",
                    "method": "POST",
                    "body": {"token": "page-2"},
                }
            ],
        },
        {
            "features": [
                {
                    "id": f"{COLLECTION}-{DATASET}",
                    "collection": COLLECTION,
                    "properties": {"dclimate:dataset_id": DATASET},
                    "assets": {"data": {"href": "ipfs://bafy-default"}},
                }
            ]
        },
    ]
    calls: list[dict[str, Any] | None] = []

    def post(url, *, json, timeout):
        calls.append(json)
        return _Response(pages[len(calls) - 1])

    monkeypatch.setattr(stac_server.requests, "post", post)

    resolved = stac_server.resolve_cid_from_stac_server(
        COLLECTION,
        DATASET,
        server_url="https://stac.example",
    )

    assert _cid(resolved) == "bafy-default"
    assert len(calls) == 2


def test_resolvers_return_cid_and_selected_variant(monkeypatch):
    monkeypatch.setattr(
        stac_server.requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "features": [
                    _feature("latest", "bafy-latest"),
                    _feature("final", "bafy-final"),
                ]
            }
        ),
    )

    server_resolved = stac_server.resolve_cid_from_stac_server(
        COLLECTION,
        DATASET,
        server_url="https://stac.example",
    )
    catalog = _catalog_with_items(
        _item("latest", "bafy-latest"),
        _item("final", "bafy-final"),
    )
    catalog_resolved = stac_catalog.resolve_dataset_cid_from_stac(
        catalog,
        collection=COLLECTION,
        dataset=DATASET,
        organization="example",
    )

    assert all(
        hasattr(resolved, "cid") and hasattr(resolved, "variant")
        for resolved in (server_resolved, catalog_resolved)
    )
    assert server_resolved.cid == catalog_resolved.cid == "bafy-final"
    assert server_resolved.variant == catalog_resolved.variant == "final"
