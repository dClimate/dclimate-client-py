from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pystac

from dclimate_client_py import ipfs_retrieval, stac_catalog, stac_server


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


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
    monkeypatch.setattr(
        stac_server.requests,
        "post",
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
        )
        == cid
    )
    assert (
        stac_catalog.resolve_dataset_cid_from_stac(
            catalog,
            collection="chirps",
            dataset="precip-daily",
            variant="final-p05",
            organization="org",
        )
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

    monkeypatch.setattr(stac_server.requests, "post", post)

    cid = stac_server.resolve_cid_from_stac_server(
        collection=collection,
        dataset="target",
        variant="finalized",
        server_url="https://example.test",
    )

    assert cid == "bafy-page-two-target"
    assert len(calls) == 2


def test_non_network_timeout_text_is_not_a_connection_error():
    error = RuntimeError(
        "invalid Zarr metadata: variable 'timeout' has an unsupported dtype"
    )

    assert not ipfs_retrieval._is_connection_error(error)
