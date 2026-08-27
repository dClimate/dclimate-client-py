"""
Unit tests for ``list_available_datasets_from_stac_server``.

These are pure unit tests — ``httpx.MockTransport`` handles all requests, so
the tests run offline and don't depend on the public STAC server or the IPFS
gateway. Integration coverage (parity with the IPFS walker) lives in
``test_list_datasets_parity.py`` and is gated behind ``--run-integration``.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest

from dclimate_client_py import stac_server
from dclimate_client_py.stac_server import (
    list_available_datasets_from_stac_server,
    resolve_cid_from_stac_server,
)


_install_mock_client = None


@pytest.fixture(autouse=True)
def _use_managed_httpx_clients(install_httpx_mock):
    global _install_mock_client
    _install_mock_client = install_httpx_mock
    yield
    _install_mock_client = None


def _mock_response(
    request: httpx.Request, payload: Dict[str, Any], status: int = 200
) -> httpx.Response:
    return httpx.Response(status, json=payload, request=request)


def _install_mocks(monkeypatch, *, collections_body, search_body):
    """Inject canned collection/search responses through the client accessor."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path.endswith("/collections"), (
                f"unexpected GET {request.url}"
            )
            return _mock_response(request, collections_body)
        assert request.method == "POST"
        assert request.url.path.endswith("/search"), f"unexpected POST {request.url}"
        return _mock_response(request, search_body)

    assert _install_mock_client is not None
    _install_mock_client(stac_server, handler)


SAMPLE_COLLECTIONS = {
    "collections": [
        {"id": "ecmwf_era5", "title": "ECMWF ERA5 Reanalysis"},
        {"id": "noaa_gfs", "title": "NOAA GFS Forecast"},
        # Collection with no items — should be dropped from the result.
        {"id": "test_empty", "title": "Empty"},
    ]
}

SAMPLE_SEARCH = {
    "features": [
        {
            "id": "ecmwf_era5-temperature_2m-finalized",
            "collection": "ecmwf_era5",
            "bbox": [-180.0, -90.0, 180.0, 90.0],
            "properties": {
                "dclimate:dataset_id": "temperature_2m",
                "dclimate:variant": "finalized",
                "dclimate:observation": "historical",
                "dclimate:latest_dataset_cid": "ipfs://bafy-era5-t2m-finalized",
                "start_datetime": "1979-01-01T00:00:00Z",
                "end_datetime": "2024-12-31T23:00:00Z",
            },
        },
        {
            "id": "ecmwf_era5-temperature_2m-non_finalized",
            "collection": "ecmwf_era5",
            "bbox": [-180.0, -90.0, 180.0, 90.0],
            "properties": {
                "dclimate:dataset_id": "temperature_2m",
                "dclimate:variant": "non_finalized",
                "dclimate:observation": "historical",
                "dclimate:latest_dataset_cid": "bafy-era5-t2m-nonfinal",
                "start_datetime": "2025-01-01T00:00:00Z",
                "end_datetime": "2026-01-01T00:00:00Z",
            },
        },
        {
            "id": "noaa_gfs-precipitation_total-default",
            "collection": "noaa_gfs",
            "bbox": [-180.0, -90.0, 180.0, 90.0],
            "properties": {
                "dclimate:dataset_id": "precipitation_total",
                "dclimate:variant": "default",
                "dclimate:observation": "forecast",
                "dclimate:latest_dataset_cid": "ipfs://bafy-gfs-precip",
                "start_datetime": "2026-05-01T00:00:00Z",
                "end_datetime": "2026-05-16T00:00:00Z",
            },
        },
    ]
}


def test_returns_expected_shape(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body=SAMPLE_COLLECTIONS,
        search_body=SAMPLE_SEARCH,
    )

    result = list_available_datasets_from_stac_server("https://example.test")

    # Empty collection (no items) should be filtered out.
    assert set(result.keys()) == {"ecmwf_era5", "noaa_gfs"}

    era5 = result["ecmwf_era5"]
    assert era5["id"] == "ecmwf_era5"
    assert era5["title"] == "ECMWF ERA5 Reanalysis"
    assert era5["organization"] == "ecmwf"
    assert era5["types"] == ["temperature_2m"]
    assert era5["category"] == "historical"

    gfs = result["noaa_gfs"]
    assert gfs["organization"] == "noaa"
    assert gfs["category"] == "forecast"


def test_strips_ipfs_scheme_from_cid(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body=SAMPLE_COLLECTIONS,
        search_body=SAMPLE_SEARCH,
    )

    result = list_available_datasets_from_stac_server("https://example.test")
    variants = {v["variant"]: v for v in result["ecmwf_era5"]["variants"]}

    # ipfs://... should be stripped, raw CIDs should pass through.
    assert variants["finalized"]["cid"] == "bafy-era5-t2m-finalized"
    assert variants["non_finalized"]["cid"] == "bafy-era5-t2m-nonfinal"


def test_extracts_extents(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body=SAMPLE_COLLECTIONS,
        search_body=SAMPLE_SEARCH,
    )

    result = list_available_datasets_from_stac_server("https://example.test")
    variant = next(
        v for v in result["ecmwf_era5"]["variants"] if v["variant"] == "finalized"
    )

    assert variant["spatial_extent"] == {"bbox": (-180.0, -90.0, 180.0, 90.0)}
    assert variant["temporal_extent"] == {
        "start": "1979-01-01T00:00:00Z",
        "end": "2024-12-31T23:00:00Z",
    }


def test_falls_back_to_id_parsing_when_properties_missing(monkeypatch):
    """Items without dclimate:* props should be parsed from the id pattern."""
    _install_mocks(
        monkeypatch,
        collections_body={"collections": [{"id": "legacy_coll", "title": "Legacy"}]},
        search_body={
            "features": [
                {
                    "id": "legacy_coll-temperature-default",
                    "collection": "legacy_coll",
                    "properties": {},  # no dclimate:* fields
                }
            ]
        },
    )

    result = list_available_datasets_from_stac_server("https://example.test")
    assert "legacy_coll" in result
    assert result["legacy_coll"]["types"] == ["temperature"]
    variants = result["legacy_coll"]["variants"]
    assert len(variants) == 1
    assert variants[0]["dataset"] == "temperature"
    assert variants[0]["variant"] == "default"
    # No CID asset — should be absent rather than None.
    assert "cid" not in variants[0]


def test_partial_properties_use_dataset_hint_and_asset_cid(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body={
            "collections": [
                {
                    "id": "chirps",
                    "title": "CHIRPS",
                    "dclimate:types": ["precip-daily"],
                }
            ]
        },
        search_body={
            "features": [
                {
                    "id": "chirps-precip-daily-final-p05",
                    "collection": "chirps",
                    "properties": {"dclimate:dataset_id": "precip-daily"},
                    "assets": {"data": {"href": "ipfs://bafy-partial-properties"}},
                }
            ]
        },
    )

    variant = list_available_datasets_from_stac_server("https://example.test")[
        "chirps"
    ]["variants"][0]

    assert variant["dataset"] == "precip-daily"
    assert variant["variant"] == "final-p05"
    assert variant["cid"] == "bafy-partial-properties"


def test_variant_only_property_keeps_bare_hyphenated_dataset(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body={
            "collections": [
                {
                    "id": "chirps",
                    "title": "CHIRPS",
                    "dclimate:types": ["precip-daily"],
                }
            ]
        },
        search_body={
            "features": [
                {
                    "id": "chirps-precip-daily",
                    "collection": "chirps",
                    "properties": {"dclimate:variant": "default"},
                }
            ]
        },
    )

    variant = list_available_datasets_from_stac_server("https://example.test")[
        "chirps"
    ]["variants"][0]

    assert variant["dataset"] == "precip-daily"
    assert variant["variant"] == "default"


def test_unknown_collection_uses_explicit_sibling_dataset_hints(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body={"collections": []},
        search_body={
            "features": [
                {
                    "id": "new_coll-precip-daily-final-p05",
                    "collection": "new_coll",
                    "properties": {},
                },
                {
                    "id": "new_coll-precip-daily-prelim-p05",
                    "collection": "new_coll",
                    "properties": {
                        "dclimate:dataset_id": "precip-daily",
                        "dclimate:variant": "prelim-p05",
                    },
                },
            ]
        },
    )

    listing = list_available_datasets_from_stac_server("https://example.test")

    assert listing["new_coll"]["types"] == ["precip-daily"]
    assert {variant["variant"] for variant in listing["new_coll"]["variants"]} == {
        "final-p05",
        "prelim-p05",
    }


def test_category_unanimous_only(monkeypatch):
    """When items in a collection disagree on observation, category is dropped."""
    _install_mocks(
        monkeypatch,
        collections_body={"collections": [{"id": "mixed", "title": "Mixed"}]},
        search_body={
            "features": [
                {
                    "id": "mixed-a-default",
                    "collection": "mixed",
                    "properties": {
                        "dclimate:dataset_id": "a",
                        "dclimate:variant": "default",
                        "dclimate:observation": "historical",
                    },
                },
                {
                    "id": "mixed-b-default",
                    "collection": "mixed",
                    "properties": {
                        "dclimate:dataset_id": "b",
                        "dclimate:variant": "default",
                        "dclimate:observation": "forecast",
                    },
                },
            ]
        },
    )

    result = list_available_datasets_from_stac_server("https://example.test")
    assert "category" not in result["mixed"]


def test_groups_multiple_variants_under_same_dataset(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body=SAMPLE_COLLECTIONS,
        search_body=SAMPLE_SEARCH,
    )

    result = list_available_datasets_from_stac_server("https://example.test")
    era5_variants = result["ecmwf_era5"]["variants"]

    # Both variants live under the same dataset; types lists the dataset once.
    assert {v["variant"] for v in era5_variants} == {"finalized", "non_finalized"}
    assert all(v["dataset"] == "temperature_2m" for v in era5_variants)
    assert result["ecmwf_era5"]["types"] == ["temperature_2m"]


def test_resolve_cid_uses_exact_dataset_id_for_prefix_collisions(monkeypatch):
    """Base ERA5 datasets must not resolve to similarly named *_land datasets."""
    _install_mocks(
        monkeypatch,
        collections_body=SAMPLE_COLLECTIONS,
        search_body={
            "features": [
                {
                    "id": "ecmwf_era5-precipitation_total_land-finalized",
                    "collection": "ecmwf_era5",
                    "properties": {
                        "dclimate:dataset_id": "precipitation_total_land",
                        "dclimate:variant": "finalized",
                    },
                    "assets": {
                        "data": {"href": "ipfs://bafy-era5-land-precip-finalized"}
                    },
                },
                {
                    "id": "ecmwf_era5-precipitation_total-finalized",
                    "collection": "ecmwf_era5",
                    "properties": {
                        "dclimate:dataset_id": "precipitation_total",
                        "dclimate:variant": "finalized",
                    },
                    "assets": {"data": {"href": "ipfs://bafy-era5-precip-finalized"}},
                },
            ]
        },
    )

    resolved = resolve_cid_from_stac_server(
        "ecmwf_era5",
        "precipitation_total",
        "finalized",
        "https://example.test",
    )

    assert resolved.cid == "bafy-era5-precip-finalized"


def test_resolve_cid_rejects_only_prefix_dataset_match(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body=SAMPLE_COLLECTIONS,
        search_body={
            "features": [
                {
                    "id": "ecmwf_era5-wind_u_10m_land-finalized",
                    "collection": "ecmwf_era5",
                    "properties": {
                        "dclimate:dataset_id": "wind_u_10m_land",
                        "dclimate:variant": "finalized",
                    },
                    "assets": {"data": {"href": "ipfs://bafy-era5-land-wind-u"}},
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="No items found"):
        resolve_cid_from_stac_server(
            "ecmwf_era5",
            "wind_u_10m",
            "finalized",
            "https://example.test",
        )


def test_resolve_cid_legacy_id_fallback_is_exact(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body=SAMPLE_COLLECTIONS,
        search_body={
            "features": [
                {
                    "id": "ecmwf_era5-temperature_2m_land-finalized",
                    "collection": "ecmwf_era5",
                    "properties": {
                        "dclimate:variant": "finalized",
                    },
                    "assets": {"data": {"href": "ipfs://bafy-era5-land-t2m"}},
                },
                {
                    "id": "ecmwf_era5-temperature_2m-finalized",
                    "collection": "ecmwf_era5",
                    "properties": {
                        "dclimate:variant": "finalized",
                    },
                    "assets": {"data": {"href": "ipfs://bafy-era5-t2m"}},
                },
            ]
        },
    )

    resolved = resolve_cid_from_stac_server(
        "ecmwf_era5",
        "temperature_2m",
        "finalized",
        "https://example.test",
    )

    assert resolved.cid == "bafy-era5-t2m"


def test_collections_endpoint_error_propagates(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return _mock_response(request, {}, status=500)
        return _mock_response(request, {"features": []})

    assert _install_mock_client is not None
    _install_mock_client(stac_server, handler)

    with pytest.raises(httpx.HTTPStatusError):
        list_available_datasets_from_stac_server("https://example.test")


def test_organization_derived_from_underscore_prefix(monkeypatch):
    _install_mocks(
        monkeypatch,
        collections_body={
            "collections": [
                {"id": "single", "title": "Single"},  # no underscore → None
                {"id": "org_thing", "title": "OrgThing"},
            ]
        },
        search_body={
            "features": [
                {
                    "id": "single-d-default",
                    "collection": "single",
                    "properties": {
                        "dclimate:dataset_id": "d",
                        "dclimate:variant": "default",
                    },
                },
                {
                    "id": "org_thing-d-default",
                    "collection": "org_thing",
                    "properties": {
                        "dclimate:dataset_id": "d",
                        "dclimate:variant": "default",
                    },
                },
            ]
        },
    )

    result = list_available_datasets_from_stac_server("https://example.test")
    assert result["single"]["organization"] is None
    assert result["org_thing"]["organization"] == "org_thing".split("_")[0]


def test_collections_pagination_follows_next(monkeypatch):
    """``/collections`` paginates, and only the first page arrived.

    The endpoint defaults to a page size smaller than the catalogue, so a single
    unpaged request returns a well-formed but short list. The collections past
    the first page are not obviously missing -- they simply arrive with no title
    or organization, because this endpoint is their only source. That surfaced
    as a live parity test failing on the 11th collection of 14.
    """
    requested: list[str] = []

    page_two = {"collections": [{"id": "noaa_gfs", "title": "NOAA GFS Forecast"}]}
    page_one = {
        "collections": [{"id": "ecmwf_era5", "title": "ECMWF ERA5 Reanalysis"}],
        "links": [{"rel": "next", "href": "https://stac.test/collections?offset=1"}],
        "numberMatched": 2,
        "numberReturned": 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            requested.append(str(request.url))
            body = page_two if "offset=1" in str(request.url) else page_one
            return _mock_response(request, body)
        return _mock_response(request, SAMPLE_SEARCH)

    assert _install_mock_client is not None
    _install_mock_client(stac_server, handler)

    result = list_available_datasets_from_stac_server("https://stac.test")

    assert any("offset=1" in url for url in requested), (
        "second page was never requested"
    )
    # The collection from page two must carry the title only /collections knows.
    assert result["noaa_gfs"]["title"] == "NOAA GFS Forecast"


def test_collections_pagination_raises_at_foreign_origin(monkeypatch):
    """A ``next`` pointing off-origin would walk the client out of its server.

    So it is never followed -- a redirect chain could otherwise steer catalogue
    reads to a host the caller never configured. But it is not dropped silently
    either: that would end the walk exactly like a server saying it was
    finished, returning a truncated catalogue that looks whole. Refusing the
    link keeps the boundary; raising keeps the truncation visible.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            if "evil.test" in str(request.url):
                raise AssertionError(f"followed next off-origin: {request.url}")
            return _mock_response(
                request,
                {
                    "collections": [
                        {"id": "ecmwf_era5", "title": "ECMWF ERA5 Reanalysis"}
                    ],
                    "links": [{"rel": "next", "href": "https://evil.test/collections"}],
                },
            )
        return _mock_response(request, SAMPLE_SEARCH)

    assert _install_mock_client is not None
    _install_mock_client(stac_server, handler)

    # The boundary held: the off-origin href was reported, never fetched.
    # (Had it been requested, the handler would have raised AssertionError.)
    with pytest.raises(ValueError, match="configured server origin"):
        list_available_datasets_from_stac_server("https://stac.test")


def test_layout_is_carried_onto_variants(monkeypatch):
    """``dclimate:layout`` says which loader a dataset needs.

    Without it the listing describes every dataset as the same kind, and a
    caller can only discover that GHCND needs ``load_entities`` rather than
    ``load_dataset`` by opening it and being refused.
    """
    collections = {
        "collections": [
            {"id": "noaa_ghcnd", "title": "GHCNd"},
            {"id": "ecmwf_era5", "title": "ERA5"},
        ]
    }
    search = {
        "features": [
            {
                "id": "noaa_ghcnd-station_observations-default",
                "collection": "noaa_ghcnd",
                "properties": {
                    "dclimate:dataset_id": "station_observations",
                    "dclimate:variant": "default",
                    "dclimate:layout": "tabular",
                    "dclimate:latest_dataset_cid": "ipfs://bafy-ghcnd",
                },
            },
            {
                "id": "ecmwf_era5-precipitation_total-default",
                "collection": "ecmwf_era5",
                "properties": {
                    "dclimate:dataset_id": "precipitation_total",
                    "dclimate:variant": "default",
                    "dclimate:layout": "zarr",
                    "dclimate:latest_dataset_cid": "ipfs://bafy-era5",
                },
            },
        ]
    }

    _install_mocks(monkeypatch, collections_body=collections, search_body=search)

    result = list_available_datasets_from_stac_server("https://example.test")

    assert result["noaa_ghcnd"]["variants"][0]["layout"] == "tabular"
    assert result["ecmwf_era5"]["variants"][0]["layout"] == "zarr"
