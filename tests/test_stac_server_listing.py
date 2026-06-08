"""
Unit tests for ``list_available_datasets_from_stac_server``.

These are pure unit tests — ``requests.get`` / ``requests.post`` are mocked, so
the tests run offline and don't depend on the public STAC server or the IPFS
gateway. Integration coverage (parity with the IPFS walker) lives in
``test_list_datasets_parity.py`` and is gated behind ``--run-integration``.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import requests

from dclimate_client_py.stac_server import (
    list_available_datasets_from_stac_server,
)


def _mock_response(payload: Dict[str, Any], status: int = 200):
    """Build a stand-in for a ``requests.Response`` that has ``json()`` and
    ``raise_for_status()``."""

    class _Resp:
        def __init__(self) -> None:
            self.status_code = status

        def json(self):
            return payload

        def raise_for_status(self):
            if status >= 400:
                raise requests.HTTPError(f"HTTP {status}")

    return _Resp()


def _install_mocks(monkeypatch, *, collections_body, search_body):
    """Patch requests.get/post to return canned bodies based on URL."""

    def fake_get(url, *args, **kwargs):
        assert url.endswith("/collections"), f"unexpected GET {url}"
        return _mock_response(collections_body)

    def fake_post(url, *args, **kwargs):
        assert url.endswith("/search"), f"unexpected POST {url}"
        return _mock_response(search_body)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)


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


def test_collections_endpoint_error_propagates(monkeypatch):
    def failing_get(url, *args, **kwargs):
        return _mock_response({}, status=500)

    monkeypatch.setattr(requests, "get", failing_get)
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: _mock_response({"features": []})
    )

    with pytest.raises(requests.HTTPError):
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
