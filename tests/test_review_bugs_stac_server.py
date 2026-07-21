import json

import httpx
import pytest
import dclimate_client_py.stac_server as stac_server


_install_mock_client = None


@pytest.fixture(autouse=True)
def _use_managed_httpx_clients(install_httpx_mock):
    global _install_mock_client
    _install_mock_client = install_httpx_mock
    yield
    _install_mock_client = None


def _mock_search(monkeypatch, features):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://stac.example/search"
        assert json.loads(request.content) == {
            "limit": 100,
            "collections": ["ecmwf_era5"],
        }
        assert set(request.extensions["timeout"].values()) == {10}
        return httpx.Response(200, json={"features": features}, request=request)

    assert _install_mock_client is not None
    _install_mock_client(stac_server, handler)


def test_resolve_variant_falls_back_to_variant_encoded_in_item_id(monkeypatch):
    _mock_search(
        monkeypatch,
        [
            {
                "id": "ecmwf_era5-temperature-finalized",
                "collection": "ecmwf_era5",
                "properties": {"dclimate:dataset_id": "temperature"},
                "assets": {"data": {"href": "ipfs://bafy-temperature-finalized"}},
            }
        ],
    )

    resolved = stac_server.resolve_cid_from_stac_server(
        "ecmwf_era5",
        "temperature",
        variant="finalized",
        server_url="https://stac.example",
    )

    assert resolved.cid == "bafy-temperature-finalized"
    assert resolved.variant == "finalized"


def test_resolve_feature_without_properties(monkeypatch):
    _mock_search(
        monkeypatch,
        [
            {
                "id": "ecmwf_era5-temperature",
                "collection": "ecmwf_era5",
                "assets": {"data": {"href": "ipfs://bafy-temperature"}},
            }
        ],
    )

    resolved = stac_server.resolve_cid_from_stac_server(
        "ecmwf_era5",
        "temperature",
        server_url="https://stac.example",
    )

    assert resolved.cid == "bafy-temperature"
    assert resolved.variant == "default"


def test_resolve_default_variant_matches_bare_item_id(monkeypatch):
    # list_available_datasets_from_stac_server reports items without a
    # variant segment as variant "default"; resolve must accept the same
    # name so a list -> resolve round-trip works.
    _mock_search(
        monkeypatch,
        [
            {
                "id": "ecmwf_era5-temperature",
                "collection": "ecmwf_era5",
                "assets": {"data": {"href": "ipfs://bafy-temperature"}},
            }
        ],
    )

    resolved = stac_server.resolve_cid_from_stac_server(
        "ecmwf_era5",
        "temperature",
        variant="default",
        server_url="https://stac.example",
    )

    assert resolved.cid == "bafy-temperature"
    assert resolved.variant == "default"


def test_resolve_without_variant_prefers_unnamed_item_over_latest(monkeypatch):
    _mock_search(
        monkeypatch,
        [
            {
                "id": "ecmwf_era5-temperature-latest",
                "collection": "ecmwf_era5",
                "assets": {"data": {"href": "ipfs://bafy-temperature-latest"}},
            },
            {
                "id": "ecmwf_era5-temperature",
                "collection": "ecmwf_era5",
                "assets": {"data": {"href": "ipfs://bafy-temperature"}},
            },
        ],
    )

    resolved = stac_server.resolve_cid_from_stac_server(
        "ecmwf_era5",
        "temperature",
        server_url="https://stac.example",
    )

    assert resolved.cid == "bafy-temperature"
    assert resolved.variant == "default"
