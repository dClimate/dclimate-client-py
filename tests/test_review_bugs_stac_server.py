from typing import Any


import dclimate_client_py.stac_server as stac_server


class _Response:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _mock_search(monkeypatch, features):
    def post(url, *, json, timeout):
        assert url == "https://stac.example/search"
        assert json == {"limit": 100, "collections": ["ecmwf_era5"]}
        assert timeout == 10
        return _Response({"features": features})

    monkeypatch.setattr(stac_server.requests, "post", post)


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
