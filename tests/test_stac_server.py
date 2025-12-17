"""
Tests for STAC server integration module.

Tests the stac_server.py module which provides fast CID resolution via STAC server API.
These tests require a running STAC server at localhost:8081 (or STAC_SERVER_URL env var).
"""

import os
import pytest
import requests
from dclimate_client_py.stac_server import (
    resolve_cid_from_stac_server,
    STAC_SERVER_URL,
)


@pytest.fixture(scope="module")
def stac_server_url():
    """Get STAC server URL from environment or use default."""
    return os.environ.get("STAC_SERVER_URL", STAC_SERVER_URL)


@pytest.fixture(scope="module")
def check_stac_server(stac_server_url):
    """Check if STAC server is available, skip tests if not."""
    try:
        response = requests.post(
            f"{stac_server_url}/search",
            json={"limit": 1},
            timeout=5,
        )
        response.raise_for_status()
    except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
        pytest.skip(f"STAC server not available at {stac_server_url}")


@pytest.fixture(scope="module")
def available_dataset(stac_server_url, check_stac_server):
    """Get an available dataset from the STAC server for testing."""
    response = requests.post(
        f"{stac_server_url}/search",
        json={"limit": 10},
        timeout=10,
    )
    response.raise_for_status()
    features = response.json().get("features", [])

    if not features:
        pytest.skip("No datasets available in STAC server")

    # Parse first feature to extract collection/dataset/variant
    item = features[0]
    collection = item.get("collection")
    item_id = item.get("id", "")
    variant = item.get("properties", {}).get("dclimate:variant")

    # Extract dataset from item ID (format: {collection}-{dataset}-{variant})
    dataset = None
    if collection and item_id.startswith(f"{collection}-"):
        remainder = item_id[len(f"{collection}-"):]
        parts = remainder.split("-")
        if parts:
            dataset = parts[0]

    if not collection or not dataset:
        pytest.skip("Could not parse dataset info from STAC server response")

    return {
        "collection": collection,
        "dataset": dataset,
        "variant": variant,
        "item_id": item_id,
        "item": item,
    }


class TestStacServerConnection:
    """Test basic STAC server connectivity."""

    def test_default_server_url_constant(self):
        """Test that default server URL constant is set."""
        assert STAC_SERVER_URL == "https://api.stac.dclimate.net"

    def test_server_search_endpoint(self, stac_server_url, check_stac_server):
        """Test that search endpoint responds."""
        response = requests.post(
            f"{stac_server_url}/search",
            json={"limit": 1},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        assert "features" in data
        assert isinstance(data["features"], list)


class TestResolveCidFromStacServer:
    """Test the resolve_cid_from_stac_server function."""

    def test_resolve_cid_returns_string(self, stac_server_url, available_dataset):
        """Test that resolve returns a non-empty string CID."""
        cid = resolve_cid_from_stac_server(
            collection=available_dataset["collection"],
            dataset=available_dataset["dataset"],
            server_url=stac_server_url,
        )

        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_resolve_cid_no_ipfs_prefix(self, stac_server_url, available_dataset):
        """Test that returned CID has no ipfs:// prefix."""
        cid = resolve_cid_from_stac_server(
            collection=available_dataset["collection"],
            dataset=available_dataset["dataset"],
            server_url=stac_server_url,
        )

        assert not cid.startswith("ipfs://")

    def test_resolve_cid_valid_format(self, stac_server_url, available_dataset):
        """Test that returned CID has valid IPFS CID format."""
        cid = resolve_cid_from_stac_server(
            collection=available_dataset["collection"],
            dataset=available_dataset["dataset"],
            server_url=stac_server_url,
        )

        # IPFS CIDs typically start with these prefixes
        assert cid.startswith(("Qm", "bafy", "bafk", "bafz", "bafyr"))

    def test_resolve_cid_with_variant(self, stac_server_url, available_dataset):
        """Test CID resolution with specific variant."""
        if not available_dataset["variant"]:
            pytest.skip("Test dataset has no variant")

        cid = resolve_cid_from_stac_server(
            collection=available_dataset["collection"],
            dataset=available_dataset["dataset"],
            variant=available_dataset["variant"],
            server_url=stac_server_url,
        )

        assert isinstance(cid, str)
        assert len(cid) > 0
        assert not cid.startswith("ipfs://")

    def test_resolve_cid_without_variant(self, stac_server_url, available_dataset):
        """Test CID resolution without specifying variant."""
        cid = resolve_cid_from_stac_server(
            collection=available_dataset["collection"],
            dataset=available_dataset["dataset"],
            server_url=stac_server_url,
        )

        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_resolve_cid_invalid_collection_raises(self, stac_server_url, check_stac_server):
        """Test that invalid collection raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_cid_from_stac_server(
                collection="nonexistent_collection_xyz_12345",
                dataset="nonexistent_dataset",
                server_url=stac_server_url,
            )

        assert "No items found" in str(exc_info.value)

    def test_resolve_cid_invalid_dataset_raises(self, stac_server_url, available_dataset):
        """Test that invalid dataset raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_cid_from_stac_server(
                collection=available_dataset["collection"],
                dataset="nonexistent_dataset_xyz_12345",
                server_url=stac_server_url,
            )

        assert "No items found" in str(exc_info.value)

    def test_resolve_cid_invalid_variant_raises(self, stac_server_url, available_dataset):
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_cid_from_stac_server(
                collection=available_dataset["collection"],
                dataset=available_dataset["dataset"],
                variant="nonexistent_variant_xyz_12345",
                server_url=stac_server_url,
            )

        assert "Variant" in str(exc_info.value) or "not found" in str(exc_info.value)

    def test_resolve_cid_connection_error_on_bad_url(self):
        """Test that connection error is raised for unreachable server."""
        with pytest.raises((requests.ConnectionError, requests.exceptions.RequestException)):
            resolve_cid_from_stac_server(
                collection="any",
                dataset="any",
                server_url="http://127.0.0.1:59999",
            )

    def test_resolve_cid_consistent_results(self, stac_server_url, available_dataset):
        """Test that multiple calls return consistent results."""
        cid1 = resolve_cid_from_stac_server(
            collection=available_dataset["collection"],
            dataset=available_dataset["dataset"],
            server_url=stac_server_url,
        )

        cid2 = resolve_cid_from_stac_server(
            collection=available_dataset["collection"],
            dataset=available_dataset["dataset"],
            server_url=stac_server_url,
        )

        assert cid1 == cid2


class TestMultipleDatasets:
    """Test resolving multiple different datasets."""

    def test_resolve_multiple_datasets(self, stac_server_url, check_stac_server):
        """Test resolving CIDs for multiple datasets from the server."""
        # Get multiple datasets
        response = requests.post(
            f"{stac_server_url}/search",
            json={"limit": 50},
            timeout=10,
        )
        response.raise_for_status()
        features = response.json().get("features", [])

        if len(features) < 2:
            pytest.skip("Need at least 2 datasets for this test")

        # Group by collection/dataset
        datasets_seen = set()
        resolved_cids = []

        for item in features:
            collection = item.get("collection")
            item_id = item.get("id", "")

            if not collection or not item_id.startswith(f"{collection}-"):
                continue

            remainder = item_id[len(f"{collection}-"):]
            parts = remainder.split("-")
            if not parts:
                continue

            dataset = parts[0]
            key = f"{collection}/{dataset}"

            if key in datasets_seen:
                continue

            datasets_seen.add(key)

            try:
                cid = resolve_cid_from_stac_server(
                    collection=collection,
                    dataset=dataset,
                    server_url=stac_server_url,
                )
                resolved_cids.append((key, cid))
            except ValueError:
                continue

            if len(resolved_cids) >= 3:
                break

        assert len(resolved_cids) >= 1, "Should resolve at least one dataset"

        for key, cid in resolved_cids:
            assert isinstance(cid, str)
            assert len(cid) > 0
            assert not cid.startswith("ipfs://")
