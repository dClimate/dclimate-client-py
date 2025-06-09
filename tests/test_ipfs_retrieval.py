import os
import re
import json
from typing import Dict, Any  # Import Dict and Any
from unittest.mock import patch, mock_open, MagicMock, ANY

import pytest
import requests
from multiformats import CID  # Import CID

import dclimate_zarr_client.ipfs_retrieval as ipfs_retrieval
from dclimate_zarr_client.ipfs_retrieval import (
    _stac_hamt_cid_cache,
    fetch_json_from_ipns,
    get_dataset_hamt_cid_from_stac,
    get_ipns_name_hash,
    _get_dataset_by_ipfs_cid,
    update_cache_if_changed,
    _get_single_metadata,
    list_datasets,
    DatasetNotFoundError,
    IpfsConnectionError,
    StacCatalogError,
    # Import internal functions for testing
    _get_ipfs_store,
    fetch_json_from_cid,  # noqa: E402 Testing private member
    _get_host,  # noqa: E402 Testing private member
)
from py_hamt import IPFSStore  # Import IPFSStore

# Import constants/configs
from dclimate_zarr_client.client import DCLIMATE_STAC_CATALOG_IPNS
from .conftest import KNOWN_STAC_DATASET_ID, KNOWN_STAC_DATASET_ID_2

# Apply IPFS check fixture ONLY to tests that actually need functional IPFS
# Most tests in this file should be mocked unit tests.
# pytestmark = pytest.mark.usefixtures("check_ipfs_connection") # Remove module-level mark

# --- Type Hinting ---
MonkeyPatch = pytest.MonkeyPatch
MockIPFSStore = MagicMock
MockResponse = MagicMock  # Alias for requests.Response mock


# --- Fixtures ---
@pytest.fixture
def mock_ipfs_store() -> MockIPFSStore:
    """Fixture to create a mock IPFSStore instance."""
    store = MagicMock(spec=IPFSStore)
    store.gateway_uri_stem = "http://mock-gateway:8080"
    store.rpc_uri_stem = "http://mock-rpc:5001"
    return store


@pytest.fixture
def mock_requests_get(mocker) -> MagicMock:
    """Fixture to mock requests.get."""
    return mocker.patch("requests.get")


# --- Helper to get cache path ---test_get_ipfs_store_defaults
def get_cache_path():
    # Helper to get the expected cache file path within the package
    # Need to ensure this reflects the actual location used by the code
    package_dir = os.path.dirname(ipfs_retrieval.__file__)
    return os.path.join(package_dir, "cids.json")


# --- Tests for _get_ipfs_store ---


def test_get_ipfs_store_defaults(monkeypatch: MonkeyPatch):
    """Test store creation uses defaults when no args/env vars are set."""
    monkeypatch.delenv("IPFS_GATEWAY_URI_STEM", raising=False)
    monkeypatch.delenv("IPFS_RPC_URI_STEM", raising=False)
    with patch("dclimate_zarr_client.ipfs_retrieval.IPFSStore") as mock_store_class:
        store = _get_ipfs_store()
        mock_store_class.assert_called_once_with()  # Called with no args, uses internal defaults
        assert isinstance(
            store, MagicMock
        )  # We mocked the class, so instance is MagicMock


def test_get_ipfs_store_args_override_env(monkeypatch: MonkeyPatch):
    """Test that function arguments override environment variables."""
    monkeypatch.setenv("IPFS_GATEWAY_URI_STEM", "http://env-gateway:8080")
    monkeypatch.setenv("IPFS_RPC_URI_STEM", "http://env-rpc:5001")
    with patch("dclimate_zarr_client.ipfs_retrieval.IPFSStore") as mock_store_class:
        store = _get_ipfs_store(
            gateway_uri_stem="http://arg-gateway:8080",
            rpc_uri_stem="http://arg-rpc:5001",
        )
        mock_store_class.assert_called_once_with(
            gateway_uri_stem="http://arg-gateway:8080",
            rpc_uri_stem="http://arg-rpc:5001",
        )
        assert isinstance(store, MagicMock)


def test_get_ipfs_store_env_vars(monkeypatch: MonkeyPatch):
    """Test that environment variables are used when no args are provided."""
    monkeypatch.setenv("IPFS_GATEWAY_URI_STEM", "http://env-gateway:8080")
    monkeypatch.setenv("IPFS_RPC_URI_STEM", "http://env-rpc:5001")
    with patch("dclimate_zarr_client.ipfs_retrieval.IPFSStore") as mock_store_class:
        store = _get_ipfs_store()
        mock_store_class.assert_called_once_with(
            gateway_uri_stem="http://env-gateway:8080",
            rpc_uri_stem="http://env-rpc:5001",
        )
        assert isinstance(store, MagicMock)


def test_get_ipfs_store_mixed_args_env(monkeypatch: MonkeyPatch):
    """Test using a mix of args and env vars (args should take precedence)."""
    monkeypatch.setenv("IPFS_GATEWAY_URI_STEM", "http://env-gateway:8080")
    monkeypatch.delenv("IPFS_RPC_URI_STEM", raising=False)  # RPC not set in env
    with patch("dclimate_zarr_client.ipfs_retrieval.IPFSStore") as mock_store_class:
        store = _get_ipfs_store(
            rpc_uri_stem="http://arg-rpc:5001"
        )  # Provide RPC as arg
        mock_store_class.assert_called_once_with(
            gateway_uri_stem="http://env-gateway:8080",  # Gateway comes from env
            rpc_uri_stem="http://arg-rpc:5001",  # RPC comes from arg
        )
        assert isinstance(store, MagicMock)


# --- Tests for fetch_json_from_cid (Unit/Mocked) ---


def test_fetch_json_from_cid_success(mock_ipfs_store: MockIPFSStore):
    """Test successful fetching and decoding of JSON from CID."""
    valid_cid_str = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    json_data: Dict[str, Any] = {"key": "value"}
    mock_ipfs_store.load.return_value = json.dumps(json_data).encode("utf-8")
    result: Dict[str, Any] = fetch_json_from_cid(valid_cid_str, mock_ipfs_store)
    assert result == json_data
    mock_ipfs_store.load.assert_called_once()
    # Check that CID.decode was implicitly called by store.load mock (or explicitly if store expects CID obj)
    # For MagicMock, we check the arg type if needed, assuming it passes the string
    call_args = mock_ipfs_store.load.call_args[0]
    assert isinstance(call_args[0], CID)
    assert str(call_args[0]) == valid_cid_str


def test_fetch_json_from_cid_success_with_prefix(mock_ipfs_store: MockIPFSStore):
    """Test successful fetching when CID string has /ipfs/ prefix."""
    cid_str_no_prefix = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    cid_str_with_prefix = f"/ipfs/{cid_str_no_prefix}"
    json_data: Dict[str, Any] = {"key": "value"}
    mock_ipfs_store.load.return_value = json.dumps(json_data).encode("utf-8")
    result: Dict[str, Any] = fetch_json_from_cid(cid_str_with_prefix, mock_ipfs_store)
    assert result == json_data
    mock_ipfs_store.load.assert_called_once()
    call_args = mock_ipfs_store.load.call_args[0]
    assert isinstance(call_args[0], CID)
    assert str(call_args[0]) == cid_str_no_prefix  # Prefix should be stripped


def test_fetch_json_from_cid_invalid_cid_string(mock_ipfs_store: MockIPFSStore):
    """Test StacCatalogError when CID string is invalid."""
    invalid_cid_str = "this-is-not-a-cid"
    with pytest.raises(
        StacCatalogError, match=f"Failed to decode CID string '{invalid_cid_str}'"
    ):
        fetch_json_from_cid(invalid_cid_str, mock_ipfs_store)
    mock_ipfs_store.load.assert_not_called()  # Should fail before calling load


def test_fetch_json_from_cid_load_returns_none(mock_ipfs_store: MockIPFSStore):
    """Test StacCatalogError when ipfs_store.load returns None or empty bytes."""
    valid_cid_str = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    mock_ipfs_store.load.return_value = b""  # Empty bytes
    with pytest.raises(
        StacCatalogError, match=f"No data returned for CID: {valid_cid_str}"
    ):
        fetch_json_from_cid(valid_cid_str, mock_ipfs_store)
    mock_ipfs_store.load.assert_called_once()


def test_fetch_json_from_cid_json_decode_error(mock_ipfs_store: MockIPFSStore):
    """Test StacCatalogError when fetched data is not valid JSON."""
    valid_cid_str = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    mock_ipfs_store.load.return_value = b"this is not json"
    with pytest.raises(
        StacCatalogError, match=f"Failed to decode JSON from CID {valid_cid_str}"
    ):
        fetch_json_from_cid(valid_cid_str, mock_ipfs_store)
    mock_ipfs_store.load.assert_called_once()


def test_fetch_json_from_cid_timeout_error(mock_ipfs_store: MockIPFSStore):
    """Test IpfsConnectionError when ipfs_store.load raises Timeout."""
    valid_cid_str = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    mock_ipfs_store.load.side_effect = requests.exceptions.Timeout("Request timed out")
    with pytest.raises(
        IpfsConnectionError, match=f"Timeout fetching CID {valid_cid_str}"
    ):
        fetch_json_from_cid(valid_cid_str, mock_ipfs_store)
    mock_ipfs_store.load.assert_called_once()


def test_fetch_json_from_cid_connection_error(mock_ipfs_store: MockIPFSStore):
    """Test IpfsConnectionError when ipfs_store.load raises connection-related error."""
    valid_cid_str = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    # Simulate different connection error messages
    errors_to_test = [
        requests.exceptions.ConnectionError("Connection refused"),
        requests.exceptions.RequestException("Max retries exceeded"),
        requests.exceptions.RequestException("Failed to establish a new connection"),
    ]
    for error in errors_to_test:
        mock_ipfs_store.load.side_effect = error
        with pytest.raises(
            IpfsConnectionError,
            match=f"Failed to connect via IPFSStore.*to fetch CID {valid_cid_str}",
        ):
            fetch_json_from_cid(valid_cid_str, mock_ipfs_store)
        mock_ipfs_store.load.assert_called_once()
        mock_ipfs_store.load.reset_mock()  # Reset mock for next error in loop


def test_fetch_json_from_cid_generic_load_error(mock_ipfs_store: MockIPFSStore):
    """Test StacCatalogError for other exceptions during ipfs_store.load."""
    valid_cid_str = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    mock_ipfs_store.load.side_effect = RuntimeError("Some other IPFSStore error")

    with pytest.raises(
        StacCatalogError,
        match=f"Error fetching data for CID {valid_cid_str} via IPFSStore",
    ):
        fetch_json_from_cid(valid_cid_str, mock_ipfs_store)
    mock_ipfs_store.load.assert_called_once()


# --- Tests for _get_host ---
def test_get_host_default():
    """Test _get_host uses the default localhost when env var is not set."""
    # Import DEFAULT_HOST to use in assertion
    from dclimate_zarr_client.ipfs_retrieval import DEFAULT_HOST

    with patch.dict(os.environ, {}, clear=True):  # Ensure env var is not set
        assert _get_host() == DEFAULT_HOST  # Check default URI
        # If IPFS_HOST is not set, _get_host IGNORES the uri argument and returns DEFAULT_HOST
        assert _get_host("/custom/uri") == DEFAULT_HOST


def test_get_host_from_env(monkeypatch: MonkeyPatch):
    """Test _get_host uses IPFS_HOST environment variable."""
    monkeypatch.setenv("IPFS_HOST", "http://my-ipfs-node:5002")
    assert _get_host() == "http://my-ipfs-node:5002/api/v0"
    assert _get_host("/other/uri") == "http://my-ipfs-node:5002/other/uri"


# --- NEW: Tests for fetch_json_from_ipns (Mocked Error Paths) ---


class TestFetchJsonFromIpnsErrors:
    MOCK_IPNS = "k51qzi5uqu5dk89atnl883sr0g1cb2py631ckz9ng45qhk6dg0pj141jtxtx6l"
    MOCK_GATEWAY = "http://mock-gateway:8080"
    EXPECTED_URL = f"{MOCK_GATEWAY}/ipns/{MOCK_IPNS}"

    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch, mock_requests_get):
        # Mock _get_ipfs_store to return a predictable gateway
        mock_store = MagicMock(spec=IPFSStore)
        mock_store.gateway_uri_stem = self.MOCK_GATEWAY
        monkeypatch.setattr(
            ipfs_retrieval, "_get_ipfs_store", lambda *args, **kwargs: mock_store
        )
        self.mock_requests_get = mock_requests_get

    def mock_response(
        self, status_code=200, json_data=None, text=None, raise_for_status_error=None
    ) -> MockResponse:
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = status_code
        mock_resp.raise_for_status.side_effect = raise_for_status_error
        if json_data is not None:
            mock_resp.json.return_value = json_data
            # If json_data is provided, requests usually sets text as well
            mock_resp.text = json.dumps(json_data) if text is None else text
        else:
            mock_resp.json.side_effect = requests.exceptions.JSONDecodeError(
                "Expecting value", "doc", 0
            )
            mock_resp.text = text if text is not None else "Invalid JSON"
        return mock_resp

    def test_fetch_json_from_ipns_empty_name(self):
        with pytest.raises(ValueError, match="IPNS name cannot be empty"):
            fetch_json_from_ipns("")

    def test_fetch_json_from_ipns_initial_timeout_then_success(self):
        """Simulate timeout on first try, success on retry."""
        mock_resp_success = self.mock_response(
            status_code=200, json_data={"type": "Catalog"}
        )
        self.mock_requests_get.side_effect = [
            requests.exceptions.Timeout("Initial timeout"),
            mock_resp_success,
        ]

        result = fetch_json_from_ipns(self.MOCK_IPNS)

        assert result == {"type": "Catalog"}
        assert self.mock_requests_get.call_count == 2
        # Check params: first call with nocache=true, second without
        assert self.mock_requests_get.call_args_list[0].kwargs["params"] == {
            "nocache": "true"
        }
        assert self.mock_requests_get.call_args_list[1].kwargs["params"] == {}

    def test_fetch_json_from_ipns_initial_connection_error(self):
        """Simulate ConnectionError on first try (should raise immediately)."""
        self.mock_requests_get.side_effect = requests.exceptions.ConnectionError(
            "Gateway down"
        )

        with pytest.raises(IpfsConnectionError, match="Connection error fetching IPNS"):
            fetch_json_from_ipns(self.MOCK_IPNS)
        assert self.mock_requests_get.call_count == 1

    def test_fetch_json_from_ipns_initial_json_decode_then_success(self):
        """Simulate JSON decode error on first try, success on retry."""
        mock_resp_bad_json = self.mock_response(
            status_code=200, text="<html>error page</html>"
        )
        mock_resp_success = self.mock_response(
            status_code=200, json_data={"type": "Catalog"}
        )
        self.mock_requests_get.side_effect = [mock_resp_bad_json, mock_resp_success]

        result = fetch_json_from_ipns(self.MOCK_IPNS)

        assert result == {"type": "Catalog"}
        assert self.mock_requests_get.call_count == 2
        assert self.mock_requests_get.call_args_list[0].kwargs["params"] == {
            "nocache": "true"
        }
        assert self.mock_requests_get.call_args_list[1].kwargs["params"] == {}

    def test_fetch_json_from_ipns_initial_500_error_then_success(self):
        """Simulate 500 HTTP error on first try, success on retry."""
        mock_resp_500 = self.mock_response(
            status_code=500,
            text="Server Error",
            raise_for_status_error=requests.exceptions.HTTPError("500 Error"),
        )
        mock_resp_success = self.mock_response(
            status_code=200, json_data={"type": "Catalog"}
        )
        self.mock_requests_get.side_effect = [mock_resp_500, mock_resp_success]

        result = fetch_json_from_ipns(self.MOCK_IPNS)

        assert result == {"type": "Catalog"}
        assert self.mock_requests_get.call_count == 2

    def test_fetch_json_from_ipns_retry_timeout(self):
        """Simulate failure on first try (e.g., timeout) AND timeout on retry."""
        self.mock_requests_get.side_effect = [
            requests.exceptions.Timeout("Initial timeout"),
            requests.exceptions.Timeout("Retry timeout"),
        ]

        with pytest.raises(
            IpfsConnectionError, match="Timeout .* during IPNS fetch retry"
        ):
            fetch_json_from_ipns(self.MOCK_IPNS)
        assert self.mock_requests_get.call_count == 2

    def test_fetch_json_from_ipns_retry_connection_error(self):
        """Simulate failure on first try AND ConnectionError on retry."""
        self.mock_requests_get.side_effect = [
            requests.exceptions.Timeout("Initial timeout"),
            requests.exceptions.ConnectionError("Retry connection failed"),
        ]

        with pytest.raises(
            IpfsConnectionError, match="Connection error during IPNS fetch retry"
        ):
            fetch_json_from_ipns(self.MOCK_IPNS)
        assert self.mock_requests_get.call_count == 2

    def test_fetch_json_from_ipns_retry_json_decode_error(self):
        """Simulate failure on first try AND JSON decode error on retry."""
        mock_resp_bad_json_retry = self.mock_response(
            status_code=200, text="Retry also bad json"
        )
        self.mock_requests_get.side_effect = [
            requests.exceptions.Timeout("Initial timeout"),
            mock_resp_bad_json_retry,
        ]

        with pytest.raises(
            StacCatalogError, match="Invalid JSON fetching IPNS .* \\(retry\\)"
        ):
            fetch_json_from_ipns(self.MOCK_IPNS)
        assert self.mock_requests_get.call_count == 2

    def test_fetch_json_from_ipns_retry_http_error(self):
        """Simulate failure on first try AND HTTP error on retry."""
        mock_resp_503_retry = self.mock_response(
            status_code=503,
            text="Service Unavailable",
            raise_for_status_error=requests.exceptions.HTTPError("503 Error"),
        )
        self.mock_requests_get.side_effect = [
            requests.exceptions.Timeout("Initial timeout"),
            mock_resp_503_retry,
        ]

        with pytest.raises(
            StacCatalogError, match="Error fetching IPNS .* \\(retry\\) via Gateway"
        ):
            fetch_json_from_ipns(self.MOCK_IPNS)
        assert self.mock_requests_get.call_count == 2

    def test_fetch_json_from_ipns_initial_other_exception_then_success(self):
        """Simulate generic exception on first try, success on retry."""
        mock_resp_success = self.mock_response(
            status_code=200, json_data={"type": "Catalog"}
        )
        self.mock_requests_get.side_effect = [
            RuntimeError("Unexpected issue"),
            mock_resp_success,
        ]
        result = fetch_json_from_ipns(self.MOCK_IPNS)
        assert result == {"type": "Catalog"}
        assert self.mock_requests_get.call_count == 2

    def test_fetch_json_from_ipns_retry_other_exception(self):
        """Simulate failure on first try AND generic exception on retry."""
        self.mock_requests_get.side_effect = [
            requests.exceptions.Timeout("Initial timeout"),
            RuntimeError("Unexpected issue on retry"),
        ]
        with pytest.raises(
            StacCatalogError, match="Unexpected error during IPNS fetch retry"
        ):
            fetch_json_from_ipns(self.MOCK_IPNS)
        assert self.mock_requests_get.call_count == 2


# --- NEW: Tests for get_dataset_hamt_cid_from_stac (Mocked Error Paths) ---


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_root_catalog_fetch_error(mock_fetch_cid, mock_fetch_ipns):
    """Test error when fetching the root catalog fails."""
    mock_fetch_ipns.side_effect = IpfsConnectionError("Cannot connect to gateway")
    with pytest.raises(StacCatalogError, match="Failed to fetch root catalog"):
        get_dataset_hamt_cid_from_stac(
            DCLIMATE_STAC_CATALOG_IPNS, KNOWN_STAC_DATASET_ID
        )
    mock_fetch_ipns.assert_called_once()
    mock_fetch_cid.assert_not_called()


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_invalid_root_catalog_format(mock_fetch_cid, mock_fetch_ipns):
    """Test error when root catalog JSON is not a valid STAC Catalog."""
    mock_fetch_ipns.return_value = {"not": "a catalog"}
    with pytest.raises(StacCatalogError, match="Invalid root catalog format"):
        get_dataset_hamt_cid_from_stac(
            DCLIMATE_STAC_CATALOG_IPNS, KNOWN_STAC_DATASET_ID
        )
    mock_fetch_ipns.assert_called_once()
    mock_fetch_cid.assert_not_called()


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_no_child_links(mock_fetch_cid, mock_fetch_ipns):
    """Test error when root catalog has no valid child links."""
    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "root",
        "links": [{"rel": "self", "href": "."}],  # No child links
    }
    with pytest.raises(StacCatalogError, match="No valid child collection links found"):
        get_dataset_hamt_cid_from_stac(
            DCLIMATE_STAC_CATALOG_IPNS, KNOWN_STAC_DATASET_ID
        )
    mock_fetch_ipns.assert_called_once()
    mock_fetch_cid.assert_not_called()


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_invalid_child_link_formats(mock_fetch_cid, mock_fetch_ipns):
    """Test skipping various invalid child link formats."""
    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "root",
        "links": [
            {
                "rel": "child",
                "type": "application/json",
                "href": {"/": 123},
            },  # Invalid CID value
            {
                "rel": "child",
                "type": "application/json",
                "href": "/ipfs/legacy_cid_string",
            },  # Legacy string (warning expected)
            {
                "rel": "child",
                "type": "application/json",
                "href": "not_a_dict_or_ipfs_string",
            },  # Invalid format
            {"rel": "child", "type": "application/json"},  # Missing href
            {
                "rel": "child",
                "type": "application/xml",
                "href": {"/": "cid1"},
            },  # Wrong type
            {
                "rel": "item",
                "type": "application/json",
                "href": {"/": "cid2"},
            },  # Wrong rel
        ],
    }
    # Expect DatasetNotFoundError because no *valid* child links lead anywhere
    with pytest.raises(
        DatasetNotFoundError, match=f"Dataset ID '{KNOWN_STAC_DATASET_ID}' not found"
    ):
        get_dataset_hamt_cid_from_stac(
            DCLIMATE_STAC_CATALOG_IPNS, KNOWN_STAC_DATASET_ID
        )
    mock_fetch_ipns.assert_called_once()
    mock_fetch_cid.assert_called_once()  # One legacy string was allowed


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_collection_fetch_error(mock_fetch_cid, mock_fetch_ipns):
    """Test scenario where fetching a collection fails but others might succeed."""
    target_dataset = "dataset-in-good-collection"
    good_collection_cid = "bafyGoodCollection"
    bad_collection_cid = "bafyBadCollection"

    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "root",
        "links": [
            {
                "rel": "child",
                "type": "application/json",
                "href": {"/": bad_collection_cid},
            },
            {
                "rel": "child",
                "type": "application/json",
                "href": {"/": good_collection_cid},
            },
        ],
    }
    # Mock fetch_json_from_cid behavior
    good_item_cid = "bafyGoodItem"
    good_hamt_cid = "bafyGoodHAMT"

    def fetch_cid_side_effect(cid_str, store):
        if cid_str == bad_collection_cid:
            raise IpfsConnectionError("Failed to fetch bad collection")
        elif cid_str == good_collection_cid:
            return {
                "type": "Collection",
                "id": "good-collection",
                "links": [
                    {
                        "rel": "item",
                        "type": "application/json",
                        "href": {"/": good_item_cid},
                    }
                ],
            }
        elif cid_str == good_item_cid:
            return {
                "type": "Feature",
                "id": target_dataset,
                "assets": {"hamt-zarr": {"href": f"/ipfs/{good_hamt_cid}"}},
            }
        else:
            raise ValueError(f"Unexpected CID requested: {cid_str}")

    mock_fetch_cid.side_effect = fetch_cid_side_effect

    # Should succeed by finding the dataset in the good collection
    result_cid = get_dataset_hamt_cid_from_stac(
        DCLIMATE_STAC_CATALOG_IPNS, target_dataset
    )
    assert result_cid == good_hamt_cid
    assert (
        mock_fetch_cid.call_count == 3
    )  # Bad collection fetch fails, good collection + good item fetches succeed


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_invalid_collection_format(mock_fetch_cid, mock_fetch_ipns):
    """Test skipping invalid collection format."""
    collection_cid = "bafyValidCollectionLink"
    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "root",
        "links": [
            {"rel": "child", "type": "application/json", "href": {"/": collection_cid}}
        ],
    }
    mock_fetch_cid.return_value = {"not": "a collection"}  # Invalid format

    with pytest.raises(
        DatasetNotFoundError, match=f"Dataset ID '{KNOWN_STAC_DATASET_ID}' not found"
    ):
        get_dataset_hamt_cid_from_stac(
            DCLIMATE_STAC_CATALOG_IPNS, KNOWN_STAC_DATASET_ID
        )
    mock_fetch_cid.assert_called_once_with(
        collection_cid, ANY
    )  # Assuming default store used internally


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_item_fetch_error(mock_fetch_cid, mock_fetch_ipns):
    """Test scenario where fetching an item fails but the target is in another item."""
    target_dataset = KNOWN_STAC_DATASET_ID
    collection_cid = "bafyCollectionWithItems"
    bad_item_cid = "bafyBadItem"
    good_item_cid = "bafyGoodItem"
    good_hamt_cid = "bafyTargetHAMT"

    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "root",
        "links": [
            {"rel": "child", "type": "application/json", "href": {"/": collection_cid}}
        ],
    }

    def fetch_cid_side_effect(cid_str, store):
        if cid_str == collection_cid:
            return {
                "type": "Collection",
                "id": "collection",
                "links": [
                    {
                        "rel": "item",
                        "type": "application/json",
                        "href": {"/": bad_item_cid},
                    },
                    {
                        "rel": "item",
                        "type": "application/json",
                        "href": {"/": good_item_cid},
                    },
                ],
            }
        elif cid_str == bad_item_cid:
            raise StacCatalogError("Failed to fetch bad item")
        elif cid_str == good_item_cid:
            return {
                "type": "Feature",
                "id": target_dataset,
                "assets": {"hamt-zarr": {"href": f"/ipfs/{good_hamt_cid}"}},
            }
        else:
            raise ValueError(f"Unexpected CID requested: {cid_str}")

    mock_fetch_cid.side_effect = fetch_cid_side_effect

    result_cid = get_dataset_hamt_cid_from_stac(
        DCLIMATE_STAC_CATALOG_IPNS, target_dataset
    )
    assert result_cid == good_hamt_cid
    # Called for collection, bad item (failed), good item (success)
    assert mock_fetch_cid.call_count == 3


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_invalid_item_format(mock_fetch_cid, mock_fetch_ipns):
    """Test skipping invalid item format."""
    # --- FIX: Clear cache at the start of the test ---
    _stac_hamt_cid_cache.clear()

    collection_cid = "bafyCollection"
    item_cid = "bafyItem"
    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "dclimate-stac-catalog",
        "links": [
            {"rel": "child", "type": "application/json", "href": {"/": collection_cid}}
        ],
    }
    mock_fetch_cid.side_effect = [
        # First call returns collection
        {
            "type": "Collection",
            "id": "collection",
            "links": [
                {"rel": "item", "type": "application/json", "href": {"/": item_cid}}
            ],
        },
        # Second call returns invalid item
        {"not": "a feature"},
    ]
    with pytest.raises(
        DatasetNotFoundError, match=f"Dataset ID '{KNOWN_STAC_DATASET_ID}' not found"
    ):
        get_dataset_hamt_cid_from_stac(
            DCLIMATE_STAC_CATALOG_IPNS, KNOWN_STAC_DATASET_ID
        )
    assert mock_fetch_cid.call_count == 2


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_item_missing_hamt_asset(mock_fetch_cid, mock_fetch_ipns):
    """Test error if the found item doesn't have the 'hamt-zarr' asset href."""
    target_dataset = KNOWN_STAC_DATASET_ID
    collection_cid = "bafyCollection"
    item_cid = "bafyItem"
    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "root",
        "links": [
            {"rel": "child", "type": "application/json", "href": {"/": collection_cid}}
        ],
    }
    mock_fetch_cid.side_effect = [
        {
            "type": "Collection",
            "id": "collection",
            "links": [
                {"rel": "item", "type": "application/json", "href": {"/": item_cid}}
            ],
        },
        {
            "type": "Feature",
            "id": target_dataset,
            "assets": {},
        },  # Missing assets.hamt-zarr.href
    ]
    with pytest.raises(
        StacCatalogError, match="missing a valid string 'assets.hamt-zarr.href'"
    ):
        get_dataset_hamt_cid_from_stac(DCLIMATE_STAC_CATALOG_IPNS, target_dataset)
    assert mock_fetch_cid.call_count == 2


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_get_hamt_cid_item_invalid_hamt_asset_href(mock_fetch_cid, mock_fetch_ipns):
    """Test error if the hamt-zarr href is not a valid /ipfs/ string."""
    target_dataset = KNOWN_STAC_DATASET_ID
    collection_cid = "bafyCollection"
    item_cid = "bafyItem"
    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "root",
        "links": [
            {"rel": "child", "type": "application/json", "href": {"/": collection_cid}}
        ],
    }
    mock_fetch_cid.side_effect = [
        {
            "type": "Collection",
            "id": "collection",
            "links": [
                {"rel": "item", "type": "application/json", "href": {"/": item_cid}}
            ],
        },
        {
            "type": "Feature",
            "id": target_dataset,
            "assets": {"hamt-zarr": {"href": "not-an-ipfs-link"}},
        },
    ]
    with pytest.raises(
        StacCatalogError, match="missing a valid string 'assets.hamt-zarr.href'"
    ):
        get_dataset_hamt_cid_from_stac(DCLIMATE_STAC_CATALOG_IPNS, target_dataset)
    assert mock_fetch_cid.call_count == 2


# --- NEW: Tests for _get_dataset_by_ipfs_cid (Unit/Mocked Error Paths) ---


def test_get_dataset_by_ipfs_cid_empty():
    """Test ValueError if ipfs_cid is empty."""
    with pytest.raises(ValueError, match="IPFS CID cannot be empty"):
        _get_dataset_by_ipfs_cid("")


def test_get_dataset_by_ipfs_cid_invalid_format():
    """Test ValueError if ipfs_cid is not a valid CID format."""
    with pytest.raises(ValueError, match="Invalid IPFS CID format"):
        _get_dataset_by_ipfs_cid("this-is-definitely-not-a-cid")


@patch("dclimate_zarr_client.ipfs_retrieval.xr.open_zarr")
@patch("dclimate_zarr_client.ipfs_retrieval.IPFSZarr3")
@patch("dclimate_zarr_client.ipfs_retrieval.HAMT")
@patch("dclimate_zarr_client.ipfs_retrieval._get_ipfs_store")
def test_get_dataset_by_ipfs_cid_zarr_not_found(
    mock_get_store, mock_hamt, mock_ipfs_zarr3, mock_open_zarr
):
    """Test StacCatalogError if Zarr metadata (e.g., .zgroup) is missing."""
    valid_cid = (
        "bafybeicg2rebjoofv4kbyovkw7af3rpiitvnl6i7ckcywaq6za2eflbka4"  # Example CID
    )
    mock_open_zarr.side_effect = FileNotFoundError(
        "[Errno 2] No such file or directory: '.zgroup'"
    )  # Simulate xr.open_zarr error

    with pytest.raises(
        StacCatalogError, match=f"Zarr metadata not found at CID {valid_cid}"
    ):
        _get_dataset_by_ipfs_cid(valid_cid)

    mock_get_store.assert_called_once()
    mock_hamt.assert_called_once()
    mock_ipfs_zarr3.assert_called_once()
    mock_open_zarr.assert_called_once()


@patch("dclimate_zarr_client.ipfs_retrieval.xr.open_zarr")
@patch("dclimate_zarr_client.ipfs_retrieval.IPFSZarr3")
@patch("dclimate_zarr_client.ipfs_retrieval.HAMT")
@patch("dclimate_zarr_client.ipfs_retrieval._get_ipfs_store")
def test_get_dataset_by_ipfs_cid_connection_error(
    mock_get_store, mock_hamt, mock_ipfs_zarr3, mock_open_zarr
):
    """Test IpfsConnectionError if loading data fails due to connection issues."""
    valid_cid = "bafybeicg2rebjoofv4kbyovkw7af3rpiitvnl6i7ckcywaq6za2eflbka4"
    # Simulate error during xr.open_zarr which tries to read from the store
    mock_open_zarr.side_effect = requests.exceptions.ConnectionError(
        "Connection refused during zarr open"
    )

    with pytest.raises(
        IpfsConnectionError,
        match=f"IPFS connection failed while loading dataset from CID {valid_cid}",
    ):
        _get_dataset_by_ipfs_cid(valid_cid)


@patch("dclimate_zarr_client.ipfs_retrieval.xr.open_zarr")
@patch("dclimate_zarr_client.ipfs_retrieval.IPFSZarr3")
@patch("dclimate_zarr_client.ipfs_retrieval.HAMT")
@patch("dclimate_zarr_client.ipfs_retrieval._get_ipfs_store")
def test_get_dataset_by_ipfs_cid_other_runtime_error(
    mock_get_store, mock_hamt, mock_ipfs_zarr3, mock_open_zarr
):
    """Test generic RuntimeError for other failures during loading."""
    valid_cid = "bafybeicg2rebjoofv4kbyovkw7af3rpiitvnl6i7ckcywaq6za2eflbka4"
    mock_open_zarr.side_effect = Exception("Some Zarr parsing error")

    with pytest.raises(
        RuntimeError, match=f"Failed to load Zarr dataset from IPFS CID {valid_cid}"
    ):
        _get_dataset_by_ipfs_cid(valid_cid)


# --- NEW: Tests for list_datasets (Mocked Error Paths) ---


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_list_datasets_root_catalog_fetch_error(mock_fetch_cid, mock_fetch_ipns):
    """Test list_datasets fails if root catalog fetch fails."""
    mock_fetch_ipns.side_effect = StacCatalogError("Cannot fetch root")
    with pytest.raises(
        StacCatalogError,
        match="Failed to fetch or parse root catalog.*for listing datasets",
    ):
        list_datasets(root_catalog_ipns=DCLIMATE_STAC_CATALOG_IPNS)


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_list_datasets_no_collections(mock_fetch_cid, mock_fetch_ipns):
    """Test list_datasets returns empty list if no child collections found."""
    mock_fetch_ipns.return_value = {"type": "Catalog", "id": "root", "links": []}
    result = list_datasets(root_catalog_ipns=DCLIMATE_STAC_CATALOG_IPNS)
    assert result == []
    mock_fetch_cid.assert_not_called()


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_list_datasets_fails_on_bad_collection(mock_fetch_cid, mock_fetch_ipns):
    """Test list_datasets fails if any collection fails to load."""
    good_collection_cid = "bafyGoodCollection"
    bad_collection_cid = "bafyBadCollection"
    good_item_cid = "bafyGoodItem"
    good_item_id = "good-dataset-id"

    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "root",
        "links": [
            {
                "rel": "child",
                "type": "application/json",
                "href": {"/": bad_collection_cid},
            },
            {
                "rel": "child",
                "type": "application/json",
                "href": {"/": good_collection_cid},
            },
        ],
    }

    def fetch_cid_side_effect(cid_str, store):
        if cid_str == bad_collection_cid:
            raise StacCatalogError("Cannot load bad collection")
        elif cid_str == good_collection_cid:
            return {
                "type": "Collection",
                "id": "good",
                "links": [
                    {
                        "rel": "item",
                        "type": "application/json",
                        "href": {"/": good_item_cid},
                    }
                ],
            }
        elif cid_str == good_item_cid:
            return {
                "type": "Feature",
                "id": good_item_id,
                "assets": {"hamt-zarr": {"href": "/ipfs/bafyHAMT"}},
            }
        else:
            raise ValueError("Unknown CID")

    mock_fetch_cid.side_effect = fetch_cid_side_effect

    with pytest.raises(StacCatalogError, match="Cannot load bad collection"):
        list_datasets(root_catalog_ipns=DCLIMATE_STAC_CATALOG_IPNS)


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_cid")
def test_list_datasets_skips_bad_item(mock_fetch_cid, mock_fetch_ipns):
    """Test list_datasets skips items that fail to load or parse."""
    collection_cid = "bafyCollection"
    bad_item_cid = "bafyBadItem"
    good_item_cid = "bafyGoodItem"
    invalid_format_item_cid = "bafyInvalidFormatItem"
    missing_id_item_cid = "bafyMissingIdItem"

    good_item_id = "good-dataset-id"

    mock_fetch_ipns.return_value = {
        "type": "Catalog",
        "id": "root",
        "links": [
            {"rel": "child", "type": "application/json", "href": {"/": collection_cid}}
        ],
    }

    def fetch_cid_side_effect(cid_str, store):
        if cid_str == collection_cid:
            return {
                "type": "Collection",
                "id": "coll",
                "links": [
                    {
                        "rel": "item",
                        "type": "application/json",
                        "href": {"/": bad_item_cid},
                    },
                    {
                        "rel": "item",
                        "type": "application/json",
                        "href": {"/": good_item_cid},
                    },
                    {
                        "rel": "item",
                        "type": "application/json",
                        "href": {"/": invalid_format_item_cid},
                    },
                    {
                        "rel": "item",
                        "type": "application/json",
                        "href": {"/": missing_id_item_cid},
                    },
                ],
            }
        elif cid_str == bad_item_cid:
            raise IpfsConnectionError("Cannot load bad item")
        elif cid_str == good_item_cid:
            return {
                "type": "Feature",
                "id": good_item_id,
                "assets": {"hamt-zarr": {"href": "/ipfs/bafyHAMT"}},
            }
        elif cid_str == invalid_format_item_cid:
            return {"not": "a feature"}
        elif cid_str == missing_id_item_cid:
            return {"type": "Feature", "assets": {}}  # Missing ID
        else:
            raise ValueError("Unknown CID")

    mock_fetch_cid.side_effect = fetch_cid_side_effect

    result = list_datasets(root_catalog_ipns=DCLIMATE_STAC_CATALOG_IPNS)
    assert result == [good_item_id]  # Only the good dataset is listed
    # Called for collection, bad item, good item, invalid item, missing ID item
    assert mock_fetch_cid.call_count == 5


@patch("dclimate_zarr_client.ipfs_retrieval.fetch_json_from_ipns")
def test_list_datasets_uses_default_root_catalog(mock_fetch_ipns):
    """Test that list_datasets uses DCLIMATE_STAC_CATALOG_IPNS if none provided."""
    # Mock fetch_ipns to return a minimal valid catalog
    mock_fetch_ipns.return_value = {"type": "Catalog", "id": "root", "links": []}
    list_datasets()  # Call without root_catalog_ipns argument
    # Assert fetch_json_from_ipns was called with the default IPNS name
    mock_fetch_ipns.assert_called_once_with(
        DCLIMATE_STAC_CATALOG_IPNS, gateway_uri_stem=None
    )


# --- Tests for Legacy/Utility Functions (get_ipns_name_hash, update_cache_if_changed covered) ---
# --- NEW: Tests for other Legacy/Utility Functions (Mocked) ---


def test_get_single_metadata_success(mock_requests_get: MagicMock):
    """Test _get_single_metadata successfully fetches and parses JSON."""
    ipfs_hash = "QmSomeHash"
    expected_metadata = {"prop": "value", "links": []}
    mock_response = MagicMock(spec=requests.Response)
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = expected_metadata
    mock_requests_get.return_value = mock_response

    metadata = _get_single_metadata(ipfs_hash)

    assert metadata == expected_metadata
    expected_url_pattern = rf".*/ipfs/{ipfs_hash}"
    call_args, call_kwargs = mock_requests_get.call_args
    assert re.match(expected_url_pattern, call_args[0]), (
        f"URL {call_args[0]} does not match pattern {expected_url_pattern}"
    )
    mock_response.raise_for_status.assert_called_once()
    mock_response.json.assert_called_once()


def test_get_single_metadata_http_error(mock_requests_get: MagicMock):
    """Test _get_single_metadata raises HTTPError."""
    ipfs_hash = "QmSomeHash"
    mock_response = MagicMock(spec=requests.Response)
    http_error = requests.exceptions.HTTPError("404 Not Found")
    mock_response.raise_for_status.side_effect = http_error
    mock_requests_get.return_value = mock_response

    with pytest.raises(requests.exceptions.HTTPError):
        _get_single_metadata(ipfs_hash)


# --- Tests for get_ipns_name_hash (Legacy/Utility Function) ---
# These tests remain as UNIT tests, mocking requests and file system,
# as they test the specific fallback logic of this function, not STAC traversal.


def test_get_ipns_name_hash_success_from_endpoint():
    """Test successfully getting a hash from the endpoint."""
    endpoint_data = {"cpc-precip-conus": "bafy-real-or-mock-hash", "other": "hash2"}
    with patch("requests.get") as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = endpoint_data
        # Mock file operations to simulate cache update
        with (
            patch("builtins.open", mock_open()) as _,
            patch("os.path.dirname", return_value="/fake/dir"),
            patch(
                "dclimate_zarr_client.ipfs_retrieval.update_cache_if_changed"
            ) as mock_update,
        ):
            ipns_name_hash = ipfs_retrieval.get_ipns_name_hash("cpc-precip-conus")
            assert ipns_name_hash == "bafy-real-or-mock-hash"
            mock_update.assert_called_once_with(
                endpoint_data
            )  # Verify cache update logic is called


# Success from Local Cache
def test_get_ipns_name_hash_fallback_success():
    """
    Test that when the remote CID endpoint is unreachable OR the key isn't found,
    we successfully fall back to the local cids_cache.json.
    """
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated Error")
    ):
        with patch("os.path.exists", return_value=True):
            fake_json = '{"cpc-precip-conus":"bafkreihashfromlocalfile"}'
            # Mock open within the context of the function call
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=fake_json),
                create=True,
            ):
                ipns_name_hash = get_ipns_name_hash("cpc-precip-conus")
                assert ipns_name_hash == "bafkreihashfromlocalfile"


# Failure When Local Cache Does Not Exist
def test_get_ipns_name_hash_fallback_no_file():
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated Error")
    ):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(DatasetNotFoundError):
                get_ipns_name_hash("some-nonexistent-key")


# Failure When Local Cache Exists But Missing Key
def test_get_ipns_name_hash_fallback_key_not_found_in_file():
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated Error")
    ):
        with patch("os.path.exists", return_value=True):
            fake_json = '{"some-other-key":"bafkreihashfromlocalfile"}'
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=fake_json),
                create=True,
            ):
                with pytest.raises(DatasetNotFoundError, match="Invalid dataset name"):
                    get_ipns_name_hash("cpc-precip-conus")


def test_get_ipns_name_hash_endpoint_missing_key():
    endpoint_data = {"some-other-key": "bafy-other"}
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = endpoint_data
        mock_get.return_value.raise_for_status.return_value = None
        # Mock local file to also not have the key
        with patch("os.path.exists", return_value=True):
            fake_json = '{"still-some-other-key":"bafkrei-another"}'
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=fake_json),
                create=True,
            ):
                # Mock cache update to avoid file write attempt during assert
                with patch(
                    "dclimate_zarr_client.ipfs_retrieval.update_cache_if_changed"
                ):
                    with pytest.raises(
                        DatasetNotFoundError, match="Invalid dataset name"
                    ):
                        get_ipns_name_hash("cpc-precip-conus")


def test_get_ipns_name_hash_endpoint_malformed_json():
    with patch("requests.get") as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.side_effect = json.JSONDecodeError(
            "Expecting value", "doc", 0
        )
        # Provide valid local cache
        with patch("os.path.exists", return_value=True):
            fake_json = '{"cpc-precip-conus":"bafkreihashfromlocalfile"}'
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=fake_json),
                create=True,
            ):
                ipns_name_hash = get_ipns_name_hash("cpc-precip-conus")
                assert ipns_name_hash == "bafkreihashfromlocalfile"


def test_get_ipns_name_hash_local_cache_malformed_json():
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated error")
    ):
        with patch("os.path.exists", return_value=True):
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data="INVALID JSON!!"),
                create=True,
            ):
                # Expect DatasetNotFoundError because parsing fails during fallback
                with pytest.raises(DatasetNotFoundError):
                    get_ipns_name_hash("cpc-precip-conus")


def test_get_ipns_name_hash_local_cache_empty():
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated error")
    ):
        with patch("os.path.exists", return_value=True):
            with patch(
                "dclimate_zarr_client.ipfs_retrieval.open",
                mock_open(read_data=""),
                create=True,
            ):
                # Expect DatasetNotFoundError because parsing fails during fallback
                with pytest.raises(DatasetNotFoundError):
                    get_ipns_name_hash("cpc-precip-conus")


# --- Functional Test for list_datasets (using STAC) ---


@pytest.mark.ipfs  # Mark as IPFS dependent
def test_list_datasets_functional():
    """Test listing datasets functionally by traversing the STAC catalog."""
    try:
        # Call the actual function pointing to the root catalog
        dataset_list = ipfs_retrieval.list_datasets(
            root_catalog_ipns=DCLIMATE_STAC_CATALOG_IPNS
        )
        assert isinstance(dataset_list, list)
        assert len(dataset_list) > 6  # Expect a reasonable number of datasets
        assert all(isinstance(item, str) for item in dataset_list)
        # Check for known datasets that should be present
        assert KNOWN_STAC_DATASET_ID in dataset_list
        assert KNOWN_STAC_DATASET_ID_2 in dataset_list
        print(f"Functional list_datasets found {len(dataset_list)} datasets.")

    except (IpfsConnectionError, StacCatalogError) as e:
        pytest.fail(f"Functional list_datasets failed: {e}")
    except Exception as e:
        pytest.fail(f"Functional list_datasets failed with unexpected error: {e}")


# def test_list_datasets_endpoint_malformed_json():
#     """
#     Test that if the endpoint returns invalid JSON,
#     list_datasets() attempts fallback.
#     """
#     with patch("requests.get") as mock_requests_get:
#         mock_requests_get.return_value.text = "INVALID JSON!!"
#         mock_requests_get.return_value.raise_for_status.return_value = None
#         mock_requests_get.return_value.json.side_effect = json.JSONDecodeError(
#             "Expecting value", "doc", 0
#         )

#         # Provide valid local cache
#         with patch("os.path.exists", return_value=True):
#             fake_json = '{"ds1":"bafy1","ds2":"bafy2"}'
#             with patch("builtins.open", mock_open(read_data=fake_json)):
#                 datasets = list_datasets()
#                 assert datasets == ["ds1", "ds2"]


# def test_list_datasets_no_fallback_file():
#     """
#     Test that if the endpoint fails AND no local cache file is found,
#     list_datasets() raises RuntimeError.
#     """
#     with patch("requests.get") as mock_requests_get:
#         mock_requests_get.side_effect = requests.RequestException("Simulated error")

#         with patch("os.path.exists", return_value=False):
#             with pytest.raises(RuntimeError) as exc:
#                 list_datasets()
#             assert "Failed to retrieve dataset list" in str(exc.value)


# def test_list_datasets_local_cache_malformed_json():
#     """
#     Test that if the endpoint fails AND local cache file has malformed JSON,
#     list_datasets() raises RuntimeError because it cannot parse the local file.
#     """
#     with patch("requests.get") as mock_requests_get:
#         mock_requests_get.side_effect = requests.RequestException("Simulated error")

#         with patch("os.path.exists", return_value=True):
#             # Malformed JSON
#             with patch("builtins.open", mock_open(read_data="INVALID JSON")):
#                 with pytest.raises(RuntimeError) as exc:
#                     list_datasets()
#                 assert "Failed to retrieve dataset list" in str(exc.value)


# def test_geo_temporal_query():
#     ds_bytes = geo_temporal_query(
#         "cpc-precip-conus",
#         point_kwargs={"latitude": 40.875, "longitude": -104.875},
#         time_range=[datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 1)],
#     )
#     assert ds_bytes["data"][0] == 0.7991394996643066


# def test_get_dataset_by_ipns_hash_no_as_of():
#     """
#     Test that `get_dataset_by_ipns_hash` can select without a specified as_of time
#     """
#     ds = ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH)
#     assert ds.attrs["order_created"] == 4


# def test_get_dataset_by_ipns_hash_with_as_of():
#     """
#     Test that `get_dataset_by_ipns_hash` can select by specified as_of times
#     """
#     creation_times = [
#         datetime.datetime(2022, 7, 26, 19, 17, 55),
#         datetime.datetime(2022, 7, 26, 19, 19, 41),
#         datetime.datetime(2022, 7, 26, 19, 20, 45),
#         datetime.datetime(2022, 7, 26, 19, 22, 46),
#     ]
#     for i, time in enumerate(creation_times):
#         ds = ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH, as_of=time)
#         assert ds.attrs["order_created"] == i + 1


# def test_get_dataset_by_ipns_hash_with_bad_as_of():
#     """
#     Test that `get_dataset_by_ipns_hash` fails when provided an invalid `as_of` parameter
#     """
#     creation_time = datetime.datetime(2022, 7, 26, 19, 17, 53)
#     with pytest.raises(NoMetadataFoundError):
#         ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH, as_of=creation_time)


# --- Tests for update_cache_if_changed (Utility Function) ---
# These remain unit tests using mocks for file I/O.


# --- Helper Functions for Mocking --- #


def mock_exists_true(*args, **kwargs) -> bool:
    return True


def mock_exists_false(*args, **kwargs) -> bool:
    return False


# Use monkeypatch fixture for modifying builtins like open
def test_update_cache_no_update(monkeypatch: MonkeyPatch):
    cached_data = {"dataset": "hash1"}
    new_data = {"dataset": "hash1"}
    file_path = get_cache_path()

    # Mock os.path.exists needed if called before open
    monkeypatch.setattr("os.path.exists", mock_exists_true)

    m = mock_open(read_data=json.dumps(cached_data))
    monkeypatch.setattr("builtins.open", m)  # Correct: Patch built-in open

    update_cache_if_changed(new_data)

    # Assert open was called once for reading, and never for writing
    m.assert_called_once_with(file_path, "r")


def test_update_cache_update(monkeypatch: MonkeyPatch):
    cached_data = {"dataset": "hash1"}
    new_data = {"dataset": "hash2"}
    file_path = get_cache_path()

    # Mock os.path.exists needed if called before open
    monkeypatch.setattr("os.path.exists", mock_exists_true)

    m = mock_open(read_data=json.dumps(cached_data))
    # Patch open in the correct namespace where it's called
    monkeypatch.setattr("builtins.open", m)  # Correct: Patch built-in open

    update_cache_if_changed(new_data)

    # Assert open was called twice: once for read, once for write.
    assert m.call_count == 2
    calls = m.call_args_list
    assert calls[0].args == (file_path, "r")
    assert calls[1].args == (file_path, "w")
    # Check the content written (optional, but good practice)
    # Ensure the mock handle captures the write
    # Note: mock_open's write checking can be tricky.
    # A simpler check is often sufficient unless exact content is critical.
    # handle = m()
    # handle.write.assert_called_once_with(json.dumps(new_data))


def test_update_cache_file_not_found(monkeypatch: MonkeyPatch):
    new_data = {"dataset": "hash2"}
    file_path = get_cache_path()

    # No need to mock os.path.exists as the function doesn't use it

    # Mock 'open' to raise FileNotFoundError on first call (read), succeed on second (write)
    mock_write_handle = mock_open().return_value
    m = mock_open()
    # First call (open "r") raises FileNotFoundError
    # Second call (open "w") returns the mock handle
    m.side_effect = [FileNotFoundError, mock_write_handle]
    # Patch open in the correct namespace
    monkeypatch.setattr("builtins.open", m)  # Correct: Patch built-in open

    update_cache_if_changed(new_data)

    # Assert open was called twice: read attempt (failed), write attempt (succeeded)
    assert m.call_count == 2
    calls = m.call_args_list
    assert calls[0].args == (file_path, "r")
    assert calls[1].args == (file_path, "w")
    # Optionally check write content if needed:
    # handle = calls[1]._extract_mock_return_value() # Get the handle returned by the 2nd call
    # handle.write.assert_called_once_with(json.dumps(new_data))


def test_update_cache_decode_error(monkeypatch: MonkeyPatch):
    """Test when the existing cache file has invalid JSON."""
    new_data = {"dataset": "hash2"}
    file_path = get_cache_path()

    # Mock os.path.exists needed if called before open
    monkeypatch.setattr("os.path.exists", mock_exists_true)

    # Create separate mock handles for read and write attempts
    # The read handle will simulate having invalid data
    read_handle = mock_open(read_data="invalid json").return_value
    write_handle = mock_open().return_value  # Mock handle for the write call

    m = mock_open()
    # Define side effect: return read_handle on first call, write_handle on second
    m.side_effect = [read_handle, write_handle]
    # Patch open in the correct namespace
    monkeypatch.setattr("builtins.open", m)  # Correct: Patch built-in open

    # Mock json.load to raise error when the read_handle is passed to it
    # Patch it in the correct namespace where it's used
    mock_json_load = patch(
        "json.load",  # Patching built-in json directly
        side_effect=json.JSONDecodeError("err", "doc", 0),
    ).start()  # No need for .start()/.stop() if using 'with patch(...)'

    # Use 'with patch' for cleaner setup/teardown
    # with patch("json.load", side_effect=json.JSONDecodeError("err", "doc", 0)):
    update_cache_if_changed(new_data)

    # Assert open was called twice: read attempt (failed decode), write attempt
    assert m.call_count == 2
    calls = m.call_args_list
    assert calls[0].args == (file_path, "r")  # Read attempt
    assert calls[1].args == (file_path, "w")  # Write attempt

    # Clean up the patch for json.load if using start/stop
    mock_json_load.stop()


# --- Test Legacy get_ipns_name_hash Errors ---
# (Ensure these cover JSONDecodeError during fallback read)


def test_get_ipns_name_hash_local_cache_malformed_json_during_fallback():
    """Test DatasetNotFoundError when local cache read fails during fallback"""
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated error")
    ):
        with patch("os.path.exists", return_value=True):
            # Mock update_cache_if_changed to avoid file writes during test setup if called early
            with patch("dclimate_zarr_client.ipfs_retrieval.update_cache_if_changed"):
                # Mock open specifically within the fallback block
                with patch(
                    "dclimate_zarr_client.ipfs_retrieval.open",
                    mock_open(read_data="INVALID JSON!!"),
                    create=True,  # Allow create=True if needed by mock_open internals
                ) as mock_file_open:
                    # Mock json.load raising the error when called by get_ipns_name_hash
                    with patch(
                        "json.load", side_effect=json.JSONDecodeError("err", "doc", 0)
                    ):
                        with pytest.raises(
                            DatasetNotFoundError, match="Invalid dataset name"
                        ):
                            get_ipns_name_hash("cpc-precip-conus")
                    # Assert the mocked open was called for reading the cache
                    mock_file_open.assert_called_once_with(get_cache_path(), "r")


def test_get_ipns_name_hash_local_cache_empty_during_fallback():
    """Test DatasetNotFoundError when local cache is empty during fallback"""
    with patch(
        "requests.get", side_effect=requests.RequestException("Simulated error")
    ):
        with patch("os.path.exists", return_value=True):
            with patch("dclimate_zarr_client.ipfs_retrieval.update_cache_if_changed"):
                with patch(
                    "dclimate_zarr_client.ipfs_retrieval.open",
                    mock_open(read_data=""),  # Empty file
                    create=True,
                ) as mock_file_open:
                    # json.load will raise JSONDecodeError on empty string
                    with patch(
                        "json.load",
                        side_effect=json.JSONDecodeError("Expecting value", "", 0),
                    ):
                        with pytest.raises(
                            DatasetNotFoundError, match="Invalid dataset name"
                        ):
                            get_ipns_name_hash("cpc-precip-conus")
                    mock_file_open.assert_called_once_with(get_cache_path(), "r")
