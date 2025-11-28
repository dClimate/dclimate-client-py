"""
Comprehensive tests for STAC catalog integration module.

Tests all functions and classes in stac_catalog.py using real data from the dClimate IPFS gateway.
No mocking is used - all tests interact with actual STAC catalog data.
"""

import pytest
import pystac
from dclimate_client_py import stac_catalog


# Mark all tests in this module to use the IPFS connection check
pytestmark = pytest.mark.usefixtures("check_ipfs_connection")


class TestGetRootCatalogCid:
    """Test the get_root_catalog_cid function."""

    def test_get_root_catalog_cid_returns_string(self):
        """Test that get_root_catalog_cid returns a non-empty string CID."""
        cid = stac_catalog.get_root_catalog_cid()

        assert isinstance(cid, str)
        assert len(cid) > 0
        # Basic validation: CIDs typically start with 'Qm' or 'bafy'
        assert cid.startswith(('Qm', 'bafy', 'bafk', 'bafz'))

    def test_get_root_catalog_cid_consistent(self):
        """Test that multiple calls return consistent CID format."""
        cid1 = stac_catalog.get_root_catalog_cid()
        cid2 = stac_catalog.get_root_catalog_cid()

        # Both should be valid CIDs
        assert isinstance(cid1, str)
        assert isinstance(cid2, str)
        # The CID might change over time but format should be consistent
        assert len(cid1) > 0
        assert len(cid2) > 0


class TestIPFSStacIO:
    """Test the IPFSStacIO custom StacIO implementation."""

    def test_initialization(self):
        """Test IPFSStacIO initialization with gateway URL."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        stac_io = stac_catalog.IPFSStacIO(gateway_url)

        assert stac_io.gateway_url == gateway_url
        assert isinstance(stac_io, pystac.StacIO)

    def test_initialization_strips_trailing_slash(self):
        """Test that trailing slash is stripped from gateway URL."""
        gateway_url_with_slash = "https://ipfs-gateway.dclimate.net/"
        stac_io = stac_catalog.IPFSStacIO(gateway_url_with_slash)

        assert stac_io.gateway_url == "https://ipfs-gateway.dclimate.net"
        assert not stac_io.gateway_url.endswith('/')

    def test_read_text_with_ipfs_uri(self):
        """Test reading content from ipfs:// URI via gateway."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        stac_io = stac_catalog.IPFSStacIO(gateway_url)

        # Get a real CID from the catalog
        root_cid = stac_catalog.get_root_catalog_cid()
        ipfs_uri = f"ipfs://{root_cid}"

        # Read the content
        content = stac_io.read_text(ipfs_uri)

        assert isinstance(content, str)
        assert len(content) > 0
        # Should be valid JSON for a STAC catalog
        import json
        catalog_data = json.loads(content)
        assert "type" in catalog_data
        assert catalog_data["type"] in ["Catalog", "Collection"]

    def test_read_text_handles_ipfs_uri_correctly(self):
        """Test that IPFS URIs are properly transformed to gateway HTTP URLs."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        stac_io = stac_catalog.IPFSStacIO(gateway_url)

        # Get a real CID to test with
        root_cid = stac_catalog.get_root_catalog_cid()

        # Test that ipfs:// URI is handled
        ipfs_uri = f"ipfs://{root_cid}"
        content = stac_io.read_text(ipfs_uri)

        # Verify content is valid
        assert isinstance(content, str)
        assert len(content) > 0

        # The content should be JSON for a STAC catalog
        import json
        data = json.loads(content)
        assert "type" in data

    def test_read_text_multiple_cids(self):
        """Test reading from multiple different CIDs."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        stac_io = stac_catalog.IPFSStacIO(gateway_url)

        # Get root CID and load catalog
        root_cid = stac_catalog.get_root_catalog_cid()
        ipfs_uri = f"ipfs://{root_cid}"

        # First read
        content1 = stac_io.read_text(ipfs_uri)
        assert isinstance(content1, str)
        assert len(content1) > 0

        # Second read (should work the same)
        content2 = stac_io.read_text(ipfs_uri)
        assert isinstance(content2, str)
        assert content1 == content2

    def test_write_text_raises_not_implemented(self):
        """Test that write_text raises NotImplementedError."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        stac_io = stac_catalog.IPFSStacIO(gateway_url)

        with pytest.raises(NotImplementedError) as exc_info:
            stac_io.write_text("ipfs://somecid", "some content")

        assert "not supported" in str(exc_info.value).lower()


class TestLoadStacCatalog:
    """Test the load_stac_catalog function."""

    def test_load_catalog_with_auto_cid(self):
        """Test loading catalog with automatically fetched CID."""
        gateway_url = "https://ipfs-gateway.dclimate.net"

        catalog = stac_catalog.load_stac_catalog(gateway_url)

        assert isinstance(catalog, pystac.Catalog)
        assert catalog.id is not None
        assert catalog.description is not None

    def test_load_catalog_with_explicit_cid(self):
        """Test loading catalog with explicitly provided CID."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        root_cid = stac_catalog.get_root_catalog_cid()

        catalog = stac_catalog.load_stac_catalog(gateway_url, root_cid=root_cid)

        assert isinstance(catalog, pystac.Catalog)
        assert catalog.id is not None

    def test_loaded_catalog_has_children(self):
        """Test that loaded catalog has child links (collections)."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        catalog = stac_catalog.load_stac_catalog(gateway_url)

        child_links = list(catalog.get_child_links())
        assert len(child_links) > 0

        # At least one child should have dclimate:id
        has_dclimate_id = any(
            link.extra_fields.get("dclimate:id") for link in child_links
        )
        assert has_dclimate_id

    def test_catalog_collections_are_accessible(self):
        """Test that collections in the catalog can be resolved and accessed."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        catalog = stac_catalog.load_stac_catalog(gateway_url)

        # Get first child link
        child_links = list(catalog.get_child_links())
        assert len(child_links) > 0

        first_link = child_links[0]
        # Resolve the link
        collection = first_link.resolve_stac_object(root=catalog).target

        assert collection is not None
        # Should be a Collection or Catalog
        assert isinstance(collection, (pystac.Collection, pystac.Catalog))


class TestResolveDatasetCidFromStac:
    """Test the resolve_dataset_cid_from_stac function."""

    @pytest.fixture
    def loaded_catalog(self):
        """Fixture providing a loaded STAC catalog."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        return stac_catalog.load_stac_catalog(gateway_url)

    def test_resolve_dataset_cid_basic(self, loaded_catalog):
        """Test resolving a dataset CID from the catalog."""
        # First, find an available collection and dataset
        datasets = stac_catalog.list_available_datasets(loaded_catalog)

        # Get the first collection with datasets
        collection_id = None
        dataset_type = None
        for coll_id, info in datasets.items():
            if info["types"] and len(info["types"]) > 0:
                collection_id = coll_id
                dataset_type = info["types"][0]
                break

        # Skip if no datasets available
        if collection_id is None or dataset_type is None:
            pytest.skip("No datasets available in catalog")

        # Try to resolve the CID
        cid = stac_catalog.resolve_dataset_cid_from_stac(
            loaded_catalog,
            collection=collection_id,
            dataset=dataset_type
        )

        assert isinstance(cid, str)
        assert len(cid) > 0
        # Should not have ipfs:// prefix
        assert not cid.startswith("ipfs://")
        # Should be a valid CID format
        assert cid.startswith(('Qm', 'bafy', 'bafk', 'bafz'))

    def test_resolve_dataset_cid_with_variant(self, loaded_catalog):
        """Test resolving a dataset CID with a specific variant."""
        child_links = list(loaded_catalog.get_child_links())
        if not child_links:
            pytest.skip("No organizations/collections in catalog")

        item_with_variant = None
        organization_id = None
        collection_id = None

        # Walk org -> collection until we find an item with a variant segment
        for org_link in child_links:
            organization_id = org_link.extra_fields.get("dclimate:id")
            org_catalog = org_link.resolve_stac_object(root=loaded_catalog).target
            collection_links = list(org_catalog.get_child_links())
            if not collection_links:
                continue

            for col_link in collection_links:
                collection_id = col_link.extra_fields.get("dclimate:id")
                collection_obj = col_link.resolve_stac_object(root=org_catalog).target
                for item in collection_obj.get_items():
                    parts = item.id.split("-")
                    if len(parts) >= 3:  # collection-dataset-variant
                        item_with_variant = item
                        break
                if item_with_variant:
                    break
            if item_with_variant:
                break

        if not item_with_variant or not collection_id:
            pytest.skip("No items with variants found")

        parts = item_with_variant.id.split("-")
        item_collection = parts[0]
        item_dataset = parts[1]
        item_variant = parts[2]

        cid = stac_catalog.resolve_dataset_cid_from_stac(
            loaded_catalog,
            collection=item_collection,
            dataset=item_dataset,
            variant=item_variant,
            organization=organization_id,
        )

        assert isinstance(cid, str)
        assert len(cid) > 0
        assert not cid.startswith("ipfs://")

    def test_resolve_dataset_cid_invalid_collection(self, loaded_catalog):
        """Test that invalid collection raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            stac_catalog.resolve_dataset_cid_from_stac(
                loaded_catalog,
                collection="nonexistent_collection_xyz",
                dataset="some_dataset"
            )

        assert "not found" in str(exc_info.value).lower()
        assert "collection" in str(exc_info.value).lower()

    def test_resolve_dataset_cid_invalid_dataset(self, loaded_catalog):
        """Test that invalid dataset raises ValueError."""
        # Get a valid collection
        datasets = stac_catalog.list_available_datasets(loaded_catalog)
        if not datasets:
            pytest.skip("No datasets available")

        collection_id = list(datasets.keys())[0]

        with pytest.raises(ValueError) as exc_info:
            stac_catalog.resolve_dataset_cid_from_stac(
                loaded_catalog,
                collection=collection_id,
                dataset="nonexistent_dataset_xyz"
            )

        assert "not found" in str(exc_info.value).lower()
        assert "dataset" in str(exc_info.value).lower()

    def test_resolve_dataset_cid_invalid_variant(self, loaded_catalog):
        """Test that invalid variant raises ValueError."""
        # Get a valid collection and dataset
        datasets = stac_catalog.list_available_datasets(loaded_catalog)

        collection_id = None
        dataset_type = None
        for coll_id, info in datasets.items():
            if info["types"] and len(info["types"]) > 0:
                collection_id = coll_id
                dataset_type = info["types"][0]
                break

        if collection_id is None or dataset_type is None:
            pytest.skip("No datasets available")

        with pytest.raises(ValueError) as exc_info:
            stac_catalog.resolve_dataset_cid_from_stac(
                loaded_catalog,
                collection=collection_id,
                dataset=dataset_type,
                variant="nonexistent_variant_xyz"
            )

        assert "not found" in str(exc_info.value).lower()
        assert "variant" in str(exc_info.value).lower()


class TestListAvailableDatasets:
    """Test the list_available_datasets function."""

    @pytest.fixture
    def loaded_catalog(self):
        """Fixture providing a loaded STAC catalog."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        return stac_catalog.load_stac_catalog(gateway_url)

    def test_list_datasets_returns_dict(self, loaded_catalog):
        """Test that list_available_datasets returns a dictionary."""
        datasets = stac_catalog.list_available_datasets(loaded_catalog)

        assert isinstance(datasets, dict)
        assert len(datasets) > 0

    def test_list_datasets_structure(self, loaded_catalog):
        """Test the structure of returned datasets dictionary."""
        datasets = stac_catalog.list_available_datasets(loaded_catalog)

        # Check each entry has required fields
        for collection_id, info in datasets.items():
            assert isinstance(collection_id, str)
            assert isinstance(info, dict)

            # Check required fields
            assert "id" in info
            assert "title" in info
            assert "types" in info
            assert "organization" in info

            # Validate field types
            assert isinstance(info["id"], str)
            assert isinstance(info["title"], str)
            assert isinstance(info["types"], list)

            # ID should match the key
            assert info["id"] == collection_id

    def test_list_datasets_has_types(self, loaded_catalog):
        """Test that at least some collections have dataset types."""
        datasets = stac_catalog.list_available_datasets(loaded_catalog)

        # At least one collection should have types
        has_types = any(len(info["types"]) > 0 for info in datasets.values())
        assert has_types

    def test_list_datasets_types_are_strings(self, loaded_catalog):
        """Test that dataset types are strings."""
        datasets = stac_catalog.list_available_datasets(loaded_catalog)

        for info in datasets.values():
            for dataset_type in info["types"]:
                assert isinstance(dataset_type, str)
                assert len(dataset_type) > 0

    def test_list_datasets_includes_dclimate_collections(self, loaded_catalog):
        """Test that the results include collections with dclimate:id."""
        datasets = stac_catalog.list_available_datasets(loaded_catalog)

        child_links = list(loaded_catalog.get_child_links())
        collection_ids = []
        for link in child_links:
            link_id = link.extra_fields.get("dclimate:id")
            if not link_id:
                continue
            target = link.resolve_stac_object(root=loaded_catalog).target
            if isinstance(target, pystac.Collection):
                collection_ids.append(link_id)
            else:
                for col_link in target.get_child_links():
                    col_id = col_link.extra_fields.get("dclimate:id")
                    if col_id:
                        collection_ids.append(col_id)

        for collection_id in collection_ids:
            assert collection_id in datasets

    def test_list_datasets_excludes_links_without_dclimate_id(self, loaded_catalog):
        """Test that links without dclimate:id are excluded."""
        datasets = stac_catalog.list_available_datasets(loaded_catalog)

        # All returned collection IDs should be non-empty strings
        for collection_id in datasets.keys():
            assert collection_id is not None
            assert isinstance(collection_id, str)
            assert len(collection_id) > 0


class TestIntegrationEndToEnd:
    """Integration tests that exercise the full workflow."""

    def test_full_workflow_load_and_resolve(self):
        """Test the complete workflow: get CID, load catalog, resolve dataset."""
        # Step 1: Get root catalog CID
        root_cid = stac_catalog.get_root_catalog_cid()
        assert isinstance(root_cid, str)

        # Step 2: Load catalog
        gateway_url = "https://ipfs-gateway.dclimate.net"
        catalog = stac_catalog.load_stac_catalog(gateway_url, root_cid=root_cid)
        assert isinstance(catalog, pystac.Catalog)

        # Step 3: List available datasets
        datasets = stac_catalog.list_available_datasets(catalog)
        assert len(datasets) > 0

        # Step 4: Resolve a dataset CID
        collection_id = None
        dataset_type = None
        for coll_id, info in datasets.items():
            if info["types"] and len(info["types"]) > 0:
                collection_id = coll_id
                dataset_type = info["types"][0]
                break

        if collection_id and dataset_type:
            cid = stac_catalog.resolve_dataset_cid_from_stac(
                catalog,
                collection=collection_id,
                dataset=dataset_type
            )
            assert isinstance(cid, str)
            assert len(cid) > 0

    def test_multiple_catalog_loads_work(self):
        """Test that multiple catalog loads don't interfere with each other."""
        gateway_url = "https://ipfs-gateway.dclimate.net"

        # Load catalog twice
        catalog1 = stac_catalog.load_stac_catalog(gateway_url)
        catalog2 = stac_catalog.load_stac_catalog(gateway_url)

        # Both should be valid
        assert isinstance(catalog1, pystac.Catalog)
        assert isinstance(catalog2, pystac.Catalog)

        # Both should have children
        assert len(list(catalog1.get_child_links())) > 0
        assert len(list(catalog2.get_child_links())) > 0

    def test_catalog_navigation(self):
        """Test navigating through catalog hierarchy."""
        gateway_url = "https://ipfs-gateway.dclimate.net"
        catalog = stac_catalog.load_stac_catalog(gateway_url)

        # Get child links
        child_links = list(catalog.get_child_links())
        assert len(child_links) > 0

        # Resolve first organization then first collection
        first_org = child_links[0].resolve_stac_object(root=catalog).target
        collection_links = list(first_org.get_child_links())
        assert len(collection_links) > 0

        collection = collection_links[0].resolve_stac_object(root=first_org).target

        items = list(collection.get_items())
        if items:
            first_item = items[0]
            assert isinstance(first_item, pystac.Item)
            assert len(first_item.assets) > 0
