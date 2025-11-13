"""
Tests for dataset catalog functionality.

Tests the new load_dataset() function and related catalog features.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock

from dclimate_client_py import (
    list_dataset_catalog,
    DatasetCatalog,
    GeotemporalData,
    dClimateClient,
)
from dclimate_client_py.dclimate_zarr_errors import (
    DatasetNotFoundError,
    CollectionNotFoundError,
    VariantNotFoundError,
    InvalidSelectionError,
)
from dclimate_client_py.datasets import (
    normalize_key,
    find_collection_by_name,
    find_dataset_by_name,
    find_variant_by_name,
    get_concatenable_variants,
    find_collection_for_dataset,
    resolve_dataset_source,
    DATASET_CATALOG_INTERNAL,
)


# --- Test Catalog Utilities ---


def test_normalize_key():
    """Test key normalization for case-insensitive matching."""
    assert normalize_key("Dataset-Name") == "dataset_name"
    assert normalize_key("  DATASET NAME  ") == "dataset_name"
    assert normalize_key("dataset_name") == "dataset_name"
    assert normalize_key("Dataset Name") == "dataset_name"
    assert normalize_key("") == ""


def test_list_dataset_catalog():
    """Test listing the dataset catalog."""
    catalog = list_dataset_catalog()

    # Should return a list
    assert isinstance(catalog, list)

    # Should have at least one collection
    assert len(catalog) > 0

    # First collection should have expected structure
    first_collection = catalog[0]
    assert "collection" in first_collection
    assert "datasets" in first_collection
    assert isinstance(first_collection["datasets"], list)

    # First dataset should have expected structure
    if len(first_collection["datasets"]) > 0:
        first_dataset = first_collection["datasets"][0]
        assert "dataset" in first_dataset
        assert "variants" in first_dataset
        assert isinstance(first_dataset["variants"], list)


def test_find_collection_by_name():
    """Test finding a collection by name."""
    catalog = DATASET_CATALOG_INTERNAL

    # Should find existing collection (case-insensitive)
    coll = find_collection_by_name(catalog, "era5")
    assert coll is not None
    assert coll["collection"] == "era5"

    coll = find_collection_by_name(catalog, "ERA5")
    assert coll is not None

    coll = find_collection_by_name(catalog, "Era5")
    assert coll is not None

    # Should return None for non-existent collection
    coll = find_collection_by_name(catalog, "nonexistent")
    assert coll is None


def test_find_dataset_by_name():
    """Test finding a dataset within a collection."""
    catalog = DATASET_CATALOG_INTERNAL
    era5_collection = find_collection_by_name(catalog, "era5")

    # Should find existing dataset
    ds = find_dataset_by_name(era5_collection, "2m_temperature")
    assert ds is not None
    assert ds["dataset"] == "2m_temperature"

    # Case-insensitive
    ds = find_dataset_by_name(era5_collection, "2m_tEmperature")
    assert ds is not None

    # Should return None for non-existent dataset
    ds = find_dataset_by_name(era5_collection, "nonexistent")
    assert ds is None


def test_find_variant_by_name():
    """Test finding a variant within a dataset."""
    catalog = DATASET_CATALOG_INTERNAL
    era5_collection = find_collection_by_name(catalog, "era5")
    temp2m_dataset = find_dataset_by_name(era5_collection, "2m_temperature")

    # Should find existing variant
    variant = find_variant_by_name(temp2m_dataset, "finalized")
    assert variant is not None
    assert variant["variant"] == "finalized"

    # Case-insensitive
    variant = find_variant_by_name(temp2m_dataset, "FINALIZED")
    assert variant is not None

    # Should return None for non-existent variant
    variant = find_variant_by_name(temp2m_dataset, "nonexistent")
    assert variant is None


# def test_get_concatenable_variants():
#     """Test getting concatenable variants sorted by priority."""
#     catalog = DATASET_CATALOG_INTERNAL
#     era5_collection = find_collection_by_name(catalog, "era5")
#     temp2m_dataset = find_dataset_by_name(era5_collection, "2m_temperature")

#     concatenable = get_concatenable_variants(temp2m_dataset)

#     # Should have concatenable variants
#     assert len(concatenable) > 0

#     # Should be sorted by concat_priority
#     priorities = [v["concat_priority"] for v in concatenable]
#     assert priorities == sorted(priorities)

#     # First should have lower priority number than second
#     if len(concatenable) >= 2:
#         assert concatenable[0]["concat_priority"] < concatenable[1]["concat_priority"]


def test_find_collection_for_dataset():
    """Test auto-detecting collection from dataset name."""
    catalog = DATASET_CATALOG_INTERNAL

    # Should find collection for temp2m
    coll = find_collection_for_dataset(catalog, "2m_temperature")
    assert coll is not None
    assert coll["collection"] == "era5"

    # Should return None for non-existent dataset
    coll = find_collection_for_dataset(catalog, "nonexistent_dataset")
    assert coll is None


def test_resolve_dataset_source():
    """Test resolving dataset to source CID."""
    # Test with explicit collection and variant
    resolved = resolve_dataset_source(
        dataset_name="2m_temperature",
        collection_name="era5",
        variant_name="finalized"
    )

    assert resolved["collection"] == "era5"
    assert resolved["dataset"] == "2m_temperature"
    assert resolved["variant"] == "finalized"
    assert resolved["cid"] is not None
    assert "era5/2m_temperature/finalized" in resolved["slug"]

    # Test with auto-detected collection
    resolved = resolve_dataset_source(
        dataset_name="2m_temperature",
        variant_name="finalized"
    )

    assert resolved["collection"] == "era5"
    assert resolved["cid"] is not None


def test_resolve_dataset_source_errors():
    """Test error cases for resolve_dataset_source."""
    # Non-existent collection
    with pytest.raises(CollectionNotFoundError):
        resolve_dataset_source(
            dataset_name="2m_temperature",
            collection_name="nonexistent"
        )

    # Non-existent dataset
    with pytest.raises(DatasetNotFoundError):
        resolve_dataset_source(
            dataset_name="nonexistent",
            collection_name="era5"
        )

    # Non-existent variant
    with pytest.raises(VariantNotFoundError):
        resolve_dataset_source(
            dataset_name="2m_temperature",
            collection_name="era5",
            variant_name="nonexistent"
        )

    # Multi-variant dataset without specifying variant
    with pytest.raises(InvalidSelectionError):
        resolve_dataset_source(
            dataset_name="2m_temperature",
            collection_name="era5"
            # No variant specified
        )


# --- Test load_dataset ---


@pytest.fixture
def mock_xarray_dataset():
    """Create a mock xarray dataset."""
    # Create a simple mock dataset with time dimension
    time = np.arange("2020-01-01", "2020-01-10", dtype="datetime64[D]")
    data = np.random.randn(len(time), 10, 10)

    ds = xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], data),
        },
        coords={
            "time": time,
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(-180, 180, 10),
        },
    )
    return ds

@pytest.mark.asyncio
@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
async def test_load_dataset_direct_cid(mock_get_dataset, mock_xarray_dataset):
    """Test loading dataset with direct CID."""
    mock_get_dataset.return_value = mock_xarray_dataset

    async with dClimateClient() as client:

        result = await client.load_dataset(
            dataset="2m_temperature",
            cid="bafyr4iacuutc5bgmirkfyzn4igi2wys7e42kkn674hx3c4dv4wrgjp2k2u"
        )

        # Should call _load_dataset_from_ipfs_cid with the CID
        mock_get_dataset.assert_called_once()
        call_kwargs = mock_get_dataset.call_args[1]
        assert call_kwargs["ipfs_cid"] == "bafyr4iacuutc5bgmirkfyzn4igi2wys7e42kkn674hx3c4dv4wrgjp2k2u"

        # Should return GeotemporalData by default
        assert isinstance(result, GeotemporalData)


@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
@pytest.mark.asyncio
async def test_load_dataset_return_xarray(mock_get_dataset, mock_xarray_dataset):
    """Test loading dataset and returning raw xarray.Dataset."""
    mock_get_dataset.return_value = mock_xarray_dataset
    async with dClimateClient() as client:
        result = await client.load_dataset(
            dataset="2m_temperature",
            cid="bafyr4iacuutc5bgmirkfyzn4igi2wys7e42kkn674hx3c4dv4wrgjp2k2u",
            return_xarray=True
        )

        # Should return xarray.Dataset
        assert isinstance(result, xr.Dataset)
        assert result is mock_xarray_dataset


@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
@pytest.mark.asyncio
async def test_load_dataset_single_variant(mock_get_dataset, mock_xarray_dataset):
    """Test loading dataset with explicit variant."""
    mock_get_dataset.return_value = mock_xarray_dataset
    async with dClimateClient() as client:
        result = await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized"
        )

        # Should call _load_dataset_from_ipfs_cid
        mock_get_dataset.assert_called_once()

        # Should return GeotemporalData
        assert isinstance(result, GeotemporalData)


@pytest.mark.asyncio
async def test_load_dataset_multi_variant_no_selection():
    """Test error when dataset has multiple variants and no variant specified."""
    # ERA5 2m_temperature has multiple variants, should raise error without variant selection
    # Note: Auto-concatenation is disabled due to xarray lazy concat limitations
    with pytest.raises(InvalidSelectionError, match="multiple variants"):
        async with dClimateClient() as client:
            await client.load_dataset(
                dataset="2m_temperature",
                collection="era5",
                return_xarray=True
            )


@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
@pytest.mark.asyncio
async def test_load_dataset_single_variant_autodetect(mock_get_dataset, mock_xarray_dataset):
    """Test loading dataset with single variant (no explicit variant needed)."""
    mock_get_dataset.return_value = mock_xarray_dataset

    async with dClimateClient() as client:
        # IFS temp2m has only one variant, so should load without specifying variant
        result = await client.load_dataset(
            dataset="temperature",
            collection="ifs",
            return_xarray=True
    )

        # Should call _load_dataset_from_ipfs_cid
        mock_get_dataset.assert_called_once()

        # Should return xarray.Dataset
        assert isinstance(result, xr.Dataset)

@pytest.mark.asyncio
async def test_load_dataset_collection_not_found():
    """Test error when collection not found."""
    with pytest.raises(CollectionNotFoundError):
        async with dClimateClient() as client:
            await client.load_dataset(
                dataset="2m_temperature",
                collection="nonexistent"
            )

@pytest.mark.asyncio
async def test_load_dataset_dataset_not_found():
    """Test error when dataset not found."""
    with pytest.raises(DatasetNotFoundError):
        async with dClimateClient() as client:
            await client.load_dataset(
                dataset="nonexistent",
                collection="era5"
            )

@pytest.mark.asyncio
async def test_load_dataset_variant_not_found():
    """Test error when variant not found."""
    with pytest.raises(VariantNotFoundError):
        async with dClimateClient() as client:
            await client.load_dataset(
                dataset="2m_temperature",
                collection="era5",
                variant="nonexistent"
            )


@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
@pytest.mark.asyncio
async def test_load_dataset_with_gateway_config(mock_get_dataset, mock_xarray_dataset):
    """Test loading dataset with custom IPFS gateway configuration."""
    mock_get_dataset.return_value = mock_xarray_dataset
    async with dClimateClient(
        gateway_base_url="http://localhost:8080",
        rpc_base_url="http://localhost:5001"
    ) as client:

        result = await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized"
        )

        # Should call _load_dataset_from_ipfs_cid
        mock_get_dataset.assert_called_once()

        # Should return GeotemporalData
        assert isinstance(result, GeotemporalData)


# --- Test Concatenation Logic ---


@pytest.fixture
def mock_datasets_for_concat():
    """Create mock datasets for concatenation testing."""
    # First dataset: Jan 1-5
    time1 = np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]")
    data1 = np.random.randn(len(time1), 5, 5)
    ds1 = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data1)},
        coords={
            "time": time1,
            "lat": np.linspace(-90, 90, 5),
            "lon": np.linspace(-180, 180, 5),
        },
    )

    # Second dataset: Jan 4-10 (overlap on Jan 4-5)
    time2 = np.arange("2020-01-04", "2020-01-11", dtype="datetime64[D]")
    data2 = np.random.randn(len(time2), 5, 5)
    ds2 = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data2)},
        coords={
            "time": time2,
            "lat": np.linspace(-90, 90, 5),
            "lon": np.linspace(-180, 180, 5),
        },
    )

    return [ds1, ds2]


def test_find_split_index():
    """Test finding split index for concatenation."""
    from dclimate_client_py.concatenate import find_split_index

    # Test with datetime
    combined_coords = np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]")
    next_coords = np.arange("2020-01-04", "2020-01-11", dtype="datetime64[D]")
    last_value = combined_coords[-1]  # 2020-01-05

    split_idx = find_split_index(combined_coords, next_coords, last_value)

    # Should find index where next_coords > 2020-01-05
    # That's 2020-01-06, which is at index 2 (0=Jan4, 1=Jan5, 2=Jan6)
    assert split_idx == 2
    assert next_coords[split_idx] > last_value

    # Test with numeric
    combined_coords = np.array([1, 2, 3, 4, 5])
    next_coords = np.array([4, 5, 6, 7, 8])
    last_value = 5

    split_idx = find_split_index(combined_coords, next_coords, last_value)

    # Should find index where next_coords > 5
    # That's 6, which is at index 2
    assert split_idx == 2
    assert next_coords[split_idx] > last_value


# --- Integration-like Tests (with mocking) ---

@pytest.mark.asyncio
@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
async def test_full_workflow_explicit_variant(mock_get_dataset, mock_xarray_dataset):
    """Test full workflow: list catalog, then load specific variant."""
    mock_get_dataset.return_value = mock_xarray_dataset

    # Step 1: List catalog
    catalog = list_dataset_catalog()
    assert len(catalog) > 0

    # Step 2: Find a dataset
    era5 = [c for c in catalog if c["collection"] == "era5"][0]
    temp2m = [d for d in era5["datasets"] if d["dataset"] == "2m_temperature"][0]

    # Step 3: Load a specific variant
    variant_name = temp2m["variants"][0]["variant"]

    async with dClimateClient() as client:

        result = await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant=variant_name
        )

        assert isinstance(result, GeotemporalData)
        mock_get_dataset.assert_called_once()
    
@pytest.mark.asyncio
async def test_full_workflow_with_explicit_variant():
    """Test full workflow: list catalog, then load with explicit variant selection."""
    # Step 1: List catalog
    catalog = list_dataset_catalog()

    # Step 2: Load with explicit variant (required for multi-variant datasets)
    # Note: Auto-concatenation is disabled due to xarray lazy concat limitations
    async with dClimateClient() as client:
        result = await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized",  # Must specify variant for multi-variant datasets
            return_xarray=True
        )

        assert isinstance(result, xr.Dataset)


# --- Test dClimateClient ---


@pytest.mark.asyncio
@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
async def test_dclimate_client_basic(mock_load_dataset, mock_xarray_dataset):
    """Test basic dClimateClient usage."""
    mock_load_dataset.return_value = mock_xarray_dataset

    async with dClimateClient() as client:
        result = await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized"
        )

        # Should return GeotemporalData by default
        assert isinstance(result, GeotemporalData)

        # Should have called the internal load function
        mock_load_dataset.assert_called_once()


@pytest.mark.asyncio
@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
async def test_dclimate_client_return_xarray(mock_load_dataset, mock_xarray_dataset):
    """Test dClimateClient returning raw xarray.Dataset."""
    mock_load_dataset.return_value = mock_xarray_dataset

    async with dClimateClient() as client:
        result = await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized",
            return_xarray=True
        )

        # Should return raw xarray.Dataset
        assert isinstance(result, xr.Dataset)
        assert result is mock_xarray_dataset


@pytest.mark.asyncio
@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
async def test_dclimate_client_with_cid(mock_load_dataset, mock_xarray_dataset):
    """Test dClimateClient with direct CID."""
    mock_load_dataset.return_value = mock_xarray_dataset

    async with dClimateClient() as client:
        result = await client.load_dataset(
            dataset="test",
            cid="bafyr4iacuutc5bgmirkfyzn4igi2wys7e42kkn674hx3c4dv4wrgjp2k2u"
        )

        # Should call with the CID
        mock_load_dataset.assert_called_once()
        # Check kwargs instead since the function is called with keyword arguments
        call_kwargs = mock_load_dataset.call_args.kwargs
        assert call_kwargs["ipfs_cid"] == "bafyr4iacuutc5bgmirkfyzn4igi2wys7e42kkn674hx3c4dv4wrgjp2k2u"

        assert isinstance(result, GeotemporalData)


@pytest.mark.asyncio
@patch("dclimate_client_py.dclimate_client._load_dataset_from_ipfs_cid")
async def test_dclimate_client_custom_endpoints(mock_load_dataset, mock_xarray_dataset):
    """Test dClimateClient with custom IPFS endpoints."""
    mock_load_dataset.return_value = mock_xarray_dataset

    async with dClimateClient(
        gateway_base_url="https://ipfs.io",
        rpc_base_url="http://localhost:5001"
    ) as client:
        result = await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized"
        )

        assert isinstance(result, GeotemporalData)
        # KuboCAS should have been initialized with custom endpoints
        # (we can't easily test this without inspecting internals, but coverage will show it)


@pytest.mark.asyncio
async def test_dclimate_client_not_in_context():
    """Test that using dClimateClient outside context manager raises error."""
    client = dClimateClient()

    # Should raise error when not in async context
    with pytest.raises(RuntimeError, match="must be used as an async context manager"):
        await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized"
        )


@pytest.mark.asyncio
async def test_dclimate_client_errors():
    """Test that dClimateClient propagates catalog errors correctly."""
    async with dClimateClient() as client:
        # Non-existent collection
        with pytest.raises(CollectionNotFoundError):
            await client.load_dataset(
                dataset="2m_temperature",
                collection="nonexistent"
            )

        # Non-existent dataset
        with pytest.raises(DatasetNotFoundError):
            await client.load_dataset(
                dataset="nonexistent",
                collection="era5"
            )

        # Multi-variant without specifying variant
        with pytest.raises(InvalidSelectionError):
            await client.load_dataset(
                dataset="2m_temperature",
                collection="era5"
            )


@pytest.mark.asyncio
async def test_dclimate_client_workflow():
    """Test full workflow with dClimateClient."""

    # List catalog (synchronous)
    catalog = list_dataset_catalog()
    assert len(catalog) > 0

    # Use client to load dataset
    async with dClimateClient() as client:
        result = await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized",
            return_xarray=False
        )

        # Verify result is GeotemporalData
        assert isinstance(result, GeotemporalData)
        
        # load again with return_xarray=True
        result_xr = await client.load_dataset(
            dataset="2m_temperature",
            collection="era5",
            variant="finalized",
            return_xarray=True
        )

        # query a point
        point_dataset = result_xr.sel(latitude=45, longitude=45, time="2020-01-01", method="nearest")
        # Access the temperature data variable
        assert point_dataset["2m_temperature"].values == 274.18854

        # load again with return_xarray=True
        result_xr = await client.load_dataset(
            dataset="temperature",
            collection="aifs",
            variant="single",
            return_xarray=True
        )

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
