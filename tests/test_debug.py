"""Debug test for loading noaa_gfs:temperature_max_forecast:default dataset."""

import pytest
from dclimate_client_py.dclimate_client import dClimateClient


@pytest.mark.asyncio
async def test_load_noaa_gfs_temperature_max_forecast():
    """Test loading noaa_gfs:temperature_max_forecast:default dataset."""
    async with dClimateClient() as client:
        print("\n--- Attempting to load dataset ---")
        print("Collection: noaa_gfs")
        print("Dataset: temperature_max_forecast")
        print("Variant: default")

        try:
            dataset, metadata = await client.load_dataset(
                collection="noaa_gfs",
                dataset="temperature_max_forecast",
                variant="default",
                return_xarray=True,
            )

            print("\n--- Dataset loaded successfully ---")
            print(f"Dataset type: {type(dataset)}")
            print(f"Dataset: {dataset}")
            print(f"\nMetadata: {metadata}")

            if hasattr(dataset, 'data_vars'):
                print(f"\nData variables: {list(dataset.data_vars)}")
            if hasattr(dataset, 'coords'):
                print(f"Coordinates: {list(dataset.coords)}")
            if hasattr(dataset, 'dims'):
                print(f"Dimensions: {dict(dataset.dims)}")

        except Exception as e:
            print(f"\n--- Error loading dataset ---")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            raise


@pytest.mark.asyncio
async def test_load_noaa_gfs_with_xarray_false():
    """Test loading noaa_gfs dataset with return_xarray=False."""
    async with dClimateClient() as client:
        print("\n--- Attempting to load dataset (return_xarray=False) ---")

        try:
            dataset, metadata = await client.load_dataset(
                collection="noaa_gfs",
                dataset="temperature_max_forecast",
                variant="default",
                return_xarray=False,
            )

            print("\n--- Dataset loaded successfully ---")
            print(f"Dataset type: {type(dataset)}")
            print(f"Dataset: {dataset}")
            print(f"\nMetadata: {metadata}")

            if hasattr(dataset, 'data'):
                print(f"\nUnderlying data type: {type(dataset.data)}")
                print(f"Data vars: {list(dataset.data.data_vars) if hasattr(dataset.data, 'data_vars') else 'N/A'}")

        except Exception as e:
            print(f"\n--- Error loading dataset ---")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            raise


@pytest.mark.asyncio
async def test_list_available_datasets():
    """List available datasets to verify noaa_gfs exists."""
    async with dClimateClient() as client:
        print("\n--- Checking catalog for noaa_gfs datasets ---")

        try:
            # Try to access catalog if available
            if hasattr(client, 'catalog') or hasattr(client, 'get_catalog'):
                catalog = await client.get_catalog() if hasattr(client, 'get_catalog') else client.catalog
                print(f"Catalog: {catalog}")
            else:
                print("No catalog method available on client")

            # Try searching for noaa_gfs
            if hasattr(client, 'search'):
                results = await client.search("noaa_gfs")
                print(f"Search results for 'noaa_gfs': {results}")

        except Exception as e:
            print(f"Error accessing catalog: {type(e).__name__} - {e}")
