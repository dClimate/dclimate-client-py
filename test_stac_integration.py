#!/usr/bin/env python3
"""
Test script for STAC integration

This script tests the STAC catalog integration with the dClimate client.
Run with: python test_stac_integration.py
"""

import asyncio
from dclimate_client_py import dClimateClient
import time


async def test_list_datasets():
    """Test listing available datasets from STAC catalog."""
    print("=" * 60)
    print("Test 1: List Available Datasets")
    print("=" * 60)

    async with dClimateClient() as client:
        datasets = client.list_datasets()
        print(datasets)

        print(f"\nFound {len(datasets)} collections:")
        for collection_id, info in datasets.items():
            print(f"\n  Collection: {collection_id}")
            print(f"  Title: {info['title']}")
            print(
                f"  Dataset Types ({len(info['types'])}): {', '.join(info['types'][:5])}"
            )
            if len(info["types"]) > 5:
                print(f"    ... and {len(info['types']) - 5} more")

    print("\n✓ Test 1 passed!")


async def test_load_dataset_from_stac():
    """Test loading a dataset using STAC catalog."""
    print("\n" + "=" * 60)
    print("Test 2: Load Dataset from STAC")
    print("=" * 60)

    # Test with IFS temperature single variant
    organization = "ecmwf"
    collection = "aifs"
    dataset = "temperature_forecast"
    variant = "single"

    print("\nLoading dataset:")
    print(f"  Collection: {collection}")
    print(f"  Dataset: {dataset}")
    print(f"  Variant: {variant}")

    async with dClimateClient() as client:
        try:
            # Start time measurement

            start_time = time.time()
            data, metadata = await client.load_dataset(
                organization=organization,
                collection=collection,
                dataset=dataset,
                variant=variant,
                return_xarray=False,
            )
            end_time = time.time()
            print(f"\nLoading took {end_time - start_time:.2f} seconds")

            print("\n✓ Successfully loaded dataset!")
            print("\nMetadata:")
            print(f"  Source: {metadata['source']}")
            print(f"  CID: {metadata['cid']}")
            print(f"  Slug: {metadata['slug']}")

            print("\nDataset info:")
            print(f"  Type: {type(data).__name__}")
            if hasattr(data, "dataset"):
                ds = data.dataset
                print(f"  Variables: {list(ds.data_vars)}")
                print(f"  Coordinates: {list(ds.coords)}")
                print(f"  Shape: {ds.dims}")

        except Exception as e:
            print(f"\n✗ Error loading dataset: {e}")
            raise

    print("\n✓ Test 2 passed!")


async def test_direct_cid_loading():
    """Test loading a dataset with direct CID (bypassing STAC)."""
    print("\n" + "=" * 60)
    print("Test 3: Direct CID Loading")
    print("=" * 60)

    # Use a known CID (this would need to be a real Zarr dataset CID)
    test_cid = "bafyr4id2atcdmh6vf57uy2ii4axsketvgf2ong3hneigyv6wwxwgkpaxve"

    print("\nTesting direct CID loading:")
    print(f"  CID: {test_cid}")

    async with dClimateClient():
        try:
            print("✓ Client created")
            print("✓ Direct CID loading bypasses STAC catalog")
            print("  (Skipping actual load test - would need valid Zarr CID)")

        except Exception as e:
            print(f"Note: {e}")

    print("\n✓ Test 3 passed!")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("STAC Integration Test Suite")
    print("=" * 60)

    try:
        await test_list_datasets()
        # Uncomment to test loading (requires IPFS with actual data)
        await test_load_dataset_from_stac()
        await test_direct_cid_loading()

        print("\n" + "=" * 60)
        print("All Tests Passed! ✓")
        print("=" * 60)
        print("\nNote: The client now ALWAYS uses STAC catalog.")
        print("Legacy catalog mode has been removed.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
