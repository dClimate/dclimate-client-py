#!/usr/bin/env python3
"""
Helper script to identify and verify STAC catalog CIDs

This script helps you find the correct root catalog CID by checking
what type of STAC object each CID points to.
"""

import requests
import json
import sys


def check_cid(gateway_url: str, cid: str):
    """Check what type of STAC object a CID points to."""
    url = f"{gateway_url}/ipfs/{cid}"
    print(f"\nChecking CID: {cid}")
    print(f"URL: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        stac_type = data.get("type", "UNKNOWN")
        stac_id = data.get("id", "UNKNOWN")
        stac_version = data.get("stac_version", "UNKNOWN")

        print(f"  Type: {stac_type}")
        print(f"  ID: {stac_id}")
        print(f"  STAC Version: {stac_version}")

        if stac_type == "Catalog":
            print(f"  ✓ This is a CATALOG - can be used as root!")
            if "links" in data:
                print(f"  Children: {len([l for l in data['links'] if l.get('rel') == 'child'])}")
        elif stac_type == "Collection":
            print(f"  ✗ This is a COLLECTION - NOT a root catalog!")
        elif stac_type == "Feature":
            print(f"  ✗ This is an ITEM - NOT a root catalog!")
        else:
            print(f"  ? Unknown type")

        return data

    except requests.RequestException as e:
        print(f"  ✗ Error fetching: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON: {e}")
        return None


def main():
    gateway_url = "https://ipfs-gateway.dclimate.net"

    print("=" * 60)
    print("STAC Catalog CID Identifier")
    print("=" * 60)

    # The CID currently in the code
    current_cid = "bafkreiamnbh76x7njoh7zu7ct6uzzozv4kyb6wecefnref7hmr454rkkiu"

    print(f"\nGateway: {gateway_url}")
    print(f"\nChecking CID from get_root_catalog_cid()...")

    data = check_cid(gateway_url, current_cid)

    if data and data.get("type") == "Catalog":
        print("\n✓ The current CID is correct - it points to a Catalog!")
    elif data and data.get("type") == "Collection":
        print("\n✗ The current CID points to a Collection, not the root Catalog!")
        print("\nYou need to:")
        print("1. Re-run your STAC generator: cd dclimate-stac && npm run generate")
        print("2. Look for output: '✓ Root catalog published to IPFS: bafkrei...'")
        print("3. Update get_root_catalog_cid() with that CID")
    else:
        print("\n? Could not verify CID type")

    # If user provides CID as argument, check that too
    if len(sys.argv) > 1:
        print("\n" + "=" * 60)
        print("Checking additional CID from command line...")
        check_cid(gateway_url, sys.argv[1])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
