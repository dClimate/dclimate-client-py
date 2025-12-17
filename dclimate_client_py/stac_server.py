"""
STAC Server client for fast CID resolution.

This module provides direct access to a STAC server API for resolving dataset CIDs,
which is faster than traversing the IPFS-hosted catalog structure.
"""

from typing import Optional
import requests

STAC_SERVER_URL = "https://api.stac.dclimate.net"


def resolve_cid_from_stac_server(
    collection: str,
    dataset: str,
    variant: Optional[str] = None,
    server_url: str = STAC_SERVER_URL,
) -> str:
    """
    Resolve dataset CID via STAC server /search API.

    Uses the same API format as the frontend (POST /search with collections filter).

    Args:
        collection: Collection ID (e.g., 'ecmwf_aifs', 'ecmwf_era5')
        dataset: Dataset name (e.g., 'temperature', 'precipitation')
        variant: Optional variant name (e.g., 'ensemble', 'deterministic')
        server_url: STAC server base URL

    Returns:
        str: The IPFS CID of the Zarr dataset (without 'ipfs://' prefix)

    Raises:
        ValueError: If dataset or variant is not found
        requests.HTTPError: If the server request fails
    """
    # Search by collection
    body = {
        "limit": 100,
        "collections": [collection],
    }

    response = requests.post(f"{server_url}/search", json=body, timeout=10)
    response.raise_for_status()

    features = response.json().get("features", [])

    # Filter to matching dataset (item ID pattern: {collection}-{dataset}-{variant})
    prefix = f"{collection}-{dataset}"
    matches = [f for f in features if f["id"].startswith(prefix)]

    if not matches:
        raise ValueError(f"No items found for {collection}/{dataset}")

    # Select by variant or use default preference
    if variant:
        item = next(
            (f for f in matches if f["properties"].get("dclimate:variant") == variant),
            None,
        )
        if not item:
            raise ValueError(f"Variant '{variant}' not found for {collection}/{dataset}")
    else:
        # Prefer: default > final > finalized > latest > first match
        item = matches[0]
        for preferred in ["default", "final", "finalized", "latest"]:
            found = next(
                (f for f in matches if f["properties"].get("dclimate:variant") == preferred),
                None,
            )
            if found:
                item = found
                break

    # Extract CID from asset
    href = item.get("assets", {}).get("data", {}).get("href", "")
    if href.startswith("ipfs://"):
        return href.replace("ipfs://", "")
    if href:
        return href

    raise ValueError(f"Item '{item['id']}' has no data asset")
