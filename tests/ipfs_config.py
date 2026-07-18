"""Shared endpoint settings for IPFS integration tests."""

from __future__ import annotations

import os


IPFS_GATEWAY_URL = (
    os.environ.get("IPFS_GATEWAY_URI_STEM")
    or os.environ.get("DCLIMATE_IPFS_GATEWAY")
    or "http://127.0.0.1:8080"
).rstrip("/")

IPFS_RPC_URL = os.environ.get("IPFS_RPC_URI_STEM", "http://127.0.0.1:5001").rstrip("/")

STAC_CATALOG_URL = os.environ.get(
    "DCLIMATE_STAC_CATALOG_URL", "https://ipfs-gateway.dclimate.net/stac"
)
