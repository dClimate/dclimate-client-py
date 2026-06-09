"""
Parity test between the two ``list_available_datasets`` paths:

- IPFS walker: ``load_stac_catalog(gateway) → list_available_datasets(...)``
- STAC server: ``list_available_datasets_from_stac_server(stac_url)``

Both should return the same dict-of-collections structure: the same collection
ids, the same (collection, dataset, variant) triples, the same organization /
title / CID / extents per entry.

Intentional differences (asserted only as documented gaps):

- The IPFS walker can include datasets with zero variants (org links declare
  dataset slugs that may not yet have STAC items). The STAC-server path won't
  see those. Filter to ≥1-variant datasets before comparing pairs.

Integration test — hits the live ``api.stac.dclimate.net`` and
``ipfs-gateway.dclimate.net``. Skips gracefully if either is unreachable or if
the IPFS walker returns an empty catalog (gateway-degraded mode).

Run with::

    pytest tests/test_list_datasets_parity.py --run-integration

The autouse ``check_ipfs_connection`` fixture in ``conftest.py`` points at a
local IPFS daemon by default; override via the ``IPFS_GATEWAY_URI_STEM``
environment variable to point at the public gateway.
"""

import json
import os
from typing import Any, Dict

import pytest
import requests

from dclimate_client_py.stac_catalog import (
    load_stac_catalog,
    list_available_datasets,
)
from dclimate_client_py.stac_server import (
    list_available_datasets_from_stac_server,
    STAC_SERVER_URL,
)


pytestmark = pytest.mark.integration


STAC_URL = os.environ.get("STAC_SERVER_URL", STAC_SERVER_URL)
PUBLIC_IPFS_GATEWAY = os.environ.get(
    "DCLIMATE_IPFS_GATEWAY", "https://ipfs-gateway.dclimate.net"
)


def _probe(url: str, *, post: bool = False, timeout: float = 10.0) -> bool:
    try:
        if post:
            resp = requests.post(url, json={"limit": 1}, timeout=timeout)
        else:
            resp = requests.get(url, timeout=timeout)
        return resp.ok
    except (requests.ConnectionError, requests.Timeout, requests.RequestException):
        return False


@pytest.fixture(scope="module")
def stac_catalog() -> Dict[str, Dict[str, Any]]:
    if not _probe(f"{STAC_URL}/collections"):
        pytest.skip(f"STAC server unreachable at {STAC_URL}")
    return list_available_datasets_from_stac_server(STAC_URL)


@pytest.fixture(scope="module")
def ipfs_catalog() -> Dict[str, Dict[str, Any]]:
    if not _probe(f"{PUBLIC_IPFS_GATEWAY}/stac"):
        pytest.skip(f"IPFS gateway unreachable at {PUBLIC_IPFS_GATEWAY}")
    try:
        catalog = load_stac_catalog(gateway_url=PUBLIC_IPFS_GATEWAY)
    except Exception as exc:  # noqa: BLE001 — surface any pystac/network error as a skip
        pytest.skip(f"IPFS catalog load failed: {exc}")
    return list_available_datasets(catalog)


@pytest.fixture(scope="module")
def both_catalogs(stac_catalog, ipfs_catalog):
    # The IPFS walker is permissive — it logs 504s on per-organization fetches
    # but still returns a valid (possibly empty) result. Treat empty-while-
    # STAC-is-not as degraded and skip rather than passing vacuously.
    if not ipfs_catalog and stac_catalog:
        pytest.skip(
            f"IPFS catalog returned 0 collections while STAC server returned "
            f"{len(stac_catalog)} — gateway can serve the root pointer but not "
            f"the deeper catalog tree. Re-run when the gateway is healthy."
        )
    return stac_catalog, ipfs_catalog


def _ipfs_pairs_with_variants(ipfs_catalog) -> set:
    pairs = set()
    for coll_id, entry in ipfs_catalog.items():
        seen = {}
        for v in entry.get("variants", []):
            seen.setdefault(v["dataset"], 0)
            seen[v["dataset"]] += 1
        for dataset_name, count in seen.items():
            if count > 0:
                pairs.add(f"{coll_id}/{dataset_name}")
    return pairs


def _triples(catalog) -> set:
    out = set()
    for coll_id, entry in catalog.items():
        for v in entry.get("variants", []):
            out.add(f"{coll_id}/{v['dataset']}/{v['variant']}")
    return out


def _by_triple(catalog, key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for coll_id, entry in catalog.items():
        for v in entry.get("variants", []):
            out[f"{coll_id}/{v['dataset']}/{v['variant']}"] = v.get(key)
    return out


def test_same_collection_ids(both_catalogs):
    stac_catalog, ipfs_catalog = both_catalogs
    assert sorted(stac_catalog.keys()) == sorted(ipfs_catalog.keys())


def test_same_organization_per_collection(both_catalogs):
    stac_catalog, ipfs_catalog = both_catalogs
    for coll_id, ipfs_entry in ipfs_catalog.items():
        stac_entry = stac_catalog.get(coll_id)
        if stac_entry is None:
            continue
        assert stac_entry["organization"] == ipfs_entry["organization"], (
            f"organization mismatch for {coll_id}: "
            f"stac={stac_entry['organization']!r} ipfs={ipfs_entry['organization']!r}"
        )


def test_same_title_per_collection(both_catalogs):
    stac_catalog, ipfs_catalog = both_catalogs
    for coll_id, ipfs_entry in ipfs_catalog.items():
        stac_entry = stac_catalog.get(coll_id)
        if stac_entry is None:
            continue
        assert stac_entry["title"] == ipfs_entry["title"], (
            f"title mismatch for {coll_id}: "
            f"stac={stac_entry['title']!r} ipfs={ipfs_entry['title']!r}"
        )


def test_same_pairs(both_catalogs):
    stac_catalog, ipfs_catalog = both_catalogs
    stac_pairs = set()
    for coll_id, entry in stac_catalog.items():
        for v in entry.get("variants", []):
            stac_pairs.add(f"{coll_id}/{v['dataset']}")
    # IPFS catalog's `types` may include datasets with zero items; filter to
    # those that actually have at least one variant.
    ipfs_pairs = _ipfs_pairs_with_variants(ipfs_catalog)
    assert sorted(stac_pairs) == sorted(ipfs_pairs)


def test_same_triples(both_catalogs):
    stac_catalog, ipfs_catalog = both_catalogs
    assert sorted(_triples(stac_catalog)) == sorted(_triples(ipfs_catalog))


def test_cids_agree(both_catalogs):
    stac_catalog, ipfs_catalog = both_catalogs
    stac_cids = _by_triple(stac_catalog, "cid")
    ipfs_cids = _by_triple(ipfs_catalog, "cid")

    mismatches = []
    for key, ipfs_cid in ipfs_cids.items():
        stac_cid = stac_cids.get(key)
        if stac_cid != ipfs_cid:
            mismatches.append((key, stac_cid, ipfs_cid))

    # CIDs can transiently disagree if the hourly cron republishes between
    # our two reads. Allow up to 2 to absorb that; anything more is real.
    if len(mismatches) > 2:
        detail = "\n".join(
            f"  {k}\n    STAC: {s}\n    IPFS: {i}" for k, s, i in mismatches[:10]
        )
        raise AssertionError(f"CID mismatch in {len(mismatches)} variants:\n{detail}")


def test_bbox_agrees(both_catalogs):
    stac_catalog, ipfs_catalog = both_catalogs
    stac_bboxes = _by_triple(stac_catalog, "spatial_extent")
    ipfs_bboxes = _by_triple(ipfs_catalog, "spatial_extent")

    for key, ipfs_bbox in ipfs_bboxes.items():
        stac_bbox = stac_bboxes.get(key)

        # bbox is a TypedDict {"bbox": (lo, la, hi, la)} — compare structurally.
        # Tuples vs lists may differ across paths; normalize to tuple of floats.
        def _norm(b):
            if b is None:
                return None
            coords = b.get("bbox") if isinstance(b, dict) else b
            return tuple(float(x) for x in coords) if coords is not None else None

        assert _norm(stac_bbox) == _norm(ipfs_bbox), f"bbox mismatch for {key}"


def test_temporal_extent_agrees(both_catalogs):
    stac_catalog, ipfs_catalog = both_catalogs

    # Compare as instants, not strings — STAC API normalizes ISO timestamps to
    # no-millis ("...:00Z") while the IPFS catalog preserves explicit millis
    # ("...:00.000Z"). Same point in time, different serialization.
    import datetime as dt

    def _to_ms(s):
        if s is None:
            return None
        try:
            return int(
                dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000
            )
        except (ValueError, AttributeError):
            return None

    def _norm(extent):
        if extent is None:
            return None
        return {
            "start": _to_ms(extent.get("start")),
            "end": _to_ms(extent.get("end")),
        }

    stac_temp = _by_triple(stac_catalog, "temporal_extent")
    ipfs_temp = _by_triple(ipfs_catalog, "temporal_extent")

    mismatches = []
    for key, ipfs_ext in ipfs_temp.items():
        stac_ext = stac_temp.get(key)
        if _norm(stac_ext) != _norm(ipfs_ext):
            mismatches.append((key, _norm(stac_ext), _norm(ipfs_ext)))

    # Forecast datasets republish often enough that temporal extents can drift
    # mid-test. Allow a tiny tolerance for that case.
    if len(mismatches) > 2:
        detail = "\n".join(
            f"  {k}\n    STAC: {json.dumps(s)}\n    IPFS: {json.dumps(i)}"
            for k, s, i in mismatches[:10]
        )
        raise AssertionError(
            f"Temporal extent mismatch in {len(mismatches)} variants:\n{detail}"
        )


def test_category_agrees(both_catalogs):
    stac_catalog, ipfs_catalog = both_catalogs
    # IPFS walker reads category from `dclimate:collections:<category>` on the
    # org link; STAC server rolls it up from item `dclimate:observation` (only
    # when items unanimously agree). Both should agree on collections that have
    # a category.
    for coll_id, ipfs_entry in ipfs_catalog.items():
        stac_entry = stac_catalog.get(coll_id)
        if stac_entry is None:
            continue
        assert stac_entry.get("category") == ipfs_entry.get("category"), (
            f"category mismatch for {coll_id}: "
            f"stac={stac_entry.get('category')!r} ipfs={ipfs_entry.get('category')!r}"
        )
