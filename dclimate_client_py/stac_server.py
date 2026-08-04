"""
STAC Server client for fast CID resolution.

This module provides direct access to a STAC server API for resolving dataset CIDs,
which is faster than traversing the IPFS-hosted catalog structure.
"""

import asyncio
import weakref
from collections.abc import AsyncIterator, Iterator
from json import dumps
from threading import Lock
from typing import Any, Dict, Iterable, NamedTuple, Optional, Set
from urllib.parse import urljoin, urlsplit

import httpx

from .datasets import SpatialExtent, TemporalExtent

STAC_SERVER_URL = "https://api.stac.dclimate.net"


_MAX_SEARCH_PAGES = 50
_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_LOCK = Lock()
_ASYNC_HTTP_CLIENTS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, httpx.AsyncClient
] = weakref.WeakKeyDictionary()
_ASYNC_HTTP_CLIENT_LOCK = Lock()


_SearchBody = Optional[Dict[str, Any]]
_SearchHeaders = Dict[str, str]
_SearchRequest = tuple[str, str, _SearchBody, _SearchHeaders]
_SearchPageKey = tuple[str, str, str, str]


def _client() -> httpx.Client:
    """Return the process-wide pooled client used for synchronous STAC calls."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = httpx.Client(timeout=30, follow_redirects=True)
    return _HTTP_CLIENT


def _async_client() -> httpx.AsyncClient:
    """Return the pooled async STAC client for the current event loop."""
    loop = asyncio.get_running_loop()
    with _ASYNC_HTTP_CLIENT_LOCK:
        client = _ASYNC_HTTP_CLIENTS.get(loop)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=30, follow_redirects=True)
            _ASYNC_HTTP_CLIENTS[loop] = client
    return client


async def aclose_stac_server_client() -> None:
    """Close the pooled async STAC client owned by the current event loop."""
    loop = asyncio.get_running_loop()
    with _ASYNC_HTTP_CLIENT_LOCK:
        client = _ASYNC_HTTP_CLIENTS.pop(loop, None)
    if client is not None:
        await client.aclose()


class ResolvedDataset(NamedTuple):
    cid: str
    variant: str


def _dataset_and_variant_from_item_id(
    feature_id: str,
    collection: str,
    dataset: Optional[str] = None,
    variant: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    prefix = f"{collection}-"
    remainder = (
        feature_id[len(prefix) :] if feature_id.startswith(prefix) else feature_id
    )
    if dataset is not None:
        if remainder == dataset:
            return dataset, None
        dataset_prefix = f"{dataset}-"
        if remainder.startswith(dataset_prefix):
            return dataset, remainder[len(dataset_prefix) :] or None
        return None, None

    if variant is not None:
        variant_suffix = f"-{variant}"
        if remainder.endswith(variant_suffix):
            return remainder[: -len(variant_suffix)] or None, variant
        return None, None

    parsed_dataset, separator, variant = remainder.partition("-")
    return parsed_dataset or None, (variant or None) if separator else None


def _dataset_id_from_item_id(
    feature_id: str,
    collection: str,
    dataset: Optional[str] = None,
) -> Optional[str]:
    parsed_dataset, _ = _dataset_and_variant_from_item_id(
        feature_id, collection, dataset
    )
    return parsed_dataset


def _dataset_and_variant_from_known_datasets(
    feature_id: str,
    collection: str,
    known_datasets: Iterable[str],
) -> tuple[Optional[str], Optional[str]]:
    """Parse an item id using the longest matching known dataset id.

    Hyphens delimit both dataset ids and variants, so an unhinted id such as
    ``chirps-precip-daily-final-p05`` is inherently ambiguous. Collection
    metadata and explicit sibling-item properties provide the missing hint.
    """
    candidates = sorted(
        {value for value in known_datasets if isinstance(value, str) and value},
        key=len,
        reverse=True,
    )
    for candidate in candidates:
        parsed_dataset, parsed_variant = _dataset_and_variant_from_item_id(
            feature_id, collection, dataset=candidate
        )
        if parsed_dataset is not None:
            return parsed_dataset, parsed_variant
    return _dataset_and_variant_from_item_id(feature_id, collection)


def _feature_variant(
    feature: Dict[str, Any], collection: str, dataset: Optional[str] = None
) -> Optional[str]:
    props = feature.get("properties") or {}
    variant = props.get("dclimate:variant")
    if variant:
        return variant

    feature_id = feature.get("id")
    if not isinstance(feature_id, str):
        return None
    _, parsed_variant = _dataset_and_variant_from_item_id(
        feature_id, collection, dataset
    )
    return parsed_variant


def _feature_matches_dataset(
    feature: Dict[str, Any],
    collection: str,
    dataset: str,
    known_datasets: Iterable[str] = (),
) -> bool:
    feature_collection = feature.get("collection")
    if feature_collection and feature_collection != collection:
        return False

    props = feature.get("properties") or {}
    dataset_id = props.get("dclimate:dataset_id")
    if dataset_id:
        return dataset_id == dataset

    feature_id = feature.get("id")
    if not isinstance(feature_id, str):
        return False
    variant = props.get("dclimate:variant")
    if variant:
        parsed_dataset, _ = _dataset_and_variant_from_item_id(
            feature_id, collection, variant=variant
        )
        if parsed_dataset is not None:
            return parsed_dataset == dataset
        # The id does not encode the property variant (e.g. a bare id with
        # properties variant "default") — fall through to dataset matching.
    if known_datasets:
        parsed_dataset, _ = _dataset_and_variant_from_known_datasets(
            feature_id, collection, known_datasets
        )
        return parsed_dataset == dataset
    return _dataset_id_from_item_id(feature_id, collection, dataset) == dataset


def _search_page_key(
    method: str,
    url: str,
    body: _SearchBody,
    headers: _SearchHeaders,
) -> _SearchPageKey:
    """Return a stable key used to stop repeated pagination requests."""
    return (
        method,
        url,
        dumps(body, sort_keys=True, default=str),
        dumps(headers, sort_keys=True, default=str),
    )


def _search_request_kwargs(
    method: str,
    body: _SearchBody,
    headers: _SearchHeaders,
    timeout: int,
) -> Dict[str, Any]:
    """Build transport-independent request arguments for a search page."""
    if method == "POST":
        request_kwargs: Dict[str, Any] = {"json": body, "timeout": timeout}
    else:
        request_kwargs = {"params": body or None, "timeout": timeout}
    if headers:
        request_kwargs["headers"] = headers
    return request_kwargs


def _url_origin(url: str) -> tuple[str, str, Optional[int]]:
    """Return a normalized URL origin for STAC pagination validation."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not scheme or not hostname:
        raise ValueError(f"STAC pagination link is not an absolute URL: {url!r}")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(
            f"STAC pagination link has an invalid port: {url!r}"
        ) from error
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    return scheme, hostname, port


def _next_search_request(
    server_url: str,
    current_url: str,
    original_body: Dict[str, Any],
    page: Dict[str, Any],
) -> Optional[_SearchRequest]:
    """Plan the next same-origin STAC request from a search response page."""
    next_link = next(
        (
            link
            for link in page.get("links", []) or []
            if link.get("rel") == "next" and link.get("href")
        ),
        None,
    )
    if next_link is None:
        return None

    href = next_link["href"]
    if not isinstance(href, str):
        raise ValueError("STAC pagination link href must be a string")
    next_url = urljoin(current_url, href)
    parsed_next_url = urlsplit(next_url)
    if (
        _url_origin(next_url) != _url_origin(server_url)
        or parsed_next_url.username is not None
        or parsed_next_url.password is not None
    ):
        raise ValueError(
            "STAC pagination link must use the configured server origin "
            f"{_url_origin(server_url)!r}: {next_url!r}"
        )

    method = str(next_link.get("method", "GET")).upper()
    if method not in {"GET", "POST"}:
        return None

    linked_headers = next_link.get("headers")
    # A server-provided next link may carry continuation credentials. Forward
    # those headers only on a validated, encrypted connection to the same
    # origin; plaintext endpoints still paginate without linked headers.
    request_headers = (
        linked_headers
        if parsed_next_url.scheme.lower() == "https"
        and isinstance(linked_headers, dict)
        else {}
    )

    linked_body = next_link.get("body")
    if isinstance(linked_body, dict):
        # STAC API next-link contract: with "merge": true the linked body
        # extends the original request (keeping filters like "collections");
        # otherwise it replaces it wholesale.
        request_body = (
            {**original_body, **linked_body} if next_link.get("merge") else linked_body
        )
    elif next_link.get("merge"):
        # ``merge: true`` without a link body still carries the original
        # search filters to the next request.
        request_body = dict(original_body)
    else:
        request_body = None

    return next_url, method, request_body, request_headers


def _search_pages(
    server_url: str,
    body: Dict[str, Any],
    timeout: int,
) -> Iterator[Dict[str, Any]]:
    """Yield bounded STAC search pages while following ``rel=next`` links."""
    url = f"{server_url.rstrip('/')}/search"
    method = "POST"
    request_body: _SearchBody = body
    request_headers: _SearchHeaders = {}
    seen: Set[_SearchPageKey] = set()

    for _ in range(_MAX_SEARCH_PAGES):
        page_key = _search_page_key(method, url, request_body, request_headers)
        if page_key in seen:
            return
        seen.add(page_key)

        request_kwargs = _search_request_kwargs(
            method, request_body, request_headers, timeout
        )
        if method == "POST":
            response = _client().post(url, **request_kwargs)
        else:
            response = _client().get(url, **request_kwargs)
        response.raise_for_status()
        page = response.json()
        yield page

        if not (page.get("features") or []):
            return
        next_request = _next_search_request(server_url, url, body, page)
        if next_request is None:
            return
        url, method, request_body, request_headers = next_request
    else:
        # Reaching the bound with a valid next link means the result is
        # incomplete. Surface that explicitly so callers can use their
        # catalog fallback instead of accepting truncated search results.
        raise ValueError(
            f"STAC search reached its page limit of {_MAX_SEARCH_PAGES} "
            "while another next link was present"
        )


async def _asearch_pages(
    server_url: str,
    body: Dict[str, Any],
    timeout: int,
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Yield bounded STAC search pages without blocking the event loop."""
    http_client = client or _async_client()
    url = f"{server_url.rstrip('/')}/search"
    method = "POST"
    request_body: _SearchBody = body
    request_headers: _SearchHeaders = {}
    seen: Set[_SearchPageKey] = set()

    for _ in range(_MAX_SEARCH_PAGES):
        page_key = _search_page_key(method, url, request_body, request_headers)
        if page_key in seen:
            return
        seen.add(page_key)

        request_kwargs = _search_request_kwargs(
            method, request_body, request_headers, timeout
        )
        if method == "POST":
            response = await http_client.post(url, **request_kwargs)
        else:
            response = await http_client.get(url, **request_kwargs)
        response.raise_for_status()
        page = response.json()
        yield page

        if not (page.get("features") or []):
            return
        next_request = _next_search_request(server_url, url, body, page)
        if next_request is None:
            return
        url, method, request_body, request_headers = next_request
    else:
        raise ValueError(
            f"STAC search reached its page limit of {_MAX_SEARCH_PAGES} "
            "while another next link was present"
        )


def _resolve_dataset_from_features(
    collection: str,
    dataset: str,
    variant: Optional[str],
    features: Iterable[Dict[str, Any]],
) -> ResolvedDataset:
    """Resolve a dataset from STAC search features shared by both clients."""
    feature_list = list(features)

    # An item with no variant segment/property is what the listing API
    # reports as the "default" variant — keep resolve symmetric with list.
    def _effective_variant(feature: Dict[str, Any]) -> str:
        return _feature_variant(feature, collection, dataset) or "default"

    known_datasets = {
        dataset_id
        for feature in feature_list
        if isinstance(feature, dict)
        for dataset_id in [(feature.get("properties") or {}).get("dclimate:dataset_id")]
        if isinstance(dataset_id, str) and dataset_id
    }
    known_datasets.add(dataset)
    # Filter to the exact dataset. A prefix match would conflate datasets such
    # as ``precip`` and a known hyphenated dataset ``precip-daily``.
    matches = [
        feature
        for feature in feature_list
        if _feature_matches_dataset(
            feature, collection, dataset, known_datasets=known_datasets
        )
    ]
    if not matches:
        raise ValueError(f"No items found for {collection}/{dataset}")

    # Select by variant or use default preference
    if variant is not None:
        item = next(
            (f for f in matches if _effective_variant(f) == variant),
            None,
        )
        if not item:
            raise ValueError(
                f"Variant '{variant}' not found for {collection}/{dataset}"
            )
    else:
        # Prefer: default > final > finalized > latest > first match
        item = matches[0]
        for preferred in ["default", "final", "finalized", "latest"]:
            found = next(
                (f for f in matches if _effective_variant(f) == preferred),
                None,
            )
            if found:
                item = found
                break

    # Extract CID from asset
    selected_variant = variant if variant is not None else _effective_variant(item)
    href = item.get("assets", {}).get("data", {}).get("href", "")
    if href.startswith("ipfs://"):
        return ResolvedDataset(href.replace("ipfs://", ""), selected_variant)
    if href:
        return ResolvedDataset(href, selected_variant)

    raise ValueError(f"Item '{item['id']}' has no data asset")


def resolve_cid_from_stac_server(
    collection: str,
    dataset: str,
    variant: Optional[str] = None,
    server_url: str = STAC_SERVER_URL,
) -> ResolvedDataset:
    """
    Resolve dataset CID via STAC server /search API.

    Changed in 0.6: returns ResolvedDataset; variant='' is treated as an
    explicit (unresolvable) variant rather than no-variant.

    Uses the same API format as the frontend (POST /search with collections filter).

    Args:
        collection: Collection ID (e.g., 'ecmwf_aifs', 'ecmwf_era5')
        dataset: Dataset name (e.g., 'temperature', 'precipitation')
        variant: Optional variant name (e.g., 'ensemble', 'deterministic')
        server_url: STAC server base URL

    Returns:
        ResolvedDataset: The IPFS CID and selected variant

    Raises:
        ValueError: If dataset or variant is not found
        httpx.HTTPError: If the server request fails
    """
    body = {
        "limit": 100,
        "collections": [collection],
    }
    features = [
        feature
        for page in _search_pages(server_url, body, timeout=10)
        for feature in page.get("features", []) or []
    ]
    return _resolve_dataset_from_features(collection, dataset, variant, features)


async def aresolve_cid_from_stac_server(
    collection: str,
    dataset: str,
    variant: Optional[str] = None,
    server_url: str = STAC_SERVER_URL,
    *,
    client: Optional[httpx.AsyncClient] = None,
) -> ResolvedDataset:
    """Resolve a dataset CID natively asynchronously via the STAC API.

    When ``client`` is omitted, calls reuse a pooled ``httpx.AsyncClient``
    scoped to the current event loop. Call ``aclose_stac_server_client`` when
    that loop shuts down. Injected clients remain caller-owned and are never
    closed by this function.

    Args:
        collection: Collection ID (e.g., 'ecmwf_aifs', 'ecmwf_era5')
        dataset: Dataset name (e.g., 'temperature', 'precipitation')
        variant: Optional variant name (e.g., 'ensemble', 'deterministic')
        server_url: STAC server base URL
        client: Optional caller-owned pooled async HTTP client

    Returns:
        ResolvedDataset: The IPFS CID and selected variant

    Raises:
        ValueError: If dataset or variant is not found
        httpx.HTTPError: If the server request fails
    """
    body = {
        "limit": 100,
        "collections": [collection],
    }
    features: list[Dict[str, Any]] = []
    async for page in _asearch_pages(server_url, body, timeout=10, client=client):
        features.extend(page.get("features", []) or [])
    return _resolve_dataset_from_features(collection, dataset, variant, features)


def _strip_ipfs_scheme(cid: Optional[str]) -> Optional[str]:
    if not cid:
        return None
    return cid.replace("ipfs://", "", 1) if cid.startswith("ipfs://") else cid


def list_available_datasets_from_stac_server(
    server_url: str = STAC_SERVER_URL,
) -> Dict[str, Dict[str, Any]]:
    """
    List all datasets/variants by querying a STAC API server directly.

    Fast path that mirrors ``list_available_datasets`` (the IPFS walker) without
    traversing the IPFS-hosted catalog tree. It queries:

    1. ``GET  /collections`` — collection ids, titles
    2. ``POST /search``      — items, with dataset/variant/CID in properties

    Returns the same dict-of-dicts shape as the IPFS walker so callers don't
    need to know which path produced it.

    Notes
    -----
    - Organization is derived from the ``{org}_{name}`` collection-id convention
      (e.g. ``noaa_aigfs`` → ``noaa``). The IPFS walker reads it from a
      ``dclimate:id`` field on an org-level link; the STAC API doesn't expose
      organizations as first-class entities, so we approximate.
    - Category (``historical`` / ``forecast``) is rolled up from item
      ``dclimate:observation`` properties — only when every item in the
      collection agrees, to avoid picking a misleading value when items
      disagree.
    - Search pagination is bounded to avoid looping on malformed ``next`` links.
    """
    collections_resp = _client().get(f"{server_url}/collections", timeout=10)
    collections_resp.raise_for_status()
    collections_body = collections_resp.json()

    # Accumulator per collection. Built up from /collections then enriched by
    # the /search response. Collections that have no items end up filtered out
    # at the end — matches the IPFS walker, which only surfaces collections
    # that have actual datasets.
    accumulators: Dict[str, Dict[str, Any]] = {}

    for coll in collections_body.get("collections", []) or []:
        coll_id = coll.get("id")
        if not isinstance(coll_id, str):
            continue
        organization = coll_id.split("_", 1)[0] if "_" in coll_id else None
        accumulators[coll_id] = {
            "id": coll_id,
            "title": coll.get("title") or coll_id,
            "organization": organization,
            "observations": set(),
            "datasets": {},  # dataset_name -> { variant_name -> variant_entry }
            "known_datasets": set(),
        }

        summaries = coll.get("summaries") or {}
        declared_datasets = coll.get("dclimate:types") or summaries.get(
            "dclimate:dataset_id", []
        )
        if isinstance(declared_datasets, list):
            accumulators[coll_id]["known_datasets"].update(
                value for value in declared_datasets if isinstance(value, str) and value
            )

    search_features = [
        feature
        for page in _search_pages(server_url, {"limit": 100}, timeout=15)
        for feature in page.get("features", []) or []
    ]
    dataset_hints: Dict[str, Set[str]] = {}
    for feature in search_features:
        collection_id = feature.get("collection")
        props = feature.get("properties") or {}
        dataset_id = props.get("dclimate:dataset_id")
        if (
            isinstance(collection_id, str)
            and isinstance(dataset_id, str)
            and dataset_id
        ):
            dataset_hints.setdefault(collection_id, set()).add(dataset_id)

    for collection_id, hints in dataset_hints.items():
        if collection_id in accumulators:
            accumulators[collection_id]["known_datasets"].update(hints)

    for feature in search_features:
        feature_id = feature.get("id", "")
        collection_id = feature.get("collection")
        if not collection_id and isinstance(feature_id, str) and "-" in feature_id:
            collection_id = feature_id.split("-", 1)[0]
        if not collection_id:
            continue

        entry = accumulators.get(collection_id)
        if entry is None:
            organization = (
                collection_id.split("_", 1)[0] if "_" in collection_id else None
            )
            entry = {
                "id": collection_id,
                "title": collection_id,
                "organization": organization,
                "observations": set(),
                "datasets": {},
                "known_datasets": set(dataset_hints.get(collection_id, set())),
            }
            accumulators[collection_id] = entry

        props = feature.get("properties") or {}
        observation = props.get("dclimate:observation")
        if isinstance(observation, str) and observation:
            entry["observations"].add(observation)

        # Prefer explicit dclimate:* properties; fall back to id-parsing for
        # items that pre-date the property convention.
        property_dataset = props.get("dclimate:dataset_id")
        property_variant = props.get("dclimate:variant")
        if not isinstance(feature_id, str):
            parsed_dataset, parsed_variant = None, None
        elif property_dataset:
            parsed_dataset, parsed_variant = _dataset_and_variant_from_item_id(
                feature_id, collection_id, dataset=property_dataset
            )
        elif property_variant:
            parsed_dataset, parsed_variant = _dataset_and_variant_from_item_id(
                feature_id, collection_id, variant=property_variant
            )
            if parsed_dataset is None:
                parsed_dataset, _ = _dataset_and_variant_from_known_datasets(
                    feature_id, collection_id, entry["known_datasets"]
                )
                parsed_variant = property_variant
        else:
            parsed_dataset, parsed_variant = _dataset_and_variant_from_known_datasets(
                feature_id, collection_id, entry["known_datasets"]
            )
        dataset_name = property_dataset or parsed_dataset
        variant_name = props.get("dclimate:variant") or parsed_variant or "default"
        if not dataset_name:
            continue

        cid = _strip_ipfs_scheme(props.get("dclimate:latest_dataset_cid"))
        if not cid:
            cid = _strip_ipfs_scheme(
                (feature.get("assets") or {}).get("data", {}).get("href")
            )

        variant_entry: Dict[str, Any] = {
            "dataset": dataset_name,
            "variant": variant_name,
        }
        if cid:
            variant_entry["cid"] = cid

        bbox = feature.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            variant_entry["spatial_extent"] = SpatialExtent(
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            )

        start_dt = props.get("start_datetime") or props.get("datetime")
        end_dt = props.get("end_datetime") or props.get("datetime")
        if start_dt is not None or end_dt is not None:
            variant_entry["temporal_extent"] = TemporalExtent(
                start=start_dt, end=end_dt
            )

        dataset_variants = entry["datasets"].setdefault(dataset_name, {})
        dataset_variants[variant_name] = variant_entry

    result: Dict[str, Dict[str, Any]] = {}
    for collection_id, entry in accumulators.items():
        if not entry["datasets"]:
            continue

        variants_flat = []
        for _, variants in entry["datasets"].items():
            variants_flat.extend(variants.values())

        out: Dict[str, Any] = {
            "id": entry["id"],
            "title": entry["title"],
            "types": sorted(entry["datasets"].keys()),
            "organization": entry["organization"],
            "variants": variants_flat,
        }

        # Only roll up to a collection-level category when every item in the
        # collection agrees. Mixed observations are a meaningful ambiguity —
        # leave the key absent rather than picking a misleading value.
        observations: Set[str] = entry["observations"]
        if len(observations) == 1:
            out["category"] = next(iter(observations))

        result[collection_id] = out

    return result
