"""
STAC Catalog Integration for dClimate

This module provides integration with STAC (SpatioTemporal Asset Catalog) format
for discovering and accessing dClimate datasets stored on IPFS.
"""

from os import PathLike, fspath
from typing import Optional, Dict, List, Set, Tuple, Any, cast
import logging
import weakref
from threading import Lock
from urllib.parse import urlsplit

import httpx
import pystac

from .datasets import SpatialExtent, TemporalExtent
from .stac_server import (
    ResolvedDataset,
    _dataset_and_variant_from_item_id,
    _dataset_and_variant_from_known_datasets,
)

logger = logging.getLogger(__name__)
STAC_CATALOG_URL = "https://ipfs-gateway.dclimate.net/stac"
_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_LOCK = Lock()


def _client() -> httpx.Client:
    """Return the process-wide pooled client used for synchronous STAC reads."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = httpx.Client(timeout=30, follow_redirects=True)
    return _HTTP_CLIENT


def get_root_catalog_cid(
    catalog_url: str = STAC_CATALOG_URL,
    *,
    headers: Optional[Dict[str, str]] = None,
    auth: Optional[Tuple[str, str]] = None,
) -> str:
    """
    Get the root STAC catalog CID.

    Fetches the latest catalog CID from the dClimate IPFS gateway API.

    Args:
        catalog_url: URL of the dClimate STAC root-CID pointer endpoint.
        headers: Optional default headers for the request. The pointer endpoint
            lives on the IPFS gateway host, so authenticated gateways need the
            same credentials here as for ``/ipfs`` reads.
        auth: Optional ``(username, password)`` basic-auth pair for the request.

    Returns:
        str: The IPFS CID of the root STAC catalog

    Raises:
        httpx.HTTPError: If the API request fails
        KeyError: If the response doesn't contain the expected 'cid' field
    """
    # The pooled client is process-wide and gateway-agnostic, so credentials are
    # applied per-request rather than baked into the shared client.
    response = _client().get(
        catalog_url,
        timeout=30,
        headers=headers,
        auth=auth if auth is not None else httpx.USE_CLIENT_DEFAULT,
    )
    response.raise_for_status()
    data = response.json()
    return data["cid"]


def _extract_collections_from_org_link(link: pystac.Link) -> Set[str]:
    """
    Pull collection identifiers from an organization link.

    The new STAC layout stores organization-level metadata on each child link of
    the root catalog, including the collections that belong to that org grouped
    by historical/forecast buckets.
    """
    collections: Set[str] = set()
    for key, value in (link.extra_fields or {}).items():
        if not key.startswith("dclimate:collections"):
            continue
        if isinstance(value, list):
            collections.update([v for v in value if isinstance(v, str)])
    return collections


def _extract_datasets_for_collection(
    link: pystac.Link, collection_id: str
) -> List[str]:
    """
    Get dataset slugs belonging to a specific collection from an org link.

    Dataset slugs are stored as "<collection_id>/<dataset>" strings.
    """
    datasets: List[str] = []
    for slug in link.extra_fields.get("dclimate:datasets", []) or []:
        if not isinstance(slug, str):
            continue
        prefix, _, ds = slug.partition("/")
        if prefix == collection_id and ds:
            datasets.append(ds)
    return datasets


def _resolve_child_by_dclimate_id(
    parent: pystac.Catalog, child_id: str
) -> Tuple[Optional[pystac.Catalog], Optional[pystac.Link]]:
    """Resolve a catalog child by its dclimate:id extra field."""
    for link in parent.get_child_links():
        if link.extra_fields.get("dclimate:id") == child_id:
            return cast(
                pystac.Catalog, link.resolve_stac_object(root=parent).target
            ), link
    return None, None


def _resolve_child_by_collection_slug(
    parent: pystac.Catalog, collection_slug: str
) -> Tuple[Optional[pystac.Catalog], Optional[pystac.Link]]:
    """Resolve a catalog child by collection slug.

    Only supports the new layout where root children are organizations that
    declare collections via their extra fields. Uses a None-safe extractor
    to avoid TypeError when collection lists are missing.
    """
    # New layout only: root children are orgs that declare collections
    for link in parent.get_child_links():
        collections = _extract_collections_from_org_link(link)
        if collection_slug not in collections:
            continue

        org_catalog = cast(pystac.Catalog, link.resolve_stac_object(root=parent).target)
        if org_catalog is None:
            continue

        for col_link in org_catalog.get_child_links():
            if col_link.extra_fields.get("dclimate:id") == collection_slug:
                return cast(
                    pystac.Catalog,
                    col_link.resolve_stac_object(root=org_catalog).target,
                ), col_link

    return None, None


class IPFSStacIO(pystac.StacIO):
    """
    Custom StacIO implementation that resolves ipfs:// URIs via HTTP gateway.

    This allows pystac to transparently load STAC catalogs, collections, and items
    that are stored on IPFS and referenced using ipfs:// protocol URIs.
    """

    def __init__(
        self,
        gateway_url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Tuple[str, str]] = None,
    ):
        """
        Initialize the IPFS STAC I/O handler.

        Args:
            gateway_url: Base URL of the IPFS HTTP gateway (e.g., 'https://ipfs-gateway.dclimate.net')
            headers: Optional default headers applied to every gateway request.
                Required for authenticated gateways so catalog fallback does not
                fail with 401.
            auth: Optional ``(username, password)`` basic-auth pair applied to
                every gateway request.
        """
        self.gateway_url = gateway_url.rstrip("/")
        # Per-instance client so each catalog owns its pool lifecycle
        # (closed via weakref.finalize when the catalog is collected).
        # httpx.Client is thread-safe across asyncio.to_thread workers.
        # Credentials are baked into the client so every gateway read carries
        # them, mirroring the KuboCAS data path.
        self.client = httpx.Client(
            timeout=30,
            follow_redirects=True,
            headers=headers,
            auth=auth,
        )

    def read_text(self, source: str | PathLike[str], *args, **kwargs) -> str:
        """
        Read text content from a source URI.

        If the source starts with 'ipfs://', resolves it via the HTTP gateway.
        HTTP(S) sources are fetched directly by the owned HTTP client.

        Args:
            source: URI to read from (e.g., 'ipfs://bafkrei...' or 'https://...')

        Returns:
            str: The text content

        Raises:
            httpx.HTTPError: If the HTTP request fails
        """
        source_text = fspath(source)
        if source_text.startswith("ipfs://"):
            cid = source_text.replace("ipfs://", "")
            url = f"{self.gateway_url}/ipfs/{cid}"
            response = self.client.get(url, timeout=30)
            response.raise_for_status()
            return response.text

        scheme = urlsplit(source_text).scheme.lower()
        if scheme in {"http", "https"}:
            response = self.client.get(source_text)
            response.raise_for_status()
            return response.text

        unsupported_scheme = scheme or "<none>"
        raise ValueError(
            f"Unsupported STAC source scheme '{unsupported_scheme}': {source_text}"
        )

    def write_text(self, dest: str | PathLike[str], txt: str, *args, **kwargs) -> None:
        """
        Write text content is not supported for IPFS.

        Raises:
            NotImplementedError: Always, as IPFS is read-only in this context
        """
        raise NotImplementedError("Writing to IPFS is not supported via StacIO")

    def close(self) -> None:
        """Close the pooled HTTP client owned by this I/O handler."""
        self.client.close()


def load_stac_catalog(
    gateway_url: str,
    root_cid: Optional[str] = None,
    catalog_url: str = STAC_CATALOG_URL,
    *,
    headers: Optional[Dict[str, str]] = None,
    auth: Optional[Tuple[str, str]] = None,
) -> pystac.Catalog:
    """
    Load the dClimate STAC catalog from IPFS.

    Args:
        gateway_url: Base URL of the IPFS HTTP gateway
        root_cid: Optional IPFS CID of the root catalog. If None, fetches via get_root_catalog_cid()
        catalog_url: Root-CID pointer endpoint used when ``root_cid`` is omitted.
        headers: Optional default headers for every gateway/pointer request.
            Pass the same credentials used for the KuboCAS data path so catalog
            fallback works against authenticated gateways.
        auth: Optional ``(username, password)`` basic-auth pair for every
            gateway/pointer request.

    Returns:
        pystac.Catalog: The loaded STAC catalog with all links and references

    Raises:
        httpx.HTTPError: If fetching from IPFS fails
        pystac.STACError: If the catalog structure is invalid
    """
    if root_cid is None:
        root_cid = get_root_catalog_cid(catalog_url, headers=headers, auth=auth)

    # Bind the I/O handler to this catalog. Avoid pystac's process-global
    # default, because concurrent clients may use different gateways.
    stac_io = IPFSStacIO(gateway_url, headers=headers, auth=auth)

    # Load the root catalog
    catalog_uri = f"ipfs://{root_cid}"
    try:
        catalog = pystac.Catalog.from_file(catalog_uri, stac_io=stac_io)
    except BaseException:
        stac_io.close()
        raise

    # Keep the pool alive for lazy link resolution, then release it when the
    # returned catalog is no longer in use.
    weakref.finalize(catalog, stac_io.close)

    return catalog


def resolve_dataset_cid_from_stac(
    catalog: pystac.Catalog,
    collection: str,
    dataset: str,
    variant: Optional[str] = None,
    organization: Optional[str] = None,
) -> ResolvedDataset:
    """
    Resolve a dataset to its IPFS CID by querying the STAC catalog.

    Changed in 0.6: returns ResolvedDataset.

    This function navigates the STAC catalog structure to find the specific dataset variant
    and extracts the Zarr data CID from the STAC Item's assets.

    Args:
        catalog: The loaded STAC catalog
        collection: Collection ID (e.g., 'ifs', 'era5', 'aifs'). Can be prefixed
            with the organization (e.g., 'ecmwf_era5') or unprefixed when an
            organization is supplied separately.
        dataset: Dataset name (e.g., 'temperature', 'precipitation')
        variant: Optional variant name (e.g., 'single', 'ensemble'). Required for multi-variant datasets
        organization: Optional organization/agency id (e.g., 'ecmwf'). When
            provided, collection will be resolved inside this organization's
            catalog. When omitted, the organization is inferred from the root
            catalog metadata.

    Returns:
        ResolvedDataset: The IPFS CID and selected variant

    Raises:
        ValueError: If collection, dataset, or variant is not found in the catalog
    """
    # Resolve organization (new catalog layout) or fall back to legacy layout
    org_catalog: Optional[pystac.Catalog] = None
    org_link: Optional[pystac.Link] = None
    resolved_collection_id = collection

    # If an organization is provided, load it first
    if organization:
        org_catalog, org_link = _resolve_child_by_dclimate_id(catalog, organization)
        if org_catalog is None:
            raise ValueError(f"Organization '{organization}' not found in STAC catalog")

        # Accept either plain collection name or the prefixed id
        if not resolved_collection_id.startswith(f"{organization}_"):
            resolved_collection_id = f"{organization}_{resolved_collection_id}"
        collection_obj, _ = _resolve_child_by_dclimate_id(
            org_catalog, resolved_collection_id
        )
        if collection_obj is None and resolved_collection_id != collection:
            collection_obj, _ = _resolve_child_by_dclimate_id(org_catalog, collection)
            if collection_obj is not None:
                resolved_collection_id = collection
        if collection_obj is None:
            raise ValueError(
                f"Collection '{collection}' not found under organization '{organization}'"
            )
    else:
        for candidate_link in catalog.get_child_links():
            if resolved_collection_id in _extract_collections_from_org_link(
                candidate_link
            ):
                org_link = candidate_link
                break
        # First, try legacy layout where collections hang off the root catalog
        collection_obj, _ = _resolve_child_by_collection_slug(
            catalog, resolved_collection_id
        )
        if collection_obj is None:
            org_msg = (
                f" under organization '{org_link.extra_fields.get('dclimate:id')}'"
                if org_link
                else ""
            )
            raise ValueError(
                f"Collection '{collection}' not found in STAC catalog{org_msg}"
            )

    # Find the item matching dataset and variant
    items = list(collection_obj.get_items())
    known_datasets = {
        value
        for value in collection_obj.extra_fields.get("dclimate:types", []) or []
        if isinstance(value, str) and value
    }
    if org_link is not None:
        known_datasets.update(
            _extract_datasets_for_collection(org_link, resolved_collection_id)
        )
    known_datasets.update(
        property_dataset
        for item in items
        for property_dataset in [(item.properties or {}).get("dclimate:dataset_id")]
        if isinstance(property_dataset, str) and property_dataset
    )
    # Metadata can be incomplete and mention only a shorter sibling (for
    # example ``precip`` but not ``precip-daily``). The requested name is
    # nevertheless a valid parsing hint, matching the STAC-server resolver.
    known_datasets.add(dataset)

    candidates = []
    selected_item = None
    selected_variant = None
    for item in items:
        # Item IDs follow pattern: "{collection_id}-{dataset}" or "-{variant}"
        properties = item.properties or {}
        property_dataset = properties.get("dclimate:dataset_id")
        property_variant = properties.get("dclimate:variant")
        if property_dataset:
            item_dataset = property_dataset
            _, parsed_variant = _dataset_and_variant_from_item_id(
                item.id, collection_obj.id, dataset=property_dataset
            )
        elif property_variant:
            item_dataset, parsed_variant = _dataset_and_variant_from_item_id(
                item.id, collection_obj.id, variant=property_variant
            )
            if item_dataset is None:
                # The id does not encode the property variant (e.g. a bare
                # id with properties variant "default") — parse with the
                # requested-dataset hint instead.
                item_dataset, parsed_variant = _dataset_and_variant_from_item_id(
                    item.id, collection_obj.id, dataset
                )
        else:
            item_dataset, parsed_variant = _dataset_and_variant_from_known_datasets(
                item.id, collection_obj.id, known_datasets
            )
        # A bare item (no variant segment/property) is what the listing APIs
        # report as the "default" variant — keep resolve symmetric with list
        # and with the STAC-server resolver.
        item_variant = property_variant or parsed_variant or "default"

        if item_dataset != dataset:
            continue

        candidates.append((item_variant, item))

        if variant is not None and item_variant == variant:
            selected_item = item
            selected_variant = variant
            break
    if variant is not None:
        if not selected_item:
            raise ValueError(
                f"Dataset '{dataset}' with variant '{variant}' not found in collection '{collection_obj.id}'"
            )
    else:
        if not candidates:
            raise ValueError(
                f"Dataset '{dataset}' not found in collection '{collection_obj.id}'"
            )
        # If multiple variants exist and none specified, pick a sensible default
        preferred_order = ["default", "final", "finalized", "latest"]
        selected_variant, selected_item = candidates[0]
        for preferred in preferred_order:
            for cand_variant, cand_item in candidates:
                if cand_variant == preferred:
                    selected_item = cand_item
                    selected_variant = cand_variant
                    break
            else:
                continue
            break

    assert selected_variant is not None
    if "data" in selected_item.assets:
        href = selected_item.assets["data"].href
        if href.startswith("ipfs://"):
            href = href.replace("ipfs://", "")
        return ResolvedDataset(href, selected_variant)

    raise ValueError(f"Item '{selected_item.id}' does not have a 'data' asset")


def _extract_item_extents(
    item: pystac.Item,
) -> Tuple[Optional[SpatialExtent], Optional[TemporalExtent]]:
    """Extract spatial and temporal extents from a STAC item."""
    spatial: Optional[SpatialExtent] = None
    temporal: Optional[TemporalExtent] = None

    bbox = item.bbox
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        spatial = SpatialExtent(
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        )

    props = item.properties or {}
    start_dt = props.get("start_datetime") or props.get("datetime")
    end_dt = props.get("end_datetime") or props.get("datetime")
    if start_dt is not None or end_dt is not None:
        temporal = TemporalExtent(start=start_dt, end=end_dt)

    return spatial, temporal


def list_available_datasets(catalog: pystac.Catalog) -> Dict[str, Dict[str, Any]]:
    """
    List all available datasets from the STAC catalog.

    Returns a dictionary mapping collection IDs to their metadata, including
    the dataset types available in each collection. Supports both the legacy
    layout (collections as root children) and the new layout where the root
    contains organizations that own collections.

    Each dataset variant includes spatial and temporal extents extracted from
    the STAC items (bbox and start_datetime/end_datetime properties).

    Args:
        catalog: The loaded STAC catalog

    Returns:
        dict: Dictionary keyed by collection id with at least:
            - id: Collection id (may include organization prefix)
            - title: Collection title
            - types: Dataset names within the collection
            - organization: Owning organization id (None for legacy layout)
            - category: Optional category tag (e.g., historical, forecast)
            - variants: List of dicts with variant, dataset, cid, spatial_extent, temporal_extent
    """
    result: Dict[str, Dict[str, Any]] = {}

    for link in catalog.get_child_links():
        child_id = link.extra_fields.get("dclimate:id")
        if not child_id:
            continue

        # New layout: root children are organizations with nested collections
        is_org = link.extra_fields.get("dclimate:type") == "organization" or bool(
            _extract_collections_from_org_link(link)
        )

        if is_org:
            org_id = child_id
            org_title = link.title or org_id
            org_catalog = cast(
                pystac.Catalog, link.resolve_stac_object(root=catalog).target
            )

            # Map collection -> category (historical/forecast/etc.)
            collection_categories: Dict[str, str] = {}
            for key, value in (link.extra_fields or {}).items():
                if not key.startswith("dclimate:collections:"):
                    continue
                category = key.split(":", 2)[-1]
                if isinstance(value, list):
                    for coll in value:
                        if isinstance(coll, str):
                            collection_categories[coll] = category

            # Derive dataset types from the dataset slugs on the org link
            datasets_by_collection: Dict[str, Set[str]] = {}
            for slug in link.extra_fields.get("dclimate:datasets", []) or []:
                if not isinstance(slug, str) or "/" not in slug:
                    continue
                coll_id, ds = slug.split("/", 1)
                datasets_by_collection.setdefault(coll_id, set()).add(ds)

            # Walk each collection hanging off this org
            for col_link in org_catalog.get_child_links():
                collection_id = col_link.extra_fields.get("dclimate:id")
                if not collection_id:
                    continue

                types = sorted(datasets_by_collection.get(collection_id, []))
                if not types:
                    types = col_link.extra_fields.get("dclimate:types", [])

                entry: Dict[str, Any] = {
                    "id": collection_id,
                    "title": col_link.title or collection_id,
                    "types": types,
                    "organization": org_id,
                    "organization_title": org_title,
                    "variants": [],
                }

                if collection_id in collection_categories:
                    entry["category"] = collection_categories[collection_id]

                # Resolve items to extract per-variant extents
                try:
                    col_catalog = cast(
                        pystac.Catalog,
                        col_link.resolve_stac_object(root=org_catalog).target,
                    )
                    if col_catalog is not None:
                        items = list(col_catalog.get_items())
                        known_datasets = set(types)
                        known_datasets.update(
                            property_dataset
                            for item in items
                            for property_dataset in [
                                (item.properties or {}).get("dclimate:dataset_id")
                            ]
                            if isinstance(property_dataset, str) and property_dataset
                        )
                        for item in items:
                            # Prefer explicit dclimate:* properties (like the
                            # server lister); fall back to the shared
                            # hyphen-aware id parsing, which is ambiguous for
                            # hyphenated dataset ids without a dataset hint.
                            # Bare items are the "default" variant, matching
                            # the resolvers.
                            props = item.properties or {}
                            property_dataset = props.get("dclimate:dataset_id")
                            property_variant = props.get("dclimate:variant")
                            if property_dataset:
                                parsed_dataset, parsed_variant = (
                                    _dataset_and_variant_from_item_id(
                                        item.id,
                                        collection_id,
                                        dataset=property_dataset,
                                    )
                                )
                            elif property_variant:
                                parsed_dataset, parsed_variant = (
                                    _dataset_and_variant_from_item_id(
                                        item.id,
                                        collection_id,
                                        variant=property_variant,
                                    )
                                )
                                if parsed_dataset is None:
                                    parsed_dataset, parsed_variant = (
                                        _dataset_and_variant_from_known_datasets(
                                            item.id,
                                            collection_id,
                                            known_datasets,
                                        )
                                    )
                            else:
                                parsed_dataset, parsed_variant = (
                                    _dataset_and_variant_from_known_datasets(
                                        item.id,
                                        collection_id,
                                        known_datasets,
                                    )
                                )
                            item_dataset = property_dataset or parsed_dataset
                            if item_dataset is None:
                                continue
                            item_variant = (
                                property_variant or parsed_variant or "default"
                            )

                            cid: Optional[str] = None
                            if "data" in item.assets:
                                href = item.assets["data"].href
                                cid = (
                                    href.replace("ipfs://", "")
                                    if href.startswith("ipfs://")
                                    else href
                                )

                            spatial, temporal = _extract_item_extents(item)
                            variant_entry: Dict[str, Any] = {
                                "dataset": item_dataset,
                                "variant": item_variant,
                            }
                            if cid:
                                variant_entry["cid"] = cid
                            if spatial:
                                variant_entry["spatial_extent"] = spatial
                            if temporal:
                                variant_entry["temporal_extent"] = temporal
                            entry["variants"].append(variant_entry)
                            known_datasets.add(item_dataset)
                        entry["types"] = sorted(known_datasets)
                except Exception:
                    logger.debug(
                        "Could not resolve items for collection %s",
                        collection_id,
                        exc_info=True,
                    )

                result[collection_id] = entry
        else:
            # Legacy layout: root children are collections
            types = link.extra_fields.get("dclimate:types", [])
            result[child_id] = {
                "id": child_id,
                "title": link.title or child_id,
                "types": types,
                "organization": None,
                "variants": [],
            }

    return result
