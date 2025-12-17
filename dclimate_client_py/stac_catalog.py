"""
STAC Catalog Integration for dClimate

This module provides integration with STAC (SpatioTemporal Asset Catalog) format
for discovering and accessing dClimate datasets stored on IPFS.
"""

from typing import Optional, Dict, List, Set, Tuple
import requests
import pystac


def get_root_catalog_cid() -> str:
    """
    Get the root STAC catalog CID.

    Fetches the latest catalog CID from the dClimate IPFS gateway API.

    Returns:
        str: The IPFS CID of the root STAC catalog

    Raises:
        requests.HTTPError: If the API request fails
        KeyError: If the response doesn't contain the expected 'cid' field
    """
    url = "https://ipfs-gateway.dclimate.net/stac"
    response = requests.get(url)
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
            return link.resolve_stac_object(root=parent).target, link
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

        org_catalog = link.resolve_stac_object(root=parent).target
        if org_catalog is None:
            continue

        for col_link in org_catalog.get_child_links():
            if col_link.extra_fields.get("dclimate:id") == collection_slug:
                return col_link.resolve_stac_object(root=org_catalog).target, col_link

    return None, None

class IPFSStacIO(pystac.StacIO):
    """
    Custom StacIO implementation that resolves ipfs:// URIs via HTTP gateway.

    This allows pystac to transparently load STAC catalogs, collections, and items
    that are stored on IPFS and referenced using ipfs:// protocol URIs.
    """

    def __init__(self, gateway_url: str):
        """
        Initialize the IPFS STAC I/O handler.

        Args:
            gateway_url: Base URL of the IPFS HTTP gateway (e.g., 'https://ipfs-gateway.dclimate.net')
        """
        self.gateway_url = gateway_url.rstrip('/')

    def read_text(self, source: str, *args, **kwargs) -> str:
        """
        Read text content from a source URI.

        If the source starts with 'ipfs://', resolves it via the HTTP gateway.
        Otherwise, delegates to the default StacIO implementation.

        Args:
            source: URI to read from (e.g., 'ipfs://bafkrei...' or 'https://...')

        Returns:
            str: The text content

        Raises:
            requests.HTTPError: If the HTTP request fails
        """
        if source.startswith("ipfs://"):
            cid = source.replace("ipfs://", "")
            url = f"{self.gateway_url}/ipfs/{cid}"
            response = requests.get(url)
            response.raise_for_status()
            return response.text

        # Fall back to default behavior for HTTP/HTTPS URLs
        return super().read_text(source, *args, **kwargs)

    def write_text(self, dest: str, txt: str, *args, **kwargs) -> None:
        """
        Write text content is not supported for IPFS.

        Raises:
            NotImplementedError: Always, as IPFS is read-only in this context
        """
        raise NotImplementedError("Writing to IPFS is not supported via StacIO")


def load_stac_catalog(
    gateway_url: str,
    root_cid: Optional[str] = None
) -> pystac.Catalog:
    """
    Load the dClimate STAC catalog from IPFS.

    Args:
        gateway_url: Base URL of the IPFS HTTP gateway
        root_cid: Optional IPFS CID of the root catalog. If None, fetches via get_root_catalog_cid()

    Returns:
        pystac.Catalog: The loaded STAC catalog with all links and references

    Raises:
        requests.HTTPError: If fetching from IPFS fails
        pystac.STACError: If the catalog structure is invalid
    """
    if root_cid is None:
        root_cid = get_root_catalog_cid()

    # Set up custom IPFS I/O handler
    stac_io = IPFSStacIO(gateway_url)
    pystac.StacIO.set_default(lambda: stac_io)

    # Load the root catalog
    catalog_uri = f"ipfs://{root_cid}"
    catalog = pystac.Catalog.from_file(catalog_uri)

    return catalog


def resolve_dataset_cid_from_stac(
    catalog: pystac.Catalog,
    collection: str,
    dataset: str,
    variant: Optional[str] = None,
    organization: Optional[str] = None,
) -> str:
    """
    Resolve a dataset to its IPFS CID by querying the STAC catalog.

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
        str: The IPFS CID of the Zarr dataset (without 'ipfs://' prefix)

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
        if collection_obj is None:
            raise ValueError(
                f"Collection '{collection}' not found under organization '{organization}'"
            )
    else:
        # First, try legacy layout where collections hang off the root catalog
        collection_obj, _ = _resolve_child_by_collection_slug(catalog, resolved_collection_id)
        # # Otherwise, infer the organization by scanning org metadata on the root catalog
        # if collection_obj is None:
        #     for candidate_link in catalog.get_child_links():
        #         print(candidate_link)
        #         org_id = candidate_link.extra_fields.get("dclimate:id")
        #         print(f"Organization ID: {org_id}")
        #         if not org_id:
        #             continue

        #         declared_collections = _extract_collections_from_org_link(candidate_link)
        #         dataset_collections = {
        #             slug.split("/", 1)[0]
        #             for slug in candidate_link.extra_fields.get("dclimate:datasets", [])
        #             if isinstance(slug, str) and "/" in slug
        #         }
        #         declared_collections.update(dataset_collections)

        #         if resolved_collection_id in declared_collections:
        #             org_link = candidate_link
        #             org_catalog = candidate_link.resolve_stac_object(root=catalog).target
        #             break

        #         prefixed = f"{org_id}_{collection}"
        #         if prefixed in declared_collections:
        #             resolved_collection_id = prefixed
        #             org_link = candidate_link
        #             org_catalog = candidate_link.resolve_stac_object(root=catalog).target
        #             break

        # if collection_obj is None and org_catalog:
        #     collection_obj, _ = _resolve_child_by_dclimate_id(
        #         org_catalog, resolved_collection_id
        #     )

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
    candidates = []
    for item in collection_obj.get_items():
        # Item IDs follow pattern: "{collection_id}-{dataset}" or "-{variant}"
        item_id = item.id
        prefix = f"{collection_obj.id}-"
        remainder = item_id[len(prefix):] if item_id.startswith(prefix) else item_id
        parts = remainder.split("-")
        item_dataset = parts[0] if parts else remainder
        item_variant = parts[1] if len(parts) > 1 else None

        if item_dataset != dataset:
            continue

        candidates.append((item_variant, item))

        if variant is not None and item_variant == variant:
            selected_item = item
            break
    else:
        selected_item = None

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
        preferred_order = ["default", "final", "finalized", "latest", None]
        selected_item = candidates[0][1]
        for preferred in preferred_order:
            for cand_variant, cand_item in candidates:
                if cand_variant == preferred:
                    selected_item = cand_item
                    break
            else:
                continue
            break

    if "data" in selected_item.assets:
        href = selected_item.assets["data"].href
        if href.startswith("ipfs://"):
            return href.replace("ipfs://", "")
        return href

    raise ValueError(f"Item '{selected_item.id}' does not have a 'data' asset")


def list_available_datasets(catalog: pystac.Catalog) -> Dict[str, Dict[str, any]]:
    """
    List all available datasets from the STAC catalog.

    Returns a dictionary mapping collection IDs to their metadata, including
    the dataset types available in each collection. Supports both the legacy
    layout (collections as root children) and the new layout where the root
    contains organizations that own collections.

    Args:
        catalog: The loaded STAC catalog

    Returns:
        dict: Dictionary keyed by collection id with at least:
            - id: Collection id (may include organization prefix)
            - title: Collection title
            - types: Dataset names within the collection
            - organization: Owning organization id (None for legacy layout)
            - category: Optional category tag (e.g., historical, forecast)
    """
    result: Dict[str, Dict[str, any]] = {}

    for link in catalog.get_child_links():
        child_id = link.extra_fields.get("dclimate:id")
        if not child_id:
            continue

        # New layout: root children are organizations with nested collections
        is_org = (
            link.extra_fields.get("dclimate:type") == "organization"
            or bool(_extract_collections_from_org_link(link))
        )

        if is_org:
            org_id = child_id
            org_title = link.title or org_id
            org_catalog = link.resolve_stac_object(root=catalog).target

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

                result[collection_id] = {
                    "id": collection_id,
                    "title": col_link.title or collection_id,
                    "types": types,
                    "organization": org_id,
                    "organization_title": org_title,
                }

                if collection_id in collection_categories:
                    result[collection_id]["category"] = collection_categories[collection_id]
        else:
            # Legacy layout: root children are collections
            types = link.extra_fields.get("dclimate:types", [])
            result[child_id] = {
                "id": child_id,
                "title": link.title or child_id,
                "types": types,
                "organization": None,
            }

    return result
