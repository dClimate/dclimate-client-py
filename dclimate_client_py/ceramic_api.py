"""
Ceramic dataset version and provenance helpers.

This module provides a small Python interface for the dClimate Ceramic API so
consumers can list dataset versions, select exact historical releases, resolve
citations, and build gateway URLs for reproducible access workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlsplit, urlunsplit

import httpx


HYDROGEN_CERAMIC_API_BASE_URL = "https://hydrogen.dclimate.net/api"
TRITIUM_CERAMIC_API_BASE_URL = "https://tritium.dclimate.net/api"
# Kept for callers that explicitly rely on the legacy default. STAC-aware code
# should follow the complete dclimate:versions_api URL instead.
DEFAULT_CERAMIC_API_BASE_URL = HYDROGEN_CERAMIC_API_BASE_URL
DEFAULT_IPFS_GATEWAY_BASE_URL = "https://ipfs-gateway.dclimate.net"


@dataclass(frozen=True)
class VerificationInfo:
    """Verification metadata derived from Ceramic anchoring state."""

    anchor_status: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_payload(cls, payload: Optional[Dict[str, Any]]) -> "VerificationInfo":
        payload = payload or {}
        details = dict(payload)
        anchor_status = details.pop("anchorStatus", None)
        return cls(anchor_status=anchor_status, details=details)


@dataclass(frozen=True)
class DatasetVersion:
    """One versioned dataset snapshot returned by the Ceramic API."""

    dataset: str
    cid: str
    old_cid: Optional[str] = None
    timestamp: Optional[int] = None
    stream_id: Optional[str] = None
    commit_id: Optional[str] = None
    controller_did: Optional[str] = None
    published_at: Optional[str] = None
    version_label: Optional[str] = None
    release_class: Optional[str] = None
    is_citable: Optional[bool] = None
    retention_class: Optional[str] = None
    verification: VerificationInfo = field(default_factory=VerificationInfo)

    @classmethod
    def from_api_payload(cls, payload: Dict[str, Any]) -> "DatasetVersion":
        return cls(
            dataset=payload["dataset"],
            cid=payload["cid"],
            old_cid=payload.get("oldCid"),
            timestamp=payload.get("timestamp"),
            stream_id=payload.get("streamId"),
            commit_id=payload.get("commitId"),
            controller_did=payload.get("controllerDid"),
            published_at=payload.get("publishedAt"),
            version_label=payload.get("versionLabel"),
            release_class=payload.get("releaseClass"),
            is_citable=payload.get("isCitable"),
            retention_class=payload.get("retentionClass"),
            verification=VerificationInfo.from_api_payload(payload.get("verification")),
        )


@dataclass(frozen=True)
class DatasetVersionListing:
    """Version-history response for one dataset."""

    dataset: str
    stream_id: Optional[str]
    versions: List[DatasetVersion]


@dataclass(frozen=True)
class CitationInfo:
    """Citation payload for one dataset release."""

    dataset: str
    stream_id: Optional[str]
    commit_id: Optional[str]
    cid: str
    published_at: Optional[str]
    version_label: Optional[str]
    is_citable: Optional[bool]
    retention_class: Optional[str]
    citation: str

    @classmethod
    def from_api_payload(cls, payload: Dict[str, Any]) -> "CitationInfo":
        return cls(
            dataset=payload["dataset"],
            stream_id=payload.get("streamId"),
            commit_id=payload.get("commitId"),
            cid=payload["cid"],
            published_at=payload.get("publishedAt"),
            version_label=payload.get("versionLabel"),
            is_citable=payload.get("isCitable"),
            retention_class=payload.get("retentionClass"),
            citation=payload["citation"],
        )


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _dataset_path(dataset: str) -> str:
    return quote(dataset, safe="")


def _request_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    session: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    if session is not None:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def _encode_bool(value: Optional[bool]) -> Optional[str]:
    if value is None:
        return None
    return str(value).lower()


def _append_url_path(url: str, component: str) -> str:
    """Append one encoded path component without disturbing URL query data."""
    parts = urlsplit(url)
    path = f"{parts.path.rstrip('/')}/{quote(component, safe='')}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def list_versions_from_url(
    versions_url: str,
    anchored: Optional[bool] = None,
    is_citable: Optional[bool] = None,
    version_label: Optional[str] = None,
    session: Optional[httpx.Client] = None,
) -> DatasetVersionListing:
    """List versions using the complete service URL discovered from STAC."""
    params: Dict[str, Any] = {}
    encoded_anchored = _encode_bool(anchored)
    encoded_is_citable = _encode_bool(is_citable)
    if encoded_anchored is not None:
        params["anchored"] = encoded_anchored
    if encoded_is_citable is not None:
        params["isCitable"] = encoded_is_citable
    if version_label is not None:
        params["versionLabel"] = version_label

    payload = _request_json(
        versions_url,
        params=params or None,
        session=session,
    )
    versions = [
        DatasetVersion.from_api_payload(item) for item in payload.get("versions", [])
    ]
    return DatasetVersionListing(
        dataset=payload["dataset"],
        stream_id=payload.get("streamId"),
        versions=versions,
    )


def get_exact_version_from_url(
    versions_url: str,
    commit_id: str,
    session: Optional[httpx.Client] = None,
) -> DatasetVersion:
    """Resolve an exact release from a STAC-discovered versions URL."""
    payload = _request_json(
        _append_url_path(versions_url, commit_id),
        session=session,
    )
    return DatasetVersion.from_api_payload(payload)


def get_citation_from_url(
    citation_url: str,
    session: Optional[httpx.Client] = None,
) -> CitationInfo:
    """Fetch citation metadata using the complete URL discovered from STAC."""
    return CitationInfo.from_api_payload(_request_json(citation_url, session=session))


def list_versions(
    dataset: str,
    base_url: str = DEFAULT_CERAMIC_API_BASE_URL,
    anchored: Optional[bool] = None,
    is_citable: Optional[bool] = None,
    version_label: Optional[str] = None,
    session: Optional[httpx.Client] = None,
) -> DatasetVersionListing:
    """
    List known versions for one dataset.
    """
    dataset_path = _dataset_path(dataset)
    return list_versions_from_url(
        f"{_normalize_base_url(base_url)}/datasets/{dataset_path}/versions",
        anchored=anchored,
        is_citable=is_citable,
        version_label=version_label,
        session=session,
    )


def get_exact_version(
    dataset: str,
    commit_id: str,
    base_url: str = DEFAULT_CERAMIC_API_BASE_URL,
    session: Optional[httpx.Client] = None,
) -> DatasetVersion:
    """
    Resolve one exact dataset version by commit id.
    """
    dataset_path = _dataset_path(dataset)
    commit_path = quote(commit_id, safe="")
    payload = _request_json(
        f"{_normalize_base_url(base_url)}/datasets/{dataset_path}/versions/{commit_path}",
        session=session,
    )
    return DatasetVersion.from_api_payload(payload)


def get_latest_metadata(
    dataset: str,
    base_url: str = DEFAULT_CERAMIC_API_BASE_URL,
    session: Optional[httpx.Client] = None,
) -> DatasetVersion:
    """
    Get the latest dataset metadata from the Ceramic API.
    """
    dataset_path = _dataset_path(dataset)
    payload = _request_json(
        f"{_normalize_base_url(base_url)}/datasets/{dataset_path}",
        session=session,
    )
    return DatasetVersion.from_api_payload(payload)


def get_citation(
    dataset: str,
    commit_id: Optional[str] = None,
    base_url: str = DEFAULT_CERAMIC_API_BASE_URL,
    session: Optional[httpx.Client] = None,
) -> CitationInfo:
    """
    Fetch citation metadata for a dataset release.
    """
    params = {"commitId": commit_id} if commit_id else None
    dataset_path = _dataset_path(dataset)
    payload = _request_json(
        f"{_normalize_base_url(base_url)}/datasets/{dataset_path}/citation",
        params=params,
        session=session,
    )
    return CitationInfo.from_api_payload(payload)


def filter_anchored_versions(versions: List[DatasetVersion]) -> List[DatasetVersion]:
    """Return only anchored versions from a version list."""
    return [
        version
        for version in versions
        if version.verification.anchor_status == "anchored"
    ]


def get_latest_anchored_version(
    dataset: str,
    base_url: str = DEFAULT_CERAMIC_API_BASE_URL,
    session: Optional[httpx.Client] = None,
) -> DatasetVersion:
    """
    Return the most recent anchored version for one dataset.
    """
    listing = list_versions(
        dataset=dataset,
        base_url=base_url,
        anchored=True,
        session=session,
    )
    anchored_versions = filter_anchored_versions(listing.versions)
    if not anchored_versions:
        raise ValueError(f"No anchored versions found for dataset '{dataset}'")
    return anchored_versions[-1]


def build_gateway_url(
    cid: str,
    gateway_base: str = DEFAULT_IPFS_GATEWAY_BASE_URL,
) -> str:
    """Build a gateway URL for one IPFS CID."""
    return f"{gateway_base.rstrip('/')}/ipfs/{cid}"
