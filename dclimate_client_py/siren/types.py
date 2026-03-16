"""
Siren API types and authentication strategies.

Supports two auth modes:
- API key + account ID (traditional Bearer token)
- x402 pay-per-request via wallet signature
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, Union


# ---------------------------------------------------------------------------
# EVM signer interface (mirrors @x402/evm ClientEvmSigner)
# Defined here so the SDK has zero hard dependency on x402
# ---------------------------------------------------------------------------


class EvmSigner(Protocol):
    """Protocol matching the x402 ClientEvmSigner interface."""

    @property
    def address(self) -> str: ...

    async def sign_typed_data(
        self,
        *,
        domain: dict[str, Any],
        types: dict[str, Any],
        primary_type: str,
        message: dict[str, Any],
    ) -> str: ...

    async def read_contract(
        self,
        *,
        address: str,
        abi: list[Any],
        function_name: str,
        args: list[Any] | None = None,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Auth strategies (discriminated union via dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SirenApiKeyAuth:
    """API key authentication. Falls back to env vars if fields are omitted."""

    type: Literal["api_key"] = "api_key"
    api_key: Optional[str] = None
    """Falls back to SIREN_API_KEY env var if omitted."""
    account_id: Optional[str] = None
    """Falls back to SIREN_ACCOUNT_ID env var if omitted."""


@dataclass(frozen=True)
class SirenX402Auth:
    """x402 pay-per-request authentication via wallet signature."""

    signer: EvmSigner = field(repr=False)
    type: Literal["x402"] = "x402"
    network: str = "base"
    """Chain network identifier (default: 'base')."""
    facilitator_url: Optional[str] = None
    """x402 facilitator URL (uses protocol default if omitted)."""


SirenAuth = Union[SirenApiKeyAuth, SirenX402Auth]


# ---------------------------------------------------------------------------
# Client options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SirenOptions:
    """Configuration for the Siren client."""

    auth: SirenAuth
    base_url: Optional[str] = None
    """Base URL for API-key authenticated requests (default: production Siren API)."""
    x402_base_url: Optional[str] = None
    """Base URL for x402-authenticated requests (separate service, TBD)."""


# ---------------------------------------------------------------------------
# Query types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SirenMetricQuery:
    """Query parameters for fetching metric data."""

    region_id: str
    metric: str
    start_date: Union[str, datetime.date]
    end_date: Union[str, datetime.date]


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


@dataclass
class SirenMetricDataPoint:
    """A single data point in a metric time series."""

    date: str
    value: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SirenCountry:
    """Country metadata from the Siren API."""

    id: str
    name: str
    code: str


@dataclass
class SirenRegion:
    """A geographic region from the Siren API."""

    id: str
    name: str
    region_type: str
    country_id: str
    commodity_code: str
    geo_json: str
    created_at: str
    historical_fetch_enabled: bool
    country: SirenCountry
    internal_code: Optional[str] = None
    account_id: Optional[str] = None
    extra_info: Optional[str] = None


@dataclass
class SirenRegionsResponse:
    """Paginated response from the regions endpoint."""

    items: list[SirenRegion]
    limit: int
    offset: int
    total: int
