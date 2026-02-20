"""
Siren REST API client supporting API-key and x402 payment authentication.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Callable, Coroutine, Optional

import httpx

from ..dclimate_zarr_errors import SirenApiError, X402NotInstalledError, X402PaymentError
from .types import (
    SirenAuth,
    SirenApiKeyAuth,
    SirenMetricDataPoint,
    SirenMetricQuery,
    SirenOptions,
    SirenRegion,
    SirenRegionsResponse,
    SirenCountry,
)

DEFAULT_SIREN_API_URL = "https://production-api-siren.dclimate.net/api"


def _parse_metric_response(
    body: Any,
    metric: str,
) -> list[SirenMetricDataPoint]:
    """
    Parse the Siren metric-data response format.

    The API returns: ``{"metric_name": {"2026-01-01": 0.5, "2026-01-02": 1.2, ...}}``
    We flatten this into a list of SirenMetricDataPoint objects.
    """
    if isinstance(body, list):
        return [SirenMetricDataPoint(date=item["date"], value=item["value"]) for item in body]

    if not isinstance(body, dict):
        return []

    # Try the exact metric key first, then fall back to the first key
    time_series = body.get(metric)
    if time_series is None and body:
        time_series = next(iter(body.values()))

    if not isinstance(time_series, dict):
        return []

    return [SirenMetricDataPoint(date=date, value=value) for date, value in time_series.items()]


def _format_date(date: str | datetime.date) -> str:
    """Format a date as YYYY-MM-DD string."""
    if isinstance(date, str):
        return date
    return date.isoformat()


def _is_api_key_auth(auth: SirenAuth) -> bool:
    return auth.type == "api_key"


class SirenClient:
    """
    Client for the Siren REST API.

    Supports two authentication modes:
    - API key + account ID (Bearer token)
    - x402 pay-per-request via wallet signature
    """

    def __init__(self, options: SirenOptions) -> None:
        # Resolve API key auth from env vars if not provided directly
        if isinstance(options.auth, SirenApiKeyAuth):
            api_key = options.auth.api_key or os.environ.get("SIREN_API_KEY")
            account_id = options.auth.account_id or os.environ.get("SIREN_ACCOUNT_ID")
            if not api_key:
                raise SirenApiError(
                    "Siren API key is required. Pass it as api_key or set the "
                    "SIREN_API_KEY environment variable."
                )
            if not account_id:
                raise SirenApiError(
                    "Siren account ID is required. Pass it as account_id or set the "
                    "SIREN_ACCOUNT_ID environment variable."
                )
            self._auth: SirenAuth = SirenApiKeyAuth(
                api_key=api_key, account_id=account_id
            )
        else:
            self._auth = options.auth

        self._base_url = options.base_url or DEFAULT_SIREN_API_URL
        self._x402_base_url = options.x402_base_url
        self._x402_fetch: Optional[Any] = None

    async def get_metric_data(
        self, query: SirenMetricQuery
    ) -> list[SirenMetricDataPoint]:
        """Fetch metric data for a region over a date range."""
        start_date = _format_date(query.start_date)
        end_date = _format_date(query.end_date)

        if _is_api_key_auth(self._auth):
            assert isinstance(self._auth, SirenApiKeyAuth)
            url = (
                f"{self._base_url}/metric-data-multiple/"
                f"{self._auth.account_id}/{query.region_id}/"
                f"{query.metric}/{start_date}/{end_date}"
            )
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._auth.api_key}",
                        "Content-Type": "application/json",
                    },
                )
            if response.status_code >= 400:
                raise SirenApiError(
                    f"Siren API error ({response.status_code}): {response.reason_phrase}"
                )
            body = response.json()
            return _parse_metric_response(body, query.metric)

        # x402 auth
        wrapped_fetch = await self._get_x402_fetch()
        api_base = self._x402_base_url or self._base_url
        url = f"{api_base}/metric-data/{query.region_id}/{query.metric}/{start_date}/{end_date}"
        response = await wrapped_fetch(url, method="GET")
        if response.status_code >= 400:
            raise X402PaymentError(
                f"Siren x402 request failed ({response.status_code}): {response.reason_phrase}"
            )
        body = response.json()
        return _parse_metric_response(body, query.metric)

    async def list_regions(self) -> list[SirenRegion]:
        """List available regions. Free endpoint - no payment required."""
        if _is_api_key_auth(self._auth):
            assert isinstance(self._auth, SirenApiKeyAuth)
            url = f"{self._base_url}/custom-regions/{self._auth.account_id}/custom"
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._auth.api_key}",
                        "Content-Type": "application/json",
                    },
                )
            if response.status_code >= 400:
                raise SirenApiError(
                    f"Siren API error ({response.status_code}): {response.reason_phrase}"
                )
            data = response.json()
            return _parse_regions_response(data)

        # x402 auth - listRegions is free but uses x402 base URL
        wrapped_fetch = await self._get_x402_fetch()
        api_base = self._x402_base_url or self._base_url
        url = f"{api_base}/regions"
        response = await wrapped_fetch(url, method="GET")
        if response.status_code >= 400:
            raise SirenApiError(
                f"Siren API error ({response.status_code}): {response.reason_phrase}"
            )
        data = response.json()
        return _parse_regions_response(data)

    async def _get_x402_fetch(self) -> Any:
        """
        Lazily initialize the x402-wrapped fetch function.
        Dynamically imports x402 packages so they remain optional.
        """
        if self._x402_fetch is not None:
            return self._x402_fetch

        if not hasattr(self._auth, "signer"):
            raise RuntimeError("x402 fetch requested but auth is not x402")

        try:
            from x402.fetch import wrap_fetch_with_payment  # type: ignore[import-not-found]
        except ImportError:
            raise X402NotInstalledError(
                "x402 auth requires the x402 package. Install it: pip install x402"
            )

        try:
            from x402.core import X402Client  # type: ignore[import-not-found]
        except ImportError:
            raise X402NotInstalledError(
                "x402 auth requires the x402 package. Install it: pip install x402"
            )

        try:
            from x402.evm import register_evm_schemes  # type: ignore[import-not-found]
        except ImportError:
            raise X402NotInstalledError(
                "x402 auth requires the x402 package. Install it: pip install x402"
            )

        # Build x402 client with EVM scheme registered for the signer
        client = X402Client()
        network = self._auth.network  # type: ignore[attr-defined]
        register_evm_schemes(client, self._auth.signer, network)  # type: ignore[attr-defined]

        self._x402_fetch = wrap_fetch_with_payment(httpx.AsyncClient(), client)
        return self._x402_fetch


def _parse_regions_response(data: dict[str, Any]) -> list[SirenRegion]:
    """Parse the regions API response into SirenRegion objects."""
    items = data.get("items", [])
    return [
        SirenRegion(
            id=item["id"],
            name=item["name"],
            internal_code=item.get("internal_code"),
            region_type=item.get("region_type", ""),
            account_id=item.get("account_id"),
            country_id=item.get("country_id", ""),
            commodity_code=item.get("commodity_code", ""),
            geo_json=item.get("geo_json", ""),
            extra_info=item.get("extra_info"),
            created_at=item.get("created_at", ""),
            historical_fetch_enabled=item.get("historical_fetch_enabled", False),
            country=SirenCountry(
                id=item.get("country", {}).get("id", ""),
                name=item.get("country", {}).get("name", ""),
                code=item.get("country", {}).get("code", ""),
            ),
        )
        for item in items
    ]
