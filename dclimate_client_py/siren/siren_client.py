"""
Siren REST API client supporting API-key and x402 payment authentication.
"""

from __future__ import annotations

import datetime
import inspect
import os
from typing import Any, Optional, TypeGuard
from urllib.parse import quote

import httpx

from ..dclimate_zarr_errors import SirenApiError, X402NotInstalledError, X402PaymentError
from .types import (
    SirenAuth,
    SirenApiKeyAuth,
    SirenCountry,
    SirenMetricDataPoint,
    SirenMetricQuery,
    SirenOptions,
    SirenRegion,
    SirenX402Auth,
)

DEFAULT_SIREN_API_URL = "https://production-api-siren.dclimate.net/api"
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0


def _quote_path_segment(value: str) -> str:
    """URL-encode a dynamic path segment."""
    return quote(value, safe="")


def _parse_metric_list_item(item: Any) -> SirenMetricDataPoint:
    if not isinstance(item, dict):
        raise SirenApiError("Unexpected Siren metric list format: items must be objects.")

    if "date" not in item or "value" not in item:
        raise SirenApiError(
            "Unexpected Siren metric list format: each item must include 'date' and 'value'."
        )

    extra = {k: v for k, v in item.items() if k not in {"date", "value"}}
    try:
        value = float(item["value"])
    except (TypeError, ValueError) as exc:
        raise SirenApiError(
            "Unexpected Siren metric list format: 'value' must be numeric."
        ) from exc

    return SirenMetricDataPoint(date=str(item["date"]), value=value, extra=extra)


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
        return [_parse_metric_list_item(item) for item in body]

    if not isinstance(body, dict):
        raise SirenApiError(
            "Unexpected Siren metric response format: expected an object or array."
        )

    if metric not in body:
        available_metrics = [str(key) for key in body.keys()]
        if available_metrics:
            metrics_preview = ", ".join(available_metrics[:5])
            if len(available_metrics) > 5:
                metrics_preview = f"{metrics_preview}, ..."
            raise SirenApiError(
                f"Siren API response missing requested metric '{metric}'. "
                f"Available metrics: {metrics_preview}."
            )
        raise SirenApiError(
            f"Siren API response missing requested metric '{metric}' and returned no metrics."
        )

    time_series = body[metric]

    if not isinstance(time_series, dict):
        raise SirenApiError(
            f"Unexpected Siren metric response format for '{metric}': expected an object keyed by date."
        )

    points: list[SirenMetricDataPoint] = []
    for date, value in time_series.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise SirenApiError(
                f"Unexpected Siren metric value type for date '{date}': expected numeric value."
            ) from exc
        points.append(SirenMetricDataPoint(date=str(date), value=numeric_value))
    return points


def _format_date(date: str | datetime.date) -> str:
    """Format a date as YYYY-MM-DD string."""
    if isinstance(date, str):
        return date
    return date.isoformat()


def _is_api_key_auth(auth: SirenAuth) -> TypeGuard[SirenApiKeyAuth]:
    return isinstance(auth, SirenApiKeyAuth)


def _is_x402_auth(auth: SirenAuth) -> TypeGuard[SirenX402Auth]:
    return isinstance(auth, SirenX402Auth)


def _parse_region(item: Any) -> SirenRegion:
    if not isinstance(item, dict):
        raise SirenApiError("Unexpected Siren regions response format: each item must be an object.")

    country_data = item.get("country", {})
    if not isinstance(country_data, dict):
        country_data = {}

    region_id = item.get("id")
    region_name = item.get("name")
    if not isinstance(region_id, str) or not region_id:
        raise SirenApiError("Unexpected Siren region item format: missing non-empty 'id'.")
    if not isinstance(region_name, str) or not region_name:
        raise SirenApiError("Unexpected Siren region item format: missing non-empty 'name'.")

    return SirenRegion(
        id=region_id,
        name=region_name,
        internal_code=item.get("internal_code"),
        region_type=str(item.get("region_type", "")),
        account_id=item.get("account_id"),
        country_id=str(item.get("country_id", "")),
        commodity_code=str(item.get("commodity_code", "")),
        geo_json=str(item.get("geo_json", "")),
        extra_info=item.get("extra_info"),
        created_at=str(item.get("created_at", "")),
        historical_fetch_enabled=bool(item.get("historical_fetch_enabled", False)),
        country=SirenCountry(
            id=str(country_data.get("id", "")),
            name=str(country_data.get("name", "")),
            code=str(country_data.get("code", "")),
        ),
    )


def _parse_regions_response(
    data: Any,
) -> list[SirenRegion]:
    """Parse regions API response into typed region objects."""
    if isinstance(data, dict):
        items_data = data.get("items", [])
    elif isinstance(data, list):
        items_data = data
    else:
        raise SirenApiError("Unexpected Siren regions response format: expected an object or array.")

    if not isinstance(items_data, list):
        raise SirenApiError("Unexpected Siren regions response format: 'items' must be an array.")

    return [_parse_region(item) for item in items_data]


def _build_x402_client(x402_client_class: Any, facilitator_url: str | None) -> Any:
    """
    Construct an x402 client, forwarding facilitator_url when supported by the installed x402 version.
    """
    if not facilitator_url:
        return x402_client_class()

    try:
        params = inspect.signature(x402_client_class).parameters
    except (TypeError, ValueError):
        params = {}

    if "facilitator_url" in params:
        return x402_client_class(facilitator_url=facilitator_url)
    if "facilitatorUrl" in params:
        return x402_client_class(facilitatorUrl=facilitator_url)

    # Fallback for x402 versions that expose facilitator config as an attribute.
    client = x402_client_class()
    for attr_name in ("facilitator_url", "facilitatorUrl"):
        if hasattr(client, attr_name):
            try:
                setattr(client, attr_name, facilitator_url)
            except Exception:
                pass
            break
    return client


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
        self._timeout = httpx.Timeout(DEFAULT_HTTP_TIMEOUT_SECONDS)
        self._x402_fetch: Optional[Any] = None
        self._x402_http_client: Optional[httpx.AsyncClient] = None

    async def get_metric_data(
        self, query: SirenMetricQuery
    ) -> list[SirenMetricDataPoint]:
        """Fetch metric data for a region over a date range."""
        start_date = _format_date(query.start_date)
        end_date = _format_date(query.end_date)
        encoded_region_id = _quote_path_segment(query.region_id)
        encoded_metric = _quote_path_segment(query.metric)
        encoded_start_date = _quote_path_segment(start_date)
        encoded_end_date = _quote_path_segment(end_date)

        if _is_api_key_auth(self._auth):
            encoded_account_id = _quote_path_segment(self._auth.account_id or "")
            url = (
                f"{self._base_url}/metric-data-multiple/"
                f"{encoded_account_id}/{encoded_region_id}/"
                f"{encoded_metric}/{encoded_start_date}/{encoded_end_date}"
            )
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(
                        url,
                        headers={
                            "Authorization": f"Bearer {self._auth.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
            except httpx.HTTPError as exc:
                raise SirenApiError(f"Siren API request failed: {exc}") from exc

            if response.status_code >= 400:
                raise SirenApiError(
                    f"Siren API error ({response.status_code}): {response.reason_phrase}"
                )
            try:
                body = response.json()
            except ValueError as exc:
                raise SirenApiError(
                    "Siren API returned invalid JSON for metric data response."
                ) from exc
            return _parse_metric_response(body, metric=query.metric)

        # x402 auth
        wrapped_fetch = await self._get_x402_fetch()
        api_base = self._x402_base_url or self._base_url
        url = (
            f"{api_base}/metric-data/{encoded_region_id}/{encoded_metric}/"
            f"{encoded_start_date}/{encoded_end_date}"
        )
        response = await wrapped_fetch(url, method="GET")
        if response.status_code >= 400:
            raise X402PaymentError(
                f"Siren x402 request failed ({response.status_code}): {response.reason_phrase}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise SirenApiError(
                "Siren API returned invalid JSON for metric data response."
            ) from exc
        return _parse_metric_response(body, metric=query.metric)

    async def list_regions(self) -> list[SirenRegion]:
        """List available regions. Free endpoint - no payment required."""
        if _is_api_key_auth(self._auth):
            url = f"{self._base_url}/custom-regions/{self._auth.account_id}/custom"
            headers = {
                "Authorization": f"Bearer {self._auth.api_key}",
                "Content-Type": "application/json",
            }
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                raise SirenApiError(f"Siren API request failed: {exc}") from exc

            if response.status_code >= 400:
                raise SirenApiError(
                    f"Siren API error ({response.status_code}): {response.reason_phrase}"
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise SirenApiError(
                    "Siren API returned invalid JSON for regions response."
                ) from exc
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
        try:
            data = response.json()
        except ValueError as exc:
            raise SirenApiError(
                "Siren API returned invalid JSON for regions response."
            ) from exc
        return _parse_regions_response(data)

    async def aclose(self) -> None:
        """Close any persistent x402 HTTP resources."""
        if self._x402_http_client is not None:
            await self._x402_http_client.aclose()
            self._x402_http_client = None
            self._x402_fetch = None

    async def _get_x402_fetch(self) -> Any:
        """
        Lazily initialize the x402-wrapped fetch function.
        Dynamically imports x402 packages so they remain optional.
        """
        if self._x402_fetch is not None:
            return self._x402_fetch

        if not _is_x402_auth(self._auth):
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

        # Build x402 client with EVM scheme registered for the signer.
        client = _build_x402_client(X402Client, self._auth.facilitator_url)
        register_evm_schemes(client, self._auth.signer, self._auth.network)

        self._x402_http_client = httpx.AsyncClient(timeout=self._timeout)
        self._x402_fetch = wrap_fetch_with_payment(self._x402_http_client, client)
        return self._x402_fetch
