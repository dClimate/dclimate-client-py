"""Tests for the Siren REST API client."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dclimate_client_py.dclimate_client import dClimateClient
from dclimate_client_py.dclimate_zarr_errors import SirenApiError, X402PaymentError
from dclimate_client_py.siren import SirenClient
from dclimate_client_py.siren.types import (
    SirenApiKeyAuth,
    SirenMetricQuery,
    SirenOptions,
    SirenX402Auth,
)

MOCK_REGIONS_RESPONSE = {
    "items": [
        {
            "id": "region-1",
            "name": "US Midwest",
            "internal_code": None,
            "region_type": "custom",
            "account_id": "acc-123",
            "country_id": "us",
            "commodity_code": "custom",
            "geo_json": '{"type":"Polygon"}',
            "extra_info": None,
            "created_at": "2025-01-01T00:00:00Z",
            "historical_fetch_enabled": True,
            "country": {"id": "us", "name": "United States", "code": "US"},
        }
    ],
    "limit": 100,
    "offset": 0,
    "total": 1,
}

MOCK_METRIC_DATA_DICT = {
    "average_precip": {
        "2025-01-01": 12.5,
        "2025-01-02": 13.1,
        "2025-01-03": 11.8,
    }
}

MOCK_METRIC_DATA_LIST = [
    {"date": "2025-01-01", "value": 12.5},
    {"date": "2025-01-02", "value": 13.1},
    {"date": "2025-01-03", "value": 11.8},
]


def _make_response(json_data, status_code=200):
    """Create a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.reason_phrase = "OK" if status_code < 400 else "Error"
    response.json.return_value = json_data
    return response


class TestSirenClientApiKeyAuth:
    """Tests for API key authentication."""

    @pytest.mark.asyncio
    async def test_fetches_metric_data_with_bearer_token(self):
        client = SirenClient(
            SirenOptions(auth=SirenApiKeyAuth(api_key="sk-test", account_id="acc-123"))
        )

        mock_response = _make_response(MOCK_METRIC_DATA_DICT)

        with patch(
            "dclimate_client_py.siren.siren_client.httpx.AsyncClient"
        ) as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            data = await client.get_metric_data(
                SirenMetricQuery(
                    region_id="region-1",
                    metric="average_precip",
                    start_date="2025-01-01",
                    end_date="2025-01-03",
                )
            )

        assert len(data) == 3
        assert data[0].date == "2025-01-01"
        assert data[0].value == 12.5

        call_args = instance.get.call_args
        url = call_args[0][0]
        headers = call_args[1]["headers"]
        assert (
            "/metric-data-multiple/acc-123/region-1/average_precip/2025-01-01/2025-01-03"
            in url
        )
        assert headers["Authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_lists_regions_with_bearer_token(self):
        client = SirenClient(
            SirenOptions(auth=SirenApiKeyAuth(api_key="sk-test", account_id="acc-123"))
        )

        mock_response = _make_response(MOCK_REGIONS_RESPONSE)

        with patch(
            "dclimate_client_py.siren.siren_client.httpx.AsyncClient"
        ) as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            regions = await client.list_regions()

        assert len(regions) == 1
        assert regions[0].name == "US Midwest"

        call_args = instance.get.call_args
        url = call_args[0][0]
        assert "/custom-regions/acc-123/custom" in url
        assert instance.get.call_count == 1

    @pytest.mark.asyncio
    async def test_formats_date_objects(self):
        client = SirenClient(
            SirenOptions(auth=SirenApiKeyAuth(api_key="sk-test", account_id="acc-123"))
        )

        mock_response = _make_response({"average_temp_mean": {"2025-06-01": 22.4}})

        with patch(
            "dclimate_client_py.siren.siren_client.httpx.AsyncClient"
        ) as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            await client.get_metric_data(
                SirenMetricQuery(
                    region_id="region-1",
                    metric="average_temp_mean",
                    start_date=datetime.date(2025, 6, 1),
                    end_date=datetime.date(2025, 6, 30),
                )
            )

        url = instance.get.call_args[0][0]
        assert "/2025-06-01/2025-06-30" in url

    @pytest.mark.asyncio
    async def test_throws_siren_api_error_on_non_ok_response(self):
        client = SirenClient(
            SirenOptions(auth=SirenApiKeyAuth(api_key="bad-key", account_id="acc-123"))
        )

        mock_response = _make_response({}, status_code=401)
        mock_response.reason_phrase = "Unauthorized"

        with patch(
            "dclimate_client_py.siren.siren_client.httpx.AsyncClient"
        ) as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with pytest.raises(SirenApiError):
                await client.get_metric_data(
                    SirenMetricQuery(
                        region_id="region-1",
                        metric="average_precip",
                        start_date="2025-01-01",
                        end_date="2025-01-03",
                    )
                )

    @pytest.mark.asyncio
    async def test_uses_custom_base_url(self):
        client = SirenClient(
            SirenOptions(
                auth=SirenApiKeyAuth(api_key="sk-test", account_id="acc-123"),
                base_url="https://custom-siren.example.com/api",
            )
        )

        mock_response = _make_response(MOCK_METRIC_DATA_DICT)

        with patch(
            "dclimate_client_py.siren.siren_client.httpx.AsyncClient"
        ) as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            await client.get_metric_data(
                SirenMetricQuery(
                    region_id="region-1",
                    metric="average_precip",
                    start_date="2025-01-01",
                    end_date="2025-01-03",
                )
            )

        url = instance.get.call_args[0][0]
        assert url.startswith("https://custom-siren.example.com/api")

    @pytest.mark.asyncio
    async def test_parses_list_response_format(self):
        """Test that the client handles an array response format too."""
        client = SirenClient(
            SirenOptions(auth=SirenApiKeyAuth(api_key="sk-test", account_id="acc-123"))
        )

        mock_response = _make_response(MOCK_METRIC_DATA_LIST)

        with patch(
            "dclimate_client_py.siren.siren_client.httpx.AsyncClient"
        ) as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            data = await client.get_metric_data(
                SirenMetricQuery(
                    region_id="region-1",
                    metric="average_precip",
                    start_date="2025-01-01",
                    end_date="2025-01-03",
                )
            )

        assert len(data) == 3
        assert data[0].date == "2025-01-01"
        assert data[0].value == 12.5

    @pytest.mark.asyncio
    async def test_raises_if_requested_metric_missing(self):
        client = SirenClient(
            SirenOptions(auth=SirenApiKeyAuth(api_key="sk-test", account_id="acc-123"))
        )

        mock_response = _make_response({"different_metric": {"2025-01-01": 1.0}})

        with patch(
            "dclimate_client_py.siren.siren_client.httpx.AsyncClient"
        ) as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with pytest.raises(SirenApiError, match="missing requested metric"):
                await client.get_metric_data(
                    SirenMetricQuery(
                        region_id="region-1",
                        metric="average_precip",
                        start_date="2025-01-01",
                        end_date="2025-01-03",
                    )
                )


class TestSirenClientEnvVarFallback:
    """Tests for environment variable fallback."""

    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("SIREN_API_KEY", "env-key")
        monkeypatch.setenv("SIREN_ACCOUNT_ID", "env-acc")

        client = SirenClient(SirenOptions(auth=SirenApiKeyAuth()))
        assert client is not None

    def test_throws_if_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("SIREN_API_KEY", raising=False)
        monkeypatch.delenv("SIREN_ACCOUNT_ID", raising=False)

        with pytest.raises(SirenApiError, match="SIREN_API_KEY"):
            SirenClient(SirenOptions(auth=SirenApiKeyAuth()))

    def test_throws_if_account_id_missing(self, monkeypatch):
        monkeypatch.setenv("SIREN_API_KEY", "env-key")
        monkeypatch.delenv("SIREN_ACCOUNT_ID", raising=False)

        with pytest.raises(SirenApiError, match="SIREN_ACCOUNT_ID"):
            SirenClient(SirenOptions(auth=SirenApiKeyAuth()))

    @pytest.mark.asyncio
    async def test_explicit_options_take_precedence(self, monkeypatch):
        monkeypatch.setenv("SIREN_API_KEY", "env-key")
        monkeypatch.setenv("SIREN_ACCOUNT_ID", "env-acc")

        client = SirenClient(
            SirenOptions(
                auth=SirenApiKeyAuth(api_key="explicit-key", account_id="explicit-acc")
            )
        )

        mock_response = _make_response({"m1": {"2025-01-01": 1.0}})

        with patch(
            "dclimate_client_py.siren.siren_client.httpx.AsyncClient"
        ) as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            await client.get_metric_data(
                SirenMetricQuery(
                    region_id="r1",
                    metric="m1",
                    start_date="2025-01-01",
                    end_date="2025-01-02",
                )
            )

        url = instance.get.call_args[0][0]
        headers = instance.get.call_args[1]["headers"]
        assert "/explicit-acc/" in url
        assert headers["Authorization"] == "Bearer explicit-key"


class TestSirenClientX402Auth:
    """Tests for x402 authentication."""

    @pytest.mark.asyncio
    async def test_raises_x402_payment_error_on_non_ok_response(self):
        mock_signer = MagicMock()
        mock_signer.address = "0x1234567890abcdef1234567890abcdef12345678"

        client = SirenClient(
            SirenOptions(
                auth=SirenX402Auth(signer=mock_signer, network="base"),
                x402_base_url="https://x402-siren.example.com",
            )
        )

        mock_response = _make_response({}, status_code=402)
        mock_response.reason_phrase = "Payment Required"
        wrapped_fetch = AsyncMock(return_value=mock_response)

        with patch.object(
            client, "_get_x402_fetch", AsyncMock(return_value=wrapped_fetch)
        ):
            with pytest.raises(X402PaymentError, match="x402 request failed"):
                await client.get_metric_data(
                    SirenMetricQuery(
                        region_id="region-1",
                        metric="average_precip",
                        start_date="2025-01-01",
                        end_date="2025-01-03",
                    )
                )


class TestDClimateClientSirenIntegration:
    """Tests for Siren methods on the main dClimateClient."""

    @pytest.mark.asyncio
    async def test_exposes_get_metric_data(self):
        mock_response = _make_response(MOCK_METRIC_DATA_DICT)

        client = dClimateClient(
            siren=SirenOptions(
                auth=SirenApiKeyAuth(api_key="sk-test", account_id="acc-123")
            )
        )

        with patch(
            "dclimate_client_py.siren.siren_client.httpx.AsyncClient"
        ) as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            data = await client.get_metric_data(
                SirenMetricQuery(
                    region_id="region-1",
                    metric="average_precip",
                    start_date="2025-01-01",
                    end_date="2025-01-03",
                )
            )

        assert len(data) == 3

    @pytest.mark.asyncio
    async def test_throws_when_siren_not_configured_get_metric_data(self):
        client = dClimateClient()

        with pytest.raises(RuntimeError, match="Siren is not configured"):
            await client.get_metric_data(
                SirenMetricQuery(
                    region_id="region-1",
                    metric="average_precip",
                    start_date="2025-01-01",
                    end_date="2025-01-03",
                )
            )

    @pytest.mark.asyncio
    async def test_throws_when_siren_not_configured_list_regions(self):
        client = dClimateClient()

        with pytest.raises(RuntimeError, match="Siren is not configured"):
            await client.list_regions()
