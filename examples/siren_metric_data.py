"""
Example: Fetch Siren metric data using the dClimate client.

Prerequisites:
  - Create a .env file with SIREN_API_KEY and SIREN_ACCOUNT_ID
  - Or set them as environment variables

Run:
  uv run python examples/siren_metric_data.py
"""

import asyncio
import json
import pathlib
from datetime import date

from dotenv import load_dotenv

from dclimate_client_py import (
    dClimateClient,
    SirenApiKeyAuth,
    SirenMetricQuery,
    SirenOptions,
)

# Load .env from project root
load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")


async def main():
    client = dClimateClient(
        siren=SirenOptions(
            auth=SirenApiKeyAuth(),  # reads from SIREN_API_KEY & SIREN_ACCOUNT_ID env vars
        ),
    )

    start_date = date(2025, 12, 31)
    end_date = date(2026, 12, 31)

    region_id = "4c59966e-8653-4534-a640-5b0e9be3de81"
    metric = "average_precip"

    print(f"Fetching {metric} for region {region_id}")
    print(f"Date range: {start_date} to {end_date}")

    data = await client.get_metric_data(
        SirenMetricQuery(
            region_id=region_id,
            metric=metric,
            start_date=start_date,
            end_date=end_date,
        )
    )

    print("Response:")
    print(json.dumps([{"date": dp.date, "value": dp.value} for dp in data], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
