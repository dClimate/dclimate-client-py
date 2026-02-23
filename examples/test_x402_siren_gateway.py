"""
Test script: make an x402-paid request to Siren via a local gateway.

Prerequisites:
  1. Install project dependencies:
     uv sync

  2. Provide wallet credentials via ONE of:
     - PRIVATE_KEY env var (hex, with or without 0x prefix)
     - .env file with PRIVATE_KEY=0x...
     - mnemonic.txt file in project root (BIP-39 mnemonic phrase)

  3. Start the x402 gateway:
     cd ../x402-gateway && npm start

Usage:
  uv run python examples/test_x402_siren_gateway.py

Environment variables:
  PRIVATE_KEY      Wallet private key (hex)
  GATEWAY_URL      Gateway base URL (default: http://localhost:8080)
  REGION_ID        Siren region ID to query
  METRIC           Metric name (default: average_precip)
  NETWORK          Payment network (default: base-sepolia)
  FACILITATOR_URL  Optional x402 facilitator URL override
"""

from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from dclimate_client_py import (
    SirenClient,
    SirenMetricQuery,
    SirenOptions,
    SirenX402Auth,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8080")
REGION_ID = os.getenv("REGION_ID", "4c59966e-8653-4534-a640-5b0e9be3de81")
METRIC = os.getenv("METRIC", "average_precip")
NETWORK = os.getenv("NETWORK", "base-sepolia")
FACILITATOR_URL = os.getenv("FACILITATOR_URL")


def _load_private_key() -> str | None:
    private_key = os.getenv("PRIVATE_KEY")
    if not private_key:
        return None
    return private_key if private_key.startswith("0x") else f"0x{private_key}"


def _load_mnemonic() -> str | None:
    mnemonic_path = PROJECT_ROOT / "mnemonic.txt"
    if not mnemonic_path.exists():
        return None

    mnemonic = mnemonic_path.read_text(encoding="utf-8").strip()
    return mnemonic or None


def _create_signer() -> object:
    try:
        from eth_account import Account
        from x402.mechanisms.evm.signers import EthAccountSigner
    except ImportError as exc:
        raise RuntimeError(
            "x402 EVM dependencies are missing. Run `uv sync` to install project dependencies."
        ) from exc

    private_key = _load_private_key()
    mnemonic = _load_mnemonic()

    if private_key:
        return EthAccountSigner(Account.from_key(private_key))

    if mnemonic:
        if not hasattr(Account, "from_mnemonic"):
            raise RuntimeError(
                "eth_account.from_mnemonic is unavailable in this environment."
            )
        Account.enable_unaudited_hdwallet_features()
        return EthAccountSigner(Account.from_mnemonic(mnemonic))

    raise RuntimeError(
        "No wallet credentials found. Provide PRIVATE_KEY env var, .env file, or mnemonic.txt"
    )


async def main() -> None:
    print("=== x402 Siren Gateway Test (Python) ===\n")
    print(f"Gateway:  {GATEWAY_URL}")
    print(f"Network:  {NETWORK}")
    print(f"Region:   {REGION_ID}")
    print(f"Metric:   {METRIC}\n")

    signer = _create_signer()
    print(f"Wallet:   {signer.address}\n")

    client = SirenClient(
        SirenOptions(
            auth=SirenX402Auth(
                signer=signer,  # type: ignore[arg-type]
                network=NETWORK,
                facilitator_url=FACILITATOR_URL,
            ),
            x402_base_url=f"{GATEWAY_URL}/v1/siren",
        )
    )

    try:
        print("--- List Regions (free) ---")
        try:
            regions = await client.list_regions()
            print(f"Found {len(regions)} regions")
            if regions:
                print(f"  First: {regions[0].name} (id: {regions[0].id})")
            print()
        except Exception as err:
            print(f"list_regions failed: {err}\n")

        print("--- Get Metric Data (x402 paid) ---")
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        started_at = perf_counter()

        try:
            data = await client.get_metric_data(
                SirenMetricQuery(
                    region_id=REGION_ID,
                    metric=METRIC,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            elapsed_ms = int((perf_counter() - started_at) * 1000)
            print(f"Received {len(data)} data points")
            print(f"Request time: {elapsed_ms} ms ({elapsed_ms / 1000:.2f} s)")
            if data:
                print(f"  First: {data[0].date} = {data[0].value}")
                print(f"  Last:  {data[-1].date} = {data[-1].value}")
        except Exception as err:
            elapsed_ms = int((perf_counter() - started_at) * 1000)
            print(f"get_metric_data failed: {err}")
            print(
                f"Request time before failure: {elapsed_ms} ms ({elapsed_ms / 1000:.2f} s)"
            )
    finally:
        await client.aclose()

    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
