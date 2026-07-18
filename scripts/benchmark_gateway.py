#!/usr/bin/env python3
"""Benchmark bounded dClimate gateway reads using the py-hamt #58 methodology.

The benchmark compares HTTP/2 with HTTP/1.1 while holding Kubo request
concurrency and the dataset sample constant.  The companion infrastructure
change is to raise nginx's ``keepalive_requests`` from its default of 1000 on
the dClimate gateway: large dataset opens otherwise cause GOAWAY churn.
Benchmark HTTP/2 on and off both before and after that gateway change.
"""

import argparse
import json
import statistics
import time
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--http2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable HTTP/2 for gateway requests (default: enabled)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="maximum concurrent Kubo gateway requests (default: 32)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="number of timed dataset opens (default: 3)",
    )
    parser.add_argument("--collection", default="cpc-precip-conus")
    parser.add_argument("--dataset", default="precip")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--gateway", default="https://ipfs-gateway.dclimate.net")
    parser.add_argument("--stac-server", default="https://api.stac.dclimate.net")
    return parser


async def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    from dclimate_client_py import dClimateClient

    elapsed_times: list[float] = []
    bytes_read: list[int] = []

    for repetition in range(1, args.repetitions + 1):
        started = time.perf_counter()
        async with dClimateClient(
            gateway_base_url=args.gateway,
            rpc_base_url=args.gateway,
            stac_server_url=args.stac_server,
            concurrency=args.concurrency,
            client_factory=lambda: httpx.AsyncClient(
                http2=args.http2,
                timeout=60.0,
            ),
        ) as client:
            dataset, _metadata = await client.load_dataset(
                collection=args.collection,
                dataset=args.dataset,
                variant=args.variant,
                return_xarray=True,
            )
            bounded_indexers = {
                dimension: slice(0, min(size, 8))
                for dimension, size in dataset.sizes.items()
            }
            sample = dataset.isel(bounded_indexers).load()
            sample_bytes = sample.nbytes
        elapsed = time.perf_counter() - started
        elapsed_times.append(elapsed)
        bytes_read.append(sample_bytes)
        print(
            f"repetition {repetition}/{args.repetitions}: "
            f"{elapsed:.3f} s ({sample_bytes} sample bytes)"
        )

    median = statistics.median(elapsed_times)
    print(f"median: {median:.3f} s")
    return {
        "http2": args.http2,
        "concurrency": args.concurrency,
        "repetitions": args.repetitions,
        "collection": args.collection,
        "dataset": args.dataset,
        "variant": args.variant,
        "gateway": args.gateway,
        "stac_server": args.stac_server,
        "times_seconds": elapsed_times,
        "median_seconds": median,
        "sample_bytes": bytes_read,
    }


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    import asyncio

    result = asyncio.run(_benchmark(args))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
