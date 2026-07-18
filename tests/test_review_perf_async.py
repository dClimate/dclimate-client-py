import asyncio
import time
from unittest.mock import AsyncMock

import httpx
import xarray as xr

from dclimate_client_py import dclimate_client, stac_catalog, stac_server


async def test_load_dataset_does_not_stall_event_loop(monkeypatch, install_httpx_mock):
    fake_cid = "bafy-fake-dataset-cid"

    def slow_stac_search(request: httpx.Request) -> httpx.Response:
        time.sleep(0.25)
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "id": "example_temperature_default",
                        "collection": "example",
                        "properties": {
                            "dclimate:dataset_id": "temperature",
                            "dclimate:variant": "default",
                        },
                        "assets": {"data": {"href": f"ipfs://{fake_cid}"}},
                    }
                ]
            },
            request=request,
        )

    async def load_from_ipfs(**kwargs):
        assert kwargs["ipfs_cid"] == fake_cid
        return xr.Dataset({"temperature": ("time", [21.0])}, coords={"time": [0]})

    install_httpx_mock(stac_server, slow_stac_search)
    monkeypatch.setattr(dclimate_client, "_load_dataset_from_ipfs_cid", load_from_ipfs)

    client = dclimate_client.dClimateClient(stac_server_url="https://stac.invalid")
    client._kubo_cas = AsyncMock()

    loop = asyncio.get_running_loop()
    heartbeat_times = [loop.time()]
    stop_heartbeat = asyncio.Event()

    async def heartbeat():
        while not stop_heartbeat.is_set():
            await asyncio.sleep(0.01)
            heartbeat_times.append(loop.time())

    heartbeat_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    dataset, metadata = await client.load_dataset(
        collection="example",
        dataset="temperature",
        variant="default",
        return_xarray=True,
    )
    stop_heartbeat.set()
    await heartbeat_task

    assert dataset["temperature"].values.tolist() == [21.0]
    assert metadata["cid"] == fake_cid
    max_tick_gap = max(
        later - earlier for earlier, later in zip(heartbeat_times, heartbeat_times[1:])
    )
    assert max_tick_gap < 0.15, f"event loop stalled for {max_tick_gap:.3f}s"


def test_ipfs_stac_io_reuses_client(install_httpx_mock):
    get_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        get_calls.append(str(request.url))
        return httpx.Response(200, text="{}", request=request)

    pooled_client = install_httpx_mock(stac_catalog, handler)

    stac_io = stac_catalog.IPFSStacIO("https://gateway.invalid")
    for index in range(5):
        assert stac_io.read_text(f"ipfs://fake-cid-{index}") == "{}"

    assert stac_io.client is pooled_client
    assert len(get_calls) == 5
