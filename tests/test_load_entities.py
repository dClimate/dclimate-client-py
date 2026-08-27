"""``client.load_entities``: catalog resolution for entity datasets.

Mirrors ``dclimate-client-js`` ``tests/load-entities.test.ts``.

What is under test is the dispatch this method adds -- the layout guard, the
``column_key`` default, and the metadata it assembles -- not STAC resolution,
which ``test_stac_server.py`` already covers against the live server. So
``_resolve_dataset_details`` is stubbed rather than the network.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from dclimate_client_py.dclimate_client import dClimateClient
from dclimate_client_py.dclimate_zarr_errors import DatasetNotFoundError
from dclimate_client_py.stac_server import ResolvedDatasetDetails


def _resolved(**over: Any) -> ResolvedDatasetDetails:
    fields: dict[str, Any] = {
        "cid": "bafyr4ieoihgvnl5rvu6eh2fqduapjtz7wjp3e7kdtfxjospmavi5lgkoq4",
        "variant": "default",
        "layout": "tabular",
        "commit_id": "k1commit",
        "stream_id": "kjstream",
        "version_label": "2026-08-26",
    }
    fields.update(over)
    return ResolvedDatasetDetails(**fields)


class _Field:
    def __init__(self, name: str) -> None:
        self.name = name


def _stub(
    monkeypatch: pytest.MonkeyPatch,
    client: dClimateClient,
    resolved: Optional[ResolvedDatasetDetails] = None,
) -> dict[str, Any]:
    """Stub resolution and capture what reaches ``entities.load``."""
    captured: dict[str, Any] = {}

    async def fake_resolve(**_kwargs: Any) -> ResolvedDatasetDetails:
        return resolved if resolved is not None else _resolved()

    async def fake_load(cid: str, **kwargs: Any) -> object:
        captured["cid"] = cid
        captured.update(kwargs)
        captured["called"] = True
        return object()

    monkeypatch.setattr(client, "_resolve_dataset_details", fake_resolve)

    class _Entities:
        load = staticmethod(fake_load)

    monkeypatch.setattr(type(client), "entities", property(lambda _self: _Entities()))
    return captured


@pytest.mark.asyncio
async def test_refuses_a_gridded_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Zarr CID opened as an entity dataset dies inside a manifest parse.

    That reads as corruption and sends the caller to the publisher rather than
    to their own call, so the layout is checked before the reader sees it.
    """
    client = dClimateClient()
    captured = _stub(monkeypatch, client, _resolved(layout="zarr"))

    with pytest.raises(DatasetNotFoundError, match="not an entity dataset"):
        await client.load_entities(
            collection="ecmwf_era5", dataset="precipitation_total"
        )
    assert "called" not in captured


@pytest.mark.asyncio
async def test_refuses_an_item_with_no_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absence is not permission.

    Entity support postdates ``dclimate:layout``, so an item without the field
    is a gridded one from before the convention -- there is no such thing as a
    legacy entity dataset to accommodate.
    """
    client = dClimateClient()
    captured = _stub(monkeypatch, client, _resolved(layout=None))

    with pytest.raises(DatasetNotFoundError, match="'gridded' dataset"):
        await client.load_entities(collection="ecmwf_era5", dataset="reanalysis")
    assert "called" not in captured


@pytest.mark.asyncio
async def test_supplies_no_column_key_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``column_key`` renames columns; it does not gate access to them.

    Without one every column is still readable under the schema's own field
    names, which are what the dataset stores and so are never wrong. A default
    here would be a guess at a dataset's publishing profile -- right for GHCND,
    silently wrong for a profile like NDBC's ``.spec`` feed that publishes
    ``SwH``.
    """
    client = dClimateClient()
    captured = _stub(monkeypatch, client)

    await client.load_entities(collection="noaa_ghcnd", dataset="station_observations")
    assert "column_key" not in captured


@pytest.mark.asyncio
async def test_forwards_a_callers_column_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NDBC preserves mixed case like ``SwH``, so a caller must be able to say
    so explicitly."""
    client = dClimateClient()
    captured = _stub(monkeypatch, client)

    await client.load_entities(
        collection="noaa_ndbc",
        dataset="buoy_observations",
        column_key=lambda field: field.name,
    )
    assert captured["column_key"](_Field("SwH")) == "SwH"


@pytest.mark.asyncio
async def test_returns_snapshot_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A settlement or citation has to say which snapshot it ran against, not
    just "whatever was newest that day"."""
    client = dClimateClient()
    _stub(monkeypatch, client)

    _dataset, metadata = await client.load_entities(
        collection="noaa_ghcnd", dataset="station_observations"
    )

    assert metadata["collection"] == "noaa_ghcnd"
    assert metadata["dataset"] == "station_observations"
    assert metadata["variant"] == "default"
    assert metadata["organization"] == "noaa"
    assert metadata["source"] == "stac"
    assert metadata["commit_id"] == "k1commit"
    assert metadata["stream_id"] == "kjstream"
    assert metadata["version_label"] == "2026-08-26"
    assert metadata["slug"] == "noaa_ghcnd/station_observations/default"


@pytest.mark.asyncio
async def test_passes_gateway_through(monkeypatch: pytest.MonkeyPatch) -> None:
    client = dClimateClient()
    captured = _stub(monkeypatch, client)

    await client.load_entities(
        collection="noaa_ghcnd",
        dataset="station_observations",
        gateway_url="https://override.example",
    )
    assert captured["gateway_url"] == "https://override.example"
