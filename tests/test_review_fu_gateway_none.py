"""Regression coverage for catalog fallback with a defaulted IPFS gateway."""

from __future__ import annotations

from typing import Any

import pystac

from dclimate_client_py import stac_catalog
from dclimate_client_py.dclimate_client import dClimateClient


DEFAULT_PUBLIC_GATEWAY = "https://ipfs-gateway.dclimate.net"


async def test_alist_datasets_uses_public_gateway_when_gateway_is_none(
    monkeypatch,
) -> None:
    catalog = pystac.Catalog(id="root", description="Minimal root catalog")
    loaded_hrefs: list[str] = []
    constructed_gateways: list[Any] = []

    monkeypatch.setattr(
        stac_catalog,
        "get_root_catalog_cid",
        lambda: "bafy-review-root",
    )

    def fake_from_file(
        cls: type[pystac.Catalog],
        href: str,
        stac_io: pystac.StacIO | None = None,
    ) -> pystac.Catalog:
        loaded_hrefs.append(href)
        return catalog

    monkeypatch.setattr(
        pystac.Catalog,
        "from_file",
        classmethod(fake_from_file),
    )

    original_init = stac_catalog.IPFSStacIO.__init__

    def recording_init(
        self: stac_catalog.IPFSStacIO,
        gateway_url: str,
    ) -> None:
        constructed_gateways.append(gateway_url)
        original_init(self, gateway_url)

    monkeypatch.setattr(stac_catalog.IPFSStacIO, "__init__", recording_init)

    client = dClimateClient(gateway_base_url=None, stac_server_url=None)
    client._kubo_cas = object()
    previous_default_io = pystac.StacIO._default_io
    try:
        datasets = await client.alist_datasets()
    finally:
        pystac.StacIO.set_default(previous_default_io)

    assert datasets == {}
    assert loaded_hrefs == ["ipfs://bafy-review-root"]
    assert constructed_gateways == [DEFAULT_PUBLIC_GATEWAY]
