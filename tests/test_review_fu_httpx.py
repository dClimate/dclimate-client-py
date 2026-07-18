"""Migration pins for consolidating package HTTP on httpx."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import Any

import httpx
import pystac
import xarray as xr

import dclimate_client_py.dclimate_client as dclimate_client_module
from dclimate_client_py import stac_catalog
from dclimate_client_py.dclimate_client import dClimateClient
from dclimate_client_py.stac_catalog import IPFSStacIO
from dclimate_client_py.stac_server import ResolvedDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "dclimate_client_py"
FORBIDDEN_HTTP_PACKAGES = {"requests", "urllib3"}


def test_package_sources_do_not_import_requests_or_urllib3() -> None:
    offenders: list[str] = []

    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        imports_forbidden_package = any(
            (
                isinstance(node, ast.Import)
                and any(
                    alias.name.split(".", 1)[0] in FORBIDDEN_HTTP_PACKAGES
                    for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.split(".", 1)[0] in FORBIDDEN_HTTP_PACKAGES
            )
            for node in ast.walk(tree)
        )
        if imports_forbidden_package:
            offenders.append(str(source_path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "package sources still use legacy HTTP packages:\n" + "\n".join(offenders)
    )


def test_project_dependencies_do_not_include_requests_or_urllib3() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    dependencies = pyproject["project"]["dependencies"]
    forbidden_dependencies = [
        dependency
        for dependency in dependencies
        if re.split(r"[<>=!~;\s\[]", dependency, maxsplit=1)[0]
        .lower()
        .replace("_", "-")
        in FORBIDDEN_HTTP_PACKAGES
    ]

    assert not forbidden_dependencies, (
        "[project] dependencies still include legacy HTTP packages: "
        + ", ".join(forbidden_dependencies)
    )


async def test_load_dataset_falls_back_when_stac_transport_is_unreachable(
    monkeypatch,
) -> None:
    """Non-regression pin: transport errors must still enter catalog fallback."""
    catalog = pystac.Catalog(id="root", description="Fallback catalog")
    loaded_hrefs: list[str] = []
    catalog_resolutions: list[dict[str, Any]] = []

    monkeypatch.setattr(
        stac_catalog,
        "get_root_catalog_cid",
        lambda catalog_url=stac_catalog.STAC_CATALOG_URL: "bafy-review-root",
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
    monkeypatch.setattr(
        stac_catalog,
        "list_available_datasets",
        lambda loaded_catalog: {"review_collection": {"types": ["temperature"]}},
    )

    def resolve_from_catalog(**kwargs: Any) -> ResolvedDataset:
        catalog_resolutions.append(kwargs)
        return ResolvedDataset("bafy-fallback-dataset", "default")

    monkeypatch.setattr(
        stac_catalog,
        "resolve_dataset_cid_from_stac",
        resolve_from_catalog,
    )

    async def fake_load_dataset(**kwargs: Any) -> xr.Dataset:
        return xr.Dataset({"temperature": ("x", [1.0])})

    monkeypatch.setattr(
        dclimate_client_module,
        "_load_dataset_from_ipfs_cid",
        fake_load_dataset,
    )

    client = dClimateClient(
        gateway_base_url="https://gateway.example",
        stac_server_url="http://127.0.0.1:9",
    )
    client._kubo_cas = object()
    previous_default_io = pystac.StacIO._default_io
    try:
        _, metadata = await client.load_dataset(
            dataset="temperature",
            collection="review_collection",
            return_xarray=True,
        )
    finally:
        pystac.StacIO.set_default(previous_default_io)

    assert loaded_hrefs == ["ipfs://bafy-review-root"]
    assert len(catalog_resolutions) == 1
    assert metadata["cid"] == "bafy-fallback-dataset"


def test_ipfs_stac_io_owns_httpx_client() -> None:
    stac_io = IPFSStacIO("https://gateway.example")

    assert isinstance(getattr(stac_io, "client", None), httpx.Client)
