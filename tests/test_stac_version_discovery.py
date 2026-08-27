from unittest.mock import Mock

import pytest
from dataclasses import replace
import pystac
import xarray as xr
import httpx

from dclimate_client_py import dclimate_client as client_module
from dclimate_client_py import stac_catalog, stac_server
from dclimate_client_py.ceramic_api import DatasetVersion, DatasetVersionListing
from dclimate_client_py.dclimate_client import dClimateClient
from dclimate_client_py.dclimate_zarr_errors import (
    ConflictingResolutionSelectionError,
    MultiresolutionSelectionRequiredError,
    ResolutionNotAvailableError,
)


def _feature(properties, *, asset_fields=None):
    return {
        "type": "Feature",
        "id": "noaa_aigfs-wind_u_forecast-operational",
        "collection": "noaa_aigfs",
        "properties": properties,
        "assets": {"data": {"href": "ipfs://bafy-current", **(asset_fields or {})}},
    }


def test_stac_server_details_preserve_discovered_hydrogen_urls(monkeypatch):
    versions_url = "https://hydrogen.dclimate.net/api/datasets/aigfs-wind-u/versions"
    feature = _feature(
        {
            "dclimate:dataset_id": "wind_u_forecast",
            "dclimate:variant": "operational",
            "dclimate:versions_api": versions_url,
            "dclimate:commit_id": "commit-1",
            "dclimate:is_citable": False,
            "dclimate:retention_class": "ephemeral",
        },
        asset_fields={
            "dclimate:zarr_group": "0",
            "dclimate:spatial_resolution": "500m",
        },
    )
    monkeypatch.setattr(
        stac_server,
        "_search_pages",
        lambda *args, **kwargs: iter([{"features": [feature]}]),
    )

    details = stac_server.resolve_dataset_from_stac_server(
        "noaa_aigfs", "wind_u_forecast", "operational"
    )

    assert details.cid == "bafy-current"
    assert details.versions_api == versions_url
    assert details.commit_id == "commit-1"
    assert details.is_citable is False
    assert details.retention_class == "ephemeral"
    assert details.zarr_resolutions == ()
    # The old API remains a two-field tuple for backwards compatibility.
    assert stac_server.resolve_cid_from_stac_server(
        "noaa_aigfs", "wind_u_forecast", "operational"
    ) == stac_server.ResolvedDataset("bafy-current", "operational")


@pytest.mark.parametrize("include_alias", [True, False])
def test_stac_server_treats_named_assets_as_three_resolution_choices(
    monkeypatch, include_alias
):
    assets = {
        f"data-{resolution}": {
            "href": "ipfs://bafy-fpar",
            "dclimate:zarr_group": group,
            "dclimate:spatial_resolution": resolution,
        }
        for resolution, group in (("500m", "0"), ("2km", "1"), ("8km", "2"))
    }
    assets["data-500m-alias"] = dict(assets["data-500m"])
    if include_alias:
        assets["data"] = {
            **assets["data-500m"],
            "title": "Legacy compatibility alias",
        }
    feature = _feature(
        {
            "dclimate:dataset_id": "wind_u_forecast",
            "dclimate:variant": "operational",
        }
    )
    feature["assets"] = assets
    monkeypatch.setattr(
        stac_server,
        "_search_pages",
        lambda *args, **kwargs: iter([{"features": [feature]}]),
    )

    details = stac_server.resolve_dataset_from_stac_server(
        "noaa_aigfs", "wind_u_forecast", "operational"
    )

    assert len(details.zarr_resolutions) == 3
    assert {choice.asset_key for choice in details.zarr_resolutions} == {
        "data-500m",
        "data-2km",
        "data-8km",
    }


def test_ipfs_stac_details_preserve_discovered_tritium_url():
    catalog = pystac.Catalog(id="root", description="root")
    organization = pystac.Catalog(id="ecmwf", description="ECMWF")
    collection = pystac.Collection(
        id="ecmwf_era5",
        description="ERA5",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-180, -90, 180, 90]]),
            pystac.TemporalExtent([[None, None]]),
        ),
    )
    item = pystac.Item(
        id="ecmwf_era5-temperature_2m-finalized",
        geometry=None,
        bbox=None,
        datetime=None,
        properties={
            "start_datetime": "1940-01-01T00:00:00Z",
            "end_datetime": "2026-01-01T00:00:00Z",
            "dclimate:dataset_id": "temperature_2m",
            "dclimate:variant": "finalized",
            "dclimate:versions_api": (
                "https://tritium.dclimate.net/api/datasets/"
                "era5-temperature-2m-finalized/versions"
            ),
        },
    )
    data_asset = pystac.Asset(href="ipfs://bafy-era5")
    data_asset.extra_fields["dclimate:zarr_group"] = "0"
    data_asset.extra_fields["dclimate:spatial_resolution"] = "500m"
    item.add_asset("data-500m", data_asset)
    duplicate_asset = pystac.Asset(href="ipfs://bafy-era5")
    duplicate_asset.extra_fields.update(data_asset.extra_fields)
    item.add_asset("data-500m-alias", duplicate_asset)
    collection.add_item(item)
    collection_link = organization.add_child(collection)
    collection_link.extra_fields["dclimate:id"] = "ecmwf_era5"
    organization_link = catalog.add_child(organization)
    organization_link.extra_fields.update(
        {
            "dclimate:id": "ecmwf",
            "dclimate:collections:historical": ["ecmwf_era5"],
        }
    )

    details = stac_catalog.resolve_dataset_from_stac(
        catalog, "ecmwf_era5", "temperature_2m", "finalized"
    )

    assert details.cid == "bafy-era5"
    assert details.versions_api == (
        "https://tritium.dclimate.net/api/datasets/"
        "era5-temperature-2m-finalized/versions"
    )
    assert details.zarr_resolutions == (
        stac_server.ZarrResolution("data-500m", "500m", "0"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("resolution", "group"), [("500m", None), (None, "/2/")])
async def test_client_requires_explicit_resolution_or_group(
    monkeypatch, resolution, group
):
    details = stac_server.ResolvedDatasetDetails(
        cid="bafy-grouped",
        variant="default",
        versions_api="https://versions.test/datasets/pyramid/versions",
        provenance_api="https://versions.test/datasets/pyramid/provenance",
        citation_api="https://versions.test/datasets/pyramid/citation",
        stream_id="stream-1",
        commit_id="commit-1",
        version_label="2026-08",
        is_citable=False,
        retention_class="permanent",
        zarr_resolutions=(
            stac_server.ZarrResolution("data-500m", "500m", "0"),
            stac_server.ZarrResolution("data-2km", "2km", "2"),
        ),
    )

    async def aresolve(**kwargs):
        return details

    monkeypatch.setattr(
        client_module,
        "aresolve_dataset_from_stac_server",
        aresolve,
    )

    observed_groups = []

    async def load_dataset_from_ipfs_cid(**kwargs):
        observed_groups.append(kwargs["zarr_group"])
        normalized = kwargs["zarr_group"]
        return xr.Dataset(attrs={"_ipfs_zarr_group": normalized})

    monkeypatch.setattr(
        client_module,
        "_load_dataset_from_ipfs_cid",
        load_dataset_from_ipfs_cid,
    )
    client = dClimateClient(stac_server_url="https://stac.test")
    client._kubo_cas = object()

    _, metadata = await client.load_dataset(
        dataset="pyramid",
        collection="test_grouped",
        resolution=resolution,
        zarr_group=group,
        return_xarray=True,
    )

    expected_group = "0" if resolution == "500m" else "2"
    assert observed_groups == [expected_group]
    assert metadata["zarr_group"] == expected_group
    assert metadata["resolution"] == (resolution or "2km")
    assert metadata["versions_api"] == details.versions_api
    assert metadata["provenance_api"] == details.provenance_api
    assert metadata["citation_api"] == details.citation_api
    assert metadata["stream_id"] == "stream-1"
    assert metadata["commit_id"] == "commit-1"
    assert metadata["version_label"] == "2026-08"
    assert metadata["is_citable"] is False
    assert metadata["retention_class"] == "permanent"


@pytest.mark.asyncio
async def test_client_rejects_ambiguous_or_invalid_resolution_selection(monkeypatch):
    details = stac_server.ResolvedDatasetDetails(
        cid="bafy-grouped",
        variant="default",
        zarr_resolutions=(
            stac_server.ZarrResolution("data-500m", "500m", "0"),
            stac_server.ZarrResolution("data-2km", "2km", "1"),
        ),
    )

    async def aresolve(**kwargs):
        return details

    monkeypatch.setattr(client_module, "aresolve_dataset_from_stac_server", aresolve)
    client = dClimateClient(stac_server_url="https://stac.test")
    client._kubo_cas = object()

    with pytest.raises(MultiresolutionSelectionRequiredError) as required:
        await client.load_dataset(dataset="pyramid", collection="test_grouped")
    assert required.value.available_resolutions == ("500m", "2km")

    with pytest.raises(ResolutionNotAvailableError, match="10km"):
        await client.load_dataset(
            dataset="pyramid", collection="test_grouped", resolution="10km"
        )

    with pytest.raises(ConflictingResolutionSelectionError):
        await client.load_dataset(
            dataset="pyramid",
            collection="test_grouped",
            resolution="500m",
            zarr_group="0",
        )


@pytest.mark.asyncio
async def test_client_lists_versions_from_stac_url(monkeypatch):
    client = dClimateClient()
    details = stac_server.ResolvedDatasetDetails(
        cid="bafy-current",
        variant="operational",
        versions_api="https://hydrogen.test/api/datasets/aigfs-wind-u/versions",
    )

    async def resolve_details(*args, **kwargs):
        return details

    listing = DatasetVersionListing("aigfs-wind-u", "stream-1", [])
    request = Mock(return_value=listing)
    monkeypatch.setattr(client, "_resolve_dataset_details", resolve_details)
    monkeypatch.setattr(
        "dclimate_client_py.dclimate_client.list_versions_from_url", request
    )

    result = await client.list_dataset_versions(
        "noaa_aigfs",
        "wind_u_forecast",
        "operational",
        anchored=True,
    )

    assert result is listing
    request.assert_called_once_with(
        details.versions_api,
        anchored=True,
        is_citable=None,
        version_label=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "versions_url",
    [
        "https://hydrogen.dclimate.net/api/datasets/aigfs-wind-u/versions",
        "https://tritium.dclimate.net/api/datasets/aigfs-wind-u/versions",
    ],
)
async def test_client_gets_exact_version_from_stac_url(monkeypatch, versions_url):
    client = dClimateClient()
    details = stac_server.ResolvedDatasetDetails(
        cid="bafy-current",
        variant="operational",
        versions_api=versions_url,
    )

    async def resolve_details(*args, **kwargs):
        return details

    exact = DatasetVersion(dataset="aigfs-wind-u", cid="bafy-exact")
    request = Mock(return_value=exact)
    monkeypatch.setattr(client, "_resolve_dataset_details", resolve_details)
    monkeypatch.setattr(client_module, "get_exact_version_from_url", request)

    result = await client.get_dataset_version(
        "noaa_aigfs",
        "wind_u_forecast",
        "commit/with spaces?and=query#fragment",
        "operational",
    )

    assert result is exact
    request.assert_called_once_with(
        versions_url, "commit/with spaces?and=query#fragment"
    )


@pytest.mark.asyncio
async def test_client_propagates_exact_version_http_error(monkeypatch):
    client = dClimateClient()

    async def resolve_details(*args, **kwargs):
        return stac_server.ResolvedDatasetDetails(
            cid="bafy-current",
            variant="operational",
            versions_api="https://hydrogen.test/datasets/aigfs/versions",
        )

    request = httpx.Request("GET", "https://hydrogen.test/datasets/aigfs/versions/c")
    response = httpx.Response(503, request=request)
    error = httpx.HTTPStatusError("unavailable", request=request, response=response)
    monkeypatch.setattr(client, "_resolve_dataset_details", resolve_details)
    monkeypatch.setattr(
        client_module,
        "get_exact_version_from_url",
        Mock(side_effect=error),
    )

    with pytest.raises(httpx.HTTPStatusError) as raised:
        await client.get_dataset_version("noaa_aigfs", "wind_u_forecast", "commit-1")

    assert raised.value.response.status_code == 503


@pytest.mark.asyncio
async def test_client_reports_items_without_version_history(monkeypatch):
    client = dClimateClient()

    async def resolve_details(*args, **kwargs):
        return stac_server.ResolvedDatasetDetails("bafy", "default")

    monkeypatch.setattr(client, "_resolve_dataset_details", resolve_details)

    with pytest.raises(ValueError, match="Version history is not available"):
        await client.list_dataset_versions("copernicus_clms", "fpar")

    with pytest.raises(ValueError, match="Version history is not available"):
        await client.get_dataset_version("copernicus_clms", "fpar", "commit-1")


@pytest.mark.asyncio
async def test_version_catalog_fallback_normalizes_shorthand_collection(monkeypatch):
    catalog = object()
    resolved = stac_server.ResolvedDatasetDetails(
        cid="bafy-era5",
        variant="finalized",
        versions_api="https://versions.test/era5/versions",
    )
    resolver = Mock(return_value=resolved)

    monkeypatch.setattr(stac_catalog, "load_stac_catalog", Mock(return_value=catalog))
    monkeypatch.setattr(
        stac_catalog,
        "list_available_datasets",
        Mock(return_value={"ecmwf_era5": {}}),
    )
    monkeypatch.setattr(stac_catalog, "resolve_dataset_from_stac", resolver)
    client = dClimateClient(stac_server_url=None)

    details = await client._resolve_dataset_details(
        collection="era5",
        dataset="temperature_2m",
        variant="finalized",
        organization=None,
    )

    # The resolver's payload is passed through unchanged except for the
    # identity the expansion just established: `era5` was matched against the
    # catalogue as `ecmwf_era5`, and that is the name callers must report.
    assert details == replace(resolved, collection="ecmwf_era5")
    assert details.collection == "ecmwf_era5"
    resolver.assert_called_once_with(
        catalog=catalog,
        collection="ecmwf_era5",
        dataset="temperature_2m",
        variant="finalized",
        organization=None,
    )
