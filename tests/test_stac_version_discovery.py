from unittest.mock import Mock

import pytest
import pystac
import xarray as xr

from dclimate_client_py import dclimate_client as client_module
from dclimate_client_py import stac_catalog, stac_server
from dclimate_client_py.ceramic_api import DatasetVersionListing
from dclimate_client_py.dclimate_client import dClimateClient


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
        asset_fields={"dclimate:zarr_group": "/0/"},
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
    assert details.zarr_group == "/0/"
    # The old API remains a two-field tuple for backwards compatibility.
    assert stac_server.resolve_cid_from_stac_server(
        "noaa_aigfs", "wind_u_forecast", "operational"
    ) == stac_server.ResolvedDataset("bafy-current", "operational")


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
            "dclimate:default_zarr_group": "1",
        },
    )
    data_asset = pystac.Asset(href="ipfs://bafy-era5")
    data_asset.extra_fields["dclimate:zarr_group"] = "0"
    item.add_asset("data", data_asset)
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
    assert details.zarr_group == "0"

    data_asset.extra_fields.pop("dclimate:zarr_group")
    fallback_details = stac_catalog.resolve_dataset_from_stac(
        catalog, "ecmwf_era5", "temperature_2m", "finalized"
    )
    assert fallback_details.zarr_group == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("explicit_group", "expected_group"),
    [(None, "0"), ("/2/", "/2/")],
)
async def test_client_prefers_explicit_group_over_stac(
    monkeypatch, explicit_group, expected_group
):
    details = stac_server.ResolvedDatasetDetails(
        cid="bafy-grouped", variant="default", zarr_group="0"
    )

    monkeypatch.setattr(
        client_module,
        "resolve_dataset_from_stac_server",
        lambda **kwargs: details,
    )

    observed_groups = []

    async def load_dataset_from_ipfs_cid(**kwargs):
        observed_groups.append(kwargs["zarr_group"])
        normalized = kwargs["zarr_group"].strip("/")
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
        zarr_group=explicit_group,
        return_xarray=True,
    )

    assert observed_groups == [expected_group]
    assert metadata["zarr_group"] == expected_group.strip("/")


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
async def test_client_reports_items_without_version_history(monkeypatch):
    client = dClimateClient()

    async def resolve_details(*args, **kwargs):
        return stac_server.ResolvedDatasetDetails("bafy", "default")

    monkeypatch.setattr(client, "_resolve_dataset_details", resolve_details)

    with pytest.raises(ValueError, match="Version history is not available"):
        await client.list_dataset_versions("copernicus_clms", "fpar")
