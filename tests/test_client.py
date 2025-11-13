import datetime
import pathlib
import unittest
import unittest.mock
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import dclimate_client_py.client as client
from dclimate_client_py.dclimate_zarr_errors import (
    SelectionTooLargeError,
    ConflictingGeoRequestError,
    NoDataFoundError,
    ConflictingAggregationRequestError,
    InvalidExportFormatError,
    InvalidForecastRequestError,
    DatasetNotFoundError,
    IpfsConnectionError,
)
from xarray.core.variable import MissingDimensionsError

# Import constants from conftest
from .conftest import (
    KNOWN_STAC_DATASET_ID,
    KNOWN_STAC_COORD_LAT,
    KNOWN_STAC_COORD_LON,
    KNOWN_STAC_DATE,
    KNOWN_STAC_DATE_END,
    KNOWN_STAC_FORECAST_ID,
)

# --- Test Markers ---
# pytestmark = pytest.mark.client # Mark tests specific to the client module
# Apply IPFS check fixture to relevant tests/module if not session-wide autouse
pytestmark = pytest.mark.usefixtures("check_ipfs_connection")

# Keep S3 sample path if needed for S3 tests
SAMPLE_ZARRS = pathlib.Path(__file__).parent / "etc" / "sample_zarrs"


@unittest.mock.patch("dclimate_client_py.client.get_dataset_from_s3")
def test_load_s3(get_dataset_from_s3, dataset):
    get_dataset_from_s3.return_value = dataset

    data = client.load_s3("bucket_name", "dataset_name")
    assert data.data is dataset

    get_dataset_from_s3.assert_called_once_with("bucket_name", "dataset_name")


def patched_get_dataset_from_s3(dataset_name: str, bucket_name: str) -> xr.Dataset:
    zip_path = None

    # Explicit checks for known test datasets
    if (
        "gfs_temp_max" in dataset_name
    ):  # Match the specific forecast dataset used in the test
        zip_path = pathlib.Path(__file__).parent / "etc" / "forecast_retrieval_test.zip"
    elif (
        "era5_wind_100m_u" in dataset_name
    ):  # Match the specific standard dataset used in the test
        zip_path = pathlib.Path(__file__).parent / "etc" / "retrieval_test.zip"
    # Add more elif checks here if other specific test datasets are needed

    if zip_path is None:
        # Fallback or raise specific error if name not recognized

        raise ValueError(
            f"Test patch does not have a specific zip file configured for dataset: {dataset_name}"
        )

    if not zip_path.exists():
        raise FileNotFoundError(f"Required test zip file not found: {zip_path}")

    try:
        with zarr.storage.ZipStore(str(zip_path), mode="r") as in_zarr:
            ds = xr.open_zarr(in_zarr)
            ds.attrs.setdefault("update_in_progress", False)
            # Add minimal necessary coords if needed, avoid adding forecast_reference_time here
            if "latitude" not in ds.coords:
                ds = ds.assign_coords(latitude=np.arange(40, 50))
            if "longitude" not in ds.coords:
                ds = ds.assign_coords(longitude=np.arange(-120, -110))
    except Exception as e:
        raise ValueError(
            f"Failed to load test Zarr from {zip_path} for dataset '{dataset_name}': {type(e).__name__} - {e}"
        ) from e


# --- Fixture using the updated patch ---
@pytest.fixture(scope="module")
def patch_s3(module_mocker):
    """
    Patch S3 dataset retrieval functions in this test module.
    """
    # Patch the function where it's called in the client module
    module_mocker.patch(
        "dclimate_client_py.client.get_dataset_from_s3",
        patched_get_dataset_from_s3,
    )
    # Patch it in the s3_retrieval module too, in case it's called directly elsewhere
    module_mocker.patch(
        "dclimate_client_py.s3_retrieval.get_dataset_from_s3",
        patched_get_dataset_from_s3,
    )


# --- Functional IPFS Tests ---
@pytest.mark.ipfs
@pytest.mark.asyncio
async def test_geo_temporal_query_ipfs_functional(polygons_mask, points_mask):
    """
    Test geotemporal queries functionally using IPFS source via dClimateClient.
    Focus on basic point, rectangle, time range, and output formats.
    Aggregation tests might be slow; keep them simple or separate.
    """
    from dclimate_client_py.dclimate_client import dClimateClient

    async with dClimateClient() as dclimate:
        # Point query
        try:
            # Load the dataset once
            dataset = await dclimate.load_dataset(
                collection="era5",
                dataset="2m_temperature",
                variant="finalized",
                return_xarray=False,
            )

            # Use the specific variable if needed
            if "2m_temperature" in dataset.data.data_vars:
                dataset = dataset.use("2m_temperature")

            # Apply point query
            point_data = dataset.point(
                latitude=KNOWN_STAC_COORD_LAT,
                longitude=KNOWN_STAC_COORD_LON,
            )

            # Apply time range filter
            point_data = point_data.query(
                time_range=[KNOWN_STAC_DATE, KNOWN_STAC_DATE],
            )

            point_result = point_data.as_dict()
            assert isinstance(point_result, dict)
            assert "data" in point_result
            # Value can be float, int, or None if no data/NaN
            assert (
                isinstance(point_result["data"][0], (float, int))
                or point_result["data"][0] is None
            )
            # We know this point/date has data for cpc-precip-conus
            assert point_result["data"][0] is not None
            assert point_result["data"][0] > -1  # Precip should be >= 0

        except (
            DatasetNotFoundError,
            IpfsConnectionError,
            NoDataFoundError,
        ) as e:
            pytest.fail(f"IPFS point query failed: {e}")
        except Exception as e:
            pytest.fail(f"IPFS point query failed with unexpected error: {e}")

        # Rectangle query + NetCDF output
        try:
            # Load the dataset again for rectangle query
            dataset = await dclimate.load_dataset(
                collection="era5",
                dataset="2m_temperature",
                variant="finalized",
                return_xarray=False,
            )

            # Use the specific variable if needed
            if "2m_temperature" in dataset.data.data_vars:
                dataset = dataset.use("2m_temperature")

            # Apply rectangle query
            rect_data = dataset.rectangle(
                min_lat=KNOWN_STAC_COORD_LAT - 0.5,
                min_lon=KNOWN_STAC_COORD_LON - 0.5,
                max_lat=KNOWN_STAC_COORD_LAT + 0.5,
                max_lon=KNOWN_STAC_COORD_LON + 0.5,
            )

            # Apply time range filter
            rect_data = rect_data.query(
                time_range=[KNOWN_STAC_DATE, KNOWN_STAC_DATE_END],
            )

            rectangle_nc = rect_data.to_netcdf()
            assert isinstance(rectangle_nc, bytes)
            assert len(rectangle_nc) > 100  # Should not be empty

            # Optionally load back the netcdf to verify content
            import io
            ds_from_nc = xr.open_dataset(io.BytesIO(rectangle_nc))
            assert "2m_temperature" in ds_from_nc or len(ds_from_nc.data_vars) > 0
            assert ds_from_nc.dims["time"] == 97  # 97 hours inclusive
            assert "latitude" in ds_from_nc.dims or "lat" in ds_from_nc.dims
            assert "longitude" in ds_from_nc.dims or "lon" in ds_from_nc.dims

        except (
            DatasetNotFoundError,
            IpfsConnectionError,
            NoDataFoundError,
        ) as e:
            pytest.fail(f"IPFS rectangle query (netcdf) failed: {e}")
        except Exception as e:
            pytest.fail(f"IPFS rectangle query (netcdf) failed with unexpected error: {e}")

        # Basic spatial aggregation (mean over a small box)
        try:
            # Load the dataset again for spatial aggregation query
            dataset = await dclimate.load_dataset(
                collection="era5",
                dataset="2m_temperature",
                variant="finalized",
                return_xarray=False,
            )

            # Use the specific variable if needed
            if "2m_temperature" in dataset.data.data_vars:
                dataset = dataset.use("2m_temperature")

            # Apply rectangle query
            agg_data = dataset.rectangle(
                min_lat=KNOWN_STAC_COORD_LAT - 0.5,
                min_lon=KNOWN_STAC_COORD_LON - 0.5,
                max_lat=KNOWN_STAC_COORD_LAT + 0.5,
                max_lon=KNOWN_STAC_COORD_LON + 0.5,
            )

            # Apply spatial aggregation and time filter
            agg_data = agg_data.query(
                spatial_agg_kwargs={"agg_method": "mean"},
                time_range=[KNOWN_STAC_DATE, KNOWN_STAC_DATE],
            )

            spatial_agg_result = agg_data.as_dict()
            assert isinstance(spatial_agg_result, dict)
            assert "data" in spatial_agg_result
            assert (
                isinstance(spatial_agg_result["data"][0], (float, int))
                or spatial_agg_result["data"][0] is None
            )
            # Should have aggregated spatially, leaving only time dim (size 1 here)
            assert (
                isinstance(spatial_agg_result["data"], list)
                and len(spatial_agg_result["data"]) == 1
            )

        except (
            DatasetNotFoundError,
            IpfsConnectionError,
            NoDataFoundError,
        ) as e:
            pytest.fail(f"IPFS spatial aggregation query failed: {e}")
        except Exception as e:
            pytest.fail(f"IPFS spatial aggregation query failed with unexpected error: {e}")


# --- Error Handling Tests ---
# Note: The old geo_temporal_query with source="ipfs" is deprecated.
# Error validation tests for the new dClimateClient API are handled in
# test_geo_temporal_query_ipfs_functional and other dClimateClient-specific tests.

# --- Keep S3 Mocked Tests if necessary ---
# These tests use the patch_s3 fixture if still needed for specific client logic testing
# Otherwise, S3 testing should ideally be in test_s3_retrieval using moto/localstack


# Example: Keeping the TestClient class structure if it tests S3-specific client paths
# @pytest.mark.usefixtures("patch_s3") # Apply fixture if needed
# --- Test Class ---
# --- Test Class ---
# class TestClientS3Specific:
#     @pytest.mark.usefixtures("patch_s3")
#     def test__given_bucket_and_dataset_names__then__fetch_geo_temporal_query_from_S3(
#         self, mocker
#     ):
#         dataset_name = "era5_wind_100m_u-hourly"
#         bucket_name = "zarr-dev"
#         test_lat = 45.0
#         test_lon = -119.5

#         result = client.geo_temporal_query(
#             dataset_name=dataset_name,
#             bucket_name=bucket_name,
#             source="s3",
#             point_kwargs={"latitude": test_lat, "longitude": test_lon},
#         )
#         assert isinstance(result, dict)
#         assert "data" in result
#         assert len(result.get("times", [])) > 0
#         assert len(result.get("data", [])) > 0

#     @pytest.mark.usefixtures("patch_s3")
#     def test__given_bucket_and_dataset_names_and_forecast_reference_time_then__fetch_geo_temporal_query_from_S3(
#         self,
#         mocker,
#         forecast_ds,  # Inject the fixture to verify its contents
#     ):
#         dataset_name = "gfs_temp_max-hourly"
#         bucket_name = "zarr-dev"

#         # *** Check for 'lat'/'lon' in the fixture data ***
#         coord_lat_name = "latitude" if "latitude" in forecast_ds.coords else "lat"
#         coord_lon_name = "longitude" if "longitude" in forecast_ds.coords else "lon"

#         if (
#             coord_lat_name not in forecast_ds.coords
#             or coord_lon_name not in forecast_ds.coords
#         ):
#             pytest.fail(
#                 f"The forecast_ds fixture is missing '{coord_lat_name}' or '{coord_lon_name}' coordinates. Found: {list(forecast_ds.coords.keys())}"
#             )
#         if "forecast_reference_time" not in forecast_ds.coords:
#             pytest.fail(
#                 "The forecast_ds fixture (from forecast_retrieval_test.zip) is missing 'forecast_reference_time' coordinate."
#             )
#         if not forecast_ds["forecast_reference_time"].size > 0:
#             pytest.fail(
#                 "The forecast_ds fixture has an empty 'forecast_reference_time' coordinate."
#             )
#         if (
#             not forecast_ds[coord_lat_name].size > 0
#             or not forecast_ds[coord_lon_name].size > 0
#         ):
#             pytest.fail(
#                 f"The forecast_ds fixture has empty '{coord_lat_name}' or '{coord_lon_name}' coordinates."
#             )

#         # Use the *first* available time and coords from the fixture data
#         valid_forecast_time_dt64 = forecast_ds["forecast_reference_time"].values[0]
#         valid_forecast_time_str = pd.Timestamp(valid_forecast_time_dt64).isoformat()

#         # *** Extract using the identified coordinate names ***
#         valid_lat = forecast_ds[coord_lat_name].values[0]
#         valid_lon = forecast_ds[coord_lon_name].values[0]

#         try:
#             result = client.geo_temporal_query(
#                 dataset_name=dataset_name,
#                 source="s3",
#                 bucket_name=bucket_name,
#                 forecast_reference_time=valid_forecast_time_str,
#                 # *** Call API using 'latitude'/'longitude' kwargs ***
#                 point_kwargs={"latitude": valid_lat, "longitude": valid_lon},
#             )

#             assert isinstance(result, dict)
#             assert "data" in result
#             assert len(result.get("times", [])) > 0  # Should have forecast steps
#             assert len(result.get("data", [])) > 0
#             assert "time" in result.get("dimensions_order", [])

#         except KeyError as e:
#             pytest.fail(
#                 f"KeyError during forecast test: {e}. Check coordinates/time in forecast_retrieval_test.zip and patch logic."
#             )
#         except Exception as e:
#             pytest.fail(
#                 f"Unexpected error during forecast test: {type(e).__name__} - {e}"
#             )
