import datetime
import pathlib
import unittest
import unittest.mock
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import dclimate_zarr_client.client as client
from dclimate_zarr_client.dclimate_zarr_errors import (
    SelectionTooLargeError,
    ConflictingGeoRequestError,
    NoDataFoundError,
    ConflictingAggregationRequestError,
    InvalidExportFormatError,
    InvalidForecastRequestError,
    DatasetNotFoundError,
    IpfsConnectionError,
    StacCatalogError,
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


@unittest.mock.patch("dclimate_zarr_client.client.get_dataset_from_s3")
def test_load_s3(get_dataset_from_s3, dataset):
    get_dataset_from_s3.return_value = dataset

    data = client.load_s3("bucket_name", "dataset_name")
    assert data.data is dataset

    get_dataset_from_s3.assert_called_once_with("bucket_name", "dataset_name")


def patched_get_dataset_from_s3(dataset_name: str, bucket_name: str) -> xr.Dataset:
    print(f"Patching S3 load for: {dataset_name}")
    zip_path = None

    # Explicit checks for known test datasets
    if (
        "gfs_temp_max" in dataset_name
    ):  # Match the specific forecast dataset used in the test
        zip_path = pathlib.Path(__file__).parent / "etc" / "forecast_retrieval_test.zip"
        print(f"Identified as forecast, loading from: {zip_path}")
    elif (
        "era5_wind_100m_u" in dataset_name
    ):  # Match the specific standard dataset used in the test
        zip_path = pathlib.Path(__file__).parent / "etc" / "retrieval_test.zip"
        print(f"Identified as standard, loading from: {zip_path}")
    # Add more elif checks here if other specific test datasets are needed

    if zip_path is None:
        # Fallback or raise specific error if name not recognized
        print(
            f"WARNING: Dataset name '{dataset_name}' not explicitly handled by patch. Raising error."
        )
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

            print(
                f"Successfully opened dataset from {zip_path}. Dims: {ds.dims}, Coords: {list(ds.coords.keys())}"
            )
            # *** Add a check here to see the actual forecast_reference_time values ***
            if "forecast_reference_time" in ds.coords:
                print(
                    f"Forecast reference times in loaded data: {ds['forecast_reference_time'].values}"
                )
            else:
                print("No 'forecast_reference_time' coordinate found in loaded data.")
            return ds
    except Exception as e:
        raise ValueError(
            f"Failed to load test Zarr from {zip_path} for dataset '{dataset_name}': {type(e).__name__} - {e}"
        )


# --- Fixture using the updated patch ---
@pytest.fixture(scope="module")
def patch_s3(module_mocker):
    """
    Patch S3 dataset retrieval functions in this test module.
    """
    # Patch the function where it's called in the client module
    module_mocker.patch(
        "dclimate_zarr_client.client.get_dataset_from_s3",
        patched_get_dataset_from_s3,
    )
    # Patch it in the s3_retrieval module too, in case it's called directly elsewhere
    module_mocker.patch(
        "dclimate_zarr_client.s3_retrieval.get_dataset_from_s3",
        patched_get_dataset_from_s3,
    )


# @pytest.mark.usefixtures("patch_s3")
# def test_geo_temporal_query(polygons_mask, points_mask):
#     """
#     Test the `geo_temporal_query` method's functionalities: geographic queries,
#     aggregation methods, and export formats Geographic queries include point, rectangle,
#     circle, and polygon queries Aggregation methods include spatial, temporal, and
#     rolling temporal approaches for various mathematical operations Exports can be of
#     numpy array (default) or NetCDF format
#     """
#     point = client.geo_temporal_query(
#         dataset_name="era5_wind_100m_u-hourly",
#         bucket_name="zarr-dev",
#         point_kwargs={"latitude": 39.75, "longitude": -118.5},
#         rolling_agg_kwargs={"window_size": 5, "agg_method": "mean"},
#         point_limit=None,
#     )
#     rectangle = client.geo_temporal_query(
#         dataset_name="era5_wind_100m_u-hourly",
#         bucket_name="zarr-dev",
#         rectangle_kwargs={
#             "min_lat": 39.75,
#             "min_lon": -120.5,
#             "max_lat": 40.25,
#             "max_lon": -119.5,
#         },
#         var_name="u100",
#     )
#     rectangle_nc = client.geo_temporal_query(
#         dataset_name="era5_wind_100m_u-hourly",
#         bucket_name="zarr-dev",
#         rectangle_kwargs={
#             "min_lat": 39.75,
#             "min_lon": -120.5,
#             "max_lat": 40.25,
#             "max_lon": -119.5,
#         },
#         spatial_agg_kwargs={"agg_method": "max"},
#         output_format="netcdf",
#     )
#     circle = client.geo_temporal_query(
#         dataset_name="era5_wind_100m_u-hourly",
#         bucket_name="zarr-dev",
#         circle_kwargs={"center_lat": 40, "center_lon": -120, "radius": 150},
#         spatial_agg_kwargs={"agg_method": "std"},
#         temporal_agg_kwargs={"time_period": "day", "agg_method": "std", "time_unit": 1},
#     )
#     polygon = client.geo_temporal_query(
#         dataset_name="era5_wind_100m_u-hourly",
#         bucket_name="zarr-dev",
#         polygon_kwargs={"polygons_mask": polygons_mask, "epsg_crs": "epsg:4326"},
#         spatial_agg_kwargs={"agg_method": "mean"},
#         rolling_agg_kwargs={"window_size": 5, "agg_method": "mean"},
#     )

#     # NB, the following section is disabled for now because xarray 2024.3.0 does not support
#     # opening netcdfs as bytes directly due to a bug. Hopefully will be fixed in a later release
#     # so we can reenable the test

#     # points_arr = client.geo_temporal_query(
#     #     dataset_name="era5_wind_100m_u-hourly",
#     #     bucket_name="zarr-dev",
#     #     multiple_points_kwargs={"points_mask": points_mask, "epsg_crs": "epsg:4326"},
#     # )
#     # points_nc = client.geo_temporal_query(
#     #     dataset_name="era5_wind_100m_u-hourly",
#     #     bucket_name="zarr-dev",
#     #     multiple_points_kwargs={"points_mask": points_mask, "epsg_crs": "epsg:4326"},
#     #     output_format="netcdf",
#     # )

#     # points_nc = xr.open_dataset(points_nc)

#     # for i, (lat, lon) in enumerate(points_arr["points"]):
#     #     nc_vals = (
#     #         points_nc.where((points_nc.latitude == lat) & (points_nc.longitude == lon), drop=True)
#     #         .u100.values.flatten()
#     #         .tolist()
#     #     )
#     #     assert nc_vals == points_arr["data"][i]

#     assert point["data"][0] == pytest.approx(-2.013934326171875)
#     assert rectangle["data"][0][0][0] == pytest.approx(-1.9547119140625)
#     assert rectangle_nc[0] == 67
#     assert circle["data"][0] == pytest.approx(0.44366344809532166)
#     assert polygon["data"][0] == pytest.approx(-1.1927716255187988)

# --- Functional IPFS Tests ---


@pytest.mark.ipfs  # Add specific marker if needed
def test_load_ipfs_via_stac_functional():
    """Test loading a known dataset via STAC functionally."""
    try:
        data = client.load_ipfs_via_stac(KNOWN_STAC_DATASET_ID)
        assert isinstance(data.data, xr.Dataset)
        assert data.dataset_name == KNOWN_STAC_DATASET_ID
        assert "precip" in data.data.data_vars  # Check known variable
    except (DatasetNotFoundError, IpfsConnectionError, StacCatalogError) as e:
        pytest.fail(f"load_ipfs_via_stac failed: {e}")
    except Exception as e:
        pytest.fail(f"load_ipfs_via_stac failed with unexpected error: {e}")


@pytest.mark.ipfs
def test_geo_temporal_query_ipfs_functional(polygons_mask, points_mask):
    """
    Test the `geo_temporal_query` method functionally using IPFS source.
    Focus on basic point, rectangle, time range, and output formats.
    Aggregation tests might be slow; keep them simple or separate.
    """
    # Point query
    try:
        point_result = client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,
            source="ipfs",
            var_name="precip",  # Specify variable if needed
            point_kwargs={
                "latitude": KNOWN_STAC_COORD_LAT,
                "longitude": KNOWN_STAC_COORD_LON,
            },
            time_range=[KNOWN_STAC_DATE, KNOWN_STAC_DATE],  # Single day
            point_limit=None,  # Allow default limit
        )
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
        StacCatalogError,
        NoDataFoundError,
    ) as e:
        pytest.fail(f"IPFS point query failed: {e}")
    except Exception as e:
        pytest.fail(f"IPFS point query failed with unexpected error: {e}")

    # Rectangle query + NetCDF output
    try:
        rectangle_nc = client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,
            source="ipfs",
            var_name="precip",
            rectangle_kwargs={
                "min_lat": KNOWN_STAC_COORD_LAT - 0.1,
                "min_lon": KNOWN_STAC_COORD_LON - 0.1,
                "max_lat": KNOWN_STAC_COORD_LAT + 0.1,
                "max_lon": KNOWN_STAC_COORD_LON + 0.1,
            },
            time_range=[KNOWN_STAC_DATE, KNOWN_STAC_DATE_END],  # Small range
            output_format="netcdf",
        )
        assert isinstance(rectangle_nc, bytes)
        assert len(rectangle_nc) > 100  # Should not be empty

        # Optionally load back the netcdf to verify content
        ds_from_nc = xr.open_dataset(rectangle_nc)
        assert "precip" in ds_from_nc
        assert ds_from_nc.dims["time"] == 5  # 5 days inclusive
        assert ds_from_nc.dims["latitude"] >= 1
        assert ds_from_nc.dims["longitude"] >= 1

    except (
        DatasetNotFoundError,
        IpfsConnectionError,
        StacCatalogError,
        NoDataFoundError,
    ) as e:
        pytest.fail(f"IPFS rectangle query (netcdf) failed: {e}")
    except Exception as e:
        pytest.fail(f"IPFS rectangle query (netcdf) failed with unexpected error: {e}")

    # Basic spatial aggregation (mean over a small box)
    try:
        spatial_agg_result = client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,
            source="ipfs",
            var_name="precip",
            rectangle_kwargs={
                "min_lat": KNOWN_STAC_COORD_LAT - 0.5,
                "min_lon": KNOWN_STAC_COORD_LON - 0.5,
                "max_lat": KNOWN_STAC_COORD_LAT + 0.5,
                "max_lon": KNOWN_STAC_COORD_LON + 0.5,
            },
            spatial_agg_kwargs={"agg_method": "mean"},
            time_range=[KNOWN_STAC_DATE, KNOWN_STAC_DATE],  # Single day for speed
        )
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
        StacCatalogError,
        NoDataFoundError,
    ) as e:
        pytest.fail(f"IPFS spatial aggregation query failed: {e}")
    except Exception as e:
        pytest.fail(f"IPFS spatial aggregation query failed with unexpected error: {e}")


# --- Error Handling Tests (using IPFS source) ---

# These tests primarily check input validation or errors that occur *after* loading.
# They now need to use a valid IPFS dataset name.


def test_geo_conflicts_ipfs():
    """Test conflicting geo requests with IPFS source."""
    with pytest.raises(
        ConflictingGeoRequestError, match="more than one type of geographic query"
    ):
        client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,  # Use valid dataset
            source="ipfs",
            rectangle_kwargs={
                "min_lat": 40,
                "min_lon": -105,
                "max_lat": 41,
                "max_lon": -104,
            },
            circle_kwargs={"center_lat": 40.5, "center_lon": -104.5, "radius": 10},
        )
    with pytest.raises(
        ConflictingGeoRequestError,
        match="spatial aggregation methods on a single point",
    ):
        client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,
            source="ipfs",
            point_kwargs={
                "latitude": KNOWN_STAC_COORD_LAT,
                "longitude": KNOWN_STAC_COORD_LON,
            },
            spatial_agg_kwargs={"agg_method": "std"},
        )


@pytest.mark.skipif(
    KNOWN_STAC_FORECAST_ID is None,
    reason="No known forecast dataset ID for STAC testing",
)
def test_geo_forecast_conflicts_ipfs():
    """Test forecast-related errors with IPFS source."""
    # Requires a known forecast dataset ID available via STAC
    # Test requesting forecast dataset without specifying time
    with pytest.raises(InvalidForecastRequestError):
        client.geo_temporal_query(
            dataset_name=KNOWN_STAC_FORECAST_ID,  # Use known forecast dataset
            source="ipfs",
            rectangle_kwargs={
                "min_lat": 40,
                "min_lon": -105,
                "max_lat": 41,
                "max_lon": -104,
            },
            # Missing forecast_reference_time
        )
    # Test requesting forecast time from a non-forecast dataset
    with pytest.raises(MissingDimensionsError):
        client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,  # Use non-forecast dataset
            source="ipfs",
            forecast_reference_time="2023-01-01T00:00:00",  # Provide time
        )


def test_temp_agg_conflicts_ipfs():
    """Test conflicting temporal aggregation with IPFS source."""
    with pytest.raises(ConflictingAggregationRequestError):
        client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,
            source="ipfs",
            rectangle_kwargs={
                "min_lat": 40,
                "min_lon": -105,
                "max_lat": 41,
                "max_lon": -104,
            },
            temporal_agg_kwargs={"time_period": "day", "agg_method": "std"},
            rolling_agg_kwargs={"window_size": 5, "agg_method": "mean"},
        )


def test_invalid_export_ipfs():
    """Test invalid export format with IPFS source."""
    with pytest.raises(InvalidExportFormatError):
        client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,
            source="ipfs",
            point_kwargs={
                "latitude": KNOWN_STAC_COORD_LAT,
                "longitude": KNOWN_STAC_COORD_LON,
            },
            output_format="GRIB",  # Invalid format
        )


def test_selection_size_conflicts_ipfs(oversized_polygons_mask):
    """Test selection size limits with IPFS source."""
    # This test depends heavily on the dataset's resolution and the limit.
    # Use a small limit to force the error.
    with pytest.raises(SelectionTooLargeError, match="more than limit of 100"):
        client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,
            source="ipfs",
            # Select a relatively large area (adjust lat/lon range if needed for the dataset)
            rectangle_kwargs={
                "min_lat": 30,
                "min_lon": -90,
                "max_lat": 40,
                "max_lon": -70,
            },
            point_limit=100,  # Set a very small limit
        )


def test_no_data_in_selection_error_ipfs():
    """Test requesting data outside the dataset's actual range (functional)."""
    with pytest.raises(NoDataFoundError):
        client.geo_temporal_query(
            dataset_name=KNOWN_STAC_DATASET_ID,
            source="ipfs",
            # Use a time range known to be outside the dataset's bounds
            time_range=[datetime.datetime(1900, 1, 1), datetime.datetime(1910, 1, 1)],
            point_kwargs={
                "latitude": KNOWN_STAC_COORD_LAT,
                "longitude": KNOWN_STAC_COORD_LON,
            },
        )


# --- Keep S3 Mocked Tests if necessary ---
# These tests use the patch_s3 fixture if still needed for specific client logic testing
# Otherwise, S3 testing should ideally be in test_s3_retrieval using moto/localstack


# Example: Keeping the TestClient class structure if it tests S3-specific client paths
# @pytest.mark.usefixtures("patch_s3") # Apply fixture if needed
# --- Test Class ---
# --- Test Class ---
class TestClientS3Specific:
    @pytest.mark.usefixtures("patch_s3")
    def test__given_bucket_and_dataset_names__then__fetch_geo_temporal_query_from_S3(
        self, mocker
    ):
        dataset_name = "era5_wind_100m_u-hourly"
        bucket_name = "zarr-dev"
        test_lat = 45.0
        test_lon = -119.5

        result = client.geo_temporal_query(
            dataset_name=dataset_name,
            bucket_name=bucket_name,
            source="s3",
            point_kwargs={"latitude": test_lat, "longitude": test_lon},
        )
        assert isinstance(result, dict)
        assert "data" in result
        assert len(result.get("times", [])) > 0
        assert len(result.get("data", [])) > 0

    @pytest.mark.usefixtures("patch_s3")
    def test__given_bucket_and_dataset_names_and_forecast_reference_time_then__fetch_geo_temporal_query_from_S3(
        self,
        mocker,
        forecast_ds,  # Inject the fixture to verify its contents
    ):
        dataset_name = "gfs_temp_max-hourly"
        bucket_name = "zarr-dev"

        # *** Check for 'lat'/'lon' in the fixture data ***
        coord_lat_name = "latitude" if "latitude" in forecast_ds.coords else "lat"
        coord_lon_name = "longitude" if "longitude" in forecast_ds.coords else "lon"

        if (
            coord_lat_name not in forecast_ds.coords
            or coord_lon_name not in forecast_ds.coords
        ):
            pytest.fail(
                f"The forecast_ds fixture is missing '{coord_lat_name}' or '{coord_lon_name}' coordinates. Found: {list(forecast_ds.coords.keys())}"
            )
        if "forecast_reference_time" not in forecast_ds.coords:
            pytest.fail(
                "The forecast_ds fixture (from forecast_retrieval_test.zip) is missing 'forecast_reference_time' coordinate."
            )
        if not forecast_ds["forecast_reference_time"].size > 0:
            pytest.fail(
                "The forecast_ds fixture has an empty 'forecast_reference_time' coordinate."
            )
        if (
            not forecast_ds[coord_lat_name].size > 0
            or not forecast_ds[coord_lon_name].size > 0
        ):
            pytest.fail(
                f"The forecast_ds fixture has empty '{coord_lat_name}' or '{coord_lon_name}' coordinates."
            )

        # Use the *first* available time and coords from the fixture data
        valid_forecast_time_dt64 = forecast_ds["forecast_reference_time"].values[0]
        valid_forecast_time_str = pd.Timestamp(valid_forecast_time_dt64).isoformat()

        # *** Extract using the identified coordinate names ***
        valid_lat = forecast_ds[coord_lat_name].values[0]
        valid_lon = forecast_ds[coord_lon_name].values[0]

        print(f"Using forecast time from test file: {valid_forecast_time_str}")
        print(
            f"Using coords from test file: {coord_lat_name}={valid_lat}, {coord_lon_name}={valid_lon}"
        )

        try:
            result = client.geo_temporal_query(
                dataset_name=dataset_name,
                source="s3",
                bucket_name=bucket_name,
                forecast_reference_time=valid_forecast_time_str,
                # *** Call API using 'latitude'/'longitude' kwargs ***
                point_kwargs={"latitude": valid_lat, "longitude": valid_lon},
            )

            assert isinstance(result, dict)
            assert "data" in result
            assert len(result.get("times", [])) > 0  # Should have forecast steps
            assert len(result.get("data", [])) > 0
            assert "time" in result.get("dimensions_order", [])

        except KeyError as e:
            pytest.fail(
                f"KeyError during forecast test: {e}. Check coordinates/time in forecast_retrieval_test.zip and patch logic."
            )
        except Exception as e:
            pytest.fail(
                f"Unexpected error during forecast test: {type(e).__name__} - {e}"
            )
