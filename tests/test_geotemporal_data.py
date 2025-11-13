import datetime

import numpy as np
import pytest

from dclimate_client_py import dclimate_zarr_errors as errors
from dclimate_client_py.geotemporal_data import GeotemporalData


class TestGeotemporalData:
    @staticmethod
    def test_ctor(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        assert data.data is dataset

    @staticmethod
    def test_use_data_variable(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")

        data = data.use(data_var="var_1")
        assert np.array_equal(data.data_var, data.data.data_vars["var_1"])

        data = data.use(data_var="var_2")
        assert np.array_equal(data.data_var, data.data.data_vars["var_2"])

    @staticmethod
    def test_use_data_variable_w_dataset_name(dataset):
        data = GeotemporalData(dataset, dataset_name="top 100 penguin names")

        assert data.dataset_name == "top 100 penguin names"

        data = data.use(data_var="var_1")
        assert np.array_equal(data.data_var, data.data.data_vars["var_1"])

        data = data.use(data_var="var_2")
        assert np.array_equal(data.data_var, data.data.data_vars["var_2"])

    @staticmethod
    def test_use_data_variable_bad_var_name(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")

        with pytest.raises(KeyError):
            data.use("doesn't exist")

    @staticmethod
    def test_ambiguous_data_variable(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")

        with pytest.raises(errors.AmbiguousDataVariableError):
            data.data_var

        data.use(data_var="var_1")  # Doesn't modify existing dataset, returns new one

        with pytest.raises(errors.AmbiguousDataVariableError):
            data.data_var

    @staticmethod
    def test_single_data_variable(single_var_dataset):
        data = GeotemporalData(single_var_dataset, dataset_name="fake dataset")

        assert np.array_equal(data.data_var, data.data.data_vars["var_1"])

    @staticmethod
    def test_check_dataset_size_ok(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        data.check_dataset_size()  # doesn't raise

    @staticmethod
    def test_check_dataset_size_too_large(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        with pytest.raises(errors.SelectionTooLargeError):
            data.check_dataset_size(7999)

    @staticmethod
    def test_check_has_data_ok(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset").use(
            data_var="var_1"
        )
        data.check_has_data()  # doesn't raise

    @staticmethod
    def test_check_has_data_empty(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        begin = datetime.datetime(1900, 1, 10)
        end = datetime.datetime(1900, 1, 15)
        data = data.time_range(begin, end).use(data_var="var_1")

        with pytest.raises(errors.NoDataFoundError):
            data.check_has_data()  # doesn't raise

    @staticmethod
    def test_point(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        data = data.point(20, 190)
        assert data.data.latitude.values == 20
        assert data.data.longitude.values == 190

    @staticmethod
    def test_point_snap_to_grid(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        data = data.point(22, 192)
        assert data.data.latitude.values == 20
        assert data.data.longitude.values == 190

    @staticmethod
    def test_point_inexact_coordinates_without_snap(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        with pytest.raises(errors.NoDataFoundError):
            data.point(22, 192, snap_to_grid=False)

    @staticmethod
    def test_circle(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        data = data.circle(20, 90, 1500)
        assert np.array_equal(data.data.latitude, (150, 160, 170))
        assert np.array_equal(data.data.longitude, (260, 265, 270, 275))

    @staticmethod
    def test_rectangle(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        data = data.rectangle(20, 190, 40, 200)
        assert np.array_equal(data.data.latitude, (20, 30, 40))
        assert np.array_equal(data.data.longitude, (190, 195, 200))

    @staticmethod
    def test_time_range(dataset):
        data = GeotemporalData(dataset, dataset_name="fake dataset")
        begin = datetime.datetime(2000, 1, 10)
        end = datetime.datetime(2000, 1, 15)
        new_data = data.time_range(begin, end)

        assert new_data.data["time"][0] == np.datetime64(
            "2000-01-10T00:00:00.000000000"
        )

        assert new_data.data["time"][-1] == np.datetime64(
            "2000-01-15T00:00:00.000000000"
        )

        assert data.data["time"][0] == np.datetime64("2000-01-01T00:00:00.000000000")

    @staticmethod
    def test_forecast(forecast_ds):
        """
        Test that forceast_reference_time is properly removed from forecast datasets
        and the time dimension is narrowed to the number of forecasts
        """
        data = GeotemporalData(forecast_ds, dataset_name="fake dataset")
        data = data.forecast("2021-05-05")
        assert "forecast_reference_time" not in data.data
        assert len(data.data.time) == 5

    @staticmethod
    def test_reindex_forecast(forecast_ds):
        """
        Test that forceast_reference_time is properly removed from forecast datasets
        and the time dimension is narrowed to the number of forecasts
        """
        data = GeotemporalData(forecast_ds, dataset_name="fake dataset")
        data = data.forecast("2021-05-05")
        data = data.reindex_forecast()
        assert (
            len(data.data.time) == 7
        )  # tests that "missing" steps are successfully filled in by `reindex`

    @staticmethod
    def test_get_point_for_small_polygon(input_ds, undersized_polygons_mask):
        """
        Test that providing get_points_in_polygons a polygon_mask smaller than any grid
        cell returns a single point dataset for the point closest to that polygon's
        centroid
        """
        data = GeotemporalData(input_ds, dataset_name="fake dataset")
        data = data.polygons(polygons_mask=undersized_polygons_mask)
        assert data.data["u100"].values[0] == pytest.approx(-2.3199463)

    @staticmethod
    def test_rolling_aggregation(input_ds):
        """
        Test various descriptive statistics methods with the rolling aggregation
        approach
        """
        data = GeotemporalData(input_ds, dataset_name="fake dataset")
        mean_vals = data.rolling_aggregation(5, "mean")
        max_vals = data.rolling_aggregation(5, "max")
        std_vals = data.rolling_aggregation(5, "std")
        assert mean_vals.data["u100"].values[0][0][0] == pytest.approx(
            7.912628173828125
        )
        assert max_vals.data["u100"].values[0][0][0] == pytest.approx(8.5950927734375)
        assert std_vals.data["u100"].values[0][0][0] == pytest.approx(
            0.5272848606109619
        )

    @staticmethod
    def test_temporal_aggregation(input_ds):
        """
        Test various descriptive statistics methods with the temporal (resampling)
        aggregation approach
        """
        data = GeotemporalData(input_ds, dataset_name="fake dataset")

        # [0][0][0] returns the first value for latitude 45.0, longitue -140.0
        daily_maxs = data.temporal_aggregation(
            time_period="day", agg_method="max", time_unit=2
        )
        monthly_means = data.temporal_aggregation(
            time_period="month", agg_method="mean"
        )
        yearly_std = data.temporal_aggregation(time_period="year", agg_method="std")
        all_sum = data.temporal_aggregation(time_period="all", agg_method="sum")
        assert daily_maxs.data["u100"].values[0][0][0] == pytest.approx(8.5950927734375)
        assert monthly_means.data["u100"].values[0][0][0] == pytest.approx(
            -0.19848469, rel=1e-5
        )
        assert yearly_std.data["u100"].values[0][0][0] == pytest.approx(
            6.490322113037109
        )
        assert all_sum.data["u100"].values[0][0] == pytest.approx(-33.345367)

    @staticmethod
    def test_spatial_aggregation(input_ds):
        """
        Test various descriptive statistics methods with the spatial aggregation
        approach
        """
        data = GeotemporalData(input_ds, dataset_name="fake dataset")
        mean_vals_all_pts = data.spatial_aggregation("mean")
        min_val_rep_pt = data.spatial_aggregation("min")
        assert float(mean_vals_all_pts.data["u100"].values[0]) == pytest.approx(
            1.5880329608917236
        )
        assert float(min_val_rep_pt.data["u100"].values[0]) == pytest.approx(
            -9.5386962890625
        )
