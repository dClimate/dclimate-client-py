import datetime
import functools
import math
import numbers
import operator
import typing
from collections.abc import Mapping

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
import xarray as xr
from xarray.core.variable import MissingDimensionsError

from .dclimate_zarr_errors import (
    InvalidForecastRequestError,
)
from . import dclimate_zarr_errors as errors

# Users should not select more than this number of data points and coordinates
DEFAULT_POINT_LIMIT = 40 * 40 * 50_000
BoundsSelection = typing.Union[typing.Sequence[float], Mapping[str, typing.Any]]
GeoSelectionOptions = Mapping[str, typing.Any]


class GeotemporalData:
    """Wrapper for an Xarray dataset with extensions for geotemporal data

    Parameters
    ----------
    data: xr.Dataset
        The Xarray dataset to wrap.
    var_name: str
        The name of the data variable to use with methods that operate on a specific
        data variable.
    """

    def __init__(self, data: xr.Dataset, dataset_name: str, data_var: str = None):
        self.data = data
        self.dataset_name = dataset_name
        self._data_var = data_var

    def use(self, data_var: str) -> "GeotemporalData":
        """Set the data variable to use.

        If a dataset only has one data variable, that one will be used by default.
        Otherwise this method must be called before any operation that needs a specific
        data variable.

        Parameters
        ----------

        var_name: str
            The name of the data variable to use.

        Returns
        -------
        GeotemporalData
            A new :class:`GeotemporalData` that uses the named data variable for methods
            that operate on a specific data variable.

        Raises
        ------
        KeyError
            If the named variable is not present.
        """
        if data_var not in self.data.data_vars:
            raise KeyError(data_var)

        return type(self)(self.data, dataset_name=self.dataset_name, data_var=data_var)

    @property
    def data_var(self) -> xr.DataArray:
        """Get the current, in use data variable.

        Returns
        -------
        xr.DataArray
            The current data variable from the wrapped dataset.

        Raises
        ------
        AmbiguousDataVariableError
            If the wrapped dataset contains more than one data variable and
            :meth:`GeotemporalData.use` hasn't been called to select one.
        """
        if self._data_var is None:
            if len(self.data.data_vars) == 1:
                return next(iter(self.data.data_vars.values()))

            raise errors.AmbiguousDataVariableError

        return self.data.data_vars[self._data_var]

    def check_dataset_size(self, point_limit: int = DEFAULT_POINT_LIMIT):
        """Checks how many data points are in a dataset

        Parameters
        ----------
        point_limit, optional: limit for dataset size. Defaults to DEFAULT_POINT_LIMIT.

        Raises
        ------
        SelectionTooLargeError
            When dataset size limit is violated
        """
        num_points = functools.reduce(operator.mul, self.data.sizes.values(), 1)
        if num_points > point_limit:
            raise errors.SelectionTooLargeError(
                f"Selection of {num_points} data points is more than limit of {point_limit}"
            )

    def check_has_data(self):
        """Checks if data is all NA

        Raises
        ------
        NoDataFoundError
            When data is all NA
        """
        if self.data_var.isnull().all():
            raise errors.NoDataFoundError("Selection is empty or all NA")

    def forecast(self, forecast_reference_time: datetime.datetime) -> "GeotemporalData":
        """
        Filter a 4D forecast dataset to a 3D dataset ready for analysis

        Parameters
        ----------

        forecast_reference_time, str
            Isoformatted string representing the desire date to return all available
            forecasts for

        Returns
        -------
        GeotemporalData
            3D dataset with forecast hour added to time dimension
        """
        # select only the requested date
        data = self.data.sel(forecast_reference_time=forecast_reference_time)
        # Set time to equal the forecast time, not the forecast reference time.
        # Assumes only one forecast reference time is returned
        data = data.assign_coords(
            step=data.forecast_reference_time.values + data.step.values
        )
        # Remove forecast reference hour
        data = data.squeeze().drop_vars("forecast_reference_time")
        # Make forecasted data the time dimension
        data = data.rename({"step": "time"})
        return self._new(data)

    def reindex_forecast(self) -> "GeotemporalData":
        """
        Fill in missing forecast hours so the API call returns a full time series
        with None for missing (not forecasted) hours.
        This ensures backwards compatibility with the v3 API.

        Args:
            xr.Dataset: 3D Xarray dataset with forecast hour added to time dimension

        Returns:
            xr.Dataset: 3D Xarray dataset with the time dimension reindexed to include hours
                not forecasted
        """
        trange = pd.date_range(
            start=self.data.time[0].values, end=self.data.time[-1].values, freq="1h"
        )
        return self._new(self.data.reindex(time=trange))

    def point(
        self,
        latitude: float,
        longitude: float,
        snap_to_grid: bool = True,
        latitude_key: str = "latitude",
        longitude_key: str = "longitude",
    ) -> "GeotemporalData":
        """Gets a dataset corresponding to the full time series for a single point

        Parameters
        ----------
        latitude: float
            Latitude coordinate
        longitude: float
            Longitude coordinate
        snap_to_grid: bool, optional
            When ``True``, find nearest point to lat, lon in dataset. When ``False``,
            error out when exact lat, lon is not on dataset grid.
        latitude_key: str, optional
            Name of the latitude coordinate in the dataset.
        longitude_key: str, optional
            Name of the longitude coordinate in the dataset.

        Returns
        -------
        GeotemporalData
            New dataset restricted to single point
        """
        selection = {latitude_key: latitude, longitude_key: longitude}
        if snap_to_grid:
            data = self.data.sel(selection, method="nearest")
        else:
            try:
                data = self.data.sel(selection, method="nearest", tolerance=1e-5)
            except KeyError:
                raise errors.NoDataFoundError(
                    "User requested not to snap_to_grid, but exact coord not in dataset"
                )

        return self._new(data)

    def points(
        self,
        points_mask: gpd.array.GeometryArray,
        epsg_crs: int,
        snap_to_grid: bool = True,
    ) -> "GeotemporalData":
        mask = list(gpd.geoseries.GeoSeries(points_mask).set_crs(epsg_crs).to_crs(4326))
        lats, lons = [point.y for point in mask], [point.x for point in mask]
        lats, lons = xr.DataArray(lats, dims="point"), xr.DataArray(lons, dims="point")

        if snap_to_grid:
            data = self.data.sel(latitude=lats, longitude=lons, method="nearest")
        else:
            try:
                data = self.data.sel(
                    latitude=lats, longitude=lons, method="nearest", tolerance=1e-5
                )
            except KeyError:
                raise errors.NoDataFoundError(
                    "User requested not to snap_to_grid, but at least one coord not in dataset"
                )

        # Aggregations pull whole dataset when ds is structured as multiple points.
        # Forcing xarray to do subsetting before aggregation drastically speeds up agg
        data = data.compute()

        return self._new(data)

    def circle(
        self,
        lat: float,
        lon: float,
        radius: float,
    ) -> xr.Dataset:
        """Reduces dataset to points within radius of given center coordinates

        Parameters
        ----------

        lat: float
            Latitude coordinate of center
        lon: float
            Longitude coordinate of center
        radius: float)
            Radius of circle in kilometers

        Returns
        -------
        GeotempoeralData
            New dataset
        """
        distances = _haversine(lat, lon, self.data["latitude"], self.data["longitude"])
        data = self.data.where(distances < radius, drop=True)
        return self._new(data)

    def rectangle(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        latitude_key: str = "latitude",
        longitude_key: str = "longitude",
    ) -> "GeotemporalData":
        """Reduce dataset to points in rectangle

        Parameters
        ----------

        min_lat: float
            Southern limit of rectangle
        min_lon: float
            Western limit of rectangle
        max_lat: float
            Northern limit of rectangle
        max_lon: float
            Eastern limit of rectangle
        latitude_key: str, optional
            Name of the latitude coordinate in the dataset.
        longitude_key: str, optional
            Name of the longitude coordinate in the dataset.

        Returns
        -------
        GeotemporalData
            New dataset
        """
        try:
            latitudes = self.data[latitude_key]
            longitudes = self.data[longitude_key]
        except KeyError as exc:
            raise errors.InvalidSelectionError(
                "Latitude/longitude coordinates were not found in the dataset."
            ) from exc

        data = self.data.where(
            (latitudes >= min_lat)
            & (latitudes <= max_lat)
            & (longitudes >= min_lon)
            & (longitudes <= max_lon),
            drop=True,
        )
        return self._new(data)

    def polygons(
        self,
        polygons_mask: gpd.array.GeometryArray,
        epsg_crs: int = 4326,
        point_limit=DEFAULT_POINT_LIMIT,
    ) -> "GeotemporalData":
        """Reduces dataset to points within arbitrary shape.

        Requires rioxarray to be installed

        Parameters
        ----------
        polygons_mask: gpd.array.GeometryArray[shapely.geometry.multipolygon.MultiPolygon]
            Array of MultiPolygon shapes defining the area of interest
        epsg_crs: int
          EPSG code for polygons_mask (see
            https://en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset)

        Returns
        -------
        GeotemporalData
            New dataset
        """
        try:
            import rioxarray
        except ImportError as exc:
            raise ImportError(
                "GeotemporalData.polygons() requires rioxarray to be installed"
            ) from exc

        # If the polygon(s) are collectively smaller than the size of one grid cell,
        # clipping will return no data In this case return data from the grid cell nearest
        # to the center of the polygon
        if self.data.attrs["spatial resolution"] ** 2 > polygons_mask.union_all().area:
            return self.reduce_polygon_to_point(polygons_mask)

        # return clipped data as normal if the polygons are large enough
        spatial_data = self.data.rio.set_spatial_dims(
            x_dim="longitude", y_dim="latitude", inplace=False
        )
        spatial_data = spatial_data.rio.write_crs("epsg:4326")
        mask = gpd.geoseries.GeoSeries(polygons_mask).set_crs(epsg_crs).to_crs(4326)
        min_lon, min_lat, max_lon, max_lat = mask.total_bounds
        box_ds = self._new(spatial_data).rectangle(
            min_lat, min_lon, max_lat, max_lon
        ).data
        self._new(box_ds).check_dataset_size(point_limit=point_limit)
        try:
            shaped_ds = box_ds.rio.clip(mask, 4326, drop=True)
        except rioxarray.exceptions.NoDataInBounds:
            return self.reduce_polygon_to_point(polygons_mask)

        data_var = list(shaped_ds.data_vars)[0]
        if "grid_mapping" in shaped_ds[data_var].attrs:
            del shaped_ds[data_var].attrs["grid_mapping"]

        return self._new(shaped_ds)

    def time_range(
        self, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> "GeotemporalData":
        """Select data within a contiguous time range.

        Can be combined with spatial selectors defined above

        Parameters
        ----------

        start_time, datetime.datetime
            Beginning of time range.
        end_time, datetime.datetime
            End of time range

        Returns
        -------
        GeotemporalData
            A new dataset
        """
        data = self.data.sel(time=slice(start_time, end_time))
        return self._new(data)

    def select(self, selection: GeoSelectionOptions) -> "GeotemporalData":
        """Apply a combined point or bounds selection and optional time range.

        The selection mapping accepts:

        - ``point``: ``{"latitude": float, "longitude": float, "options": {...}}``
        - ``bounds``: ``[west, south, east, north]`` or a mapping with those keys
        - ``time_range``/``timeRange``: ``[start, end]`` or
          ``{"start": start, "end": end}``
        """
        _ensure_mapping(selection, "Selection")
        current = self

        point = _get_selection_value(selection, "point")
        bounds = _get_selection_value(selection, "bounds")
        if point is not None and bounds is not None:
            raise errors.InvalidSelectionError(
                "Use either point or bounds selection, not both."
            )

        if point is not None:
            current = current.point(**_normalize_point_selection(point))

        time_range = _get_selection_value(selection, "time_range", "timeRange")
        if time_range is not None:
            current = current.time_range(*_normalize_time_range_selection(time_range))

        if bounds is not None:
            bounds_options = _get_selection_value(
                selection, "bounds_options", "boundsOptions"
            )
            min_lat, min_lon, max_lat, max_lon, coordinate_options = (
                _normalize_bounds_selection(bounds, bounds_options)
            )
            current = current.rectangle(
                min_lat,
                min_lon,
                max_lat,
                max_lon,
                **coordinate_options,
            )

        return current

    def reduce_polygon_to_point(
        self, polygons_mask: gpd.array.GeometryArray
    ) -> "GeotemporalData":
        """Reduce data to a representative point

        Reduce data to a representative point approximately at the center of an arbitrary
        shape.

        Returns data from the nearest grid cell to this pont. NOTE a more involved
        alternative would be to return the average for values in the entire polygon at
        this point

        Parameters
        ----------
        polygons_mask: gpd.array.GeometryArray[shapely.geometry.multipolygon.MultiPolygon]
            Array of MultiPolygon shapes defining the area of interest

        Returns
        -------
        GeotemporalData
            A new dataset
        """
        pt = unary_union(polygons_mask).representative_point()
        data = self.data.sel(latitude=pt.y, longitude=pt.x, method="nearest")
        return self._new(data)

    def spatial_aggregation(
        self,
        agg_method: str,
    ) -> "GeotemporalData":
        """Reduces data using spatial aggregation

        Reduces data for all points for every time period according to the specified
        aggregation method.

        For a more nuanced treatment of spatial units use the `get_points_in_polygons`
        method.

        Parameters
        ----------
        agg_method: str
            Method to aggregate by

        Returns
        -------
        GeotemporalData
            A new dataset
        """
        _check_input_parameters(agg_method=agg_method)
        spatial_dims = [dim for dim in self.data.dims if dim != "time"]
        # Aggregate by the specified method across all time periods
        aggregator = getattr(xr.Dataset, agg_method)
        return self._new(aggregator(self.data, spatial_dims, keep_attrs=True))

    def temporal_aggregation(
        self,
        time_period: str,
        agg_method: str,
        time_unit: int = 1,
    ) -> "GeotemporalData":
        """Reduces data using temporatl aggregation.

        Reduces data according to a specified combination of time period, units of time,
        aggregation method, and/or desired spatial unit. Time-based inputs defualt to
        the entire time range and 1 unit of time, respectively. Spatial units default to
        points, i.e. every combination of latitudes/longitudes. The only alternative is
        "all". For a more nuanced treatment of spatial units use the
        `get_points_in_polygons` method.

        Parameters
        ----------

        time_period: str
            Time period to aggregate by, parsed into DateOffset objects as per
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        agg_method:str
            Method to aggregate by
        time_unit: int
            Number of time periods to aggregate by. Default is 1. Ignored if "all" time
            periods specified.

        Returns
        -------
        GeotemporalData
            A new dataset
        """
        _check_input_parameters(time_period=time_period, agg_method=agg_method)
        if time_period == "all":
            aggregator = getattr(xr.Dataset, agg_method)
            resampled_agg = aggregator(self.data, dim="time", keep_attrs=True)
        else:
            period_strings = {
                "hour": f"{time_unit}h",
                "day": f"{time_unit}D",
                "week": f"{time_unit}W",
                "month": f"{time_unit}ME",
                "quarter": f"{time_unit}QE",
                "year": f"{time_unit}YE",
            }
            # Resample by the specified time period and aggregate by the specified method
            resampled = self.data.resample(time=period_strings[time_period])
            aggregator = getattr(xr.core.resample.DatasetResample, agg_method)
            resampled_agg = aggregator(resampled, keep_attrs=True)

        return self._new(resampled_agg)

    def rolling_aggregation(
        self,
        window_size: int,
        agg_method: str,
    ) -> "GeotemporalData":
        """Reduces data using rolling aggregation.

        Reduces data to a rolling aggregate of data values along a dataset's "time"
        dimension. The size of the window and the aggregation method are specified by
        the user. Method must be one of "min", "max", "median", "mean", "std", or "sum".

        Parameters
        ----------
        window_size: int
            Size of rolling window to apply
        agg_method: str
            Method to aggregate by

        Returns
        -------
        GeotemporalData
            A new dataset
        """
        _check_input_parameters(agg_method=agg_method)
        # Aggregate by the specified method over the specified rolling window length
        rolled = self.data.rolling(time=window_size)
        rolled_agg = getattr(rolled, agg_method)(keep_attrs=True).isel(
            time=slice(window_size - 1, None)
        )
        # Remove incomplete leading windows while preserving spatial NAs.

        return self._new(rolled_agg)

    def to_netcdf(self, *args, **kwargs):
        """Serialize dataset to NetCDF byte format.

        Drops nested attributes from zarr and sets 'updating date range' if the dataset
        is currently undergoing a historical update,  then calls
        ``xarray.Dataset.to_netcdf`` with passed arguments.

        See `Xarray documentation
        <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html#xarray.Dataset.to_netcdf>`
        for parameters and return types.

        If no arguments are passed, a bytes object is returned.
        """
        ds = self.data.copy()

        try:
            if ds.update_in_progress and not ds.update_is_append_only:
                update_date_range = ds.attrs["update_date_range"]
                ds.attrs["updating date range"] = (
                    f"{update_date_range[0]}-{update_date_range[1]}"
                )
        except AttributeError:
            pass

        # remove nested and None attributes, which to_netcdf to bytes doesn't support
        for bad_key in [
            "bbox",
            "date range",
            "tags",
            "finalization date",
            "update_date_range",
        ]:
            if bad_key in ds.attrs:
                del ds.attrs[bad_key]

        return ds.to_netcdf(*args, **kwargs)

    def as_dict(self) -> dict:
        """Prepares dict containing metadata and values from dataset

        Returns
        -------
        dict
            Dict with metadata and data values included
        """
        vals = self.data_var.values
        ret_dict = {}
        dimensions = []
        ret_dict["units"] = self.data_var.attrs.get("units", "unknown")
        if "time" in self.data:
            ret_dict["times"] = (
                np.datetime_as_string(self.data.time.values, unit="s")
                .flatten()
                .tolist()
            )
            dimensions.append("time")
        if "point" in self.data.dims:
            ret_dict["points"] = list(
                zip(self.data.latitude.values, self.data.longitude.values)
            )
            ret_dict["point_coords_order"] = ["latitude", "longitude"]
            dimensions.insert(0, "point")
            ret_dict["data"] = np.where(~np.isfinite(vals), None, vals).T.tolist()
        else:
            for dim in self.data_var.dims:
                if dim != "time":
                    ret_dict[f"{dim}s"] = self.data[dim].values.flatten().tolist()
                    dimensions.append(dim)
            ret_dict["data"] = np.where(~np.isfinite(vals), None, vals).tolist()
        ret_dict["dimensions_order"] = dimensions
        try:
            if self.data.update_in_progress and not self.data.update_is_append_only:
                ret_dict["update_date_range"] = self.data.attrs["update_date_range"]
        except AttributeError:
            pass
        return ret_dict

    def query(
        self,
        forecast_reference_time: datetime.datetime = None,
        point_kwargs: dict = None,
        circle_kwargs: dict = None,
        rectangle_kwargs: dict = None,
        polygon_kwargs: dict = None,
        multiple_points_kwargs: dict = None,
        bounds: BoundsSelection = None,
        bounds_options: dict = None,
        spatial_agg_kwargs: dict = None,
        temporal_agg_kwargs: dict = None,
        rolling_agg_kwargs: dict = None,
        time_range: typing.Optional[typing.List[datetime.datetime]] = None,
        point_limit: int = DEFAULT_POINT_LIMIT,
    ) -> "GeotemporalData":
        # Filter data down temporally, then spatially, and check that the size of
        # resulting dataset fits within the limit. While a user can get the entire DS by
        # providing no filters, this will almost certainly cause the size checks to fail
        data = self
        if time_range:
            data = data.time_range(*time_range)
        if point_kwargs:
            data = data.point(**point_kwargs)
        elif circle_kwargs:
            lat = circle_kwargs.get("lat", circle_kwargs.get("center_lat"))
            lon = circle_kwargs.get("lon", circle_kwargs.get("center_lon"))
            radius = circle_kwargs.get("radius")
            data = data.circle(lat=lat, lon=lon, radius=radius)
        elif rectangle_kwargs:
            data = data.rectangle(**rectangle_kwargs)
        elif bounds is not None:
            min_lat, min_lon, max_lat, max_lon, coordinate_options = (
                _normalize_bounds_selection(bounds, bounds_options)
            )
            data = data.rectangle(
                min_lat,
                min_lon,
                max_lat,
                max_lon,
                **coordinate_options,
            )
        elif polygon_kwargs:
            data = data.polygons(**polygon_kwargs, point_limit=point_limit)
        elif multiple_points_kwargs:
            data = data.points(**multiple_points_kwargs)

        # Filter down forecast data and fill in any missing forecast dates
        if "forecast_reference_time" in data.data and not forecast_reference_time:
            raise InvalidForecastRequestError(
                "Forecast dataset requested without forecast reference time. "
                "Provide a forecast reference time or request to a different dataset if "
                "you desire observations, not projections."
            )
        if forecast_reference_time:
            if "forecast_reference_time" in data.data:
                data = data.forecast(forecast_reference_time)
                data = data.reindex_forecast()
            else:
                raise MissingDimensionsError(
                    f"Forecasts are not available for the requested dataset {data.dataset_name}"
                )

        # Check that size of reduced data won't prove too expensive to request and
        # process, according to specified limits
        data.check_dataset_size(point_limit)
        data.check_has_data()

        # Perform all requested valid aggregations. First aggregate data spatially, then
        # temporally or on a rolling basis.
        if spatial_agg_kwargs:
            data = data.spatial_aggregation(**spatial_agg_kwargs)
        if temporal_agg_kwargs:
            data = data.temporal_aggregation(**temporal_agg_kwargs)
        elif rolling_agg_kwargs:
            data = data.rolling_aggregation(**rolling_agg_kwargs)

        return data

    def _new(self, data):
        return type(self)(data, dataset_name=self.dataset_name, data_var=self._data_var)


def _ensure_mapping(value: typing.Any, label: str) -> Mapping[str, typing.Any]:
    if not isinstance(value, Mapping):
        raise errors.InvalidSelectionError(f"{label} must be a mapping.")
    return value


def _get_selection_value(
    selection: Mapping[str, typing.Any],
    *keys: str,
) -> typing.Any:
    for key in keys:
        if key in selection:
            return selection[key]
    return None


def _get_required_mapping_value(
    selection: Mapping[str, typing.Any],
    key: str,
    label: str,
) -> typing.Any:
    if key not in selection:
        raise errors.InvalidSelectionError(f"{label} must include '{key}'.")
    return selection[key]


def _coerce_finite_number(value: typing.Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
    ):
        raise errors.InvalidSelectionError(
            "Bounds selection must use finite west, south, east, and north numbers."
        )
    return float(value)


def _normalize_coordinate_options(
    options: typing.Optional[Mapping[str, typing.Any]],
) -> dict[str, str]:
    if options is None:
        return {}
    _ensure_mapping(options, "Selection options")

    normalized: dict[str, str] = {}
    latitude_key = _get_selection_value(options, "latitude_key", "latitudeKey")
    longitude_key = _get_selection_value(options, "longitude_key", "longitudeKey")

    if latitude_key is not None:
        if not isinstance(latitude_key, str):
            raise errors.InvalidSelectionError("latitude_key must be a string.")
        normalized["latitude_key"] = latitude_key
    if longitude_key is not None:
        if not isinstance(longitude_key, str):
            raise errors.InvalidSelectionError("longitude_key must be a string.")
        normalized["longitude_key"] = longitude_key

    return normalized


def _normalize_bounds_selection(
    bounds: BoundsSelection,
    fallback_options: typing.Optional[Mapping[str, typing.Any]] = None,
) -> tuple[float, float, float, float, dict[str, str]]:
    if isinstance(bounds, Mapping):
        west = _get_required_mapping_value(bounds, "west", "Bounds selection")
        south = _get_required_mapping_value(bounds, "south", "Bounds selection")
        east = _get_required_mapping_value(bounds, "east", "Bounds selection")
        north = _get_required_mapping_value(bounds, "north", "Bounds selection")
        options = bounds.get("options", fallback_options)
    elif isinstance(bounds, (str, bytes)):
        raise errors.InvalidSelectionError(
            "Bounds selection must be [west, south, east, north]."
        )
    else:
        try:
            west, south, east, north = bounds
        except (TypeError, ValueError) as exc:
            raise errors.InvalidSelectionError(
                "Bounds selection must be [west, south, east, north]."
            ) from exc
        options = fallback_options

    west = _coerce_finite_number(west, "west")
    south = _coerce_finite_number(south, "south")
    east = _coerce_finite_number(east, "east")
    north = _coerce_finite_number(north, "north")

    if west >= east:
        raise errors.InvalidSelectionError(
            f"west ({west}) must be less than east ({east})."
        )
    if south >= north:
        raise errors.InvalidSelectionError(
            f"south ({south}) must be less than north ({north})."
        )

    return (
        south,
        west,
        north,
        east,
        _normalize_coordinate_options(options),
    )


def _normalize_point_selection(
    point: Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    point = _ensure_mapping(point, "Point selection")
    latitude = _get_required_mapping_value(point, "latitude", "Point selection")
    longitude = _get_required_mapping_value(point, "longitude", "Point selection")
    options = point.get("options") or {}
    coordinate_options = _normalize_coordinate_options(options)

    snap_to_grid = _get_selection_value(options, "snap_to_grid", "snapToGrid")
    if snap_to_grid is None:
        snap_to_grid = True

    return {
        "latitude": latitude,
        "longitude": longitude,
        "snap_to_grid": snap_to_grid,
        **coordinate_options,
    }


def _normalize_time_range_selection(
    time_range: typing.Any,
) -> tuple[typing.Any, typing.Any]:
    if isinstance(time_range, Mapping):
        return (
            _get_required_mapping_value(time_range, "start", "Time range selection"),
            _get_required_mapping_value(time_range, "end", "Time range selection"),
        )
    if isinstance(time_range, (str, bytes)):
        raise errors.InvalidSelectionError(
            "Time range selection must be [start, end] or {'start': ..., 'end': ...}."
        )
    try:
        start, end = time_range
    except (TypeError, ValueError) as exc:
        raise errors.InvalidSelectionError(
            "Time range selection must be [start, end] or {'start': ..., 'end': ...}."
        ) from exc
    return start, end


def _haversine(
    lat1: typing.Union[np.ndarray, float],
    lon1: typing.Union[np.ndarray, float],
    lat2: typing.Union[np.ndarray, float],
    lon2: typing.Union[np.ndarray, float],
) -> typing.Union[np.ndarray, float]:
    """Calculates arclength distance in km between coordinate pairs,
        assuming the earth is a perfect sphere

    Args:
        lat1 (typing.Union[np.ndarray, float]): latitude coordinate for first point
        lon1 (typing.Union[np.ndarray, float]): longitude coordinate for first point
        lat2 (typing.Union[np.ndarray, float]): latitude coordinate for second point
        lon2 (typing.Union[np.ndarray, float]): longitude coordinate for second point

    Returns:
        typing.Union[np.ndarray, float]: distance between coordinate pairs in km
    """
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    # this formula sometimes produces negative values for very small distances, so we take abs
    abs_a = abs(a)
    c = 2 * np.arcsin(np.sqrt(abs_a))

    # radius of earth in km
    r = 6371
    return c * r


def _check_input_parameters(time_period=None, agg_method=None):
    """Check input parameters

    Checks whether input parameters align with permitted time periods and aggregation
    methods.

    Args:
        time_period (str, optional): a string specifying the time period to resample a
            dataset by
        agg_method (str, optional): a string specifying the aggregation method to use on
            a dataset

    Raises:
        InvalidTimePeriodError: Raised when the specified time period is not accepted
        InvalidAggregationMethodError: Raised when the specified aggregation method is
            not accepted
    """
    if time_period and time_period not in [
        "hour",
        "day",
        "week",
        "month",
        "quarter",
        "year",
        "all",
    ]:
        raise errors.InvalidTimePeriodError(
            f"Specified time period {time_period} not among permitted periods: "
            "'hour', 'day', 'week', 'month', 'quarter', 'year', 'all'"
        )
    if agg_method and agg_method not in ["min", "max", "median", "mean", "std", "sum"]:
        raise errors.InvalidAggregationMethodError(
            f"Specified method {agg_method} not among permitted methods: "
            "'min', 'max', 'median', 'mean', 'std', 'sum'"
        )
