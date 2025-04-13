"""
Functions that will map to endpoints in the flask app
"""

import datetime
import typing

from .dclimate_zarr_errors import (
    ConflictingGeoRequestError,
    ConflictingAggregationRequestError,
    InvalidExportFormatError,
)
from .geotemporal_data import GeotemporalData, DEFAULT_POINT_LIMIT
from .s3_retrieval import get_dataset_from_s3
from .ipfs_retrieval import (
    get_dataset_hamt_cid_from_stac,
    _get_dataset_by_ipfs_cid,
)

# Define the top-level dClimate STAC catalog IPNS name
DCLIMATE_STAC_CATALOG_IPNS = (
    "/ipns/k51qzi5uqu5dk89atnl883sr0g1cb2py631ckz9ng45qhk6dg0pj141jtxtx6l"
)


def load_ipfs_via_stac(
    dataset_name: str,
    # as_of: typing.Optional[datetime.datetime] = None, # Removed as_of
    gateway_uri_stem: str | None = None,
    rpc_uri_stem: str | None = None,
) -> GeotemporalData:
    """
    Load a Geotemporal dataset from IPFS/IPNS via the dClimate STAC catalog.

    This function finds the dataset's IPNS name by navigating the STAC catalog
    starting from a root IPNS name, resolves the dataset's IPNS name to its
    current IPFS CID, and then loads the Zarr dataset.

    Parameters
    ----------
    dataset_name : str
        The identifier (e.g., 'cpc-precip-conus') of the dataset as found in the STAC catalog.
    as_of: datetime.datetime, optional
        Pull in most recent data created before this time. If ``None``, just get most
        recent. Defaults to ``None``.
    gateway_uri_stem : str, optional
        Custom IPFS HTTP Gateway URI stem (e.g., "http://localhost:8080").
        If None, uses the default from py-hamt's IPFSStore.
    rpc_uri_stem : str, optional
        Custom IPFS RPC API URI stem (e.g., "http://localhost:5001").
        If None, uses the default from py-hamt's IPFSStore.

    Returns
    -------
    GeotemporalData
        A wrapper around the loaded Xarray dataset.

    Raises
    ------
    DatasetNotFoundError
        If the dataset cannot be found in the STAC catalog or lacks the HAMT asset.
    IpfsConnectionError
        If connection to IPFS fails.
    StacCatalogError
        For issues during STAC parsing or traversal.
    """
    # 1. Find the dataset's HAMT root IPFS CID from the STAC catalog
    # Calls the renamed function from ipfs_retrieval
    dataset_hamt_cid = get_dataset_hamt_cid_from_stac(
        root_catalog_ipns=DCLIMATE_STAC_CATALOG_IPNS,
        target_dataset_id=dataset_name,
        gateway_uri_stem=gateway_uri_stem,  # Pass through config
        rpc_uri_stem=rpc_uri_stem,  # Pass through config
    )

    # 2. Load the dataset using the directly obtained HAMT IPFS CID
    # NO LONGER NEED to resolve dataset IPNS name
    ds = _get_dataset_by_ipfs_cid(
        ipfs_cid=dataset_hamt_cid,
        gateway_uri_stem=gateway_uri_stem,
        rpc_uri_stem=rpc_uri_stem,
    )

    return GeotemporalData(ds, dataset_name=dataset_name)


def load_s3(
    dataset_name: str,
    bucket_name: str,
) -> GeotemporalData:
    """
    Load a Geotemporal dataset from an S3 bucket.

    Parameters
    ----------

    dataset_name: str
        The name of the dataset in the bucket.
    bucket_name: str
        S3 bucket name where the dataset is going to be fetched
    """
    ds = get_dataset_from_s3(dataset_name, bucket_name)
    return GeotemporalData(ds, dataset_name=dataset_name)


def geo_temporal_query(
    dataset_name: str,
    source: typing.Literal["ipfs", "s3"] = "ipfs",
    bucket_name: str = None,
    var_name: str = None,
    gateway_uri_stem: str | None = None,
    rpc_uri_stem: str | None = None,
    forecast_reference_time: str = None,
    point_kwargs: dict = None,
    circle_kwargs: dict = None,
    rectangle_kwargs: dict = None,
    polygon_kwargs: dict = None,
    multiple_points_kwargs: dict = None,
    spatial_agg_kwargs: dict = None,
    temporal_agg_kwargs: dict = None,
    rolling_agg_kwargs: dict = None,
    time_range: typing.Optional[typing.List[datetime.datetime]] = None,
    # as_of: typing.Optional[datetime.datetime] = None, # Removed as_of
    point_limit: int = DEFAULT_POINT_LIMIT,
    output_format: str = "array",
) -> typing.Union[dict, bytes]:
    """Filter an XArray dataset

    Filter an XArray dataset by specified spatial and/or temporal bounds and aggregate
    according to spatial and/or temporal logic, if desired. Before aggregating check
    that the filtered data fits within specified point and area maximums to avoid
    computationally expensive retrieval and processing operations. When bounds or
    aggregation logic are not provided, pass the dataset along untouched.

    Return either a numpy array of data values or a NetCDF file.

    Only one of point, circle, rectangle, or polygon kwargs may be provided. Only one of
    temporal or rolling aggregation kwargs may be provided, although they can be chained
    with spatial aggregations if desired.

    Args:
        dataset_name (str): Name used to identify the dataset within the STAC catalog (for IPFS)
                            or the dataset name in the bucket (for S3).
        source: (typing.Literal["ipfs", "s3"]): how to pull data. Defaults to "ipfs".
        bucket_name (str): S3 bucket name where the datasets are going to be fetched (required if source="s3").
        var_name (str, optional): Specific data variable to use within the dataset.
        gateway_uri_stem (str | None, optional): Custom IPFS HTTP Gateway URI stem for IPFS source.
        rpc_uri_stem (str | None, optional): Custom IPFS RPC API URI stem for IPFS source.
        forecast_reference_time (str): Isoformatted string representing the desire date
            to return all available forecasts for
        circle_kwargs (dict, optional): a dictionary of parameters relevant to a
            circular query
        rectangle_kwargs (dict, optional): a dictionary of parameters relevant to a
            rectangular query
        polygon_kwargs (dict, optional): a dictionary of parameters relevant to a
            polygonal query
        multiple_points_kwargs (dict, optional): Parameters for querying multiple specific points.
        point_kwargs (dict, optional): Parameters for querying a single point.
        spatial_agg_kwargs (dict, optional): a dictionary of parameters relevant to a
            spatial aggregation operation
        temporal_agg_kwargs (dict, optional): a dictionary of parameters relevant to a
            temporal aggregation operation
        rolling_agg_kwargs (dict, optional): a dictionary of parameters relevant to a
            rolling aggregation operation
        time_range (typing.Optional[typing.List[datetime.datetime]], optional):
            time range in which to subset data.
            Defaults to None.
        # REMOVED  as_of (typing.Optional[datetime.datetime], optional):
        #     pull in most recent data created before this time. If None, just get most
        #     recent. Defaults to None.
        point_limit (int, optional): maximum number of data points user can request.
            Defaults to DEFAULT_POINT_LIMIT.
        output_format (str, optional): Current supported formats are `array` and
            `netcdf`. Defaults to "array", which provides a dict of data
            values and coordinates.

    Returns:
        typing.Union[dict, bytes]: Output data as dict (default) or NetCDF bytes.
    """
    # Check for incompatible request parameters
    if (
        len(
            [
                kwarg_dict
                for kwarg_dict in [
                    circle_kwargs,
                    rectangle_kwargs,
                    polygon_kwargs,
                    multiple_points_kwargs,
                    point_kwargs,
                ]
                if kwarg_dict is not None
            ]
        )
        > 1
    ):
        raise ConflictingGeoRequestError(
            "User requested more than one type of geographic query, but only one can "
            "be submitted at a time"
        )
    if spatial_agg_kwargs and point_kwargs:
        raise ConflictingGeoRequestError(
            "User requested spatial aggregation methods on a single point, "
            "but these are mutually exclusive parameters. Only one may be requested at "
            "a time."
        )
    if temporal_agg_kwargs and rolling_agg_kwargs:
        raise ConflictingAggregationRequestError(
            "User requested both rolling and temporal aggregation, but these are "
            "mutually exclusive operations. Only one may be requested at a time."
        )
    if output_format not in ["array", "netcdf"]:
        raise InvalidExportFormatError(
            "User requested an invalid export format. Only 'array' or 'netcdf' "
            "permitted."
        )

    # Set defaults to avoid Nones accidentally passed by users causing a TypeError
    if not point_limit:
        point_limit = DEFAULT_POINT_LIMIT

    # Load the dataset based on the source
    if source == "ipfs":
        # *** CALL THE RENAMED FUNCTION ***
        data = load_ipfs_via_stac(
            dataset_name,
            gateway_uri_stem=gateway_uri_stem,
            rpc_uri_stem=rpc_uri_stem,
        )
    elif source == "s3":
        if not bucket_name:
            raise ValueError("bucket_name is required when source is 's3'")
        data = load_s3(dataset_name, bucket_name)
    else:
        raise ValueError("Invalid source specified. Must be 'ipfs' or 's3'.")

    # If specific variable is requested, use that
    if var_name is not None:
        data = data.use(var_name)

    # Filter data down temporally, then spatially, and check that the size of resulting
    # dataset fits within the limit. While a user can get the entire DS by providing no
    # filters, this will almost certainly cause the size checks to fail

    data = data.query(
        forecast_reference_time=forecast_reference_time,
        point_kwargs=point_kwargs,
        circle_kwargs=circle_kwargs,
        rectangle_kwargs=rectangle_kwargs,
        polygon_kwargs=polygon_kwargs,
        multiple_points_kwargs=multiple_points_kwargs,
        spatial_agg_kwargs=spatial_agg_kwargs,
        temporal_agg_kwargs=temporal_agg_kwargs,
        rolling_agg_kwargs=rolling_agg_kwargs,
        time_range=time_range,
        point_limit=point_limit,
    )

    # Export
    if output_format == "netcdf":
        return data.to_netcdf()
    else:  # "array"
        return data.as_dict()
