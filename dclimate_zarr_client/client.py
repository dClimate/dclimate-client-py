"""
Functions that will map to endpoints in the flask app
"""

import datetime
import typing
import xarray as xr

from .dclimate_zarr_errors import (
    ConflictingGeoRequestError,
    ConflictingAggregationRequestError,
    InvalidExportFormatError,
    InvalidSelectionError,
)
from .geotemporal_data import GeotemporalData, DEFAULT_POINT_LIMIT
from .s3_retrieval import get_dataset_from_s3
from .ipfs_retrieval import (
    _get_dataset_by_ipfs_cid,
)
from .datasets import (
    resolve_dataset_source,
    get_concatenable_variants,
    find_dataset_by_name,
    find_collection_by_name,
    fetch_cid_from_url,
    DATASET_CATALOG_INTERNAL,
    DatasetCatalog,
)
from .concatenate import concatenate_datasets

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
    source: typing.Literal["s3"] = "s3",
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
    if source == "s3":
        if not bucket_name:
            raise ValueError("bucket_name is required when source is 's3'")
        data = load_s3(dataset_name, bucket_name)
    else:
        raise ValueError(
            "Invalid source specified. Must be 's3'. "
            "IPFS source is deprecated - use dDClimateClient instead."
        )

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


async def load_dclimate_dataset(
    dataset: str,
    collection: typing.Optional[str] = None,
    variant: typing.Optional[str] = None,
    cid: typing.Optional[str] = None,
    gateway_uri_stem: typing.Optional[str] = None,
    rpc_uri_stem: typing.Optional[str] = None,
    return_xarray: bool = False,
    catalog: typing.Optional[DatasetCatalog] = None,
) -> typing.Union[GeotemporalData, xr.Dataset]:
    """
    Load a dClimate dataset from IPFS using the internal dataset catalog.

    This is the main entry point for loading datasets, similar to dclimate-client-js
    loadDataset(). It provides:
    - Automatic dataset resolution from catalog
    - Support for both direct CID and URL-based variants
    - Auto-concatenation of variants (e.g., finalized + non-finalized data)
    - Option to return raw xarray.Dataset or wrapped GeotemporalData

    Parameters
    ----------
    dataset : str
        Name of the dataset to load (e.g., "2m_temperature", "total_precipitation")
    collection : str, optional
        Name of the collection (e.g., "era5", "aifs"). If not provided,
        will auto-detect from catalog. Recommended to specify for clarity.
    variant : str, optional
        Specific variant to load (e.g., "finalized", "ensemble"). If not provided
        and auto_concatenate is False, will raise an error for multi-variant datasets.
    cid : str, optional
        Direct IPFS CID to load, bypassing catalog resolution. Useful for loading
        specific versions or datasets not in the catalog.
    gateway_uri_stem : str, optional
        Custom IPFS HTTP Gateway URI stem (e.g., "http://localhost:8080").
        If None, uses the default from KuboCAS.
    rpc_uri_stem : str, optional
        Custom IPFS RPC API URI stem (e.g., "http://localhost:5001").
        If None, uses the default from KuboCAS.
    return_xarray : bool, optional
        If True, return raw xarray.Dataset. If False (default), return
        GeotemporalData wrapper.
    catalog : DatasetCatalog, optional
        Custom dataset catalog to use. If None, uses DATASET_CATALOG_INTERNAL.

    .. note::
        Auto-concatenation of variants is currently disabled due to xarray's
        lazy concatenation not being fully supported. Users must explicitly
        specify a variant for datasets with multiple variants. The catalog
        still maintains concat_priority and concat_dimension metadata for
        future use when lazy concatenation becomes available.

    Returns
    -------
    Union[GeotemporalData, xr.Dataset]
        Loaded dataset, either wrapped in GeotemporalData (default) or as raw
        xarray.Dataset if return_xarray=True.

    Raises
    ------
    DatasetNotFoundError
        If dataset cannot be found in catalog
    CollectionNotFoundError
        If specified collection doesn't exist
    VariantNotFoundError
        If specified variant doesn't exist
    InvalidSelectionError
        If dataset has multiple variants and no variant is specified
    IpfsConnectionError
        If connection to IPFS fails

    Examples
    --------
    Load a single variant explicitly:

    >>> ds = load_dclimate_dataset("2m_temperature", collection="era5", variant="finalized")

    Load with direct CID:

    >>> ds = load_dclimate_dataset(
    ...     "2m_temperature",
    ...     cid="bafybeibg5o7c3hzj4eyhwvqq4fkzp6rw7gm5vu5f5qvj2p7v5zq2w2y3x4"
    ... )

    Get raw xarray.Dataset instead of GeotemporalData:

    >>> xr_ds = load_dclimate_dataset(
    ...     "2m_temperature",
    ...     collection="era5",
    ...     variant="finalized",
    ...     return_xarray=True
    ... )
    """
    if catalog is None:
        catalog = DATASET_CATALOG_INTERNAL

    # Use slug for metadata
    dataset_slug = f"{collection or 'auto'}/{dataset}/{variant or 'auto'}"

    # Case 1: Direct CID provided - bypass catalog resolution
    if cid:
        ds = await _get_dataset_by_ipfs_cid(
            ipfs_cid=cid,
            gateway_uri_stem=gateway_uri_stem,
            rpc_uri_stem=rpc_uri_stem,
        )

        if return_xarray:
            return ds
        else:
            return GeotemporalData(ds, dataset_name=dataset_slug)

    # Case 2: Normal resolution (explicit variant or single variant dataset)
    # Note: Auto-concatenation is currently disabled due to xarray lazy concat limitations
    resolved = resolve_dataset_source(
        dataset_name=dataset,
        collection_name=collection,
        variant_name=variant,
        catalog=catalog,
    )

    # Get CID either directly or from URL
    final_cid = resolved["cid"]
    if not final_cid and resolved["url"]:
        final_cid = fetch_cid_from_url(resolved["url"])

    if not final_cid:
        raise InvalidSelectionError(
            f"No CID or URL available for {resolved['slug']}. "
            f"Cannot load dataset without a source."
        )

    ds = await _get_dataset_by_ipfs_cid(
        ipfs_cid=final_cid,
        gateway_uri_stem=gateway_uri_stem,
        rpc_uri_stem=rpc_uri_stem,
    )

    if return_xarray:
        return ds
    else:
        return GeotemporalData(ds, dataset_name=resolved["slug"])
