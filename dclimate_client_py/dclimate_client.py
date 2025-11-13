"""
DClimate Client - Async context manager for loading dClimate datasets

This module provides a high-level client interface that manages IPFS connections
internally, abstracting away KuboCAS lifecycle management.
"""

import typing
import xarray as xr
from py_hamt import KuboCAS
# Import here to avoid circular imports
from .ipfs_retrieval import _load_dataset_from_ipfs_cid


from .geotemporal_data import GeotemporalData
from .datasets import (
    resolve_dataset_source,
    fetch_cid_from_url,
    DATASET_CATALOG_INTERNAL,
    DatasetCatalog,
)
from .dclimate_zarr_errors import InvalidSelectionError


class dClimateClient:
    """
    Async context manager for loading dClimate datasets from IPFS.

    This client manages IPFS connections internally via KuboCAS, so users don't
    need to manually configure or import IPFS-related dependencies.

    Parameters
    ----------
    gateway_base_url : str, optional
        IPFS HTTP Gateway base URL (e.g., "https://ipfs.io" or "http://localhost:8080").
        If None, uses KuboCAS defaults or environment variables.
    rpc_base_url : str, optional
        IPFS RPC API base URL (e.g., "http://localhost:5001").
        If None, uses KuboCAS defaults or environment variables.
    catalog : DatasetCatalog, optional
        Custom dataset catalog to use. If None, uses DATASET_CATALOG_INTERNAL.

    Examples
    --------
    Basic usage with default IPFS configuration:

    >>> async with dClimateClient() as client:
    ...     dataset = await client.load_dataset(
    ...         dataset="2m_temperature",
    ...         collection="era5",
    ...         variant="finalized"
    ...     )
    ...     # Use dataset...

    With custom IPFS endpoints:

    >>> async with dClimateClient() as client:
    ...     dataset = await client.load_dataset(
    ...         dataset="2m_temperature",
    ...         collection="era5",
    ...         variant="finalized",
    ...         return_xarray=True  # Get raw xarray.Dataset
    ...     )
    """

    def __init__(
        self,
        gateway_base_url: typing.Optional[str] = "https://ipfs-gateway.dclimate.net",
        rpc_base_url: typing.Optional[str] = "https://ipfs-gateway.dclimate.net",
        catalog: typing.Optional[DatasetCatalog] = None,
    ):
        self._gateway_base_url = gateway_base_url
        self._rpc_base_url = rpc_base_url
        self._catalog = catalog or DATASET_CATALOG_INTERNAL
        self._kubo_cas: typing.Optional[KuboCAS] = None

    async def __aenter__(self) -> "dClimateClient":
        """Initialize KuboCAS when entering async context."""
        # Create KuboCAS with configured endpoints
        self._kubo_cas = KuboCAS(
            gateway_base_url=self._gateway_base_url,
            rpc_base_url=self._rpc_base_url,
        )
        # Enter the KuboCAS context manager
        await self._kubo_cas.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up KuboCAS when exiting async context."""
        if self._kubo_cas:
            await self._kubo_cas.__aexit__(exc_type, exc_val, exc_tb)
            self._kubo_cas = None

    async def load_dataset(
        self,
        dataset: str,
        collection: typing.Optional[str] = None,
        variant: typing.Optional[str] = None,
        cid: typing.Optional[str] = None,
        return_xarray: bool = False,
    ) -> typing.Union[GeotemporalData, xr.Dataset]:
        """
        Load a dClimate dataset from IPFS using the internal dataset catalog.

        This method uses the client's managed KuboCAS instance internally,
        so no IPFS configuration is needed in the call.

        Parameters
        ----------
        dataset : str
            Name of the dataset to load (e.g., "2m_temperature", "total_precipitation")
        collection : str, optional
            Name of the collection (e.g., "era5", "aifs"). If not provided,
            will auto-detect from catalog. Recommended to specify for clarity.
        variant : str, optional
            Specific variant to load (e.g., "finalized", "ensemble"). If not provided
            and the dataset has multiple variants, an error will be raised.
        cid : str, optional
            Direct IPFS CID to load, bypassing catalog resolution. Useful for loading
            specific versions or datasets not in the catalog.
        return_xarray : bool, optional
            If True, return raw xarray.Dataset. If False (default), return
            GeotemporalData wrapper.

        Returns
        -------
        Union[GeotemporalData, xr.Dataset]
            Loaded dataset, either wrapped in GeotemporalData (default) or as raw
            xarray.Dataset if return_xarray=True.

        Raises
        ------
        RuntimeError
            If client is not being used as an async context manager
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
        >>> async with dClimateClient() as client:
        ...     dataset = await client.load_dataset(
        ...         dataset="2m_temperature",
        ...         collection="era5",
        ...         variant="finalized"
        ...     )
        ...     # Query the dataset
        ...     filtered = dataset.point(latitude=40.875, longitude=-104.875)
        """
        if not self._kubo_cas:
            raise RuntimeError(
                "dClimateClient must be used as an async context manager. "
                "Use 'async with dClimateClient() as client:'"
            )

        # Use slug for metadata
        dataset_slug = f"{collection or 'auto'}/{dataset}/{variant or 'auto'}"
        # Case 1: Direct CID provided - bypass catalog resolution
        if cid:
            ds = await _load_dataset_from_ipfs_cid(
                ipfs_cid=cid,
                kubo_cas=self._kubo_cas,
            )

            if return_xarray:
                return ds
            else:
                return GeotemporalData(ds, dataset_name=dataset_slug)

        # Case 2: Normal resolution via catalog
        resolved = resolve_dataset_source(
            dataset_name=dataset,
            collection_name=collection,
            variant_name=variant,
            catalog=self._catalog,
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

        ds = await _load_dataset_from_ipfs_cid(
            ipfs_cid=final_cid,
            kubo_cas=self._kubo_cas,
        )

        if return_xarray:
            return ds
        else:
            return GeotemporalData(ds, dataset_name=resolved["slug"])
