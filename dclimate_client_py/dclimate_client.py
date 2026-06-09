"""
DClimate Client - Async context manager for loading dClimate datasets

This module provides a high-level client interface that manages IPFS connections
internally, abstracting away KuboCAS lifecycle management.
"""

import typing
from collections.abc import Mapping

import requests
import xarray as xr
from py_hamt import KuboCAS
import pystac

# Import here to avoid circular imports
from .ipfs_retrieval import _load_dataset_from_ipfs_cid


from .geotemporal_data import GeotemporalData
from .datasets import DatasetMetadata
from .dclimate_zarr_errors import InvalidSelectionError
from .stac_catalog import (
    load_stac_catalog,
    resolve_dataset_cid_from_stac,
    list_available_datasets,
)
from .stac_server import (
    resolve_cid_from_stac_server,
    list_available_datasets_from_stac_server,
)
from .siren import SirenClient
from .siren.types import (
    SirenMetricDataPoint,
    SirenMetricQuery,
    SirenOptions,
    SirenRegion,
)


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

    Examples
    --------
    Basic usage:

    >>> async with dClimateClient() as client:
    ...     # List available datasets
    ...     datasets = client.list_datasets()
    ...     print(datasets["ifs"]["types"])
    ...
    ...     # Load a dataset
    ...     data, metadata = await client.load_dataset(
    ...         collection="ifs",
    ...         dataset="temperature",
    ...         variant="single"
    ...     )

    With custom IPFS endpoints:

    >>> async with dClimateClient(
    ...     gateway_base_url="https://custom-gateway.example.com"
    ... ) as client:
    ...     data, metadata = await client.load_dataset(
    ...         collection="ifs",
    ...         dataset="temperature",
    ...         variant="single",
    ...         return_xarray=True  # Get raw xarray.Dataset
    ...     )
    """

    def __init__(
        self,
        gateway_base_url: typing.Optional[str] = "https://ipfs-gateway.dclimate.net",
        rpc_base_url: typing.Optional[str] = "https://ipfs-gateway.dclimate.net",
        stac_server_url: typing.Optional[str] = "https://api.stac.dclimate.net",
        siren: typing.Optional[SirenOptions] = None,
    ):
        self._gateway_base_url = gateway_base_url
        self._rpc_base_url = rpc_base_url
        self._stac_server_url = stac_server_url
        self._stac_catalog: typing.Optional[pystac.Catalog] = None
        self._kubo_cas: typing.Optional[KuboCAS] = None
        # Note: STAC catalog is loaded lazily (only if STAC server fails)

        # Siren REST API client (optional)
        self._siren_client: typing.Optional[SirenClient] = None
        if siren is not None:
            self._siren_client = SirenClient(siren)

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
        if self._siren_client is not None:
            await self._siren_client.aclose()
        if self._kubo_cas:
            await self._kubo_cas.__aexit__(exc_type, exc_val, exc_tb)
            self._kubo_cas = None

    async def load_dataset(
        self,
        dataset: str,
        collection: typing.Optional[str] = None,
        variant: typing.Optional[str] = None,
        organization: typing.Optional[str] = None,
        cid: typing.Optional[str] = None,
        return_xarray: bool = False,
    ) -> typing.Union[
        typing.Tuple[GeotemporalData, DatasetMetadata],
        typing.Tuple[xr.Dataset, DatasetMetadata],
    ]:
        """
        Load a dClimate dataset from IPFS using the STAC catalog.

        This method uses the client's managed KuboCAS instance internally,
        so no IPFS configuration is needed in the call.

        Parameters
        ----------
        dataset : str
            Name of the dataset to load (e.g., "temperature", "precipitation")
        collection : str, required
            Name of the collection (e.g., "ifs", "era5", "aifs").
            Use client.list_datasets() to see available collections.
        variant : str, optional
            Specific variant to load (e.g., "single", "ensemble").
            If not provided and the dataset has multiple variants, may use default.
        organization : str, optional
            Organization/agency that owns the collection (e.g., "ecmwf", "prism").
            If provided, the collection will be resolved within that organization
            catalog. If omitted, the organization is inferred from the root catalog
            metadata where possible.
        cid : str, optional
            Direct IPFS CID to load, bypassing STAC catalog resolution.
            Useful for loading specific versions or datasets not in the catalog.
        return_xarray : bool, optional
            If True, return raw xarray.Dataset. If False (default), return
            GeotemporalData wrapper.

        Returns
        -------
        Tuple[Union[GeotemporalData, xr.Dataset], DatasetMetadata]
            A tuple containing:
            - Loaded dataset, either wrapped in GeotemporalData (default) or as raw
              xarray.Dataset if return_xarray=True.
            - Metadata dict with information about the dataset including organization,
              collection, dataset name, variant, slug, CID used, and source type.

        Raises
        ------
        RuntimeError
            If client is not being used as an async context manager
        ValueError
            If dataset cannot be found in STAC catalog
        InvalidSelectionError
            If collection parameter is not provided (when not using direct CID)
        requests.RequestException
            If connection to IPFS gateway fails

        Examples
        --------
        >>> async with dClimateClient() as client:
        ...     # List available datasets first
        ...     datasets = client.list_datasets()
        ...     print(datasets["ifs"]["types"])
        ...
        ...     # Load a dataset
        ...     data, metadata = await client.load_dataset(
        ...         collection="ifs",
        ...         dataset="temperature",
        ...         variant="single"
        ...     )
        ...
        ...     # Query the dataset
        ...     filtered = data.point(latitude=40.875, longitude=-104.875)
        """
        if not self._kubo_cas:
            raise RuntimeError(
                "dClimateClient must be used as an async context manager. "
                "Use 'async with dClimateClient() as client:'"
            )

        resolved_collection = collection
        if (
            organization
            and collection
            and not collection.startswith(f"{organization}_")
        ):
            resolved_collection = f"{organization}_{collection}"

        # Case 1: Direct CID provided - bypass catalog resolution
        if cid:
            slug_collection = resolved_collection or collection or "unknown"
            dataset_slug = (
                f"{organization}/{slug_collection}/{dataset}/{variant or 'default'}"
                if organization
                else f"{slug_collection}/{dataset}/{variant or 'default'}"
            )
            ds = await _load_dataset_from_ipfs_cid(
                ipfs_cid=cid,
                kubo_cas=self._kubo_cas,
            )

            # Build metadata for direct CID case
            metadata: DatasetMetadata = {
                "collection": resolved_collection or "unknown",
                "dataset": dataset,
                "variant": variant or "unknown",
                "slug": dataset_slug,
                "cid": cid,
                "url": None,
                "timestamp": None,
                "source": "direct_cid",
                "organization": organization
                or (
                    resolved_collection.split("_")[0]
                    if resolved_collection and "_" in resolved_collection
                    else None
                ),
            }

            if return_xarray:
                return ds, metadata
            else:
                return GeotemporalData(ds, dataset_name=dataset_slug), metadata

        # Case 2: Resolve via STAC server (fast) or STAC catalog (fallback)
        if not collection:
            raise InvalidSelectionError(
                "collection parameter is required. Use client.list_datasets() to see available collections."
            )

        final_cid = None

        # Try STAC server first (faster, avoids loading IPFS catalog)
        if self._stac_server_url:
            try:
                final_cid = resolve_cid_from_stac_server(
                    collection=resolved_collection,
                    dataset=dataset,
                    variant=variant,
                    server_url=self._stac_server_url,
                )
            except (requests.RequestException, ValueError):
                # Fall back when server lookup fails or returns no usable match.
                pass

        # Fallback: Resolve via STAC catalog from IPFS
        if final_cid is None:
            # Lazy load STAC catalog
            if self._stac_catalog is None:
                self._stac_catalog = load_stac_catalog(
                    gateway_url=self._gateway_base_url
                )

            if not organization and resolved_collection:
                available = list_available_datasets(self._stac_catalog)
                if resolved_collection not in available:
                    prefixed_matches = [
                        coll_id
                        for coll_id in available.keys()
                        if coll_id.endswith(f"_{resolved_collection}")
                    ]
                    if len(prefixed_matches) == 1:
                        resolved_collection = prefixed_matches[0]

            final_cid = resolve_dataset_cid_from_stac(
                catalog=self._stac_catalog,
                collection=resolved_collection,
                dataset=dataset,
                variant=variant,
                organization=organization,
            )

        ds = await _load_dataset_from_ipfs_cid(
            ipfs_cid=final_cid,
            kubo_cas=self._kubo_cas,
        )

        # Build metadata for STAC case
        metadata: DatasetMetadata = {
            "collection": resolved_collection,
            "dataset": dataset,
            "variant": variant or "default",
            "slug": (
                f"{organization}/{resolved_collection or collection}/{dataset}/{variant or 'default'}"
                if organization
                else f"{resolved_collection or collection}/{dataset}/{variant or 'default'}"
            ),
            "cid": final_cid,
            "url": None,
            "timestamp": None,
            "source": "stac",
            "organization": organization
            or (
                resolved_collection.split("_")[0]
                if resolved_collection and "_" in resolved_collection
                else None
            ),
        }

        if return_xarray:
            return ds, metadata
        else:
            return GeotemporalData(ds, dataset_name=metadata["slug"]), metadata

    async def select_dataset(
        self,
        *,
        request: typing.Mapping[str, typing.Any],
        selection: typing.Mapping[str, typing.Any],
        return_xarray: bool = False,
    ) -> typing.Union[
        typing.Tuple[GeotemporalData, DatasetMetadata],
        typing.Tuple[xr.Dataset, DatasetMetadata],
    ]:
        """
        Load a dClimate dataset and apply point, bounds, and/or time selections.

        Parameters
        ----------
        request : Mapping[str, Any]
            Keyword arguments accepted by :meth:`load_dataset`, such as
            ``dataset``, ``collection``, ``variant``, ``organization``, or ``cid``.
        selection : Mapping[str, Any]
            Selection mapping accepted by :meth:`GeotemporalData.select`.
        return_xarray : bool, optional
            If True, return the raw xarray dataset without applying selections.

        Returns
        -------
        Tuple[Union[GeotemporalData, xr.Dataset], DatasetMetadata]
            The selected dataset plus metadata.
        """
        if not isinstance(request, Mapping):
            raise InvalidSelectionError("request must be a mapping.")
        if not isinstance(selection, Mapping):
            raise InvalidSelectionError("selection must be a mapping.")

        load_kwargs = dict(request)
        if load_kwargs.get("cid") and "dataset" not in load_kwargs:
            load_kwargs["dataset"] = ""

        dataset_obj, metadata = await self.load_dataset(
            **load_kwargs,
            return_xarray=return_xarray,
        )
        if not isinstance(dataset_obj, GeotemporalData):
            return dataset_obj, metadata

        return dataset_obj.select(selection), metadata

    def list_datasets(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """
        List all available datasets from the STAC catalog.

        Returns a dictionary mapping collection IDs to their metadata, including
        the dataset types available in each collection.

        Returns
        -------
        dict
            Dictionary with structure:
            {
                "collection_id": {
                    "id": "collection_id",
                    "title": "Collection Title",
                    "types": ["dataset_type1", "dataset_type2", ...],
                    "organization": "org_id"  # None for legacy catalogs
                },
                ...
            }

        Examples
        --------
        >>> async with dClimateClient() as client:
        ...     datasets = client.list_datasets()
        ...     print(datasets["ecmwf_ifs"]["types"])
        ['temperature', 'precipitation', 'wind_u', 'wind_v', ...]
        """
        # STAC API first — two HTTP calls vs. hundreds of serial IPFS gateway
        # round-trips. Fall through to the IPFS walk only if the server is
        # unavailable or returns garbage. Mirrors the resolve-CID pattern in
        # load_dataset().
        if self._stac_server_url:
            try:
                return list_available_datasets_from_stac_server(self._stac_server_url)
            except (requests.RequestException, ValueError):
                pass

        # Fallback: walk the IPFS-hosted catalog.
        if self._stac_catalog is None:
            self._stac_catalog = load_stac_catalog(gateway_url=self._gateway_base_url)

        return list_available_datasets(self._stac_catalog)

    # ------------------------------------------------------------------
    # Siren REST API methods
    # ------------------------------------------------------------------

    async def get_metric_data(
        self, query: SirenMetricQuery
    ) -> list[SirenMetricDataPoint]:
        """
        Fetch Siren metric data for a region over a date range.

        Requires the client to be initialised with ``siren=SirenOptions(...)``.
        """
        if self._siren_client is None:
            raise RuntimeError(
                "Siren is not configured. Pass siren=SirenOptions(...) "
                "when creating the dClimateClient."
            )
        return await self._siren_client.get_metric_data(query)

    async def list_regions(self) -> list[SirenRegion]:
        """
        List available Siren regions.

        Requires the client to be initialised with ``siren=SirenOptions(...)``.
        """
        if self._siren_client is None:
            raise RuntimeError(
                "Siren is not configured. Pass siren=SirenOptions(...) "
                "when creating the dClimateClient."
            )
        return await self._siren_client.list_regions()
