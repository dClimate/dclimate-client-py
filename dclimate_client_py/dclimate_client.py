"""
DClimate Client - Async context manager for loading dClimate datasets

This module provides a high-level client interface that manages IPFS connections
internally, abstracting away KuboCAS lifecycle management.
"""

import asyncio
import typing
from collections.abc import Mapping

if typing.TYPE_CHECKING:
    import pystac

import httpx
import xarray as xr
from py_hamt import KuboCAS

# Import here to avoid circular imports
from .ipfs_retrieval import _load_dataset_from_ipfs_cid


from .geotemporal_data import GeotemporalData
from .datasets import DatasetMetadata
from .dclimate_zarr_errors import (
    ConflictingResolutionSelectionError,
    InvalidSelectionError,
    MultiresolutionSelectionRequiredError,
    ResolutionNotAvailableError,
)
from .stac_server import (
    ResolvedDatasetDetails,
    aresolve_dataset_from_stac_server,
    list_available_datasets_from_stac_server,
)
from .ceramic_api import (
    DatasetVersion,
    DatasetVersionListing,
    get_exact_version_from_url,
    list_versions_from_url,
)
from .siren import SirenClient
from .stations import StationsClient
from .siren.types import (
    SirenMetricDataPoint,
    SirenMetricQuery,
    SirenOptions,
    SirenRegion,
)

DEFAULT_PUBLIC_GATEWAY = "https://ipfs-gateway.dclimate.net"


def _merge_cleanup_error(
    current: BaseException | None,
    new: BaseException,
) -> BaseException:
    """Chain cleanup failures while ensuring cancellation remains dominant."""
    if current is None:
        return new
    if isinstance(current, asyncio.CancelledError) and not isinstance(
        new, asyncio.CancelledError
    ):
        new.__context__ = current.__context__
        current.__context__ = new
        return current
    new.__context__ = current
    return new


class dClimateClient:
    """
    Async context manager for loading dClimate datasets from IPFS.

    This client manages IPFS connections internally via KuboCAS, so users don't
    need to manually configure or import IPFS-related dependencies.

    Parameters
    ----------
    gateway_base_url : str, optional
        IPFS HTTP Gateway base URL (e.g., "https://ipfs.io" or "http://localhost:8080").
        If None, KuboCAS uses its own defaults while STAC-catalog fallback reads
        use ``DEFAULT_PUBLIC_GATEWAY``.
    rpc_base_url : str, optional
        IPFS RPC API base URL (e.g., "http://localhost:5001").
        If None, uses KuboCAS defaults or environment variables.
    concurrency : int, optional
        Maximum number of concurrent Kubo gateway and RPC requests.
    headers : dict[str, str], optional
        Default headers for the internally-created HTTP client.
    auth : tuple[str, str], optional
        Authentication tuple (username, password) for the internally-created client.
    max_retries : int, optional
        Maximum number of retries for retryable gateway requests.
    initial_delay : float, optional
        Initial retry delay in seconds.
    backoff_factor : float, optional
        Multiplier used for exponential retry backoff.
    client_factory : Callable[[], httpx.AsyncClient], optional
        Create a separate, fully configured HTTP client for each event loop.
        Cannot be combined with ``headers`` or ``auth``.

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
        gateway_base_url: typing.Optional[str] = DEFAULT_PUBLIC_GATEWAY,
        rpc_base_url: typing.Optional[str] = DEFAULT_PUBLIC_GATEWAY,
        stac_server_url: typing.Optional[str] = "https://api.stac.dclimate.net",
        siren: typing.Optional[SirenOptions] = None,
        *,
        concurrency: typing.Optional[int] = None,
        headers: typing.Optional[dict[str, str]] = None,
        auth: typing.Optional[tuple[str, str]] = None,
        max_retries: typing.Optional[int] = None,
        initial_delay: typing.Optional[float] = None,
        backoff_factor: typing.Optional[float] = None,
        client_factory: typing.Optional[typing.Callable[[], httpx.AsyncClient]] = None,
    ) -> None:
        if client_factory is not None and (headers is not None or auth is not None):
            raise ValueError("client_factory cannot be combined with headers or auth")

        self._gateway_base_url = gateway_base_url
        self._catalog_gateway_base_url = (
            gateway_base_url if gateway_base_url is not None else DEFAULT_PUBLIC_GATEWAY
        )
        self._rpc_base_url = rpc_base_url
        self._stac_server_url = stac_server_url
        self._concurrency = concurrency
        self._headers = headers
        self._auth = auth
        self._max_retries = max_retries
        self._initial_delay = initial_delay
        self._backoff_factor = backoff_factor
        self._client_factory = client_factory
        self._stac_catalog: typing.Optional["pystac.Catalog"] = None
        self._stac_catalog_lock = asyncio.Lock()
        self._stac_http_client: typing.Optional[httpx.AsyncClient] = None
        self._kubo_cas: typing.Optional[KuboCAS] = None
        # Note: STAC catalog is loaded lazily (only if STAC server fails)

        # Siren REST API client (optional)
        self._siren_client: typing.Optional[SirenClient] = None
        if siren is not None:
            self._siren_client = SirenClient(siren)

        # Station datasets. Unlike Siren this needs no configuration, so it is
        # built on first access rather than here -- see the `stations` property.
        self._stations_client: typing.Optional["StationsClient"] = None

    async def __aenter__(self) -> "dClimateClient":
        """Initialize KuboCAS when entering async context."""
        # Create KuboCAS with configured endpoints
        kubo_kwargs: dict[str, typing.Any] = {
            "gateway_base_url": self._gateway_base_url,
            "rpc_base_url": self._rpc_base_url,
        }
        optional_kubo_kwargs = {
            "concurrency": self._concurrency,
            "headers": self._headers,
            "auth": self._auth,
            "max_retries": self._max_retries,
            "initial_delay": self._initial_delay,
            "backoff_factor": self._backoff_factor,
            "client_factory": self._client_factory,
        }
        kubo_kwargs.update(
            {
                key: value
                for key, value in optional_kubo_kwargs.items()
                if value is not None
            }
        )
        self._kubo_cas = KuboCAS(**kubo_kwargs)
        # Enter the KuboCAS context manager
        await self._kubo_cas.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up owned HTTP and KuboCAS resources."""
        incoming_cancellation = isinstance(exc_val, asyncio.CancelledError)
        cleanup_error: BaseException | None = None

        try:
            if self._stac_http_client is not None:
                await self._stac_http_client.aclose()
        except BaseException as error:
            cleanup_error = _merge_cleanup_error(cleanup_error, error)
        finally:
            self._stac_http_client = None

        try:
            if self._siren_client is not None:
                await self._siren_client.aclose()
        except BaseException as error:
            cleanup_error = _merge_cleanup_error(cleanup_error, error)

        try:
            if self._stations_client is not None:
                await self._stations_client.aclose()
        except BaseException as error:
            cleanup_error = _merge_cleanup_error(cleanup_error, error)

        try:
            if self._kubo_cas is not None:
                await self._kubo_cas.__aexit__(exc_type, exc_val, exc_tb)
        except BaseException as kubo_error:
            cleanup_error = _merge_cleanup_error(cleanup_error, kubo_error)
        finally:
            self._kubo_cas = None

        if cleanup_error is not None:
            if incoming_cancellation and not isinstance(
                cleanup_error, asyncio.CancelledError
            ):
                # Preserve cancellation from the context body. Ordinary
                # cleanup failures remain inspectable through its context.
                exc_val.__context__ = cleanup_error
            else:
                raise cleanup_error

        return False

    def _get_stac_http_client(self) -> httpx.AsyncClient:
        """Return the pooled STAC transport owned by this client context."""
        client = self._stac_http_client
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=30, follow_redirects=False)
            self._stac_http_client = client
        return client

    @staticmethod
    def _apply_zarr_group_metadata(ds: xr.Dataset, metadata: DatasetMetadata) -> None:
        loaded_zarr_group = ds.attrs.get("_ipfs_zarr_group")
        if isinstance(loaded_zarr_group, str):
            metadata["zarr_group"] = loaded_zarr_group

    @staticmethod
    def _resolve_zarr_selection(
        resolved: ResolvedDatasetDetails,
        *,
        resolution: typing.Optional[str],
        zarr_group: typing.Optional[str],
    ) -> tuple[typing.Optional[str], typing.Optional[str]]:
        if resolution is not None and zarr_group is not None:
            raise ConflictingResolutionSelectionError(
                "Pass either resolution or zarr_group, not both."
            )

        choices = resolved.zarr_resolutions
        if resolution is not None:
            match = next(
                (choice for choice in choices if choice.resolution == resolution),
                None,
            )
            if match is None:
                available = tuple(choice.resolution for choice in choices)
                raise ResolutionNotAvailableError(
                    f"Resolution '{resolution}' is not available."
                    + (f" Choose one of: {', '.join(available)}." if available else "")
                )
            return match.group, match.resolution

        if zarr_group is not None:
            normalized_group = zarr_group.strip("/")
            if choices and normalized_group not in {choice.group for choice in choices}:
                available_groups = tuple(choice.group for choice in choices)
                raise ResolutionNotAvailableError(
                    f"Zarr group '{normalized_group}' is not available. "
                    f"Choose one of: {', '.join(available_groups)}."
                )
            selected_resolution = next(
                (
                    choice.resolution
                    for choice in choices
                    if choice.group == normalized_group
                ),
                None,
            )
            return normalized_group, selected_resolution

        if len(choices) > 1:
            available = tuple(choice.resolution for choice in choices)
            raise MultiresolutionSelectionRequiredError(
                "This dataset has multiple resolutions; pass resolution or zarr_group. "
                f"Available resolutions: {', '.join(available)}.",
                available_resolutions=available,
                available_groups=tuple(choice.group for choice in choices),
            )
        if len(choices) == 1:
            return choices[0].group, choices[0].resolution
        return None, None

    async def load_dataset(
        self,
        dataset: str,
        collection: typing.Optional[str] = None,
        variant: typing.Optional[str] = None,
        organization: typing.Optional[str] = None,
        cid: typing.Optional[str] = None,
        return_xarray: bool = False,
        zarr_group: typing.Optional[str] = None,
        shard_read_mode: typing.Literal["full", "sparse"] = "sparse",
        resolution: typing.Optional[str] = None,
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
        zarr_group : str, optional
            Explicit Zarr group to open for grouped/pyramid sharded stores.
        resolution : str, optional
            Human-readable resolution advertised by STAC. Multiresolution
            datasets require either this parameter or ``zarr_group``.
        shard_read_mode : {"full", "sparse"}, optional
            Sharded Zarr shard-index read strategy. Defaults to ``"sparse"``
            and decodes only the requested shard slot on read-only cache
            misses. ``"full"`` preserves the decoded-shard cache behavior.

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
        httpx.HTTPError
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

        # Case 1: Direct CID provided - bypass catalog resolution
        metadata: DatasetMetadata
        if cid:
            if resolution is not None and zarr_group is not None:
                raise ConflictingResolutionSelectionError(
                    "Pass either resolution or zarr_group, not both."
                )
            if resolution is not None:
                raise ResolutionNotAvailableError(
                    "resolution requires STAC metadata; pass zarr_group for a direct CID."
                )
            direct_collection = collection
            if (
                organization
                and direct_collection
                and not direct_collection.startswith(f"{organization}_")
            ):
                direct_collection = f"{organization}_{direct_collection}"
            slug_collection = direct_collection or "unknown"
            direct_variant = variant or "unknown"
            dataset_slug = (
                f"{organization}/{slug_collection}/{dataset}/{direct_variant}"
                if organization
                else f"{slug_collection}/{dataset}/{direct_variant}"
            )
            ds = await _load_dataset_from_ipfs_cid(
                ipfs_cid=cid,
                kubo_cas=self._kubo_cas,
                zarr_group=zarr_group,
                shard_read_mode=shard_read_mode,
            )

            # Build metadata for direct CID case
            metadata = {
                "collection": direct_collection or "unknown",
                "dataset": dataset,
                "variant": direct_variant,
                "slug": dataset_slug,
                "cid": cid,
                "url": None,
                "timestamp": None,
                "source": "direct_cid",
                "organization": organization
                or (
                    direct_collection.split("_")[0]
                    if direct_collection and "_" in direct_collection
                    else None
                ),
            }
            self._apply_zarr_group_metadata(ds, metadata)

            if return_xarray:
                return ds, metadata
            else:
                return GeotemporalData(ds, dataset_name=dataset_slug), metadata

        # Case 2: Resolve via STAC server (fast) or STAC catalog (fallback)
        if not collection:
            raise InvalidSelectionError(
                "collection parameter is required. Use client.list_datasets() to see available collections."
            )

        resolved_collection = collection
        if organization and not collection.startswith(f"{organization}_"):
            resolved_collection = f"{organization}_{collection}"

        resolved: typing.Optional[ResolvedDatasetDetails] = None

        # Try STAC server first (faster, avoids loading IPFS catalog)
        if self._stac_server_url:
            try:
                resolved = await aresolve_dataset_from_stac_server(
                    collection=resolved_collection,
                    dataset=dataset,
                    variant=variant,
                    server_url=self._stac_server_url,
                    client=self._get_stac_http_client(),
                )
            except (httpx.HTTPError, ValueError):
                # Fall back when server lookup fails or returns no usable match.
                pass

        # Fallback: Resolve via STAC catalog from IPFS
        if resolved is None:
            from .stac_catalog import (
                list_available_datasets,
                load_stac_catalog,
                resolve_dataset_from_stac,
            )

            # Lazy load STAC catalog
            if self._stac_catalog is None:
                async with self._stac_catalog_lock:
                    if self._stac_catalog is None:
                        self._stac_catalog = await asyncio.to_thread(
                            load_stac_catalog,
                            gateway_url=self._catalog_gateway_base_url,
                            headers=self._headers,
                            auth=self._auth,
                        )

            if not organization and resolved_collection:
                available = await asyncio.to_thread(
                    list_available_datasets, self._stac_catalog
                )
                if resolved_collection not in available:
                    prefixed_matches = [
                        coll_id
                        for coll_id in available.keys()
                        if coll_id.endswith(f"_{resolved_collection}")
                    ]
                    if len(prefixed_matches) == 1:
                        resolved_collection = prefixed_matches[0]

            resolved = await asyncio.to_thread(
                resolve_dataset_from_stac,
                catalog=self._stac_catalog,
                collection=resolved_collection,
                dataset=dataset,
                variant=variant,
                organization=organization,
            )

        assert resolved is not None

        selected_group, selected_resolution = self._resolve_zarr_selection(
            resolved,
            resolution=resolution,
            zarr_group=zarr_group,
        )

        ds = await _load_dataset_from_ipfs_cid(
            ipfs_cid=resolved.cid,
            kubo_cas=self._kubo_cas,
            zarr_group=selected_group,
            shard_read_mode=shard_read_mode,
        )

        # Build metadata for STAC case
        metadata = {
            "collection": resolved_collection,
            "dataset": dataset,
            "variant": resolved.variant,
            "slug": (
                f"{organization}/{resolved_collection or collection}/{dataset}/{resolved.variant}"
                if organization
                else f"{resolved_collection or collection}/{dataset}/{resolved.variant}"
            ),
            "cid": resolved.cid,
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
        if resolved.versions_api is not None:
            metadata["versions_api"] = resolved.versions_api
        if resolved.provenance_api is not None:
            metadata["provenance_api"] = resolved.provenance_api
        if resolved.citation_api is not None:
            metadata["citation_api"] = resolved.citation_api
        if resolved.stream_id is not None:
            metadata["stream_id"] = resolved.stream_id
        if resolved.commit_id is not None:
            metadata["commit_id"] = resolved.commit_id
        if resolved.version_label is not None:
            metadata["version_label"] = resolved.version_label
        if resolved.is_citable is not None:
            metadata["is_citable"] = resolved.is_citable
        if resolved.retention_class is not None:
            metadata["retention_class"] = resolved.retention_class
        self._apply_zarr_group_metadata(ds, metadata)
        if selected_resolution is not None:
            metadata["resolution"] = selected_resolution

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
        load_kwargs.pop("return_xarray", None)
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

        Notes
        -----
        This is a synchronous method: on the IPFS-catalog fallback path it
        performs blocking network I/O and will stall a running event loop.
        Inside async code prefer ``await client.alist_datasets()``.

        Examples
        --------
        >>> async with dClimateClient() as client:
        ...     datasets = await client.alist_datasets()
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
            except (httpx.HTTPError, ValueError):
                pass

        # Fallback: walk the IPFS-hosted catalog.
        from .stac_catalog import load_stac_catalog, list_available_datasets

        if self._stac_catalog is None:
            self._stac_catalog = load_stac_catalog(
                gateway_url=self._catalog_gateway_base_url,
                headers=self._headers,
                auth=self._auth,
            )

        return list_available_datasets(self._stac_catalog)

    async def alist_datasets(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Async variant of :meth:`list_datasets`.

        Runs the blocking STAC/catalog work in a thread so the event loop
        (and py-hamt's concurrent chunk fetches) never stall, and shares the
        catalog lazy-init lock with :meth:`load_dataset`.
        """
        if self._stac_server_url:
            try:
                return await asyncio.to_thread(
                    list_available_datasets_from_stac_server, self._stac_server_url
                )
            except (httpx.HTTPError, ValueError):
                pass

        from .stac_catalog import load_stac_catalog, list_available_datasets

        if self._stac_catalog is None:
            async with self._stac_catalog_lock:
                if self._stac_catalog is None:
                    self._stac_catalog = await asyncio.to_thread(
                        load_stac_catalog,
                        gateway_url=self._catalog_gateway_base_url,
                        headers=self._headers,
                        auth=self._auth,
                    )

        return await asyncio.to_thread(list_available_datasets, self._stac_catalog)

    async def _resolve_dataset_details(
        self,
        collection: str,
        dataset: str,
        variant: typing.Optional[str],
        organization: typing.Optional[str],
    ) -> ResolvedDatasetDetails:
        """Resolve release metadata through the hosted STAC API or IPFS fallback."""
        resolved_collection = collection
        if organization and not collection.startswith(f"{organization}_"):
            resolved_collection = f"{organization}_{collection}"

        if self._stac_server_url:
            try:
                if self._kubo_cas is None:
                    async with httpx.AsyncClient(
                        timeout=30, follow_redirects=False
                    ) as client:
                        return await aresolve_dataset_from_stac_server(
                            collection=resolved_collection,
                            dataset=dataset,
                            variant=variant,
                            server_url=self._stac_server_url,
                            client=client,
                        )
                return await aresolve_dataset_from_stac_server(
                    collection=resolved_collection,
                    dataset=dataset,
                    variant=variant,
                    server_url=self._stac_server_url,
                    client=self._get_stac_http_client(),
                )
            except (httpx.HTTPError, ValueError):
                pass

        from .stac_catalog import (
            list_available_datasets,
            load_stac_catalog,
            resolve_dataset_from_stac,
        )

        if self._stac_catalog is None:
            async with self._stac_catalog_lock:
                if self._stac_catalog is None:
                    self._stac_catalog = await asyncio.to_thread(
                        load_stac_catalog,
                        gateway_url=self._catalog_gateway_base_url,
                        headers=self._headers,
                        auth=self._auth,
                    )

        if not organization and resolved_collection:
            available = await asyncio.to_thread(
                list_available_datasets, self._stac_catalog
            )
            if resolved_collection not in available:
                prefixed_matches = [
                    coll_id
                    for coll_id in available
                    if coll_id.endswith(f"_{resolved_collection}")
                ]
                if len(prefixed_matches) == 1:
                    resolved_collection = prefixed_matches[0]

        return await asyncio.to_thread(
            resolve_dataset_from_stac,
            catalog=self._stac_catalog,
            collection=resolved_collection,
            dataset=dataset,
            variant=variant,
            organization=organization,
        )

    async def list_dataset_versions(
        self,
        collection: str,
        dataset: str,
        variant: typing.Optional[str] = None,
        organization: typing.Optional[str] = None,
        *,
        anchored: typing.Optional[bool] = None,
        is_citable: typing.Optional[bool] = None,
        version_label: typing.Optional[str] = None,
    ) -> DatasetVersionListing:
        """List releases using the version-service URL advertised by STAC."""
        details = await self._resolve_dataset_details(
            collection=collection,
            dataset=dataset,
            variant=variant,
            organization=organization,
        )
        if not details.versions_api:
            raise ValueError(
                "Version history is not available for "
                f"{collection}/{dataset}/{details.variant}"
            )
        return await asyncio.to_thread(
            list_versions_from_url,
            details.versions_api,
            anchored=anchored,
            is_citable=is_citable,
            version_label=version_label,
        )

    async def get_dataset_version(
        self,
        collection: str,
        dataset: str,
        commit_id: str,
        variant: typing.Optional[str] = None,
        organization: typing.Optional[str] = None,
    ) -> DatasetVersion:
        """Resolve one exact release through the version URL advertised by STAC."""
        details = await self._resolve_dataset_details(
            collection=collection,
            dataset=dataset,
            variant=variant,
            organization=organization,
        )
        if not details.versions_api:
            raise ValueError(
                "Version history is not available for "
                f"{collection}/{dataset}/{details.variant}"
            )
        return await asyncio.to_thread(
            get_exact_version_from_url,
            details.versions_api,
            commit_id,
        )

    # ------------------------------------------------------------------
    # Station (point-observation) datasets
    # ------------------------------------------------------------------

    @property
    def stations(self) -> StationsClient:
        """Station datasets, e.g. ``await client.stations.load(cid)``.

        Unlike :attr:`siren`, this needs no configuration -- it reads over the
        transport the client already has, so requiring an option would be
        ceremony with nothing behind it.

        Reads prefer the client's own ``KuboCAS`` when the client is open, so
        pinning, retries, and configured endpoints apply to station reads too.
        Outside the async context there is no ``KuboCAS`` yet, so this falls back
        to the plain HTTP gateway -- which is why the instance is rebuilt if it
        was first created before ``__aenter__``.
        """
        cas = self._kubo_cas
        if self._stations_client is None or (
            cas is not None and self._stations_client._cas is not cas
        ):
            self._stations_client = StationsClient(
                gateway_url=self._catalog_gateway_base_url,
                cas=cas,
            )
        return self._stations_client

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
