# public API
from importlib import import_module

from .encryption_codec import (
    EncryptionCodec,
)
from .datasets import (
    DatasetCatalog,
    CatalogCollection,
    CatalogDataset,
    DatasetVariantConfig,
    SpatialExtent,
    TemporalExtent,
)
from .stac_server import (
    ResolvedDataset,
    ZarrResolution,
    aclose_stac_server_client,
    aresolve_cid_from_stac_server,
    resolve_cid_from_stac_server,
    list_available_datasets_from_stac_server,
    STAC_SERVER_URL,
)
from .siren import (
    SirenClient,
    SirenApiKeyAuth,
    SirenX402Auth,
    SirenOptions,
    SirenMetricQuery,
    SirenMetricDataPoint,
    SirenRegion,
    SirenRegionsResponse,
    SirenCountry,
    EvmSigner,
)
from .dclimate_zarr_errors import (
    ConflictingResolutionSelectionError,
    DatasetCorruptError,
    DatasetNotFoundError,
    InvalidSelectionError,
    NoDataFoundError,
    MultiresolutionSelectionRequiredError,
    ResolutionNotAvailableError,
    SirenApiError,
    TabularNotInstalledError,
    X402PaymentError,
    X402NotInstalledError,
    ZarrClientError,
)

_LAZY_IMPORTS = {
    "load_s3": (".client", "load_s3"),
    "geo_temporal_query": (".client", "geo_temporal_query"),
    "dClimateClient": (".dclimate_client", "dClimateClient"),
    "GeotemporalData": (".geotemporal_data", "GeotemporalData"),
    "load_stac_catalog": (".stac_catalog", "load_stac_catalog"),
    "StationsClient": (".stations", "StationsClient"),
    "WrappedStationDataset": (".stations", "WrappedStationDataset"),
    "list_available_datasets": (".stac_catalog", "list_available_datasets"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


__all__ = [
    "dClimateClient",
    "load_s3",
    "geo_temporal_query",
    "GeotemporalData",
    "EncryptionCodec",
    "DatasetCatalog",
    "CatalogCollection",
    "CatalogDataset",
    "DatasetVariantConfig",
    "SpatialExtent",
    "TemporalExtent",
    "load_stac_catalog",
    "list_available_datasets",
    "ResolvedDataset",
    "ZarrResolution",
    "aclose_stac_server_client",
    "aresolve_cid_from_stac_server",
    "resolve_cid_from_stac_server",
    "list_available_datasets_from_stac_server",
    "STAC_SERVER_URL",
    "ZarrClientError",
    "DatasetCorruptError",
    "DatasetNotFoundError",
    "InvalidSelectionError",
    "NoDataFoundError",
    "MultiresolutionSelectionRequiredError",
    "ResolutionNotAvailableError",
    "ConflictingResolutionSelectionError",
    # Siren
    "SirenClient",
    "SirenApiKeyAuth",
    "SirenX402Auth",
    "SirenOptions",
    "SirenMetricQuery",
    "SirenMetricDataPoint",
    "SirenRegion",
    "SirenRegionsResponse",
    "SirenCountry",
    "EvmSigner",
    "SirenApiError",
    "X402PaymentError",
    "X402NotInstalledError",
    # Stations
    "StationsClient",
    "WrappedStationDataset",
    "TabularNotInstalledError",
]
