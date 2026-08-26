class ZarrClientError(Exception):
    """Parent class for library-level Exceptions"""


class SelectionTooLargeError(ZarrClientError):
    """Raised when user selects too many data points"""


class ConflictingGeoRequestError(ZarrClientError):
    """Raised when user requests more than one type of geographic query"""


class ConflictingAggregationRequestError(ZarrClientError):
    """Raised when user requests more than one type of geographic query"""


class NoMetadataFoundError(ZarrClientError):
    """Raised when user selects as_of before earliest existing metadata"""


class NoDataFoundError(ZarrClientError):
    """Raised when user's selection is all NA"""


class DatasetNotFoundError(ZarrClientError):
    """Raised when dataset not available over IPNS/STAC or S3"""


class InvalidForecastRequestError(ZarrClientError):
    """Raised when regular time series are requested from a forecast dataset"""


class InvalidAggregationMethodError(ZarrClientError):
    """Raised when user provides an aggregation method outside of [min, max, median,
    mean, std, sum]"""


class InvalidTimePeriodError(ZarrClientError):
    """Raised when user provides a time period outside of [hour, day, week, month,
    quarter, year]"""


class InvalidExportFormatError(ZarrClientError):
    """Raised when user specifies an export format other than [array, netcdf]"""


class BucketNotFoundError(ZarrClientError):
    """Raised when bucket does not exist in AWS S3"""


class PathNotFoundError(ZarrClientError):
    """Raised when path does not exist in AWS S3 or IPFS"""


class AmbiguousDataVariableError(ZarrClientError):
    """Raised when method that requires a specific data variable is called, the dataset
    has more than variable, and the dataset hasn't been specified by a call to
    :method:`dclimate_client_py.geotemporal_data.GeotemporalData.use`"""


class IpfsConnectionError(ZarrClientError):
    """Raised when connection to IPFS daemon or gateway fails"""


class InvalidSelectionError(ZarrClientError):
    """Raised when dataset/collection/variant selection is invalid or ambiguous"""


class MultiresolutionSelectionRequiredError(InvalidSelectionError):
    """Raised when a pyramidal dataset requires an explicit resolution or group."""

    def __init__(
        self,
        message: str,
        *,
        available_resolutions: tuple[str, ...] = (),
        available_groups: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.available_resolutions = available_resolutions
        self.available_groups = available_groups


class ResolutionNotAvailableError(InvalidSelectionError):
    """Raised when a requested resolution is not advertised by STAC."""


class ConflictingResolutionSelectionError(InvalidSelectionError):
    """Raised when both resolution and raw Zarr group are provided."""


class DatasetCorruptError(ZarrClientError):
    """Raised when the stored dataset is malformed, rather than the request against it.

    Distinct from :class:`InvalidSelectionError` because the two point at
    different culprits. An invalid selection is the caller's to fix by asking
    differently; this says the bytes behind the CID are inconsistent, so no
    rephrasing helps and the dataset's publisher is who needs to know. Reporting
    it as a bad selection would send a caller hunting for a mistake in their own
    query.
    """


class VariantNotFoundError(ZarrClientError):
    """Raised when specified variant is not found in dataset"""


class CollectionNotFoundError(ZarrClientError):
    """Raised when specified collection is not found in catalog"""


# ---------------------------------------------------------------------------
# Siren API errors
# ---------------------------------------------------------------------------


class SirenApiError(ZarrClientError):
    """Raised when the Siren REST API returns an error"""


class X402PaymentError(ZarrClientError):
    """Raised when an x402 payment request fails"""


class X402NotInstalledError(ZarrClientError):
    """Raised when x402 auth is configured but the x402 package is not installed"""


class TabularNotInstalledError(ZarrClientError):
    """Raised when entity data is requested but tabular-py is not installed"""
