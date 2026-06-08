"""
IPFS data retrieval functions for loading Zarr datasets.

This module provides functions for loading Zarr datasets from IPFS using KuboCAS
"""

import logging
import time
from typing import Any

import xarray as xr
from multiformats import CID
from opentelemetry import metrics, trace
from opentelemetry.trace import Span, Status, StatusCode
from py_hamt import HAMT, KuboCAS, ShardedZarrStore, ZarrHAMTStore

try:
    import py_hamt.instrumentation as ipfs_instrumentation
except ModuleNotFoundError as exc:
    if exc.name != "py_hamt.instrumentation":
        raise

    class _NoOpIpfsInstrumentation:
        @staticmethod
        def observe(name: str, seconds: float) -> None:
            return None

    ipfs_instrumentation = _NoOpIpfsInstrumentation()

from .dclimate_zarr_errors import (
    IpfsConnectionError,
)

# Configure logging
logger = logging.getLogger(__name__)
OTEL_ATTRIBUTE_VALUE = str | bool | int | float

_TRACER = trace.get_tracer("dclimate_client_py.ipfs_retrieval")
_METER = metrics.get_meter("dclimate_client_py.ipfs_retrieval")
_DATASET_OPEN_COUNTER = _METER.create_counter(
    "dclimate_client.ipfs.dataset_open.requests",
    unit="1",
    description="IPFS Zarr dataset open requests.",
)
_DATASET_OPEN_DURATION = _METER.create_histogram(
    "dclimate_client.ipfs.dataset_open.duration",
    unit="s",
    description="IPFS Zarr dataset open latency.",
)
_STORE_OPEN_COUNTER = _METER.create_counter(
    "dclimate_client.ipfs.store_open.requests",
    unit="1",
    description="IPFS Zarr store open attempts.",
)
_STORE_OPEN_DURATION = _METER.create_histogram(
    "dclimate_client.ipfs.store_open.duration",
    unit="s",
    description="IPFS Zarr store open attempt latency.",
)


def _attributes(
    attributes: dict[str, Any] | None = None,
) -> dict[str, OTEL_ATTRIBUTE_VALUE]:
    if not attributes:
        return {}
    cleaned: dict[str, OTEL_ATTRIBUTE_VALUE] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def _gateway_metric_attributes(
    kubo_cas: KuboCAS,
) -> dict[str, OTEL_ATTRIBUTE_VALUE]:
    return _attributes(
        {"dclimate_client.ipfs.gateway": getattr(kubo_cas, "gateway_base_url", None)}
    )


def _record_dataset_open(
    *,
    kubo_cas: KuboCAS,
    store_type: str,
    status: str,
    seconds: float,
) -> None:
    attributes = {
        **_gateway_metric_attributes(kubo_cas),
        "dclimate_client.ipfs.store_type": store_type,
        "dclimate_client.ipfs.status": status,
    }
    _DATASET_OPEN_COUNTER.add(1, attributes)
    _DATASET_OPEN_DURATION.record(seconds, attributes)


def _record_store_open(
    *,
    kubo_cas: KuboCAS,
    store_type: str,
    status: str,
    seconds: float,
) -> None:
    attributes = {
        **_gateway_metric_attributes(kubo_cas),
        "dclimate_client.ipfs.store_type": store_type,
        "dclimate_client.ipfs.status": status,
    }
    _STORE_OPEN_COUNTER.add(1, attributes)
    _STORE_OPEN_DURATION.record(seconds, attributes)


def _record_span_error(active_span: Span, exc: Exception) -> None:
    if not active_span.is_recording():
        return
    active_span.record_exception(exc)
    active_span.set_status(Status(StatusCode.ERROR, str(exc)))


# --- Zarr Dataset Loading ---


async def _load_dataset_from_ipfs_cid(
    ipfs_cid: str,
    kubo_cas: KuboCAS,
) -> xr.Dataset:
    """
    Internal function to load a Zarr dataset from IPFS using a provided KuboCAS instance.

    This function is called by both the new dClimateClient and the legacy
    _get_dataset_by_ipfs_cid function. It attempts to load as ShardedZarrStore
    first (99% of cases), then falls back to HAMT store if that fails.

    Args:
        ipfs_cid (str): The IPFS CID of the Zarr dataset's root node.
        kubo_cas (KuboCAS): An active KuboCAS instance to use for loading.

    Returns:
        xr.Dataset: The loaded dataset.

    Raises:
        ValueError: If IPFS CID is invalid.
        IpfsConnectionError: If connection to IPFS fails during loading.
        RuntimeError: Other errors during Zarr parsing or IPFS interaction.
    """
    dataset_started_at = time.perf_counter()
    dataset_status = "error"
    dataset_store_type = "unknown"
    with _TRACER.start_as_current_span(
        "dclimate_client.ipfs.load_zarr_dataset",
        attributes=_attributes(
            {
                "dclimate_client.ipfs.cid": ipfs_cid,
                **_gateway_metric_attributes(kubo_cas),
            }
        ),
    ) as dataset_span:
        try:
            if not ipfs_cid:
                raise ValueError("IPFS CID cannot be empty.")

            logger.info(f"Loading Zarr dataset from IPFS CID: {ipfs_cid}")

            # Validate CID format
            try:
                cid_obj = CID.decode(ipfs_cid)
            except Exception as decode_err:
                raise ValueError(
                    f"Invalid IPFS CID format: {ipfs_cid}. Error: {decode_err}"
                ) from decode_err

            # Try loading as ShardedZarrStore first (99% of cases)
            try:
                logger.info(
                    f"Attempting to load as ShardedZarrStore from CID: {ipfs_cid}"
                )
                sharded_start = time.perf_counter()
                with _TRACER.start_as_current_span(
                    "dclimate_client.ipfs.open_sharded_store",
                    attributes=_attributes(
                        {
                            "dclimate_client.ipfs.store_type": "ShardedZarrStore",
                            **_gateway_metric_attributes(kubo_cas),
                        }
                    ),
                ) as sharded_span:
                    try:
                        sharded_store = await ShardedZarrStore.open(
                            root_cid=ipfs_cid, cas=kubo_cas, read_only=True
                        )
                        ds = xr.open_zarr(store=sharded_store)
                    except Exception as sharded_err:
                        sharded_seconds = time.perf_counter() - sharded_start
                        _record_store_open(
                            kubo_cas=kubo_cas,
                            store_type="ShardedZarrStore",
                            status="error",
                            seconds=sharded_seconds,
                        )
                        _record_span_error(sharded_span, sharded_err)
                        raise
                    sharded_seconds = time.perf_counter() - sharded_start
                    _record_store_open(
                        kubo_cas=kubo_cas,
                        store_type="ShardedZarrStore",
                        status="ok",
                        seconds=sharded_seconds,
                    )
                ds.attrs["_ipfs_store_type"] = "ShardedZarrStore"
                ipfs_instrumentation.observe(
                    "dclimate_client.open_sharded_store_seconds",
                    sharded_seconds,
                )
                logger.info(
                    f"Successfully loaded ShardedZarrStore dataset from CID: {ipfs_cid}"
                )
                dataset_status = "ok"
                dataset_store_type = "ShardedZarrStore"
                return ds
            except Exception as sharded_err:
                # Fall back to HAMT store if sharded loading fails
                if dataset_span.is_recording():
                    dataset_span.set_attribute(
                        "dclimate_client.ipfs.sharded_fallback", True
                    )
                logger.info(
                    f"ShardedZarrStore failed, falling back to HAMT store. Error: {sharded_err}"
                )
                logger.info(f"Loading HAMT store from CID: {ipfs_cid}")
                hamt_start = time.perf_counter()
                with _TRACER.start_as_current_span(
                    "dclimate_client.ipfs.open_hamt_store",
                    attributes=_attributes(
                        {
                            "dclimate_client.ipfs.store_type": "ZarrHAMTStore",
                            **_gateway_metric_attributes(kubo_cas),
                        }
                    ),
                ) as hamt_span:
                    try:
                        hamt_store = await HAMT.build(
                            cas=kubo_cas,
                            root_node_id=cid_obj,
                            read_only=True,
                            values_are_bytes=True,
                        )

                        # Wrap with ZarrHAMTStore adapter
                        zarr_hamt_store = ZarrHAMTStore(hamt_store, read_only=True)

                        ds = xr.open_zarr(store=zarr_hamt_store)
                    except Exception as hamt_err:
                        hamt_seconds = time.perf_counter() - hamt_start
                        _record_store_open(
                            kubo_cas=kubo_cas,
                            store_type="ZarrHAMTStore",
                            status="error",
                            seconds=hamt_seconds,
                        )
                        _record_span_error(hamt_span, hamt_err)
                        raise
                    hamt_seconds = time.perf_counter() - hamt_start
                    _record_store_open(
                        kubo_cas=kubo_cas,
                        store_type="ZarrHAMTStore",
                        status="ok",
                        seconds=hamt_seconds,
                    )
                ds.attrs["_ipfs_store_type"] = "ZarrHAMTStore"
                ipfs_instrumentation.observe(
                    "dclimate_client.open_hamt_store_seconds",
                    hamt_seconds,
                )
                logger.info(f"Successfully loaded HAMT dataset from CID: {ipfs_cid}")
                dataset_status = "ok"
                dataset_store_type = "ZarrHAMTStore"
                return ds
        except IpfsConnectionError as exc:
            dataset_status = "connection_error"
            _record_span_error(dataset_span, exc)
            raise
        except ValueError as exc:
            # Re-raise ValueError as-is
            _record_span_error(dataset_span, exc)
            raise
        except Exception as e:
            # Catch other potential errors (e.g., Zarr format errors, py-hamt errors)
            # Check for connection errors
            if (
                "Connection refused" in str(e)
                or "Max retries exceeded" in str(e)
                or "Timeout" in str(e)
            ):
                dataset_status = "connection_error"
                connection_error = IpfsConnectionError(
                    f"IPFS connection failed while loading dataset from CID {ipfs_cid}. Details: {e}"
                )
                _record_span_error(dataset_span, connection_error)
                raise connection_error from e

            _record_span_error(dataset_span, e)
            logger.error(
                f"Failed to load Zarr dataset from IPFS CID {ipfs_cid}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to load Zarr dataset from IPFS CID {ipfs_cid}"
            ) from e
        finally:
            if dataset_span.is_recording():
                dataset_span.set_attribute(
                    "dclimate_client.ipfs.store_type", dataset_store_type
                )
                dataset_span.set_attribute(
                    "dclimate_client.ipfs.status", dataset_status
                )
            _record_dataset_open(
                kubo_cas=kubo_cas,
                store_type=dataset_store_type,
                status=dataset_status,
                seconds=time.perf_counter() - dataset_started_at,
            )


# Legacy wrapper for backward compatibility
async def _get_dataset_by_ipfs_cid(
    ipfs_cid: str,
    gateway_uri_stem: str | None = None,
    rpc_uri_stem: str | None = None,
) -> xr.Dataset:
    """
    Gets an xarray dataset directly from its Zarr root IPFS CID.

    This is a legacy wrapper that creates its own KuboCAS instance for
    backward compatibility. New code should use dClimateClient instead.

    Attempts to load as ShardedZarrStore first (99% of cases), then falls back
    to HAMT store if that fails.

    Args:
        ipfs_cid (str): The IPFS CID of the Zarr dataset's root node.
        gateway_uri_stem (str, optional): Custom IPFS HTTP Gateway URI stem.
        rpc_uri_stem (str, optional): Custom IPFS RPC API URI stem.

    Returns:
        xr.Dataset: The loaded dataset.

    Raises:
        IpfsConnectionError: If connection to IPFS fails during loading.
        Exception: Other errors during Zarr parsing or IPFS interaction.
    """
    async with KuboCAS(
        rpc_base_url=rpc_uri_stem, gateway_base_url=gateway_uri_stem
    ) as kubo_cas:
        return await _load_dataset_from_ipfs_cid(ipfs_cid, kubo_cas)
