"""
IPFS data retrieval functions for loading Zarr datasets.

This module provides functions for loading Zarr datasets from IPFS using KuboCAS
"""

import logging
import xarray as xr
from multiformats import CID
from py_hamt import KuboCAS, ZarrHAMTStore, ShardedZarrStore, HAMT

from .dclimate_zarr_errors import (
    IpfsConnectionError,
)

# Configure logging
logger = logging.getLogger(__name__)


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
    if not ipfs_cid:
        raise ValueError("IPFS CID cannot be empty.")

    logger.info(f"Loading Zarr dataset from IPFS CID: {ipfs_cid}")

    try:
        # Validate CID format
        try:
            cid_obj = CID.decode(ipfs_cid)
        except Exception as decode_err:
            raise ValueError(
                f"Invalid IPFS CID format: {ipfs_cid}. Error: {decode_err}"
            ) from decode_err

        # Try loading as ShardedZarrStore first (99% of cases)
        try:
            logger.info(f"Attempting to load as ShardedZarrStore from CID: {ipfs_cid}")
            sharded_store = await ShardedZarrStore.open(
                root_cid=ipfs_cid, cas=kubo_cas, read_only=True
            )
            ds = xr.open_zarr(store=sharded_store)
            logger.info(f"Successfully loaded ShardedZarrStore dataset from CID: {ipfs_cid}")
            return ds
        except Exception as sharded_err:
            # Fall back to HAMT store if sharded loading fails
            logger.info(
                f"ShardedZarrStore failed, falling back to HAMT store. Error: {sharded_err}"
            )
            logger.info(f"Loading HAMT store from CID: {ipfs_cid}")
            hamt_store = await HAMT.build(
                cas=kubo_cas,
                root_node_id=cid_obj,
                read_only=True,
                values_are_bytes=True,
            )

            # Wrap with ZarrHAMTStore adapter
            zarr_hamt_store = ZarrHAMTStore(hamt_store, read_only=True)

            ds = xr.open_zarr(store=zarr_hamt_store)
            logger.info(f"Successfully loaded HAMT dataset from CID: {ipfs_cid}")
            return ds
    except IpfsConnectionError:
        # Re-raise IpfsConnectionError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        # Catch other potential errors (e.g., Zarr format errors, py-hamt errors)
        # Check for connection errors
        if (
            "Connection refused" in str(e)
            or "Max retries exceeded" in str(e)
            or "Timeout" in str(e)
        ):
            raise IpfsConnectionError(
                f"IPFS connection failed while loading dataset from CID {ipfs_cid}. Details: {e}"
            ) from e

        logger.error(
            f"Failed to load Zarr dataset from IPFS CID {ipfs_cid}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise RuntimeError(
            f"Failed to load Zarr dataset from IPFS CID {ipfs_cid}"
        ) from e


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
