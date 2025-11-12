"""
Concatenation utilities for combining dataset variants.

Implements smart concatenation logic similar to dclimate-client-js,
avoiding duplicate data by finding split indices where new data begins.
"""

import logging
import typing
from datetime import datetime
import numpy as np
import xarray as xr

from .dclimate_zarr_errors import NoDataFoundError

logger = logging.getLogger(__name__)


def find_split_index(
    combined_coords: typing.Any,
    next_coords: typing.Any,
    last_coord_value: typing.Any,
) -> int:
    """
    Find the index in next_coords where values are strictly greater than last_coord_value.

    This prevents duplicate data when concatenating datasets.

    Args:
        combined_coords: Coordinates from the combined dataset (for reference)
        next_coords: Coordinates from the next variant to concatenate
        last_coord_value: The last coordinate value from the combined dataset

    Returns:
        Index in next_coords where to start slicing (first value > last_coord_value)

    Raises:
        NoDataFoundError: If no new data found in next variant
    """
    # Convert coordinates to numpy array for easier comparison
    next_array = np.array(next_coords)

    # Handle different data types
    if isinstance(last_coord_value, (datetime, np.datetime64)):
        # Convert to numpy datetime64 for comparison
        if isinstance(last_coord_value, datetime):
            last_coord_value = np.datetime64(last_coord_value)

        # Find where next coords are strictly after last coord
        mask = next_array > last_coord_value
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise NoDataFoundError(
                f"No new data found in next variant after {last_coord_value}"
            )

        return int(indices[0])

    elif isinstance(last_coord_value, (int, float, np.number)):
        # Numeric comparison
        mask = next_array > last_coord_value
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise NoDataFoundError(
                f"No new data found in next variant after {last_coord_value}"
            )

        return int(indices[0])

    elif isinstance(last_coord_value, str):
        # String comparison (lexicographic)
        mask = next_array > last_coord_value
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise NoDataFoundError(
                f"No new data found in next variant after '{last_coord_value}'"
            )

        return int(indices[0])

    else:
        # For other types, try direct comparison
        try:
            mask = next_array > last_coord_value
            indices = np.where(mask)[0]

            if len(indices) == 0:
                raise NoDataFoundError(
                    f"No new data found in next variant after {last_coord_value}"
                )

            return int(indices[0])
        except Exception as e:
            raise TypeError(
                f"Cannot compare coordinate type {type(last_coord_value).__name__}: {e}"
            )


async def concatenate_datasets(
    datasets: typing.List[xr.Dataset],
    dimension: str = "time",
) -> xr.Dataset:
    """
    Concatenate multiple xarray datasets along a dimension, avoiding duplicate data.

    Implements smart concatenation logic:
    1. Start with first dataset (highest priority)
    2. For each subsequent dataset:
       - Find the last coordinate value in combined dataset
       - Find split index in next dataset where coords > last coord
       - Slice next dataset to only include new data
       - Concatenate sliced dataset

    Args:
        datasets: List of xarray datasets to concatenate (in priority order)
        dimension: Dimension to concatenate along (default: "time")

    Returns:
        Concatenated xarray dataset

    Raises:
        ValueError: If datasets list is empty or dimension not found
        NoDataFoundError: If a variant contains no new data
    """
    if not datasets:
        raise ValueError("Cannot concatenate empty list of datasets")

    if len(datasets) == 1:
        logger.info("Only one dataset provided, returning it without concatenation")
        return datasets[0]

    # Check that all datasets have the dimension
    for i, ds in enumerate(datasets):
        if dimension not in ds.dims:
            raise ValueError(
                f"Dataset {i} does not have dimension '{dimension}'. "
                f"Available dimensions: {list(ds.dims.keys())}"
            )

    logger.info(f"Concatenating {len(datasets)} datasets along '{dimension}' dimension")

    # Start with the first dataset (highest priority)
    combined = datasets[0]
    logger.debug(
        f"Starting with dataset 1/{len(datasets)}, "
        f"{dimension} range: {combined[dimension].values[0]} to {combined[dimension].values[-1]}"
    )

    # Concatenate each subsequent dataset
    for i, next_ds in enumerate(datasets[1:], start=2):
        # Get the last coordinate value from combined dataset
        last_coord_value = combined[dimension].values[-1]

        # Get coordinates from next dataset
        next_coords = next_ds[dimension].values

        logger.debug(
            f"Processing dataset {i}/{len(datasets)}, "
            f"{dimension} range: {next_coords[0]} to {next_coords[-1]}"
        )

        # Find where to split the next dataset
        try:
            split_index = find_split_index(
                combined[dimension].values,
                next_coords,
                last_coord_value,
            )

            logger.debug(
                f"Found split index {split_index} (out of {len(next_coords)} coords) "
                f"for dataset {i}, new data starts at: {next_coords[split_index]}"
            )

            # Slice the next dataset to only include new data
            sliced_next = next_ds.isel({dimension: slice(split_index, None)})

            logger.debug(
                f"Sliced dataset {i} to {len(sliced_next[dimension])} new coords"
            )

            # Concatenate with combined dataset
            new_combined = xr.concat(
                [combined, sliced_next],
                dim=dimension,
            )
            logger.debug(
                f"After concatenating dataset {i}, total {dimension} coords: {len(combined[dimension])}"
            )

        except NoDataFoundError as e:
            logger.warning(
                f"Skipping dataset {i} as it contains no new data: {e}"
            )
            # Continue to next dataset
            continue

    logger.info(
        f"Concatenation complete. Final dataset has {len(combined[dimension])} "
        f"{dimension} coordinates ranging from {combined[dimension].values[0]} "
        f"to {combined[dimension].values[-1]}"
    )

    return combined
