import xarray as xr
from .base import BaseLoader
from dclimate_zarr_client.ipfs_retrieval import get_dataset_stac
import numpy as np
import pandas as pd
from dclimate_zarr_client.registry import get_cid_from_registry

class ERA5Loader(BaseLoader):
    collection = "ERA5"

    # Helps get the finalized or unfinalized dataset CID
    async def _get_dataset_type_cid(self, stac: dict, type: str) -> str:
        links = stac.get("links", [])
        dataset_cid = None
        for link in links:
            if link.get("rel") == "item" and link.get("title") == type:
                dataset_cid = link.get("href")["/"]
                break
        finalized_stac_item = await self._load_stac_json_from_cid(dataset_cid)
        finalized_stac_cid = finalized_stac_item.get("assets", {}).get("sharded-zarr", {}).get("href", None)
        return finalized_stac_cid

    async def load(self, dataset: str, options: dict) -> xr.Dataset:
        finalized_only = options.get('finalized_only', False)
        root_cid = get_cid_from_registry()

        # # 1. --- Get STAC metadata for the dataset collection ---
        dataset_stac = await get_dataset_stac(
            stac_root_cid=root_cid,
            collection=self.collection,
            dataset=dataset,
            gateway_uri_stem=self.gateway_url
        )

        # # 2. --- Lazily Open Finalized Dataset ---
        finalized_stac_cid = await self._get_dataset_type_cid(dataset_stac, "finalized")
        ds_finalized = await self._load_zarr_from_cid(finalized_stac_cid)

        if finalized_only:
            return ds_finalized
        
        # # 2. --- Lazily Open Unfinalized Dataset ---
        unfinalized_stac_cid = await self._get_dataset_type_cid(dataset_stac, "non-finalized")
        ds_unfinalized = await self._load_zarr_from_cid(unfinalized_stac_cid)

        # 3. --- Perform Lazy Slicing and Concatenation ---
        # This operation is lazy. It reads only the 'time' coordinate, not the full data.
        finalization_date = np.datetime64(ds_finalized.time.max().values)
        start_time = finalization_date + np.timedelta64(1, 'h')
        ds_unfinalized_sliced = ds_unfinalized.sel(time=slice(start_time, None))
        combined_ds = xr.concat([ds_finalized, ds_unfinalized_sliced], dim="time")
        
        return combined_ds
      