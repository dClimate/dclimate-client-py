from abc import ABC, abstractmethod
import xarray as xr
from web3 import Web3
from py_hamt import KuboCAS, ShardedZarrStore
from multiformats import CID
import json

class BaseLoader(ABC):

    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url

    async def _load_zarr_from_cid(self, cid: str) -> xr.Dataset:
        """Loads a sharded Zarr store from a given CID."""
        async with KuboCAS(gateway_base_url=self.gateway_url) as kubo_cas:
            store = await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=True,
                root_cid=cid
            )
            return xr.open_zarr(store=store, chunks="auto")

    @abstractmethod
    async def load(self, dataset: str, options: dict) -> xr.Dataset:
        """Abstract method to be implemented by each strategy."""
        pass