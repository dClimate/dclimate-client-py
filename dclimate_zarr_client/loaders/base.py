from abc import ABC, abstractmethod
import xarray as xr
from web3 import Web3
from py_hamt import KuboCAS, ShardedZarrStore
from multiformats import CID
import json

STAC_REGISTRY_ABI = """
[
    {
        "inputs": [{"internalType": "string", "name": "initialCid", "type": "string"}],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": false, "internalType": "string", "name": "newCid", "type": "string"},
            {"indexed": true, "internalType": "address", "name": "updater", "type": "address"}
        ],
        "name": "CidUpdated",
        "type": "event"
    },
    {
        "inputs": [],
        "name": "getCid",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "newOwner", "type": "address"}],
        "name": "changeOwner",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]
"""

class BaseLoader(ABC):

    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url

    async def _load_stac_json_from_cid(self, cid: str) -> dict:
        """Loads a STAC JSON from a given CID."""
        async with KuboCAS(gateway_base_url=self.gateway_url) as kubo_cas:
            try:
                item_bytes = await kubo_cas.load(CID.decode(cid))
                item = json.loads(item_bytes)
                return item
            except Exception as e:
                print(f"Error loading STAC JSON from CID {cid}: {e}")
                raise

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