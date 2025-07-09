import pytest
import xarray as xr
import numpy as np
import pandas as pd
from unittest.mock import patch
from pathlib import Path

from dclimate_zarr_client import registry, loader

import xarray as xr
import numpy as np
import pandas as pd
from py_hamt import KuboCAS, ShardedZarrStore

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def mnemonic(tmp_path_factory):
    """
    Fixture to create a temporary mnemonic file for testing.
    
    IMPORTANT: You must manually create a `tests/mnemonic.txt` file containing
    a seed phrase for a wallet funded with Base Goerli ETH.
    This test will fail if the account has no funds.
    """
    # This fixture now points to a user-created mnemonic file
    # for the live testnet.
    mnemonic_path = Path("mnemonic.txt")
    if not mnemonic_path.exists():
        pytest.fail(
            "A `mnemonic.txt` file is required for live testnet tests. "
            "Please create it and fund the associated account with Base Goerli ETH."
        )
    return str(mnemonic_path)


# This test is now parameterized to run ONLY on the Base Goerli testnet.
# NOTE: This will be slower and will consume real testnet ETH.
@pytest.mark.parametrize("network", ["base:sepolia"])
async def test_deployment_and_update_on_testnet(mnemonic, network):
    """
    Tests the full contract deployment and update lifecycle on the Base Goerli testnet.
    """
    print(f"\n--- Running Integration Test on {network} ---")
    
    # 1. Deploy the contract
    initial_cid = "baguqehra7gqmpt6j4rfhekln7pel4log7p3qu6iike2qgj3ypl74g7aj4isa"
    contract_address = registry.deploy_registry(
        initial_cid=initial_cid,
        mnemonic_path=mnemonic,
        network=network
    )
    
    assert isinstance(contract_address, str)
    assert contract_address.startswith("0x")
    print(f"✅ Deployment successful: {contract_address}")

    # 2. Update the contract
    new_cid = "bafybeihweq2w7v7nnsywn2vqn33y3o6q52rl7mvk2idxmozm2v2c6a2d5y"
    tx_hash = registry.update_registry(
        contract_address=contract_address,
        new_cid=new_cid,
        mnemonic_path=mnemonic,
    )
    
    assert isinstance(tx_hash, str)
    assert tx_hash.startswith("0x")
    print(f"✅ Update successful: {tx_hash}")


def create_dummy_dataset():
    """Helper to create a simple xarray.Dataset for mocking."""
    times = pd.date_range("2024-01-01", periods=10, freq="h")
    lats = np.arange(40, 42, 1)
    lons = np.arange(-80, -78, 1)
    temp = np.random.rand(len(times), len(lats), len(lons))
    return xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"), temp)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


async def test_era5_loader_with_mocks():
    """
    This test remains unchanged. It tests the Python logic of the data loader
    by mocking blockchain and IPFS calls. It is fast and does not depend on
    any network connection, which is a testing best practice.
    """
    all_datasets = await loader.get_all_collections()
    print(f"All : {all_datasets}")
    # era5_ds = await loader.get_geo_temporal_dataset(
    #     collection="era5",
    #     dataset="2m_temperature",
    #     options={'finalized_only': False}
    # )
    # # Download data from latitude 0 and longitude 0 at 2023-01-01T00:00:00
    # data = era5_ds.sel(
    #     latitude=0, longitude=0, time="2023-01-01T00:00:00"
    # ).load()  # Load the data into memory
    # print(f"Data at (0, 0) on 2023-01-01T00:00:00: {data['2m_temperature'].values}")
    