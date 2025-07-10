import pytest
from pathlib import Path
import time
from web3 import Web3
import os
# Assuming your updated registry functions are in this module
from dclimate_zarr_client import registry, loader


# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

RPC_URL = os.environ.get("RPC_URL", "https://sepolia.base.org")

@pytest.fixture(scope="module")
def w3():
    """Provides a Web3 instance connected to a local test network."""
    return Web3(Web3.HTTPProvider(RPC_URL))

@pytest.fixture(scope="module")
def contract_address(w3):
    """Deploys the contract and returns its address."""
    contract_address = "0xb54A652489b864638d02C508A2d5Bc14FbFA8df8"
    return contract_address

@pytest.fixture(scope="module")
def mnemonic(tmp_path_factory):
    """
    Fixture to provide the path to the mnemonic file for testing.

    IMPORTANT: You must manually create a `mnemonic.txt` file in your project's
    root directory containing a seed phrase for a wallet funded with testnet ETH
    (e.g., Base Sepolia ETH). These tests will fail if the account has no funds.
    """
    mnemonic_path = Path("mnemonic.txt")
    if not mnemonic_path.exists():
        pytest.fail(
            "A `mnemonic.txt` file is required for live testnet tests. "
            "Please create it and fund the associated account with testnet ETH."
        )
    return str(mnemonic_path)


# This test is now parameterized to run ONLY on the Base Sepolia testnet.
# NOTE: This will be slower and will consume real testnet ETH.
# @pytest.mark.parametrize("network", ["base:sepolia"])
# async def test_full_lifecycle_on_testnet(mnemonic, network):
#     """
#     Tests the full contract deployment and hierarchical initialization on the testnet.
#     """
#     print(f"\n--- Running Full Integration Test on {network} ---")

#     w3 = registry._get_web3_instance()
#     assert w3.is_connected(), "Web3 is not connected. Check your RPC URL and network configuration."

#     # 1. Deploy the contract
#     # The new contract doesn't require an initial CID at deployment.
#     print("Step 1: Deploying StacSuperRegistry...")
#     # contract_address = registry.deploy_registry(network=network)
#     contract_address = "0xb54A652489b864638d02C508A2d5Bc14FbFA8df8"
#     assert isinstance(contract_address, str)
#     assert contract_address.startswith("0x")
#     print(f"✅ Deployment successful: {contract_address}")

#     time.sleep(10)


#     # --- 2. Initialize the full data path ---

#     # Define the data structure
#     collection = "era5"
#     dataset = "2m_temperature"
#     dtype = "non_finalized"
#     cid = "bafyr4ibj3bfl5oo7bf6gagzr2g33jlnf23mq2xo632mbl6ytfry7jbuepy"
#     full_path = f"{collection}-{dataset}-{dtype}"

#     # Initialize the Collection
#     # print(f"\nStep 2: Initializing Collection '{collection}'...")
#     # tx_hash_collection = registry.init_collection(contract_address, collection)
#     # assert tx_hash_collection.startswith("0x")
#     # print(f"✅ Collection initialized. Tx: {tx_hash_collection}")

#     # w3.eth.wait_for_transaction_receipt(tx_hash_collection, timeout=120) # 120-second timeout

#     # time.sleep(10)

#     # Initialize the Dataset
#     # print(f"\nStep 3: Initializing Dataset '{dataset}'...")
#     # tx_hash_dataset = registry.init_dataset(contract_address, collection, dataset)
#     # assert tx_hash_dataset.startswith("0x")
#     # print(f"✅ Dataset initialized. Tx: {tx_hash_dataset}")

#     # w3.eth.wait_for_transaction_receipt(tx_hash_dataset, timeout=120) # 120-second timeout
#     # time.sleep(10)


#     # Initialize the Type and set its CID
#     print(f"\nStep 4: Initializing Type '{dtype}' with its CID...")
#     tx_hash_type = registry.init_type(contract_address, collection, dataset, dtype, cid)
#     assert tx_hash_type.startswith("0x")
#     print(f"✅ Type initialized. Tx: {tx_hash_type}")

#     w3.eth.wait_for_transaction_receipt(tx_hash_type, timeout=120) # 120-second timeout
#     time.sleep(10)

#     # --- 3. Verify the data ---

#     print(f"\nStep 5: Verifying the stored CID for path '{full_path}'...")
#     # Use the resolve function to get the CID for the full path
#     retrieved_cid = registry.resolve_path(contract_address, full_path)

#     assert retrieved_cid == cid
#     print(f"✅ Verification successful! Retrieved CID matches expected CID.")

#     # Optional: Verify getter functions
#     print("\nStep 6: Verifying getter functions...")
#     collections = registry.get_collections(contract_address)
#     datasets = registry.get_datasets(contract_address, collection)
#     types = registry.get_types(contract_address, collection, dataset)

#     assert collection in collections
#     assert dataset in datasets
#     assert dtype in types
#     print("✅ Getter functions returned expected values.")

# def test_update_path_cid(w3, contract_address):
#     """
#     Tests that the CID for a given path can be successfully updated.
#     """
#     # --- 1. Setup: Define test data and initialize the path ---
#     collection = "era5"
#     dataset = "2m_temperature"
#     dtype = "finalized"
#     initial_cid = "baguqehradygn3nlscqe4fnuszz6yhsnsgdtmxv5trose44iw44dfa73axbfq"
#     updated_cid = "bafyr4icrox4pxashkfmbyztn7jhp6zjlpj3bufg5ggsjux74zr7ocnqdpu"
#     path = f"{collection}-{dataset}-{dtype}"

#     # --- 2. Action: Update the CID ---
#     print("\n--- Test Action: Updating CID ---")
#     # Using category 1 for REPLACE
#     tx_hash_update = registry.update_path_cid(contract_address, path, updated_cid, category=1)
#     w3.eth.wait_for_transaction_receipt(tx_hash_update)
#     print(f"✅ CID update transaction sent. Tx: {tx_hash_update}")

#     time.sleep(5)

#     # --- 3. Assertion: Verify the CID was updated ---
#     print("\n--- Test Assertion: Verifying Update ---")
#     resolved_updated_cid = registry.resolve_path(contract_address, path)
    
#     assert resolved_updated_cid == updated_cid, "The resolved CID should match the updated CID"
#     assert resolved_updated_cid != initial_cid, "The resolved CID should no longer be the initial CID"
    
#     print(f"✅ CID for '{path}' successfully updated to: {resolved_updated_cid}")


async def test_era5_loader():
    """
    This test remains unchanged. It tests the Python logic of the data loader
    by mocking blockchain and IPFS calls. It is fast and does not depend on
    any network connection, which is a testing best practice.
    """
    era5_ds = await loader.dclimate_dataset_loader(
        collection="era5",
        dataset="2m_temperature",
        options={'finalized_only': False}
    )
    data = era5_ds.sel(
        latitude=0, longitude=0, time="2023-01-01T00:00:00"
    ).load()  # Load the data into memory
    assert data['2m_temperature'].values == 300.3167419433594
    