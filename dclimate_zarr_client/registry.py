import os
import json
from pathlib import Path
from eth_account import Account
from ape import accounts, Project, networks
from ape.exceptions import AccountsError
from web3 import Web3
from typing import List

# --- Configuration ---

# Path to the directory containing the compiled contract artifacts
CONTRACTS_PROJECT_PATH = Path(__file__).parent / "contracts"

# Load environment variables
STAC_REGISTRY_CONTRACT_ADDRESS = os.environ.get("STAC_REGISTRY_CONTRACT_ADDRESS", "0xb54A652489b864638d02C508A2d5Bc14FbFA8df8")
RPC_URL = os.environ.get("RPC_URL", "https://sepolia.base.org")
MNEMONIC_PATH = os.environ.get("MNEMONIC_PATH", "mnemonic.txt")


# --- Helper Functions ---

def _get_web3_instance() -> Web3:
    """Initializes and returns a Web3 instance."""
    return Web3(Web3.HTTPProvider(RPC_URL))

def _get_contract_abi() -> list:
    """Loads the contract ABI from the project's build artifacts."""
    abi_path = CONTRACTS_PROJECT_PATH / "compiled" / "contracts" / "StacRegistry.json"
    if not abi_path.is_file():
        raise FileNotFoundError(f"ABI file not found at: {abi_path}. Please compile your contract.")
    with open(abi_path) as f:
        contract_json = json.load(f)
        return contract_json['contractTypes']['StacRegistry']['abi']

def _load_owner_account():
    """Loads the owner account from a mnemonic file for web3.py usage."""
    mnemonic_file = Path(MNEMONIC_PATH)
    if not mnemonic_file.is_file():
        raise FileNotFoundError(f"Mnemonic file not found at: {MNEMONIC_PATH}")
    
    Account.enable_unaudited_hdwallet_features()
    mnemonic = mnemonic_file.read_text().strip()
    return Account.from_mnemonic(mnemonic)

def _load_deployer_account_ape():
    """Loads a deployer account from a mnemonic for Ape usage."""
    mnemonic_file = Path(MNEMONIC_PATH)
    alias = f"etl_deployer_{mnemonic_file.stem}"
    try:
        return accounts.load(alias)
    except AccountsError:
        print(f"Ape account '{alias}' not found. Importing from mnemonic...")
        return accounts.import_account_from_mnemonic(
            account_alias=alias,
            mnemonic=mnemonic_file.read_text().strip(),
            passphrase=os.environ.get("APE_ACCOUNT_PASSPHRASE", "")
        )

def _execute_transaction(function_call):
    """Builds, signs, and sends a transaction for the given function call."""
    w3 = _get_web3_instance()
    owner_account = _load_owner_account()
    
    tx = function_call.build_transaction({
        'chainId': w3.eth.chain_id,
        'from': owner_account.address,
        'nonce': w3.eth.get_transaction_count(owner_account.address),
        'gas': 5000000,  # Increased default gas limit for more complex transactions
        'gasPrice': w3.eth.gas_price,
    })
    
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=owner_account.key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    print(f"✓ Transaction successful! Hash: {receipt.transactionHash.hex()}")
    return receipt.transactionHash.hex()


# --- Deployment Function ---

def deploy_registry(network: str = "base:sepolia") -> str:
    """Deploys the StacRegistry contract using Ape."""
    project = Project(path=CONTRACTS_PROJECT_PATH)
    deployer = _load_deployer_account_ape()

    with networks.parse_network_choice(network) as provider:
        print(f"Using network: {provider.name}")
        print(f"Deployer address: {deployer.address}")

        StacRegistry = project.StacRegistry

        print("Deploying StacRegistry...")
        contract_instance = deployer.deploy(StacRegistry)

        print(f"✓ Deployment successful! Contract Address: {contract_instance.address}")
        return contract_instance.address

# --- Read-Only (Getter) Functions ---

def get_stac_root_cid(contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> str:
    """Retrieves the top-level STAC Root CID."""
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    cid = contract.functions.stacRootCid().call()
    print(f"STAC Root CID: {cid}")
    return cid

def resolve_path(path: str, contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> str:
    """Resolves a 'collection-dataset-type' path to its CID."""
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    cid = contract.functions.resolve(path).call()
    print(f"CID for path '{path}': {cid}")
    return cid

def get_collections(contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> List[str]:
    """Retrieves the list of all collection names."""
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    collections = contract.functions.getCollections().call()
    print(f"Available Collections: {collections}")
    return collections

def get_datasets(collection_name: str, contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> List[str]:
    """Retrieves the list of all dataset names for a given collection."""
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    datasets = contract.functions.getDatasets(collection_name).call()
    print(f"Available Datasets in '{collection_name}': {datasets}")
    return datasets

def get_types(collection_name: str, dataset_name: str, contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> List[str]:
    """Retrieves the list of all type names for a given dataset."""
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    types = contract.functions.getTypes(collection_name, dataset_name).call()
    print(f"Available Types in '{collection_name}/{dataset_name}': {types}")
    return types


# --- State-Changing (Update/Init) Functions ---

def update_stac_root_cid(new_cid: str, contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> str:
    """Updates the top-level STAC Root CID."""
    print(f"Updating STAC Root CID to: {new_cid}")
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    function_call = contract.functions.updateStacRoot(new_cid)
    return _execute_transaction(function_call)

def update_path_cid(path: str, new_cid: str, category: int = 1, contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> str:
    """
    Updates the CID for a specific 'collection-dataset-type' path.
    Category: 0 for APPEND, 1 for REPLACE.
    """
    print(f"Updating CID for path '{path}' to: {new_cid}")
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    function_call = contract.functions.updateCid(path, new_cid, category)
    return _execute_transaction(function_call)

def init_collection(collection_name: str, contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> str:
    """Initializes a new collection."""
    print(f"Initializing collection: {collection_name}")
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    function_call = contract.functions.initCollection(collection_name)
    return _execute_transaction(function_call)

def init_dataset(collection_name: str, dataset_name: str, contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> str:
    """Initializes a new dataset within a collection."""
    print(f"Initializing dataset '{dataset_name}' in collection '{collection_name}'")
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    function_call = contract.functions.initDataset(collection_name, dataset_name)
    return _execute_transaction(function_call)

def init_type(collection_name: str, dataset_name: str, type_name: str, initial_cid: str, contract_address: str = STAC_REGISTRY_CONTRACT_ADDRESS) -> str:
    """Initializes a new type for a dataset with an initial CID."""
    print(f"Initializing type '{type_name}' for '{collection_name}/{dataset_name}' with CID: {initial_cid}")
    w3 = _get_web3_instance()
    contract = w3.eth.contract(address=contract_address, abi=_get_contract_abi())
    function_call = contract.functions.initType(collection_name, dataset_name, type_name, initial_cid)
    return _execute_transaction(function_call)