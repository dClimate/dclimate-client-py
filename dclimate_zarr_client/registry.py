import os
from pathlib import Path
from eth_account import Account
from ape import accounts, Project, networks
from ape.exceptions import AccountsError
from ape_accounts import import_account_from_mnemonic
from web3 import Web3

# The path to the embedded Ape project
CONTRACTS_PROJECT_PATH = Path(__file__).parent / "contracts"
STAC_REGISTRY_CONTRACT_ADDRESS = os.environ.get("STAC_REGISTRY_CONTRACT_ADDRESS", "0xabe7441E21bDb6cCf4E517E0072c5E962F5f0B2d")
RPC_URL = os.environ.get("RPC_URL", "https://sepolia.base.org")
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
    },
    {
        "inputs": [{"internalType": "string", "name": "newCid", "type": "string"}],
        "name": "updateCid",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]
"""


def _get_web3_instance() -> Web3:
    """Initializes and returns a Web3 instance."""
    return Web3(Web3.HTTPProvider(RPC_URL))

def _load_account_from_mnemonic_ape(mnemonic_path: str):
    """Loads a deployer/owner account from a mnemonic file."""
    mnemonic_file = Path(mnemonic_path)
    if not mnemonic_file.is_file():
        raise FileNotFoundError(f"Mnemonic file not found at: {mnemonic_path}")
    
    # Use a unique alias to avoid collisions if multiple mnemonics are used
    alias = f"etl_deployer_{mnemonic_file.stem}"
    
    try:
        # Try to load if it already exists in Ape's accounts
        return accounts.load(alias)
    except KeyError:
        print(f"Account '{alias}' not found. Importing from mnemonic...")
        return import_account_from_mnemonic(
            alias=alias, # Note the parameter name change to 'account_alias'
            mnemonic=mnemonic_file.read_text().strip(),
            passphrase=os.environ.get("APE_ACCOUNT_PASSPHRASE", "test") # Provide a default
        )

def _load_account_from_mnemonic(mnemonic_path: str):
    """Loads an account from a mnemonic file and returns the account object."""
    mnemonic_file = Path(mnemonic_path)
    if not mnemonic_file.is_file():
        raise FileNotFoundError(f"Mnemonic file not found at: {mnemonic_path}")
    
    Account.enable_unaudited_hdwallet_features()
    mnemonic = mnemonic_file.read_text().strip()
    return Account.from_mnemonic(mnemonic)

def deploy_registry(initial_cid: str, mnemonic_path: str, network: str = "base:sepolia") -> str:
    """
    Deploys the StacRegistry contract programmatically.

    Args:
        initial_cid (str): The initial root CID of the STAC catalog.
        mnemonic_path (str): The path to the mnemonic.txt file for the deployer account.
        network (str): The network to deploy on (e.g., 'base:sepolia').

    Returns:
        str: The address of the newly deployed contract.
    """
    project = Project(path=CONTRACTS_PROJECT_PATH)
    deployer = _load_account_from_mnemonic_ape(mnemonic_path)

    with networks.parse_network_choice(network) as provider:
        print(f"Using network: {provider.name}")
        print(f"Deployer address: {deployer.address}")
        
        StacRegistry = project.StacRegistry
        
        print(f"Deploying StacRegistry with initial CID: {initial_cid}")
        contract_instance = deployer.deploy(StacRegistry, initial_cid)
        
        print(f"✓ Deployment successful!")
        return contract_instance.address

def _get_web3_instance() -> Web3:
    """Initializes and returns a Web3 instance."""
    return Web3(Web3.HTTPProvider(RPC_URL))

def _load_account_from_mnemonic(mnemonic_path: str):
    """Loads an account from a mnemonic file and returns the account object."""
    mnemonic_file = Path(mnemonic_path)
    if not mnemonic_file.is_file():
        raise FileNotFoundError(f"Mnemonic file not found at: {mnemonic_path}")
    
    Account.enable_unaudited_hdwallet_features()
    mnemonic = mnemonic_file.read_text().strip()
    return Account.from_mnemonic(mnemonic)

def update_registry(contract_address: str, new_cid: str, mnemonic_path: str) -> str:
    """
    Updates the STAC CID in an existing StacRegistry contract using web3.py.
    """
    w3 = _get_web3_instance()
    owner_account = _load_account_from_mnemonic(mnemonic_path)
    
    contract = w3.eth.contract(address=contract_address, abi=STAC_REGISTRY_ABI)
    
    print(f"Using network: {RPC_URL}")
    print(f"Owner address: {owner_account.address}")
    print(f"Updating STAC CID on contract {contract_address} to: {new_cid}")

    # 1. Build the transaction
    nonce = w3.eth.get_transaction_count(owner_account.address)
    tx = contract.functions.updateCid(new_cid).build_transaction({
        'chainId': w3.eth.chain_id,
        'from': owner_account.address,
        'nonce': nonce,
        'gas': 200000,  # You might need to adjust gas estimation
        'gasPrice': w3.eth.gas_price,
    })

    # 2. Sign the transaction
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=owner_account.key)

    # 3. Send the raw transaction
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    
    # 4. Wait for the transaction receipt (optional, but good practice)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    print("✓ Update successful!")
    print(f"Transaction hash: {receipt.transactionHash.hex()}")
    
    return receipt.transactionHash.hex()

def get_cid_from_registry() -> str:
    """
    Retrieves the current STAC CID from the StacRegistry contract.
    (This function already uses web3.py correctly)
    """
    w3 = _get_web3_instance()
    contract = w3.eth.contract(
        address=STAC_REGISTRY_CONTRACT_ADDRESS,
        abi=STAC_REGISTRY_ABI
    )
    stac_cid = contract.functions.getCid().call()
    print(f"Current STAC CID from registry: {stac_cid}")
    return stac_cid
