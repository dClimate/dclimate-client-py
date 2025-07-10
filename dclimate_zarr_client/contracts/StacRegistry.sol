// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 * For production, consider using OpenZeppelin's Ownable contract for more features.
 */
contract Ownable {
    address public owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}


/**
 * @title StacRegistry (Developer-Friendly Version)
 * @dev A central registry for STAC data. This version is optimized for off-chain
 * developer convenience, returning human-readable string arrays.
 * WARNING: The getter functions are gas-intensive and should NOT be called from other contracts.
 */
contract StacRegistry is Ownable {

    enum UpdateCategory { APPEND, REPLACE }

    string public stacRootCid;

    // --- Core Data Storage ---
    // collection -> dataset -> type -> cid
    mapping(bytes32 => mapping(bytes32 => mapping(bytes32 => string))) private _cids;

    // --- Arrays for Human-Readable Getters ---
    string[] private _collectionNames;
    mapping(bytes32 => string[]) private _datasetNames; // Key: keccak256(collectionName)
    mapping(bytes32 => string[]) private _typeNames;    // Key: keccak256(collectionName + datasetName)

    // --- Events ---
    event StacRootUpdated(string newCid, address indexed updater);
    event CidUpdated(string collection, string dataset, string dataType, string newCid, UpdateCategory category, address indexed updater);
    event CollectionInitialized(string collectionName, address indexed initializer);
    event DatasetInitialized(string collectionName, string datasetName, address indexed initializer);
    event TypeInitialized(string collectionName, string datasetName, string typeName, address indexed initializer);


    // --- Initialization Functions (Owner Only) ---

    function initCollection(string calldata collectionName) external onlyOwner {
        bytes32 collectionKey = keccak256(abi.encodePacked(collectionName));
        require(_datasetNames[collectionKey].length == 0, "Collection already exists");
        _collectionNames.push(collectionName);
        emit CollectionInitialized(collectionName, msg.sender);
    }

    function initDataset(string calldata collectionName, string calldata datasetName) external onlyOwner {
        bytes32 collectionKey = keccak256(abi.encodePacked(collectionName));
        bytes32 typeMapKey = keccak256(abi.encodePacked(collectionName, datasetName));
        require(_typeNames[typeMapKey].length == 0, "Dataset already exists");
        
        _datasetNames[collectionKey].push(datasetName);
        emit DatasetInitialized(collectionName, datasetName, msg.sender);
    }

    function initType(
        string calldata collectionName,
        string calldata datasetName,
        string calldata typeName,
        string calldata initialCid
    ) external onlyOwner {
        bytes32 collectionKey = keccak256(abi.encodePacked(collectionName));
        bytes32 datasetKey = keccak256(abi.encodePacked(datasetName));
        bytes32 typeKey = keccak256(abi.encodePacked(typeName));
        bytes32 typeMapKey = keccak256(abi.encodePacked(collectionName, datasetName));

        require(bytes(_cids[collectionKey][datasetKey][typeKey]).length == 0, "Type already exists");

        _cids[collectionKey][datasetKey][typeKey] = initialCid;
        _typeNames[typeMapKey].push(typeName);
        emit TypeInitialized(collectionName, datasetName, typeName, msg.sender);
    }

    // --- Update Functions (Owner Only) ---

    function updateStacRoot(string calldata newCid) external onlyOwner {
        stacRootCid = newCid;
        emit StacRootUpdated(newCid, msg.sender);
    }

    function updateCid(string calldata path, string calldata newCid, UpdateCategory category) external onlyOwner {
        (string memory collection, string memory dataset, string memory dataType) = _parsePath(path);
        
        bytes32 collectionKey = keccak256(abi.encodePacked(collection));
        bytes32 datasetKey = keccak256(abi.encodePacked(dataset));
        bytes32 typeKey = keccak256(abi.encodePacked(dataType));

        require(bytes(_cids[collectionKey][datasetKey][typeKey]).length > 0, "Path does not resolve to an existing type");

        _cids[collectionKey][datasetKey][typeKey] = newCid;
        emit CidUpdated(collection, dataset, dataType, newCid, category, msg.sender);
    }
    
    function changeOwner(address newOwner) external onlyOwner {
        transferOwnership(newOwner);
    }

    // --- Read-Only Functions (Human-Readable) ---

    /**
     * @notice Resolves a path string to its stored CID.
     */
    function resolve(string calldata path) external view returns (string memory) {
        (string memory collection, string memory dataset, string memory dataType) = _parsePath(path);
        bytes32 collectionKey = keccak256(abi.encodePacked(collection));
        bytes32 datasetKey = keccak256(abi.encodePacked(dataset));
        bytes32 typeKey = keccak256(abi.encodePacked(dataType));

        return _cids[collectionKey][datasetKey][typeKey];
    }
    
    /**
     * @notice Gets all initialized collection names as readable strings.
     * @dev WARNING: Do not call this from another smart contract.
     */
    function getCollections() external view returns (string[] memory) {
        return _collectionNames;
    }

    /**
     * @notice Gets all dataset names for a given collection as readable strings.
     * @dev WARNING: Do not call this from another smart contract.
     */
    function getDatasets(string calldata collectionName) external view returns (string[] memory) {
        bytes32 collectionKey = keccak256(abi.encodePacked(collectionName));
        return _datasetNames[collectionKey];
    }

    /**
     * @notice Gets all type names for a given dataset as readable strings.
     * @dev WARNING: Do not call this from another smart contract.
     */
    function getTypes(string calldata collectionName, string calldata datasetName) external view returns (string[] memory) {
        bytes32 typeMapKey = keccak256(abi.encodePacked(collectionName, datasetName));
        return _typeNames[typeMapKey];
    }


    // --- Internal Helpers ---

    function _parsePath(string calldata path) internal pure returns (string memory collection, string memory dataset, string memory dataType) {
        bytes memory pathBytes = bytes(path);
        uint256 firstDelimiterIndex = 0;
        for (uint256 i = 0; i < pathBytes.length; i++) {
            if (pathBytes[i] == '-') {
                firstDelimiterIndex = i;
                break;
            }
        }
        require(firstDelimiterIndex > 0, "Invalid path format: no first '-' found.");
        uint256 lastDelimiterIndex = 0;
        for (uint256 i = pathBytes.length - 1; i > 0; i--) {
            if (pathBytes[i] == '-') {
                lastDelimiterIndex = i;
                break;
            }
        }
        require(lastDelimiterIndex > firstDelimiterIndex, "Invalid path format: no second '-' found or delimiters misplaced.");
        
        // Use the _slice helper function to extract the parts
        collection = string(_slice(pathBytes, 0, firstDelimiterIndex));
        dataset = string(_slice(pathBytes, firstDelimiterIndex + 1, lastDelimiterIndex));
        dataType = string(_slice(pathBytes, lastDelimiterIndex + 1, pathBytes.length));

        require(bytes(collection).length > 0, "Collection part cannot be empty");
        require(bytes(dataset).length > 0, "Dataset part cannot be empty");
        require(bytes(dataType).length > 0, "Type part cannot be empty");
    }

    /**
     * @dev Slices a bytes array.
     * @param data The bytes array to slice.
     * @param start The starting index.
     * @param end The ending index.
     * @return A new bytes array containing the slice.
     */
    function _slice(bytes memory data, uint256 start, uint256 end) internal pure returns (bytes memory) {
        require(end >= start, "Slice: end cannot be less than start");
        uint256 len = end - start;
        bytes memory result = new bytes(len);
        for (uint256 i = 0; i < len; i++) {
            result[i] = data[start + i];
        }
        return result;
    }
}