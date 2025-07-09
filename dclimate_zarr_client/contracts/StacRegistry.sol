// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title StacRegistry
 * @dev A simple contract to store and manage the root CID for a STAC catalog.
 * The owner of the contract is the only one who can update the CID.
 */
contract StacRegistry {
    address public owner;
    string public stacCid;

    event CidUpdated(string newCid, address indexed updater);

    /**
     * @dev Sets the initial STAC CID and the contract owner.
     * @param initialCid The initial root CID of the STAC catalog.
     */
    constructor(string memory initialCid) {
        owner = msg.sender;
        stacCid = initialCid;
    }

    /**
     * @dev Allows the owner to update the STAC CID.
     * Emits a CidUpdated event.
     * @param newCid The new root CID to store.
     */
    function updateCid(string memory newCid) public {
        require(msg.sender == owner, "Only the owner can update the CID");
        stacCid = newCid;
        emit CidUpdated(newCid, msg.sender);
    }

    /**
     * @dev A public view function to retrieve the current STAC CID.
     * @return The current root CID string.
     */
    function getCid() public view returns (string memory) {
        return stacCid;
    }

    // function to change the owner of the contract
    function changeOwner(address newOwner) public {
        require(msg.sender == owner, "Only the owner can change ownership");
        require(newOwner != address(0), "New owner cannot be the zero address");
        owner = newOwner;
    }
}