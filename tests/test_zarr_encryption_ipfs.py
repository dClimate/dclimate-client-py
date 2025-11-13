import os
import shutil

from multiformats import CID
import numpy as np
import pandas as pd
import xarray as xr
import pytest
import zarr
import json

# from zarr.codecs import BytesCodec
from numcodecs.zarr3 import Blosc

from Crypto.Random import get_random_bytes

from py_hamt import HAMT, KuboCAS, ZarrHAMTStore

from dclimate_client_py.encryption_codec import EncryptionCodec


# Fixture to provide the encryption key used for initial dataset creation
# This assumes random_zarr_dataset uses this key
@pytest.fixture(scope="session")
def original_encryption_key() -> bytes:
    # Use a fixed key for the session or generate once if needed,
    # but ensure it's the *same* key used by random_zarr_dataset fixture
    # For simplicity, let's generate it here. The fixture needs to use this.
    # In a real scenario, this might come from a shared config or setup.
    return get_random_bytes(32)


@pytest.fixture
def random_zarr_dataset(
    tmp_path, original_encryption_key: bytes
) -> tuple[str, xr.Dataset]:
    """Creates a random xarray Dataset and saves it to a temporary zarr store.
    Returns:
        tuple: (dataset_path, expected_data)
            - dataset_path: Path to the zarr store
            - expected_data: The original xarray Dataset for comparison
    """
    # Create temporary directory for zarr store
    zarr_path = str(tmp_path / "random.zarr")

    # Coordinates of the random data
    times = pd.date_range("2024-01-01", periods=100)
    lats = np.linspace(-90, 90, 18)
    lons = np.linspace(-180, 180, 36)

    # Create random variables with different shapes
    temp = np.random.randn(len(times), len(lats), len(lons))
    precip = np.random.gamma(2, 0.5, size=(len(times), len(lats), len(lons)))

    # Create the dataset
    ds = xr.Dataset(
        {
            "temp": (
                ["time", "lat", "lon"],
                temp,
                {"units": "celsius", "long_name": "Surface Temperature"},
            ),
            "precip": (
                ["time", "lat", "lon"],
                precip,
                {"units": "mm/day", "long_name": "Daily Precipitation"},
            ),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={"description": "Test dataset with random weather data"},
    )

    # Set the encryption key for the class
    EncryptionCodec.set_encryption_key(original_encryption_key)
    # Register the codec
    zarr.registry.register_codec("xchacha20poly1305", EncryptionCodec)

    # Apply the encryption codec to the dataset with a selected header
    encoding = {
        "temp": {
            "compressors": [
                Blosc(cname="lz4", clevel=5),
                EncryptionCodec(header="dclimate-Zarr"),
            ],
        }
    }

    ds.to_zarr(
        zarr_path, mode="w", encoding=encoding, consolidated=False
    )  # Use consolidated=False for IPFS HAMT stores typically

    # Navigate to validate the codec does not have the key
    temp_path = os.path.join(zarr_path, "temp")
    # Open the config.json file to check the codec
    with open(os.path.join(temp_path, "zarr.json"), "r") as f:
        json_string = f.read()
        json_zarr = json.loads(json_string)
        # Check if the codec is in the config.json
        assert json_zarr["codecs"][2]["name"] == "xchacha20poly1305"
        # Check if the header is in the config.json
        assert json_zarr["codecs"][2]["configuration"]["header"] == "dclimate-Zarr"
        # Check no other keys are in the codec
        assert len(json_zarr["codecs"][2]["configuration"]) == 1
        assert len(json_zarr["codecs"][2]) == 2

    yield zarr_path, ds

    # Cleanup
    shutil.rmtree(tmp_path)


def test_bad_encryption_keys():
    # Assert failure Encryption key must be set before using EncryptionCodec
    with pytest.raises(ValueError):
        EncryptionCodec(header="dClimate-Zarr")
    # Assert failure Encryption key must be 32 bytes (64 hex characters)
    with pytest.raises(ValueError):
        EncryptionCodec.set_encryption_key(get_random_bytes(31))


def test_compute_encoded_size():
    # Example function to compute the encoded size of a dataset
    EncryptionCodec.set_encryption_key(get_random_bytes(32))
    assert EncryptionCodec().compute_encoded_size(100, "RANDOM VALUE") == 140


@pytest.mark.asyncio
async def test_upload_then_read(
    random_zarr_dataset: tuple[str, xr.Dataset], original_encryption_key: bytes
):
    zarr_path, expected_ds = random_zarr_dataset

    test_ds = xr.open_zarr(zarr_path, consolidated=False)  # Match write setting

    # Check length of compressor is just 1 for default ZstdCodec
    assert len(test_ds["precip"].encoding["compressors"]) == 1
    # Ensure EncryptionCodec is there
    assert isinstance(test_ds["temp"].encoding["compressors"][1], EncryptionCodec)

    # Prepare Writing to IPFS
    async with KuboCAS() as kubo_cas:
        hamt1 = await HAMT.build(
            cas=kubo_cas,
            values_are_bytes=True,
        )
        ipfszarr3 = ZarrHAMTStore(hamt1)

        test_ds.to_zarr(
            store=ipfszarr3,
            mode="w",
            consolidated=False,  # Match read setting
        )
        await hamt1.make_read_only()

        hamt1_root: CID = hamt1.root_node_id  # type: ignore

        # Read the dataset from IPFS
        hamt1_read = await HAMT.build(
            cas=kubo_cas,
            root_node_id=hamt1_root,
            read_only=True,
            values_are_bytes=True,
        )
        hamt1_read_z3 = ZarrHAMTStore(hamt1_read, read_only=True)

        loaded_ds1 = xr.open_zarr(
            store=hamt1_read_z3, consolidated=False
        )  # Match write setting

        # Assert the values are the same and can be read in the context
        # Check if the values of 'temp' and 'precip' are equal in all datasets
        assert np.array_equal(loaded_ds1["temp"].values, expected_ds["temp"].values), (
            "Temp values in loaded_ds1 and expected_ds are not identical!"
        )
        assert np.array_equal(loaded_ds1["precip"].values, expected_ds["precip"].values), (
            "Precip values in loaded_ds1 and expected_ds are not identical!"
        )

        # Attempt to read with the WRONG key
        # Create new encryption filter but with a different encryption key
        wrong_encryption_key = get_random_bytes(32)
        EncryptionCodec.set_encryption_key(wrong_encryption_key)
        zarr.registry.register_codec("xchacha20poly1305", EncryptionCodec)

        assert wrong_encryption_key != original_encryption_key  # Ensure it's different

        loaded_failure = xr.open_zarr(store=hamt1_read_z3, consolidated=False)

        # Accessing data should raise an exception since we don't have the correct encryption key
        with pytest.raises(Exception):
            _ = loaded_failure["temp"].values

        # Check that you can still read the precip since it was not encrypted
        assert loaded_failure["precip"].values[0][0][0]

        assert "temp" in loaded_ds1
        assert "precip" in loaded_ds1
        assert loaded_ds1.temp.attrs["units"] == "celsius"

        assert loaded_ds1.temp.shape == expected_ds.temp.shape
