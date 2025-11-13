import asyncio
from typing import Self
from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer import Buffer
from zarr.core.common import JSON
from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes


class EncryptionCodec(BytesBytesCodec):
    """A Zarr v3 codec implementing XChaCha20-Poly1305 encryption."""

    codec_id = "xchacha20poly1305"
    _encryption_key = None

    def __init__(self, header: str = "dclimate-Zarr"):
        """Initialize the codec with an optional header for associated data.

        Args:
            header (str): Associated data for AEAD authentication. Defaults to "dclimate-Zarr".
        """
        self.header = header
        self._encoded_header = header.encode()
        if self._encryption_key is None:
            raise ValueError("Encryption key must be set before using EncryptionCodec.")

    @classmethod
    def set_encryption_key(cls, encryption_key: bytes):
        """Set the encryption key dynamically (once per runtime)."""
        if len(encryption_key) != 32:  # 32 bytes = 64 hex chars
            raise ValueError("Encryption key must be 32 bytes")
        cls._encryption_key = encryption_key

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Create an instance from a Zarr v3 configuration dictionary.

        Args:
            data (dict): Configuration with optional 'header' key.

        Returns:
            Self: An instance of EncryptionCodec.
        """
        configuration = data.get("configuration", {})
        header = configuration.get("header", "dclimate-Zarr")
        return cls(header=header)

    def to_dict(self) -> dict[str, JSON]:
        """Serialize the codec configuration for Zarr v3 metadata.

        Returns:
            dict: A dictionary with 'name' and 'configuration'.
        """
        return {"name": self.codec_id, "configuration": {"header": self.header}}

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec) -> Buffer:
        """Asynchronously decrypt a chunk.

        Args:
            chunk_bytes (Buffer): Encrypted bytes (nonce + tag + ciphertext).
            chunk_spec: Array specification with buffer prototype.

        Returns:
            Buffer: Decrypted plaintext bytes.
        """
        buf = chunk_bytes.to_bytes()

        def decrypt():
            nonce, tag, ciphertext = buf[:24], buf[24:40], buf[40:]
            cipher = ChaCha20_Poly1305.new(key=self._encryption_key, nonce=nonce)
            cipher.update(self._encoded_header)
            return cipher.decrypt_and_verify(ciphertext, tag)

        plaintext = await asyncio.to_thread(decrypt)
        return chunk_spec.prototype.buffer.from_bytes(plaintext)

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec) -> Buffer:
        """Asynchronously encrypt a chunk.

        Args:
            chunk_bytes (Buffer): Plaintext bytes to encrypt.
            chunk_spec: Array specification with buffer prototype.

        Returns:
            Buffer: Encrypted bytes (nonce + tag + ciphertext).
        """
        raw = chunk_bytes.to_bytes()

        def encrypt():
            nonce = get_random_bytes(24)
            cipher = ChaCha20_Poly1305.new(key=self._encryption_key, nonce=nonce)
            cipher.update(self._encoded_header)
            ciphertext, tag = cipher.encrypt_and_digest(raw)
            return nonce + tag + ciphertext

        encoded = await asyncio.to_thread(encrypt)
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec) -> int:
        """Calculate the size of the encoded data.

        Args:
            input_byte_length (int): Size of the input bytes.
            chunk_spec: Array specification (unused in this calculation).

        Returns:
            int: Encoded size (input size + 40 bytes for nonce and tag).
        """
        return input_byte_length + 40  # 24 bytes nonce + 16 bytes tag
