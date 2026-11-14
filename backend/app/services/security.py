import base64
import binascii
import json
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
from fastapi import HTTPException


def _decode_key(encoded_key: str | None) -> bytes:
    if not encoded_key:
        raise HTTPException(status_code=503, detail="Biometric encryption is not configured.")
    try:
        key = base64.urlsafe_b64decode(encoded_key.encode("ascii"))
    except (ValueError, UnicodeEncodeError, binascii.Error) as exc:
        raise HTTPException(status_code=503, detail="Biometric encryption key is invalid.") from exc
    if len(key) != 32:
        raise HTTPException(status_code=503, detail="Biometric encryption key must decode to 32 bytes.")
    return key


def encrypt_embedding(embedding: tuple[float, ...], encoded_key: str | None) -> tuple[bytes, bytes]:
    key = _decode_key(encoded_key)
    nonce = secrets.token_bytes(12)
    plaintext = json.dumps(embedding, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return AESGCM(key).encrypt(nonce, plaintext, None), nonce


def decrypt_embedding(ciphertext: bytes, nonce: bytes, encoded_key: str | None) -> tuple[float, ...]:
    key = _decode_key(encoded_key)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
        values = json.loads(plaintext)
    except (InvalidTag, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Biometric enrollment could not be decrypted.") from exc
    if not isinstance(values, list) or not values:
        raise HTTPException(status_code=503, detail="Stored biometric embedding is invalid.")
    if not all(isinstance(value, (float, int)) for value in values):
        raise HTTPException(status_code=503, detail="Stored biometric embedding is invalid.")
    return tuple(float(value) for value in values)
