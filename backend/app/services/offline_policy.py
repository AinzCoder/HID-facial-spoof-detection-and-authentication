import base64
import binascii
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from app.schemas import OfflinePolicyClaims, SignedOfflinePolicy


class OfflinePolicyError(ValueError):
    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


def _decode_key(encoded: str | None, expected_length: int) -> bytes:
    if not encoded:
        raise ValueError("Offline signing key is not configured.")
    try:
        raw = base64.b64decode(
            encoded.encode("ascii") + b"=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("Offline signing key encoding is invalid.") from exc
    if len(raw) != expected_length:
        raise ValueError("Offline signing key has an invalid length.")
    return raw


def _private_key(encoded: str | None) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(_decode_key(encoded, 32))


def _public_key(encoded: str | None) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(_decode_key(encoded, 32))


def _canonical_claims(claims: OfflinePolicyClaims) -> bytes:
    return json.dumps(
        claims.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def validate_key_pair(private_key: str, public_key: str) -> None:
    private_raw = _private_key(private_key).public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    if private_raw != _decode_key(public_key, 32):
        raise ValueError("Offline Ed25519 private and public keys do not match.")


def validate_public_key(public_key: str) -> None:
    _public_key(public_key)


def sign_policy(
    claims: OfflinePolicyClaims,
    private_key: str | None,
    public_key: str | None,
) -> SignedOfflinePolicy:
    validate_key_pair(private_key or "", public_key or "")
    signature = _private_key(private_key).sign(_canonical_claims(claims))
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return SignedOfflinePolicy(claims=claims, signature=encoded_signature)


def verify_policy(
    policy: SignedOfflinePolicy,
    public_key: str | None,
) -> OfflinePolicyClaims:
    try:
        key = _public_key(public_key)
        signature = base64.b64decode(
            policy.signature.encode("ascii") + b"=" * (-len(policy.signature) % 4),
            altchars=b"-_",
            validate=True,
        )
        key.verify(signature, _canonical_claims(policy.claims))
    except (ValueError, TypeError, UnicodeEncodeError, binascii.Error, InvalidSignature) as exc:
        raise OfflinePolicyError("offline_policy_invalid") from exc
    return policy.claims
