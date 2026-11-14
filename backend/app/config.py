from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HID_", env_file=".env", extra="ignore")

    app_name: str = "HID Photo Spoof Detection API"
    environment: str = "development"
    database_url: str = "sqlite:///./hid_spoof.db"
    max_image_bytes: int = 10 * 1024 * 1024
    max_video_bytes: int = 50 * 1024 * 1024
    session_ttl_seconds: int = 600
    challenge_ttl_seconds: int = 120
    pad_genuine_threshold: float = 0.8
    pad_spoof_threshold: float = 0.9
    face_detection_threshold: float = 0.8
    face_detection_enabled: bool = False
    face_detection_model_path: str | None = None
    face_landmarker_model_path: str | None = None
    identity_match_threshold: float = 0.8
    identity_mismatch_threshold: float = 0.5
    mask_clear_threshold: float = 0.8
    challenge_pass_threshold: float = 0.8
    biometric_encryption_key: str | None = Field(default=None, repr=False)
    campus_schedule_admin_token: str | None = Field(
        default=None, repr=False, min_length=32, max_length=512
    )
    offline_enabled: bool = False
    offline_allow_enabled: bool = False
    offline_database_url: str | None = None
    offline_policy_ttl_seconds: int = 86400
    offline_signing_private_key: str | None = Field(default=None, repr=False)
    offline_signing_public_key: str | None = Field(default=None, repr=False)
    offline_max_clock_skew_seconds: int = 300
    offline_backend_health_url: str | None = None
    offline_connectivity_timeout_seconds: int = 2
    offline_sync_token: str | None = Field(
        default=None, repr=False, min_length=32, max_length=512
    )
    offline_sync_backend_url: str | None = None
    offline_sync_timeout_seconds: int = 10

    @field_validator(
        "pad_genuine_threshold",
        "pad_spoof_threshold",
        "face_detection_threshold",
        "identity_match_threshold",
        "identity_mismatch_threshold",
        "mask_clear_threshold",
        "challenge_pass_threshold",
    )
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("Thresholds must be between 0 and 1.")
        return value

    @field_validator(
        "max_image_bytes",
        "max_video_bytes",
        "session_ttl_seconds",
        "challenge_ttl_seconds",
        "offline_policy_ttl_seconds",
        "offline_connectivity_timeout_seconds",
        "offline_sync_timeout_seconds",
    )
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Limits and TTL values must be positive.")
        return value

    @field_validator("offline_max_clock_skew_seconds")
    @classmethod
    def validate_clock_skew(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Offline maximum clock skew must not be negative.")
        return value

    @field_validator("offline_backend_health_url", "offline_sync_backend_url")
    @classmethod
    def validate_offline_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if not parsed.hostname or (
            parsed.scheme != "https" and not (
                parsed.scheme == "http" and parsed.hostname in local_hosts
            )
        ):
            raise ValueError("Offline backend URLs must use HTTPS except for loopback development.")
        return value

    @model_validator(mode="after")
    def validate_identity_threshold_order(self):
        if self.identity_mismatch_threshold >= self.identity_match_threshold:
            raise ValueError("Identity mismatch threshold must be lower than identity match threshold.")
        if self.offline_allow_enabled:
            if not self.offline_enabled:
                raise ValueError("Offline ALLOW requires offline mode to be enabled.")
            if not self.offline_signing_public_key:
                raise ValueError("Offline ALLOW requires an Ed25519 public verification key.")
        if self.offline_signing_public_key:
            from app.services.offline_policy import validate_public_key

            validate_public_key(self.offline_signing_public_key)
        if self.offline_signing_private_key and self.offline_signing_public_key:
            from app.services.offline_policy import validate_key_pair

            validate_key_pair(
                self.offline_signing_private_key,
                self.offline_signing_public_key,
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
