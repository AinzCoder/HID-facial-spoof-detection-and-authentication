from datetime import datetime, time, timezone
from enum import StrEnum
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DecisionOutcome(StrEnum):
    ALLOW = "ALLOW"
    STEP_UP = "STEP_UP"
    DENY = "DENY"


class ChallengeType(StrEnum):
    TURN_HEAD_LEFT = "turn_head_left"
    TURN_HEAD_RIGHT = "turn_head_right"
    LOOK_UP = "look_up"
    BLINK = "blink"


class CampusScheduleDay(StrEnum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class SessionCreateRequest(BaseModel):
    subject_id: str | None = Field(default=None, min_length=1, max_length=128)


class SessionResponse(BaseModel):
    session_id: str
    status: str
    challenge: ChallengeType
    challenge_expires_at: datetime
    expires_at: datetime


class EnrollmentResponse(BaseModel):
    subject_id: str
    status: str
    model_status: str
    model_name: str
    model_version: str
    processing_time_ms: float


class DecisionResponse(BaseModel):
    session_id: str
    outcome: DecisionOutcome
    reason_code: str
    reason: str
    model_status: dict[str, str]
    processing_ms: float
    challenge: ChallengeType | None = None
    challenge_status: str | None = None


class VerificationStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    status: str
    latest_outcome: DecisionOutcome | None
    latest_reason_code: str | None
    challenge: ChallengeType
    challenge_status: str
    challenge_expires_at: datetime
    expires_at: datetime
    results: list[DecisionResponse]


class CampusAccessScheduleRequest(BaseModel):
    permitted_days: list[CampusScheduleDay] = Field(min_length=1, max_length=7)
    access_start_time: time
    access_end_time: time
    timezone: str = Field(min_length=1, max_length=64)
    enabled: bool = True

    @field_validator("permitted_days")
    @classmethod
    def validate_unique_days(
        cls, value: list[CampusScheduleDay]
    ) -> list[CampusScheduleDay]:
        if len(set(value)) != len(value):
            raise ValueError("Permitted days must not contain duplicates.")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("Timezone must be a valid IANA timezone.") from exc
        return value

    @model_validator(mode="after")
    def validate_schedule_window(self):
        if (
            self.access_start_time.tzinfo is not None
            or self.access_end_time.tzinfo is not None
        ):
            raise ValueError("Access times must be local wall-clock times without UTC offsets.")
        if self.access_start_time >= self.access_end_time:
            raise ValueError("Access end time must be later than start time.")
        return self


class CampusAccessScheduleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    subject_id: str
    permitted_days: list[CampusScheduleDay]
    access_start_time: time
    access_end_time: time
    timezone: str
    enabled: bool


class CampusScheduleEnabledRequest(BaseModel):
    enabled: bool


class OfflinePolicyPublishRequest(BaseModel):
    allow_offline: bool


class OfflinePolicyClaims(BaseModel):
    policy_id: str = Field(min_length=1, max_length=36)
    subject_id: str = Field(min_length=1, max_length=128)
    permitted_days: list[CampusScheduleDay] = Field(min_length=1, max_length=7)
    access_start_time: time
    access_end_time: time
    timezone: str = Field(min_length=1, max_length=64)
    enabled: bool
    allow_offline: bool
    issued_at: datetime
    expires_at: datetime

    @field_validator("permitted_days")
    @classmethod
    def validate_unique_days(
        cls, value: list[CampusScheduleDay]
    ) -> list[CampusScheduleDay]:
        if len(set(value)) != len(value):
            raise ValueError("Permitted days must not contain duplicates.")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("Timezone must be a valid IANA timezone.") from exc
        return value

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_aware_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Policy timestamps must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_policy(self):
        if (
            self.access_start_time.tzinfo is not None
            or self.access_end_time.tzinfo is not None
            or self.access_start_time >= self.access_end_time
        ):
            raise ValueError("Policy access times must form a same-day local window.")
        if self.expires_at <= self.issued_at:
            raise ValueError("Policy expiry must be later than its issue time.")
        return self


class SignedOfflinePolicy(BaseModel):
    algorithm: Literal["Ed25519"] = "Ed25519"
    claims: OfflinePolicyClaims
    signature: str = Field(min_length=1, max_length=256)


class OfflinePolicyResponse(BaseModel):
    policy: SignedOfflinePolicy


class OfflineStatusResponse(BaseModel):
    enabled: bool
    offline_allow_enabled: bool
    policy_verification_key_ready: bool
    policy_issuance_key_ready: bool
    local_inference_provider: str
    policy_ttl_seconds: int
    sync_configured: bool
    backend_available: bool | None


class OfflineDecisionResponse(BaseModel):
    event_id: str
    subject_id: str
    mode: Literal["offline"] = "offline"
    outcome: DecisionOutcome
    reason_code: str
    reason: str
    policy_id: str | None = None
    model_status: dict[str, str]


class OfflineSyncEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=36)
    subject_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    mode: Literal["offline"]
    outcome: DecisionOutcome
    reason_code: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    policy_id: str | None = Field(default=None, max_length=36)

    @field_validator("occurred_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Event timestamps must be timezone-aware.")
        return value.astimezone(timezone.utc)


class OfflineSyncBatch(BaseModel):
    events: list[OfflineSyncEvent] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_event_ids(self):
        event_ids = [event.event_id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("A sync batch must not contain duplicate event IDs.")
        return self


class OfflineSyncResponse(BaseModel):
    accepted_event_ids: list[str]


class OfflineSyncStatusResponse(BaseModel):
    attempted: int
    synchronized: int
    pending: int
    sync_configured: bool


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
