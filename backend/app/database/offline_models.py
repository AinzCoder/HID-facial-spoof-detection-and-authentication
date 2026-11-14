from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, event, inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class OfflineBase(DeclarativeBase):
    pass


class CachedOfflinePolicy(OfflineBase):
    __tablename__ = "cached_offline_policies"

    subject_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    policy_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    claims_json: Mapped[str] = mapped_column(Text, nullable=False)
    signature: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OfflineVerificationEvent(OfflineBase):
    __tablename__ = "offline_verification_events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="offline")
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    sync_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    sync_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_sync_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_sync_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)


@event.listens_for(OfflineVerificationEvent, "before_update")
def prevent_offline_event_mutation(_mapper, _connection, target) -> None:
    immutable_fields = (
        "event_id",
        "subject_id",
        "occurred_at",
        "mode",
        "outcome",
        "reason_code",
        "policy_id",
    )
    state = inspect(target)
    if any(state.attrs[field].history.has_changes() for field in immutable_fields):
        raise ValueError("Offline audit event contents are immutable.")


@event.listens_for(OfflineVerificationEvent, "before_delete")
def prevent_pending_event_deletion(_mapper, _connection, target) -> None:
    if target.sync_status == "pending":
        raise ValueError("Unsynchronized offline audit events cannot be deleted.")
