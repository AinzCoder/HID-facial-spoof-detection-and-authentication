from datetime import datetime, time

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    Time,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class VerificationSession(Base):
    __tablename__ = "verification_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="created")
    challenge_type: Mapped[str] = mapped_column(String(32))
    challenge_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    challenge_status: Mapped[str] = mapped_column(String(32), default="assigned")
    challenge_used: Mapped[bool] = mapped_column(default=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    verification_submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    latest_outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    latest_reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)

    results: Mapped[list["VerificationResult"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Enrollment(Base):
    __tablename__ = "enrollments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    encrypted_embedding: Mapped[bytes] = mapped_column(LargeBinary)
    encryption_nonce: Mapped[bytes] = mapped_column(LargeBinary)
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CampusAccessSchedule(Base):
    __tablename__ = "campus_access_schedules"

    subject_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("enrollments.subject_id", ondelete="CASCADE"),
        primary_key=True,
    )
    permitted_days: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    access_start_time: Mapped[time] = mapped_column(Time, nullable=False)
    access_end_time: Mapped[time] = mapped_column(Time, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class VerificationResult(Base):
    __tablename__ = "verification_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("verification_sessions.id", ondelete="CASCADE"), index=True
    )
    outcome: Mapped[str] = mapped_column(String(16))
    reason_code: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text)
    model_status_json: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[str] = mapped_column(Text)
    media_kind: Mapped[str] = mapped_column(String(16))
    processing_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    session: Mapped[VerificationSession] = relationship(back_populates="results")


class OfflineSyncReceipt(Base):
    __tablename__ = "offline_sync_receipts"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
