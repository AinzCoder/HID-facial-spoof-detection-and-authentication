from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.database.models import CampusAccessSchedule
from app.database.offline_models import CachedOfflinePolicy, OfflineVerificationEvent
from app.schemas import DecisionOutcome, OfflinePolicyClaims, SignedOfflinePolicy
from app.services.campus_access import apply_campus_schedule
from app.services.decision import Decision, ReasonCode
from app.services.offline_policy import OfflinePolicyError, verify_policy


@dataclass(frozen=True)
class OfflineEvaluation:
    decision: Decision
    policy_id: str | None


def _step_up(reason_code: ReasonCode, reason: str) -> Decision:
    return Decision(DecisionOutcome.STEP_UP, reason_code, reason)


def _aware_utc(value: datetime) -> datetime | None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def apply_offline_policy(
    biometric_decision: Decision,
    subject_id: str,
    policy: SignedOfflinePolicy | None,
    now: datetime,
    settings: Settings,
) -> OfflineEvaluation:
    if biometric_decision.outcome != DecisionOutcome.ALLOW:
        return OfflineEvaluation(biometric_decision, None)

    if not settings.offline_enabled:
        return OfflineEvaluation(
            _step_up(ReasonCode.OFFLINE_DISABLED, "Offline verification is disabled."),
            None,
        )
    if not settings.offline_allow_enabled:
        return OfflineEvaluation(
            _step_up(
                ReasonCode.OFFLINE_ALLOW_DISABLED,
                "Offline ALLOW is disabled by configuration.",
            ),
            None,
        )
    if not settings.offline_signing_public_key:
        return OfflineEvaluation(
            _step_up(
                ReasonCode.OFFLINE_SIGNING_KEY_UNAVAILABLE,
                "The offline policy verification key is unavailable.",
            ),
            None,
        )
    if policy is None:
        return OfflineEvaluation(
            _step_up(ReasonCode.OFFLINE_POLICY_MISSING, "No signed offline policy is cached."),
            None,
        )

    try:
        claims = verify_policy(policy, settings.offline_signing_public_key)
    except OfflinePolicyError:
        return OfflineEvaluation(
            _step_up(ReasonCode.OFFLINE_POLICY_INVALID, "The cached offline policy is invalid."),
            None,
        )

    if claims.subject_id != subject_id:
        return OfflineEvaluation(
            _step_up(
                ReasonCode.OFFLINE_POLICY_SUBJECT_MISMATCH,
                "The cached policy belongs to a different subject.",
            ),
            None,
        )

    current_time = _aware_utc(now)
    if current_time is None:
        return OfflineEvaluation(
            _step_up(
                ReasonCode.OFFLINE_POLICY_CLOCK_SKEW,
                "The local clock cannot be safely evaluated.",
            ),
            claims.policy_id,
        )
    if (
        current_time.timestamp() + settings.offline_max_clock_skew_seconds
        < claims.issued_at.timestamp()
    ):
        return OfflineEvaluation(
            _step_up(
                ReasonCode.OFFLINE_POLICY_CLOCK_SKEW,
                "The local clock is earlier than the signed policy issue time.",
            ),
            claims.policy_id,
        )
    if claims.expires_at - claims.issued_at > timedelta(
        seconds=settings.offline_policy_ttl_seconds
    ):
        return OfflineEvaluation(
            _step_up(
                ReasonCode.OFFLINE_POLICY_INVALID,
                "The signed policy validity period exceeds the configured maximum.",
            ),
            claims.policy_id,
        )
    if current_time >= claims.expires_at:
        return OfflineEvaluation(
            _step_up(ReasonCode.OFFLINE_POLICY_EXPIRED, "The signed offline policy has expired."),
            claims.policy_id,
        )
    if not claims.enabled:
        return OfflineEvaluation(
            _step_up(ReasonCode.OFFLINE_POLICY_DISABLED, "The signed offline policy is disabled."),
            claims.policy_id,
        )
    if not claims.allow_offline:
        return OfflineEvaluation(
            _step_up(
                ReasonCode.OFFLINE_POLICY_NOT_PERMITTED,
                "The signed policy does not permit offline access.",
            ),
            claims.policy_id,
        )

    schedule = CampusAccessSchedule(
        subject_id=claims.subject_id,
        permitted_days=[day.value for day in claims.permitted_days],
        access_start_time=claims.access_start_time,
        access_end_time=claims.access_end_time,
        timezone=claims.timezone,
        enabled=claims.enabled,
    )
    return OfflineEvaluation(
        apply_campus_schedule(biometric_decision, schedule, current_time),
        claims.policy_id,
    )


def policy_from_record(record: CachedOfflinePolicy) -> SignedOfflinePolicy:
    claims = OfflinePolicyClaims.model_validate_json(record.claims_json)
    if claims.subject_id != record.subject_id or claims.policy_id != record.policy_id:
        raise ValueError("Cached policy record does not match its signed claims.")
    return SignedOfflinePolicy(claims=claims, signature=record.signature)


def store_signed_policy(
    db: Session,
    subject_id: str,
    policy: SignedOfflinePolicy,
    now: datetime,
    settings: Settings,
) -> OfflinePolicyClaims:
    claims = verify_policy(policy, settings.offline_signing_public_key)
    if claims.subject_id != subject_id:
        raise ValueError("Policy subject does not match the cache subject.")

    current_time = _aware_utc(now)
    if current_time is None:
        raise ValueError("Current time must be timezone-aware.")
    if (
        current_time.timestamp() + settings.offline_max_clock_skew_seconds
        < claims.issued_at.timestamp()
    ):
        raise ValueError("Policy issue time is too far in the future.")
    if current_time >= claims.expires_at:
        raise ValueError("Expired policies cannot be cached.")
    if claims.expires_at - claims.issued_at > timedelta(
        seconds=settings.offline_policy_ttl_seconds
    ):
        raise ValueError("Policy validity exceeds the configured maximum.")

    existing = db.get(CachedOfflinePolicy, subject_id)
    if existing is not None:
        previous = policy_from_record(existing)
        verify_policy(previous, settings.offline_signing_public_key)
        if claims.issued_at < previous.claims.issued_at or (
            claims.issued_at == previous.claims.issued_at
            and claims.policy_id != previous.claims.policy_id
        ):
            raise ValueError("Older policies cannot replace a newer cached policy.")
        if (
            claims.policy_id == previous.claims.policy_id
            and (claims != previous.claims or policy.signature != previous.signature)
        ):
            raise ValueError("A policy ID cannot be reused with different claims.")
    row = existing or CachedOfflinePolicy(
        subject_id=subject_id,
        policy_id=claims.policy_id,
        claims_json="",
        signature=policy.signature,
        updated_at=current_time,
    )
    row.policy_id = claims.policy_id
    row.claims_json = claims.model_dump_json()
    row.signature = policy.signature
    row.updated_at = current_time
    db.add(row)
    db.commit()
    return claims


def get_cached_policy(db: Session, subject_id: str) -> SignedOfflinePolicy | None:
    record = db.scalar(
        select(CachedOfflinePolicy).where(CachedOfflinePolicy.subject_id == subject_id)
    )
    if record is None:
        return None
    return policy_from_record(record)


def record_offline_event(
    session_factory: sessionmaker[Session],
    *,
    event_id: str,
    subject_id: str,
    occurred_at: datetime,
    decision: Decision,
    policy_id: str | None,
) -> None:
    stored_time = _aware_utc(occurred_at)
    if stored_time is None:
        raise ValueError("Offline event timestamp must be timezone-aware.")
    event = OfflineVerificationEvent(
        event_id=event_id,
        subject_id=subject_id,
        occurred_at=stored_time,
        mode="offline",
        outcome=decision.outcome.value,
        reason_code=decision.reason_code.value,
        policy_id=policy_id,
        sync_status="pending",
        sync_attempt_count=0,
    )
    with session_factory() as db:
        db.add(event)
        db.commit()


