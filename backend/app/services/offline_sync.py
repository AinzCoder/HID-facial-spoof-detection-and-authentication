from datetime import datetime, timezone
import json
import logging
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import OfflineSyncReceipt
from app.database.offline_models import OfflineVerificationEvent
from app.schemas import OfflineSyncBatch, OfflineSyncEvent, OfflineSyncStatusResponse

logger = logging.getLogger(__name__)
SyncTransport = Callable[[str, str, dict, float], dict]
HealthProbe = Callable[[str, float], bool]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _event_payload(event: OfflineSyncEvent) -> dict:
    return event.model_dump(mode="json")


def _canonical_payload(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _send_sync_request(endpoint: str, token: str, payload: dict, timeout: float) -> dict:
    request = Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _probe_backend(health_url: str, timeout: float) -> bool:
    request = Request(health_url, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return 200 <= response.status < 400


def detect_backend_availability(
    health_url: str | None,
    timeout_seconds: float,
    probe: HealthProbe | None = None,
) -> bool | None:
    if not health_url:
        return None
    try:
        return (probe or _probe_backend)(health_url, timeout_seconds)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return False


def receive_sync_batch(db: Session, batch: OfflineSyncBatch, now: datetime) -> list[str]:
    accepted: list[str] = []
    new_receipts: list[OfflineSyncReceipt] = []
    for event in batch.events:
        payload = _event_payload(event)
        payload_json = _canonical_payload(payload)
        existing = db.get(OfflineSyncReceipt, event.event_id)
        if existing is not None:
            if existing.payload_json != payload_json:
                raise ValueError("An event ID was reused with different event data.")
        else:
            new_receipts.append(
                OfflineSyncReceipt(
                    event_id=event.event_id,
                    subject_id=event.subject_id,
                    occurred_at=event.occurred_at,
                    mode=event.mode,
                    outcome=event.outcome.value,
                    reason_code=event.reason_code,
                    policy_id=event.policy_id,
                    payload_json=payload_json,
                    received_at=_as_utc(now),
                )
            )
        accepted.append(event.event_id)

    try:
        db.add_all(new_receipts)
        db.commit()
    except IntegrityError:
        db.rollback()
        for event in batch.events:
            existing = db.get(OfflineSyncReceipt, event.event_id)
            if (
                existing is None
                or existing.payload_json != _canonical_payload(_event_payload(event))
            ):
                raise
    return accepted


def sync_pending_events(
    session_factory: sessionmaker[Session],
    *,
    endpoint: str | None,
    token: str | None,
    timeout_seconds: int,
    now: datetime,
    transport: SyncTransport | None = None,
) -> OfflineSyncStatusResponse:
    if not endpoint or not token:
        with session_factory() as db:
            count = len(
                db.scalars(
                    select(OfflineVerificationEvent.event_id).where(
                        OfflineVerificationEvent.sync_status == "pending"
                    )
                ).all()
            )
        return OfflineSyncStatusResponse(
            attempted=0,
            synchronized=0,
            pending=count,
            sync_configured=False,
        )

    with session_factory() as db:
        events = db.scalars(
            select(OfflineVerificationEvent)
            .where(OfflineVerificationEvent.sync_status == "pending")
            .order_by(OfflineVerificationEvent.occurred_at)
            .limit(100)
        ).all()
        if not events:
            return OfflineSyncStatusResponse(
                attempted=0, synchronized=0, pending=0, sync_configured=True
            )
        event_ids = [event.event_id for event in events]
        for event in events:
            event.sync_attempt_count += 1
            event.last_sync_attempt_at = _as_utc(now)
            event.last_sync_error_code = None
        db.commit()
        batch = OfflineSyncBatch(
            events=[
                OfflineSyncEvent(
                    event_id=event.event_id,
                    subject_id=event.subject_id,
                    occurred_at=_as_utc(event.occurred_at),
                    mode="offline",
                    outcome=event.outcome,
                    reason_code=event.reason_code,
                    policy_id=event.policy_id,
                )
                for event in events
            ]
        )

    try:
        response = (transport or _send_sync_request)(
            endpoint,
            token,
            batch.model_dump(mode="json"),
            float(timeout_seconds),
        )
        accepted = set(response.get("accepted_event_ids", []))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Offline event synchronization failed (%s)", type(exc).__name__)
        accepted = set()

    with session_factory() as db:
        for event_id in event_ids:
            event = db.get(OfflineVerificationEvent, event_id)
            if event is None or event.sync_status != "pending":
                continue
            if event_id in accepted:
                event.sync_status = "synced"
                event.last_sync_error_code = None
            else:
                event.last_sync_error_code = "sync_unconfirmed"
        db.commit()
        pending_count = len(
            db.scalars(
                select(OfflineVerificationEvent.event_id).where(
                    OfflineVerificationEvent.sync_status == "pending"
                )
            ).all()
        )
    synchronized_count = sum(event_id in accepted for event_id in event_ids)
    return OfflineSyncStatusResponse(
        attempted=len(event_ids),
        synchronized=synchronized_count,
        pending=pending_count,
        sync_configured=True,
    )
