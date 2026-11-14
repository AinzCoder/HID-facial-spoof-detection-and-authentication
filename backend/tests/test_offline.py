import base64
import io
from datetime import datetime, time, timedelta, timezone
import uuid
from urllib.error import URLError

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from PIL import Image

from app.config import Settings
from app.database.models import CampusAccessSchedule, Enrollment, OfflineSyncReceipt
from app.database.offline_models import OfflineVerificationEvent
from app.main import create_app
from app.schemas import (
    CampusScheduleDay,
    DecisionOutcome,
    OfflinePolicyClaims,
)
from app.services.decision import Decision, ReasonCode, decide
from app.services.inference import ComponentOutput, ModelState, VerificationInference
from app.services.offline import apply_offline_policy, record_offline_event
from app.services.offline_policy import (
    OfflinePolicyError,
    sign_policy,
    verify_policy,
)
from app.services.offline_sync import detect_backend_availability, sync_pending_events
from app.services.security import encrypt_embedding


NOW = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
SUBJECT = "offline-subject"
ADMIN_TOKEN = "test-campus-schedule-admin-token-0123456789"
SYNC_TOKEN = "test-offline-sync-token-0123456789"
PRIVATE_RAW = bytes(range(32))
PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(PRIVATE_RAW)
PUBLIC_RAW = PRIVATE_KEY.public_key().public_bytes(
    serialization.Encoding.Raw,
    serialization.PublicFormat.Raw,
)
PRIVATE_ENCODED = base64.urlsafe_b64encode(PRIVATE_RAW).decode("ascii").rstrip("=")
PUBLIC_ENCODED = base64.urlsafe_b64encode(PUBLIC_RAW).decode("ascii").rstrip("=")


def image_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), color=(100, 120, 140))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def component(label, score=0.99, matched=None, state=ModelState.REAL):
    return ComponentOutput(
        state=state,
        label=label,
        score=score,
        matched=matched,
        model_name="synthetic-test-provider" if state == ModelState.REAL else None,
    )


def inference(mode="genuine", state=ModelState.REAL):
    pad_label = "spoof" if mode == "spoof" else "genuine"
    identity_label = "mismatch" if mode == "mismatch" else "matched"
    identity_match = mode != "mismatch"
    pad_score = 0.99 if mode == "spoof" else 0.99
    score = 0.1 if mode == "mismatch" else 0.99
    return VerificationInference(
        face_detection=component("face", state=state),
        pad=component(pad_label, pad_score, state=state),
        identity=component(identity_label, score, identity_match, state=state),
        temporal=component("live", state=state),
        mask=component("no_mask", state=state),
    )


class OfflineProvider:
    """Synthetic test fixture; it does not load or execute biometric models."""

    def __init__(self, mode="genuine"):
        self.mode = mode

    def create_embedding(self, media_path, media_type):
        from app.services.inference import EmbeddingOutput

        return EmbeddingOutput(ModelState.UNAVAILABLE)

    def verify(self, media_path, media_type, enrollment_embedding):
        if self.mode == "mock":
            return inference(state=ModelState.MOCK)
        return inference(self.mode)

    def validate_challenge(self, media_path, media_type, challenge):
        return component("failed", state=ModelState.UNAVAILABLE)


def settings_for(tmp_path, *, offline_allow=True, sync_token=SYNC_TOKEN):
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'primary.db').as_posix()}",
        offline_database_url=f"sqlite:///{(tmp_path / 'offline.db').as_posix()}",
        biometric_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
        campus_schedule_admin_token=ADMIN_TOKEN,
        offline_sync_token=sync_token,
        offline_enabled=True,
        offline_allow_enabled=offline_allow,
        offline_signing_private_key=PRIVATE_ENCODED if offline_allow else None,
        offline_signing_public_key=PUBLIC_ENCODED if offline_allow else None,
    )


def policy_settings():
    return Settings(
        offline_enabled=True,
        offline_allow_enabled=True,
        offline_signing_private_key=PRIVATE_ENCODED,
        offline_signing_public_key=PUBLIC_ENCODED,
    )


def make_policy(
    *,
    subject_id=SUBJECT,
    now=NOW,
    enabled=True,
    allow_offline=True,
    issued_at=None,
    expires_at=None,
    days=None,
    start=time(9, 0),
    end=time(17, 0),
    zone="UTC",
):
    claims = OfflinePolicyClaims(
        policy_id=str(uuid.uuid4()),
        subject_id=subject_id,
        permitted_days=days or [CampusScheduleDay.MONDAY],
        access_start_time=start,
        access_end_time=end,
        timezone=zone,
        enabled=enabled,
        allow_offline=allow_offline,
        issued_at=issued_at or now - timedelta(minutes=1),
        expires_at=expires_at or now + timedelta(hours=1),
    )
    return sign_policy(claims, PRIVATE_ENCODED, PUBLIC_ENCODED)


def decision_for(mode="genuine", state=ModelState.REAL, challenge_result=None):
    return decide(
        inference(mode, state),
        policy_settings(),
        media_kind="image",
        challenge_result=challenge_result,
        challenge_required=challenge_result is not None,
    )


def test_offline_mode_and_allow_are_disabled_by_default():
    settings = Settings()
    assert settings.offline_enabled is False
    assert settings.offline_allow_enabled is False
    result = apply_offline_policy(
        Decision(DecisionOutcome.ALLOW, ReasonCode.VERIFIED, "verified"),
        SUBJECT,
        make_policy(),
        NOW,
        settings,
    )
    assert result.decision.outcome == DecisionOutcome.STEP_UP
    assert result.decision.reason_code == ReasonCode.OFFLINE_DISABLED


def test_ed25519_signed_policy_verifies():
    policy = make_policy()
    assert verify_policy(policy, PUBLIC_ENCODED) == policy.claims


def test_invalid_policy_signature_is_rejected():
    policy = make_policy()
    tampered = policy.model_copy(update={"signature": "AAAA"})
    with pytest.raises(OfflinePolicyError):
        verify_policy(tampered, PUBLIC_ENCODED)


def test_expired_policy_cannot_allow():
    policy = make_policy(expires_at=NOW - timedelta(seconds=1))
    result = apply_offline_policy(
        decision_for(), SUBJECT, policy, NOW, policy_settings()
    )
    assert result.decision.outcome == DecisionOutcome.STEP_UP
    assert result.decision.reason_code == ReasonCode.OFFLINE_POLICY_EXPIRED


def test_policy_longer_than_configured_ttl_cannot_allow():
    policy = make_policy(expires_at=NOW + timedelta(days=2))
    result = apply_offline_policy(
        decision_for(), SUBJECT, policy, NOW, policy_settings()
    )
    assert result.decision.outcome == DecisionOutcome.STEP_UP
    assert result.decision.reason_code == ReasonCode.OFFLINE_POLICY_INVALID


def test_disabled_policy_cannot_allow():
    policy = make_policy(enabled=False)
    result = apply_offline_policy(
        decision_for(), SUBJECT, policy, NOW, policy_settings()
    )
    assert result.decision.reason_code == ReasonCode.OFFLINE_POLICY_DISABLED


def test_subject_mismatch_cannot_allow():
    policy = make_policy(subject_id="another-subject")
    result = apply_offline_policy(
        decision_for(), SUBJECT, policy, NOW, policy_settings()
    )
    assert result.decision.reason_code == ReasonCode.OFFLINE_POLICY_SUBJECT_MISMATCH


def test_missing_policy_steps_up():
    result = apply_offline_policy(
        decision_for(), SUBJECT, None, NOW, policy_settings()
    )
    assert result.decision.outcome == DecisionOutcome.STEP_UP
    assert result.decision.reason_code == ReasonCode.OFFLINE_POLICY_MISSING


def test_outside_schedule_steps_up():
    after_hours = NOW.replace(hour=18)
    result = apply_offline_policy(
        decision_for(),
        SUBJECT,
        make_policy(now=after_hours),
        after_hours,
        policy_settings(),
    )
    assert result.decision.outcome == DecisionOutcome.STEP_UP
    assert result.decision.reason_code == ReasonCode.OUTSIDE_ACCESS_SCHEDULE


@pytest.mark.parametrize(
    ("hour", "expected"),
    [(9, DecisionOutcome.ALLOW), (17, DecisionOutcome.STEP_UP)],
)
def test_offline_schedule_start_is_inclusive_and_end_is_exclusive(hour, expected):
    current_time = NOW.replace(hour=hour)
    result = apply_offline_policy(
        decision_for(),
        SUBJECT,
        make_policy(now=current_time),
        current_time,
        policy_settings(),
    )
    assert result.decision.outcome == expected


def test_offline_weekday_restriction_steps_up():
    policy = make_policy(days=[CampusScheduleDay.TUESDAY])
    result = apply_offline_policy(
        decision_for(), SUBJECT, policy, NOW, policy_settings()
    )
    assert result.decision.reason_code == ReasonCode.OUTSIDE_ACCESS_SCHEDULE


def test_clock_skew_beyond_configured_limit_steps_up():
    policy = make_policy(
        issued_at=NOW + timedelta(minutes=10),
        expires_at=NOW + timedelta(hours=2),
    )
    result = apply_offline_policy(
        decision_for(), SUBJECT, policy, NOW, policy_settings()
    )
    assert result.decision.outcome == DecisionOutcome.STEP_UP
    assert result.decision.reason_code == ReasonCode.OFFLINE_POLICY_CLOCK_SKEW


def test_in_schedule_allow_requires_explicit_runtime_and_policy_permission():
    valid_policy = make_policy()
    result = apply_offline_policy(
        decision_for(), SUBJECT, valid_policy, NOW, policy_settings()
    )
    assert result.decision.outcome == DecisionOutcome.ALLOW

    disabled_runtime = apply_offline_policy(
        decision_for(),
        SUBJECT,
        valid_policy,
        NOW,
        Settings(offline_enabled=True, offline_allow_enabled=False),
    )
    assert disabled_runtime.decision.outcome == DecisionOutcome.STEP_UP
    assert disabled_runtime.decision.reason_code == ReasonCode.OFFLINE_ALLOW_DISABLED

    disallowed_policy = make_policy(allow_offline=False)
    disallowed = apply_offline_policy(
        decision_for(), SUBJECT, disallowed_policy, NOW, policy_settings()
    )
    assert disallowed.decision.outcome == DecisionOutcome.STEP_UP
    assert disallowed.decision.reason_code == ReasonCode.OFFLINE_POLICY_NOT_PERMITTED


def test_confirmed_spoof_remains_deny_outside_schedule():
    biometric = decision_for("spoof")
    result = apply_offline_policy(
        biometric,
        SUBJECT,
        make_policy(),
        NOW.replace(hour=23),
        policy_settings(),
    )
    assert result.decision.outcome == DecisionOutcome.DENY
    assert result.decision.reason_code == ReasonCode.CONFIRMED_SPOOF


def test_identity_mismatch_remains_deny():
    result = apply_offline_policy(
        decision_for("mismatch"), SUBJECT, make_policy(), NOW, policy_settings()
    )
    assert result.decision.outcome == DecisionOutcome.DENY
    assert result.decision.reason_code == ReasonCode.IDENTITY_MISMATCH


def test_biometric_step_up_is_not_overridden_by_valid_policy():
    biometric = decide(
        inference("genuine"),
        policy_settings(),
        media_kind="image",
        challenge_result=component("failed", state=ModelState.UNAVAILABLE),
        challenge_required=True,
    )
    result = apply_offline_policy(
        biometric, SUBJECT, make_policy(), NOW, policy_settings()
    )
    assert result.decision.outcome == DecisionOutcome.STEP_UP
    assert result.decision.reason_code == ReasonCode.CHALLENGE_REQUIRED


def test_mock_inference_cannot_allow_with_valid_policy():
    result = apply_offline_policy(
        decision_for(state=ModelState.MOCK),
        SUBJECT,
        make_policy(),
        NOW,
        policy_settings(),
    )
    assert result.decision.outcome == DecisionOutcome.STEP_UP


def test_failed_or_unavailable_challenge_cannot_allow():
    challenge = component("failed", score=0.99, state=ModelState.UNAVAILABLE)
    biometric = decide(
        inference(),
        policy_settings(),
        media_kind="image",
        challenge_result=challenge,
        challenge_required=True,
    )
    result = apply_offline_policy(
        biometric, SUBJECT, make_policy(), NOW, policy_settings()
    )
    assert result.decision.outcome == DecisionOutcome.STEP_UP
    assert result.decision.reason_code == ReasonCode.CHALLENGE_REQUIRED


def test_missing_or_mismatched_signing_key_prevents_offline_allow():
    with pytest.raises(ValidationError):
        Settings(offline_enabled=True, offline_allow_enabled=True)
    with pytest.raises(ValueError):
        Settings(
            offline_enabled=True,
            offline_allow_enabled=True,
            offline_signing_private_key=PRIVATE_ENCODED,
            offline_signing_public_key=PRIVATE_ENCODED,
        )


def test_offline_backend_urls_require_tls_outside_loopback():
    with pytest.raises(ValidationError):
        Settings(offline_sync_backend_url="http://backend.example/v1/offline/events/sync")
    local = Settings(offline_backend_health_url="http://127.0.0.1:8000/health")
    assert local.offline_backend_health_url is not None


def test_offline_verifier_can_be_configured_with_public_key_only():
    verifier_settings = Settings(
        offline_enabled=True,
        offline_allow_enabled=True,
        offline_signing_public_key=PUBLIC_ENCODED,
    )
    result = apply_offline_policy(
        decision_for(), SUBJECT, make_policy(), NOW, verifier_settings
    )
    assert result.decision.outcome == DecisionOutcome.ALLOW


def test_offline_policy_evaluates_local_timezone_and_weekday():
    tuesday_utc = datetime(2026, 1, 6, 1, 0, tzinfo=timezone.utc)
    policy = make_policy(
        now=tuesday_utc,
        days=[CampusScheduleDay.MONDAY],
        start=time(16, 0),
        end=time(18, 0),
        zone="America/Los_Angeles",
    )
    result = apply_offline_policy(
        decision_for(), SUBJECT, policy, tuesday_utc, policy_settings()
    )
    assert result.decision.outcome == DecisionOutcome.ALLOW


def test_backend_unavailability_probe_is_fail_safe():
    assert detect_backend_availability(None, 1) is None
    assert detect_backend_availability(
        "https://backend.invalid/health",
        1,
        probe=lambda _url, _timeout: False,
    ) is False


def test_policy_cache_rejects_stale_policy(tmp_path):
    settings = settings_for(tmp_path)
    newer = make_policy(issued_at=NOW)
    older = make_policy(issued_at=NOW - timedelta(minutes=1))
    with TestClient(create_app(settings, OfflineProvider(), clock=lambda: NOW)) as client:
        token = {"Authorization": f"Bearer {SYNC_TOKEN}"}
        first = client.put(
            f"/v1/offline/policies/{SUBJECT}",
            headers=token,
            json=newer.model_dump(mode="json"),
        )
        assert first.status_code == 200
        stale = client.put(
            f"/v1/offline/policies/{SUBJECT}",
            headers=token,
            json=older.model_dump(mode="json"),
        )
        assert stale.status_code == 409
        same_issue_time = client.put(
            f"/v1/offline/policies/{SUBJECT}",
            headers=token,
            json=make_policy(issued_at=NOW).model_dump(mode="json"),
        )
        assert same_issue_time.status_code == 409


def _add_enrollment(client):
    key = client.app.state.settings.biometric_encryption_key
    ciphertext, nonce = encrypt_embedding((0.1, 0.2), key)
    with client.app.state.session_factory() as db:
        db.add(
            Enrollment(
                id=str(uuid.uuid4()),
                subject_id=SUBJECT,
                encrypted_embedding=ciphertext,
                encryption_nonce=nonce,
                model_name="test-only",
                model_version="test-only",
                created_at=NOW,
            )
        )
        db.commit()
        db.add(
            CampusAccessSchedule(
                subject_id=SUBJECT,
                permitted_days=["monday"],
                access_start_time=time(9, 0),
                access_end_time=time(17, 0),
                timezone="UTC",
                enabled=True,
            )
        )
        db.commit()


def _publish_policy(client):
    response = client.post(
        f"/v1/admin/offline/policies/{SUBJECT}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"allow_offline": True},
    )
    assert response.status_code == 200, response.text
    return response.json()["policy"]


def _offline_verify(client):
    return client.post(
        f"/v1/offline/verifications/{SUBJECT}",
        files={"media_file": ("face.png", image_bytes(), "image/png")},
    )


def test_local_event_survives_app_restart(tmp_path):
    settings = settings_for(tmp_path, offline_allow=False)
    with TestClient(create_app(settings, OfflineProvider(), clock=lambda: NOW)) as client:
        response = _offline_verify(client)
        assert response.status_code == 200
        assert response.json()["outcome"] == "STEP_UP"
        event_id = response.json()["event_id"]
    with TestClient(create_app(settings, OfflineProvider(), clock=lambda: NOW)) as restarted:
        with restarted.app.state.offline_session_factory() as db:
            event = db.get(OfflineVerificationEvent, event_id)
            assert event is not None
            assert event.mode == "offline"
            assert event.sync_status == "pending"
            assert not hasattr(event, "embedding")


def test_offline_api_allows_only_after_real_inference_and_signed_schedule(tmp_path):
    settings = settings_for(tmp_path)
    with TestClient(
        create_app(settings, OfflineProvider(), clock=lambda: NOW)
    ) as client:
        _add_enrollment(client)
        _publish_policy(client)
        response = _offline_verify(client)
        assert response.status_code == 200
        assert response.json()["outcome"] == "ALLOW"
        assert response.json()["policy_id"]
        with client.app.state.offline_session_factory() as db:
            event = db.get(OfflineVerificationEvent, response.json()["event_id"])
            assert event.outcome == "ALLOW"
            assert event.sync_status == "pending"


def test_offline_api_mock_provider_cannot_allow(tmp_path):
    settings = settings_for(tmp_path)
    with TestClient(
        create_app(settings, OfflineProvider("mock"), clock=lambda: NOW)
    ) as client:
        _add_enrollment(client)
        _publish_policy(client)
        response = _offline_verify(client)
        assert response.json()["outcome"] == "STEP_UP"


def test_offline_audit_storage_failure_downgrades_allow(tmp_path):
    settings = settings_for(tmp_path)
    with TestClient(
        create_app(settings, OfflineProvider(), clock=lambda: NOW)
    ) as client:
        _add_enrollment(client)
        _publish_policy(client)
        original_factory = client.app.state.offline_session_factory
        calls = 0

        class BrokenSession:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def add(self, _event):
                return None

            def commit(self):
                raise SQLAlchemyError("local storage unavailable")

        def fail_on_event_write():
            nonlocal calls
            calls += 1
            return original_factory() if calls == 1 else BrokenSession()

        client.app.state.offline_session_factory = fail_on_event_write
        response = _offline_verify(client)
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["reason_code"] == "offline_storage_error"


def test_server_sync_endpoint_is_idempotent(tmp_path):
    settings = settings_for(tmp_path, offline_allow=False)
    with TestClient(create_app(settings, clock=lambda: NOW)) as client:
        event_id = str(uuid.uuid4())
        payload = {
            "events": [
                {
                    "event_id": event_id,
                    "subject_id": SUBJECT,
                    "occurred_at": NOW.isoformat(),
                    "mode": "offline",
                    "outcome": "STEP_UP",
                    "reason_code": "offline_policy_missing",
                    "policy_id": None,
                }
            ]
        }
        headers = {"Authorization": f"Bearer {SYNC_TOKEN}"}
        first = client.post("/v1/offline/events/sync", headers=headers, json=payload)
        second = client.post("/v1/offline/events/sync", headers=headers, json=payload)
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == second.json() == {"accepted_event_ids": [event_id]}
        with client.app.state.session_factory() as db:
            assert db.query(OfflineSyncReceipt).count() == 1


def _pending_event(factory):
    event_id = str(uuid.uuid4())
    record_offline_event(
        factory,
        event_id=event_id,
        subject_id=SUBJECT,
        occurred_at=NOW,
        decision=Decision(
            DecisionOutcome.STEP_UP,
            ReasonCode.OFFLINE_POLICY_MISSING,
            "policy missing",
        ),
        policy_id=None,
    )
    return event_id


def test_offline_audit_payload_is_immutable(tmp_path):
    settings = settings_for(tmp_path, offline_allow=False)
    with TestClient(create_app(settings, clock=lambda: NOW)) as client:
        event_id = _pending_event(client.app.state.offline_session_factory)
        with client.app.state.offline_session_factory() as db:
            event = db.get(OfflineVerificationEvent, event_id)
            event.outcome = DecisionOutcome.ALLOW.value
            with pytest.raises(ValueError):
                db.commit()
            db.rollback()
            event = db.get(OfflineVerificationEvent, event_id)
            db.delete(event)
            with pytest.raises(ValueError):
                db.commit()


def test_failed_sync_retains_event_for_retry(tmp_path):
    settings = settings_for(tmp_path, offline_allow=False)
    with TestClient(create_app(settings, clock=lambda: NOW)) as client:
        event_id = _pending_event(client.app.state.offline_session_factory)

        def fail_sync(_endpoint, _token, _payload, _timeout):
            raise URLError("backend unavailable")

        result = sync_pending_events(
            client.app.state.offline_session_factory,
            endpoint="https://backend.invalid/v1/offline/events/sync",
            token=SYNC_TOKEN,
            timeout_seconds=1,
            now=NOW,
            transport=fail_sync,
        )
        assert result.synchronized == 0
        assert result.pending == 1
        with client.app.state.offline_session_factory() as db:
            event = db.get(OfflineVerificationEvent, event_id)
            assert event.sync_status == "pending"
            assert event.sync_attempt_count == 1


def test_successful_sync_marks_event_after_server_acceptance(tmp_path):
    settings = settings_for(tmp_path, offline_allow=False)
    with TestClient(create_app(settings, clock=lambda: NOW)) as client:
        event_id = _pending_event(client.app.state.offline_session_factory)

        def accept_sync(_endpoint, _token, payload, _timeout):
            return {"accepted_event_ids": [payload["events"][0]["event_id"]]}

        result = sync_pending_events(
            client.app.state.offline_session_factory,
            endpoint="https://backend.invalid/v1/offline/events/sync",
            token=SYNC_TOKEN,
            timeout_seconds=1,
            now=NOW,
            transport=accept_sync,
        )
        assert result.synchronized == 1
        assert result.pending == 0
        with client.app.state.offline_session_factory() as db:
            event = db.get(OfflineVerificationEvent, event_id)
            assert event.sync_status == "synced"
            assert event.sync_attempt_count == 1


def test_offline_status_reports_default_safe_configuration(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'primary.db').as_posix()}",
        offline_database_url=f"sqlite:///{(tmp_path / 'offline.db').as_posix()}",
    )
    with TestClient(create_app(settings)) as client:
        status = client.get("/v1/offline/status").json()
        assert status["enabled"] is False
        assert status["offline_allow_enabled"] is False
        assert status["local_inference_provider"] == "UNAVAILABLE"


def test_offline_policy_endpoints_require_sync_auth(tmp_path):
    settings = settings_for(tmp_path)
    with TestClient(create_app(settings, clock=lambda: NOW)) as client:
        denied = client.get(f"/v1/offline/policies/{SUBJECT}")
        assert denied.status_code == 401

