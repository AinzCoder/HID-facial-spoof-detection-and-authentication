import base64
import io
from datetime import datetime, timezone
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.database.models import CampusAccessSchedule, Enrollment
from app.main import create_app
from app.services.inference import ComponentOutput, EmbeddingOutput, ModelState, VerificationInference
from app.services.security import encrypt_embedding


ADMIN_TOKEN = "test-campus-schedule-admin-token-0123456789"
SUBJECT_ID = "schedule-user"
MONDAY_MORNING = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)


class ScheduleProvider:
    """Synthetic REAL-labelled fixture for access-policy tests only."""

    def __init__(self, mode: str = "genuine"):
        self.mode = mode

    def create_embedding(self, media_path, media_type):
        return EmbeddingOutput(
            state=ModelState.REAL,
            embedding=(0.1, 0.2),
            model_name="test-identity",
            model_version="test-only",
            processing_time_ms=1.0,
            reason_code="ok",
        )

    def verify(self, media_path, media_type, enrollment_embedding):
        component = lambda label, score, matched=None: ComponentOutput(
            state=ModelState.MOCK if self.mode == "mock" else ModelState.REAL,
            label=label,
            score=score,
            matched=matched,
            model_name="test-provider",
        )
        pad = (
            component("spoof", 0.99)
            if self.mode == "spoof"
            else component("uncertain", 0.55)
            if self.mode == "step-up"
            else component("genuine", 0.99)
        )
        identity = (
            component("mismatch", 0.1, False)
            if self.mode == "mismatch"
            else component("matched", 0.99, True)
        )
        return VerificationInference(
            component("face", 0.99),
            pad,
            identity,
            component("live", 0.99),
            component("no_mask", 0.99),
        )

    def validate_challenge(self, media_path, media_type, challenge):
        if self.mode == "challenge-pass":
            return ComponentOutput(
                state=ModelState.REAL,
                label="passed",
                score=0.99,
                model_name="test-challenge",
            )
        return ComponentOutput(
            state=ModelState.UNAVAILABLE,
            error_code="challenge_validator_unavailable",
        )


def make_client(
    tmp_path,
    *,
    now: datetime = MONDAY_MORNING,
    mode: str = "genuine",
    admin_token: str | None = ADMIN_TOKEN,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'campus-schedule.db').as_posix()}",
        biometric_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
        campus_schedule_admin_token=admin_token,
    )
    return TestClient(
        create_app(settings, ScheduleProvider(mode), clock=lambda: now)
    )


def add_enrollment(client, subject_id: str = SUBJECT_ID) -> None:
    key = client.app.state.settings.biometric_encryption_key
    ciphertext, nonce = encrypt_embedding((0.1, 0.2), key)
    with client.app.state.session_factory() as db:
        db.add(
            Enrollment(
                id=str(uuid.uuid4()),
                subject_id=subject_id,
                encrypted_embedding=ciphertext,
                encryption_nonce=nonce,
                model_name="test-only",
                model_version="test-only",
                created_at=MONDAY_MORNING,
            )
        )
        db.commit()


def schedule_payload(**overrides):
    payload = {
        "permitted_days": ["monday"],
        "access_start_time": "09:00:00",
        "access_end_time": "17:00:00",
        "timezone": "UTC",
        "enabled": True,
    }
    payload.update(overrides)
    return payload


def admin_headers(token: str = ADMIN_TOKEN):
    return {"Authorization": f"Bearer {token}"}


def put_schedule(client, **overrides):
    return client.put(
        f"/v1/admin/campus-schedules/{SUBJECT_ID}",
        headers=admin_headers(),
        json=schedule_payload(**overrides),
    )


def submit_verification(client, subject_id: str = SUBJECT_ID):
    session = client.post("/v1/sessions", json={"subject_id": subject_id})
    assert session.status_code == 201
    image = Image.new("RGB", (16, 16), color=(100, 120, 140))
    data = io.BytesIO()
    image.save(data, format="PNG")
    response = client.post(
        f"/v1/verifications/{session.json()['session_id']}",
        files={"media_file": ("face.png", data.getvalue(), "image/png")},
    )
    return response


def provision_schedule(client, **overrides):
    add_enrollment(client)
    response = put_schedule(client, **overrides)
    assert response.status_code == 200, response.text
    return response.json()


def test_real_biometric_inside_enabled_schedule_allows(tmp_path):
    with make_client(tmp_path) as client:
        provision_schedule(client)
        response = submit_verification(client)
        assert response.status_code == 200
        assert response.json()["outcome"] == "ALLOW"
        assert response.json()["reason_code"] == "verified"


def test_real_biometric_outside_schedule_steps_up(tmp_path):
    now = datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc)
    with make_client(tmp_path, now=now) as client:
        provision_schedule(client)
        response = submit_verification(client)
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["reason_code"] == "outside_access_schedule"


@pytest.mark.parametrize(
    ("mode", "reason"),
    [("spoof", "confirmed_spoof"), ("mismatch", "identity_mismatch")],
)
def test_biometric_deny_is_preserved_outside_schedule(tmp_path, mode, reason):
    now = datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc)
    with make_client(tmp_path, now=now, mode=mode) as client:
        provision_schedule(client)
        response = submit_verification(client)
        assert response.json()["outcome"] == "DENY"
        assert response.json()["reason_code"] == reason


def test_biometric_step_up_is_preserved_inside_schedule(tmp_path):
    with make_client(tmp_path, mode="step-up") as client:
        provision_schedule(client)
        response = submit_verification(client)
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["reason_code"] == "uncertain_model_result"


def test_disabled_schedule_steps_up(tmp_path):
    with make_client(tmp_path) as client:
        provision_schedule(client, enabled=False)
        response = submit_verification(client)
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["reason_code"] == "campus_schedule_disabled"


def test_missing_schedule_steps_up(tmp_path):
    with make_client(tmp_path) as client:
        add_enrollment(client)
        response = submit_verification(client)
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["reason_code"] == "campus_schedule_missing"


def test_weekday_restriction_steps_up(tmp_path):
    with make_client(tmp_path) as client:
        provision_schedule(client, permitted_days=["tuesday"])
        response = submit_verification(client)
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["reason_code"] == "outside_access_schedule"


def test_weekend_restriction_steps_up(tmp_path):
    saturday = datetime(2026, 1, 10, 10, 0, tzinfo=timezone.utc)
    with make_client(tmp_path, now=saturday) as client:
        provision_schedule(client, permitted_days=["monday", "tuesday", "wednesday", "thursday", "friday"])
        response = submit_verification(client)
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["reason_code"] == "outside_access_schedule"


def test_timezone_controls_local_day_and_time(tmp_path):
    # Tuesday 01:00 UTC is Monday 17:00 in Los Angeles in January.
    utc_tuesday = datetime(2026, 1, 6, 1, 0, tzinfo=timezone.utc)
    with make_client(tmp_path, now=utc_tuesday) as client:
        provision_schedule(
            client,
            permitted_days=["monday"],
            access_start_time="16:00:00",
            access_end_time="18:00:00",
            timezone="America/Los_Angeles",
        )
        response = submit_verification(client)
        assert response.json()["outcome"] == "ALLOW"


@pytest.mark.parametrize(
    ("now", "expected_outcome", "expected_reason"),
    [
        (
            datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc),
            "ALLOW",
            "verified",
        ),
        (
            datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc),
            "STEP_UP",
            "outside_access_schedule",
        ),
    ],
)
def test_schedule_start_is_inclusive_and_end_is_exclusive(
    tmp_path, now, expected_outcome, expected_reason
):
    with make_client(tmp_path, now=now) as client:
        provision_schedule(client)
        response = submit_verification(client)
        assert response.json()["outcome"] == expected_outcome
        assert response.json()["reason_code"] == expected_reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"permitted_days": []},
        {"permitted_days": ["monday", "monday"]},
        {"permitted_days": ["funday"]},
        {"access_start_time": "09:00:00", "access_end_time": "09:00:00"},
        {"access_start_time": "22:00:00", "access_end_time": "06:00:00"},
        {"access_start_time": "09:00:00+02:00"},
        {"timezone": "Mars/Olympus"},
    ],
)
def test_invalid_schedule_inputs_are_rejected(tmp_path, overrides):
    with make_client(tmp_path) as client:
        add_enrollment(client)
        response = put_schedule(client, **overrides)
        assert response.status_code == 422


def test_schedule_lookup_evaluation_failure_steps_up(tmp_path):
    with make_client(tmp_path) as client:
        provision_schedule(client)
        with client.app.state.session_factory() as db:
            schedule = db.get(CampusAccessSchedule, SUBJECT_ID)
            schedule.timezone = "Invalid/Timezone"
            db.commit()
        response = submit_verification(client)
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["reason_code"] == "campus_schedule_evaluation_error"


def test_passing_challenge_cannot_override_outside_schedule(tmp_path):
    now = datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc)
    with make_client(tmp_path, now=now, mode="challenge-pass") as client:
        provision_schedule(client)
        initial = submit_verification(client)
        assert initial.json()["outcome"] == "STEP_UP"
        session_id = initial.json()["session_id"]

        image = Image.new("RGB", (16, 16), color=(100, 120, 140))
        data = io.BytesIO()
        image.save(data, format="PNG")
        challenged = client.post(
            f"/v1/challenges/{session_id}/response",
            files={"media_file": ("challenge.png", data.getvalue(), "image/png")},
        )
        assert challenged.status_code == 200
        assert challenged.json()["outcome"] == "STEP_UP"
        assert challenged.json()["reason_code"] == "outside_access_schedule"


def test_admin_schedule_endpoints_require_configured_bearer_token(tmp_path):
    with make_client(tmp_path, admin_token=None) as client:
        disabled = put_schedule(client)
        assert disabled.status_code == 503

    with make_client(tmp_path / "authorized", admin_token=ADMIN_TOKEN) as client:
        add_enrollment(client)
        unauthorized = client.put(
            f"/v1/admin/campus-schedules/{SUBJECT_ID}",
            json=schedule_payload(),
        )
        assert unauthorized.status_code == 401
        invalid_token = client.put(
            f"/v1/admin/campus-schedules/{SUBJECT_ID}",
            headers=admin_headers("wrong-token"),
            json=schedule_payload(),
        )
        assert invalid_token.status_code == 401


def test_schedule_create_update_get_and_enable_disable_endpoints(tmp_path):
    with make_client(tmp_path) as client:
        add_enrollment(client)
        created = put_schedule(client)
        assert created.status_code == 200
        assert created.json()["subject_id"] == SUBJECT_ID
        assert created.json()["permitted_days"] == ["monday"]

        updated = put_schedule(client, permitted_days=["monday", "friday"])
        assert updated.status_code == 200
        assert updated.json()["permitted_days"] == ["monday", "friday"]

        fetched = client.get(
            f"/v1/admin/campus-schedules/{SUBJECT_ID}",
            headers=admin_headers(),
        )
        assert fetched.status_code == 200
        assert fetched.json() == updated.json()

        disabled = client.patch(
            f"/v1/admin/campus-schedules/{SUBJECT_ID}/status",
            headers=admin_headers(),
            json={"enabled": False},
        )
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False

        enabled = client.patch(
            f"/v1/admin/campus-schedules/{SUBJECT_ID}/status",
            headers=admin_headers(),
            json={"enabled": True},
        )
        assert enabled.status_code == 200
        assert enabled.json()["enabled"] is True


def test_schedule_management_requires_an_existing_enrollment(tmp_path):
    with make_client(tmp_path) as client:
        response = put_schedule(client)
        assert response.status_code == 404


def test_injected_biometric_mock_never_allows_with_schedule(tmp_path):
    with make_client(tmp_path, mode="mock") as client:
        provision_schedule(client)
        response = submit_verification(client)
        assert response.json()["outcome"] != "ALLOW"
