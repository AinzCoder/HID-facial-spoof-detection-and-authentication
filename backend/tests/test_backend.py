import base64
import io
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.database.models import Enrollment, VerificationSession
from app.main import create_app
from app.schemas import DecisionOutcome
from app.services.decision import ReasonCode, decide
from app.services.inference import (
    ComponentOutput,
    EmbeddingOutput,
    ModelState,
    VerificationInference,
)
from app.services.security import encrypt_embedding


def image_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (16, 16), color=(120, 80, 60))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def mp4_bytes() -> bytes:
    return struct.pack(">I4s", 16, b"ftyp") + b"isom0000" + struct.pack(">I4s", 8, b"mdat")


def output(state=ModelState.MOCK, label=None, score=None, matched=None):
    return ComponentOutput(state=state, label=label, score=score, matched=matched, model_name="test-adapter")


class FixtureProvider:
    def __init__(self, mode="mock"):
        self.mode = mode
        self.last_media_path = None

    def create_embedding(self, media_path, media_type):
        self.last_media_path = media_path
        if self.mode == "unavailable":
            return EmbeddingOutput(ModelState.UNAVAILABLE)
        if self.mode == "contract-real":
            return EmbeddingOutput(
                ModelState.REAL,
                (0.1, 0.2),
                "test-contract",
                model_version="test-1",
                processing_time_ms=1.0,
                reason_code="ok",
            )
        return EmbeddingOutput(ModelState.MOCK, (0.1, 0.2), "test-only")

    def verify(self, media_path, media_type, enrollment_embedding):
        self.last_media_path = media_path
        if self.mode == "error":
            raise RuntimeError("deliberate test inference error")
        if self.mode == "invalid-score":
            invalid = output(ModelState.REAL, "genuine", float("nan"))
            return VerificationInference(invalid, invalid, invalid, invalid, invalid)
        if self.mode == "slow-uncertain":
            sleep(0.1)
        if self.mode in {"uncertain", "slow-uncertain"}:
            state = ModelState.REAL
            return VerificationInference(
                output(state, "face", 0.99),
                output(state, "uncertain", 0.55),
                output(state, "matched", 0.99, True),
                output(state, "live", 0.99),
                output(state, "no_mask", 0.99),
            )
        if self.mode == "spoof":
            return VerificationInference(
                output(ModelState.REAL, "face", 0.99),
                output(ModelState.REAL, "spoof", 0.99),
                output(ModelState.REAL, "matched", 0.99, True),
                output(ModelState.REAL, "live", 0.99),
                output(ModelState.REAL, "no_mask", 0.99),
            )
        if self.mode == "mismatch":
            return VerificationInference(
                output(ModelState.REAL, "face", 0.99),
                output(ModelState.REAL, "genuine", 0.99),
                output(ModelState.REAL, "mismatch", 0.1, False),
                output(ModelState.REAL, "live", 0.99),
                output(ModelState.REAL, "no_mask", 0.99),
            )
        if self.mode == "contract-real":
            state = ModelState.REAL
            return VerificationInference(
                output(state, "face", 0.99),
                output(state, "genuine", 0.99),
                output(state, "matched", 0.99, True),
                output(state, "live", 0.99),
                output(state, "no_mask", 0.99),
            )
        state = ModelState.MOCK
        return VerificationInference(
            output(state, "face", 0.99),
            output(state, "genuine", 0.99),
            output(state, "matched", 0.99, True),
            output(state, "live", 0.99),
            output(state, "no_mask", 0.99),
        )

    def validate_challenge(self, media_path, media_type, challenge):
        self.last_media_path = media_path
        if self.mode == "slow-uncertain":
            sleep(0.1)
        if self.mode in {"uncertain", "slow-uncertain", "contract-real"}:
            return output(ModelState.REAL, "passed", 0.99)
        return ComponentOutput(ModelState.UNAVAILABLE, error_code="not_implemented")


@pytest.fixture
def client(tmp_path):
    key = base64.urlsafe_b64encode(b"a" * 32).decode()
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        biometric_encryption_key=key,
    )
    with TestClient(create_app(settings, FixtureProvider())) as test_client:
        yield test_client


def create_session(client, subject_id="user-1"):
    response = client.post("/v1/sessions", json={"subject_id": subject_id})
    assert response.status_code == 201
    return response.json()["session_id"]


def post_media(client, path, media, content_type="image/png", field="media_file"):
    return client.post(path, files={field: ("untrusted-name.bin", media, content_type)})


def add_enrollment(client, subject_id="user-1"):
    key = client.app.state.settings.biometric_encryption_key
    ciphertext, nonce = encrypt_embedding((0.1, 0.2), key)
    with client.app.state.session_factory() as db:
        db.add(
            Enrollment(
                id="fixture-enrollment",
                subject_id=subject_id,
                encrypted_embedding=ciphertext,
                encryption_nonce=nonce,
                model_name="test-only",
                model_version="test-1",
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()


def test_app_startup_and_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    schema = client.get("/openapi.json").json()
    assert "DecisionResponse" in schema["components"]["schemas"]
    assert "ErrorResponse" in schema["components"]["schemas"]


def test_session_creation_and_database_persistence(client):
    response = client.post("/v1/sessions", json={"subject_id": "user-1"})
    assert response.status_code == 201
    body = response.json()
    assert body["session_id"]
    assert body["challenge"] in {"turn_head_left", "turn_head_right", "look_up"}
    with client.app.state.session_factory() as db:
        assert db.get(VerificationSession, body["session_id"]).subject_id == "user-1"


def test_session_request_validation(client):
    response = client.post("/v1/sessions", json={"subject_id": ""})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    missing_route = client.get("/route-that-does-not-exist")
    assert missing_route.status_code == 404
    assert missing_route.json()["code"] == "http_404"


def test_valid_image_and_mock_inference_never_allow(client):
    session_id = create_session(client)
    response = post_media(client, f"/v1/verifications/{session_id}", image_bytes())
    assert response.status_code == 200
    assert response.json()["outcome"] == DecisionOutcome.STEP_UP
    assert "MOCK" in response.json()["model_status"].values()
    assert response.json()["reason_code"] == ReasonCode.INCOMPLETE_OR_UNAVAILABLE_MODELS


def test_invalid_mime_and_malformed_image_rejected(client):
    session_id = create_session(client)
    wrong_mime = post_media(client, f"/v1/verifications/{session_id}", image_bytes(), "text/plain")
    assert wrong_mime.status_code == 415
    malformed = post_media(client, f"/v1/verifications/{session_id}", b"\x89PNG\r\n\x1a\nbroken")
    assert malformed.status_code == 400


def test_upload_size_limit_enforced(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'size.db').as_posix()}",
        max_image_bytes=16,
        max_video_bytes=32,
    )
    with TestClient(create_app(settings, FixtureProvider())) as test_client:
        session_id = create_session(test_client)
        response = post_media(test_client, f"/v1/verifications/{session_id}", image_bytes())
        assert response.status_code == 413


def test_video_container_validation_and_media_type_mismatch(client):
    session_id = create_session(client)
    valid_video = post_media(client, f"/v1/verifications/{session_id}", mp4_bytes(), "video/mp4")
    assert valid_video.status_code == 200
    assert valid_video.json()["outcome"] == "STEP_UP"

    another_session = create_session(client)
    invalid_video = post_media(client, f"/v1/verifications/{another_session}", b"\x00" * 40, "video/mp4")
    assert invalid_video.status_code == 415


def test_model_exception_is_recorded_as_error_and_steps_up(tmp_path):
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'error.db').as_posix()}")
    with TestClient(create_app(settings, FixtureProvider("error"))) as test_client:
        session_id = create_session(test_client)
        response = post_media(test_client, f"/v1/verifications/{session_id}", image_bytes())
        assert response.status_code == 200
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["model_status"]["inference"] == "ERROR"
        assert not Path(test_client.app.state.inference_provider.last_media_path).exists()


def test_invalid_model_output_is_not_persisted_as_a_success(tmp_path):
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'invalid.db').as_posix()}")
    with TestClient(create_app(settings, FixtureProvider("invalid-score"))) as test_client:
        session_id = create_session(test_client)
        response = post_media(test_client, f"/v1/verifications/{session_id}", image_bytes())
        assert response.status_code == 200
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["model_status"]["inference"] == "ERROR"


@pytest.mark.parametrize(
    ("mode", "expected", "reason"),
    [
        ("spoof", "DENY", "confirmed_spoof"),
        ("mismatch", "DENY", "identity_mismatch"),
        ("uncertain", "STEP_UP", "uncertain_model_result"),
        ("unavailable", "STEP_UP", "incomplete_or_unavailable_models"),
    ],
)
def test_verification_decision_statuses(tmp_path, mode, expected, reason):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / f'{mode}.db').as_posix()}",
        biometric_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
    )
    with TestClient(create_app(settings, FixtureProvider(mode))) as test_client:
        session_id = create_session(test_client)
        add_enrollment(test_client)
        response = post_media(test_client, f"/v1/verifications/{session_id}", image_bytes())
        assert response.status_code == 200
        assert response.json()["outcome"] == expected
        assert response.json()["reason_code"] == reason


@pytest.mark.parametrize(
    ("attack_output", "media_kind"),
    [
        ("mask_attack", "image"),
        ("replay", "video"),
    ],
)
def test_confirmed_mask_and_video_replay_are_denied(client, attack_output, media_kind):
    real = lambda label, score, matched=None: output(ModelState.REAL, label, score, matched)
    inference = VerificationInference(
        real("face", 0.99),
        real("genuine", 0.99),
        real("matched", 0.99, True),
        real(attack_output, 0.99) if media_kind == "video" else real("live", 0.99),
        real(attack_output, 0.99) if media_kind == "image" else real("no_mask", 0.99),
    )
    decision = decide(inference, client.app.state.settings, media_kind=media_kind)
    assert decision.outcome == DecisionOutcome.DENY
    assert decision.reason_code == ReasonCode.CONFIRMED_SPOOF


def test_real_model_contract_outputs_can_allow_in_decision_engine_only(client):
    real = lambda label, score, matched=None: output(ModelState.REAL, label, score, matched)
    inference = VerificationInference(
        real("face", 0.99),
        real("genuine", 0.99),
        real("matched", 0.99, True),
        real("live", 0.99),
        real("no_mask", 0.99),
    )
    decision = decide(inference, client.app.state.settings, media_kind="image")
    assert decision.outcome == DecisionOutcome.ALLOW
    with_error = VerificationInference(
        inference.face_detection,
        inference.pad,
        inference.identity,
        inference.temporal,
        inference.mask,
        errors=("model_inference_failed",),
    )
    assert decide(with_error, client.app.state.settings, media_kind="image").outcome == DecisionOutcome.STEP_UP


def test_missing_or_mock_required_outputs_never_allow(client):
    settings = client.app.state.settings
    for state in (ModelState.UNAVAILABLE, ModelState.MOCK, ModelState.ERROR):
        component = output(state, "genuine", 1.0, True)
        inference = VerificationInference(component, component, component, component, component)
        assert decide(inference, settings, media_kind="image").outcome != DecisionOutcome.ALLOW
    valid = output(ModelState.REAL, "no_mask", 0.99)
    missing_mask = VerificationInference(
        output(ModelState.REAL, "face", 0.99),
        output(ModelState.REAL, "genuine", 0.99),
        output(ModelState.REAL, "matched", 0.99, True),
        output(ModelState.REAL, "live", 0.99),
        ComponentOutput(ModelState.UNAVAILABLE),
    )
    assert decide(missing_mask, settings, media_kind="image").outcome != DecisionOutcome.ALLOW
    missing_video_temporal = VerificationInference(
        output(ModelState.REAL, "face", 0.99),
        output(ModelState.REAL, "genuine", 0.99),
        output(ModelState.REAL, "matched", 0.99, True),
        ComponentOutput(ModelState.UNAVAILABLE),
        valid,
    )
    assert decide(missing_video_temporal, settings, media_kind="video").outcome != DecisionOutcome.ALLOW


def test_challenge_can_add_server_validated_liveness_but_not_override_spoof(client):
    genuine = FixtureProvider("contract-real").verify("", "image/png", None)
    challenge = output(ModelState.REAL, "passed", 0.99)
    assert decide(
        genuine,
        client.app.state.settings,
        media_kind="image",
        challenge_result=challenge,
        challenge_required=True,
    ).outcome == DecisionOutcome.ALLOW

    failed_challenge = output(ModelState.REAL, "failed", 0.99)
    assert decide(
        genuine,
        client.app.state.settings,
        media_kind="image",
        challenge_result=failed_challenge,
        challenge_required=True,
    ).outcome == DecisionOutcome.STEP_UP

    pad_inconclusive = VerificationInference(
        genuine.face_detection,
        output(ModelState.REAL, "uncertain", 0.55),
        genuine.identity,
        genuine.temporal,
        genuine.mask,
    )
    assert decide(
        pad_inconclusive,
        client.app.state.settings,
        media_kind="image",
        challenge_result=challenge,
        challenge_required=True,
    ).outcome == DecisionOutcome.STEP_UP

    pad_unavailable = VerificationInference(
        genuine.face_detection,
        output(ModelState.UNAVAILABLE, "genuine", 0.99),
        genuine.identity,
        genuine.temporal,
        genuine.mask,
    )
    assert decide(
        pad_unavailable,
        client.app.state.settings,
        media_kind="image",
        challenge_result=challenge,
        challenge_required=True,
    ).outcome == DecisionOutcome.STEP_UP

    spoofed = FixtureProvider("spoof").verify("", "image/png", None)
    assert decide(
        spoofed,
        client.app.state.settings,
        media_kind="image",
        challenge_result=challenge,
        challenge_required=True,
    ).outcome == DecisionOutcome.DENY


def test_malformed_required_model_output_cannot_allow(client):
    inference = FixtureProvider("contract-real").verify("", "image/png", None)
    malformed_pad = ComponentOutput(
        ModelState.REAL,
        label="genuine",
        score=0.99,
        model_name=None,
    )
    malformed = VerificationInference(
        inference.face_detection,
        malformed_pad,
        inference.identity,
        inference.temporal,
        inference.mask,
    )
    assert decide(malformed, client.app.state.settings, media_kind="image").outcome != DecisionOutcome.ALLOW


@pytest.mark.parametrize(
    ("temporal", "expected"),
    [
        (ComponentOutput(ModelState.UNAVAILABLE), DecisionOutcome.STEP_UP),
        (output(ModelState.MOCK, "live", 0.99), DecisionOutcome.STEP_UP),
        (output(ModelState.ERROR, "live", 0.99), DecisionOutcome.STEP_UP),
        (output(ModelState.REAL, "uncertain", 0.99), DecisionOutcome.STEP_UP),
        (output(ModelState.REAL, "live", 0.99), DecisionOutcome.ALLOW),
    ],
)
def test_video_requires_real_successful_temporal_output(client, temporal, expected):
    real = lambda label, score, matched=None: output(ModelState.REAL, label, score, matched)
    inference = VerificationInference(
        real("face", 0.99),
        real("genuine", 0.99),
        real("matched", 0.99, True),
        temporal,
        real("no_mask", 0.99),
    )
    assert decide(inference, client.app.state.settings, media_kind="video").outcome == expected


def test_challenge_expiry_and_reuse(client):
    session_id = create_session(client)
    post_media(client, f"/v1/verifications/{session_id}", image_bytes())
    with client.app.state.session_factory() as db:
        record = db.get(VerificationSession, session_id)
        record.challenge_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
    expired = post_media(
        client,
        f"/v1/challenges/{session_id}/response",
        image_bytes(),
        field="media_file",
    )
    assert expired.status_code == 410

    other_session = create_session(client)
    post_media(client, f"/v1/verifications/{other_session}", image_bytes())
    used = post_media(
        client,
        f"/v1/challenges/{other_session}/response",
        image_bytes(),
        field="media_file",
    )
    assert used.status_code == 200
    repeated = post_media(
        client,
        f"/v1/challenges/{other_session}/response",
        image_bytes(),
        field="media_file",
    )
    assert repeated.status_code == 409


def test_concurrent_challenge_reuse_is_atomically_rejected(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'challenge-race.db').as_posix()}",
        biometric_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
    )
    with TestClient(create_app(settings, FixtureProvider("slow-uncertain"))) as test_client:
        session_id = create_session(test_client)
        add_enrollment(test_client)
        post_media(test_client, f"/v1/verifications/{session_id}", image_bytes())
        route = f"/v1/challenges/{session_id}/response"
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(
                executor.map(
                    lambda _: post_media(test_client, route, image_bytes()),
                    range(2),
                )
            )
        assert sorted(response.status_code for response in responses) == [200, 409]


def test_concurrent_verification_submission_is_atomically_rejected(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'verification-race.db').as_posix()}",
        biometric_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
    )
    with TestClient(create_app(settings, FixtureProvider("slow-uncertain"))) as test_client:
        session_id = create_session(test_client)
        add_enrollment(test_client)
        route = f"/v1/verifications/{session_id}"
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(
                executor.map(
                    lambda _: post_media(test_client, route, image_bytes()),
                    range(2),
                )
            )
        assert sorted(response.status_code for response in responses) == [200, 409]


def test_server_validated_challenge_does_not_replace_inconclusive_pad(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'challenge.db').as_posix()}",
        biometric_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
    )
    with TestClient(create_app(settings, FixtureProvider("uncertain"))) as test_client:
        session_id = create_session(test_client)
        add_enrollment(test_client)
        initial = post_media(test_client, f"/v1/verifications/{session_id}", image_bytes())
        assert initial.json()["outcome"] == "STEP_UP"
        response = post_media(
            test_client,
            f"/v1/challenges/{session_id}/response",
            image_bytes(),
        )
        assert response.status_code == 200
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["challenge_status"] == "validated"


def test_biometric_allow_without_campus_schedule_steps_up_through_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'allow.db').as_posix()}",
        biometric_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
    )
    with TestClient(create_app(settings, FixtureProvider("contract-real"))) as test_client:
        session_id = create_session(test_client)
        add_enrollment(test_client)
        response = post_media(test_client, f"/v1/verifications/{session_id}", image_bytes())
        assert response.status_code == 200
        assert response.json()["outcome"] == "STEP_UP"
        assert response.json()["reason_code"] == "campus_schedule_missing"


def test_real_enrollment_contract_encrypts_embedding(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'enrollment.db').as_posix()}",
        biometric_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
    )
    media = image_bytes()
    with TestClient(create_app(settings, FixtureProvider("contract-real"))) as test_client:
        response = test_client.post(
            "/v1/enrollments",
            data={"subject_id": "enrolled-user"},
            files={"image": ("face.png", media, "image/png")},
        )
        assert response.status_code == 201
        with test_client.app.state.session_factory() as db:
            enrollment = db.query(Enrollment).filter_by(subject_id="enrolled-user").one()
            assert media not in enrollment.encrypted_embedding
            assert enrollment.encrypted_embedding != b""
        assert not Path(test_client.app.state.inference_provider.last_media_path).exists()


def test_enrollment_unavailable_does_not_store_face_data(client):
    response = client.post(
        "/v1/enrollments",
        data={"subject_id": "new-user"},
        files={"image": ("face.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 503
    with client.app.state.session_factory() as db:
        assert db.query(Enrollment).filter_by(subject_id="new-user").first() is None


def test_session_result_retrieval_and_duplicate_verification(client):
    session_id = create_session(client)
    first = post_media(client, f"/v1/verifications/{session_id}", image_bytes())
    assert first.status_code == 200
    duplicate = post_media(client, f"/v1/verifications/{session_id}", image_bytes())
    assert duplicate.status_code == 409
    result = client.get(f"/v1/verifications/{session_id}")
    assert result.status_code == 200
    assert len(result.json()["results"]) == 1


def test_expired_session_is_rejected(client):
    session_id = create_session(client)
    with client.app.state.session_factory() as db:
        record = db.get(VerificationSession, session_id)
        record.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
    response = client.get(f"/v1/verifications/{session_id}")
    assert response.status_code == 200
    rejected = post_media(client, f"/v1/verifications/{session_id}", image_bytes())
    assert rejected.status_code == 410
