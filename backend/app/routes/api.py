import json
import logging
import math
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Request, UploadFile
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.models import (
    CampusAccessSchedule,
    Enrollment,
    VerificationResult,
    VerificationSession,
)
from app.schemas import (
    CampusAccessScheduleRequest,
    CampusAccessScheduleResponse,
    CampusScheduleEnabledRequest,
    ChallengeType,
    DecisionOutcome,
    DecisionResponse,
    EnrollmentResponse,
    ErrorResponse,
    OfflineDecisionResponse,
    OfflinePolicyClaims,
    OfflinePolicyPublishRequest,
    OfflinePolicyResponse,
    OfflineStatusResponse,
    OfflineSyncBatch,
    OfflineSyncResponse,
    OfflineSyncStatusResponse,
    SessionCreateRequest,
    SessionResponse,
    SignedOfflinePolicy,
    VerificationStatusResponse,
)
from app.services.decision import Decision, ReasonCode, decide
from app.services.campus_access import apply_campus_schedule
from app.services.inference import (
    ComponentOutput,
    EmbeddingOutput,
    ModelState,
    UnavailableInferenceProvider,
    VerificationInference,
    validate_component,
)
from app.services.media import TemporaryMedia, store_upload
from app.services.security import decrypt_embedding, encrypt_embedding
from app.services.schedule_auth import require_campus_schedule_admin
from app.services.offline import (
    OfflineEvaluation,
    apply_offline_policy,
    get_cached_policy,
    record_offline_event,
    store_signed_policy,
)
from app.services.offline_auth import require_offline_sync_auth
from app.services.offline_policy import (
    sign_policy,
    validate_key_pair,
    validate_public_key,
    verify_policy,
)
from app.services.offline_sync import (
    detect_backend_availability,
    receive_sync_batch,
    sync_pending_events,
)
from app.services.member1_face_provider import (
    decode_image_file,
    face_detection_component,
)

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/v1",
    responses={
        status_code: {"model": ErrorResponse}
        for status_code in (400, 401, 403, 404, 409, 410, 413, 415, 422, 500, 503)
    },
)
CHALLENGES = tuple(challenge for challenge in ChallengeType if challenge != ChallengeType.BLINK)


def get_db(request: Request):
    with request.app.state.session_factory() as session:
        yield session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _session_or_404(db: Session, session_id: str) -> VerificationSession:
    record = db.get(VerificationSession, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Verification session not found.")
    if _utcnow() >= _as_utc(record.expires_at):
        record.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Verification session has expired.")
    return record


def _reserve_verification(db: Session, record: VerificationSession, now: datetime) -> None:
    result = db.execute(
        update(VerificationSession)
        .where(
            VerificationSession.id == record.id,
            VerificationSession.verification_submitted_at.is_(None),
            VerificationSession.expires_at > now,
        )
        .values(verification_submitted_at=now, status="processing")
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        db.rollback()
        db.refresh(record)
        if now >= _as_utc(record.expires_at):
            record.status = "expired"
            db.commit()
            raise HTTPException(status_code=410, detail="Verification session has expired.")
        raise HTTPException(status_code=409, detail="Verification media has already been submitted for this session.")
    db.commit()
    db.refresh(record)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _serialize_output(output: ComponentOutput) -> dict:
    return {
        "state": output.state.value,
        "label": output.label,
        "score": output.score,
        "matched": output.matched,
        "model_name": output.model_name,
        "error_code": output.error_code,
        "model_version": output.model_version,
        "processing_time_ms": output.processing_time_ms,
        "reason_code": output.reason_code,
        "attack_classification": output.attack_classification,
    }


def _deserialize_output(value: dict) -> ComponentOutput:
    try:
        return ComponentOutput(
            state=ModelState(value["state"]),
            label=value.get("label"),
            score=value.get("score"),
            matched=value.get("matched"),
            model_name=value.get("model_name"),
            error_code=value.get("error_code"),
            model_version=value.get("model_version"),
            processing_time_ms=value.get("processing_time_ms"),
            reason_code=value.get("reason_code"),
            attack_classification=value.get("attack_classification"),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=500, detail="Stored verification evidence is invalid.") from exc


def _serialize_inference(result: VerificationInference) -> str:
    return json.dumps(
        {
            "face_detection": _serialize_output(result.face_detection),
            "pad": _serialize_output(result.pad),
            "identity": _serialize_output(result.identity),
            "temporal": _serialize_output(result.temporal),
            "mask": _serialize_output(result.mask),
            "errors": ["inference_error"] if result.errors else [],
        },
        allow_nan=False,
        separators=(",", ":"),
    )


def _deserialize_inference(raw: str) -> VerificationInference:
    try:
        data = json.loads(raw)
        return VerificationInference(
            face_detection=_deserialize_output(data["face_detection"]),
            pad=_deserialize_output(data["pad"]),
            identity=_deserialize_output(data["identity"]),
            temporal=_deserialize_output(data["temporal"]),
            mask=_deserialize_output(data["mask"]),
            errors=tuple(data.get("errors", [])),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=500, detail="Stored verification evidence is invalid.") from exc


def _status_payload(result: VerificationInference) -> dict[str, str]:
    statuses = result.statuses()
    if result.errors:
        statuses["inference"] = ModelState.ERROR.value
    return statuses


def _record_result(
    db: Session,
    session_record: VerificationSession,
    decision: Decision,
    inference: VerificationInference,
    media_kind: str,
    processing_ms: float,
) -> VerificationResult:
    row = VerificationResult(
        session_id=session_record.id,
        outcome=decision.outcome.value,
        reason_code=decision.reason_code.value,
        reason=decision.reason,
        model_status_json=json.dumps(_status_payload(inference), separators=(",", ":")),
        evidence_json=_serialize_inference(inference),
        media_kind=media_kind,
        processing_ms=processing_ms,
        created_at=_utcnow(),
    )
    session_record.status = decision.outcome.value.lower()
    session_record.latest_outcome = decision.outcome.value
    session_record.latest_reason_code = decision.reason_code.value
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _decision_response(
    session_id: str,
    decision: Decision,
    model_status: dict[str, str],
    processing_ms: float,
    session_record: VerificationSession,
) -> DecisionResponse:
    return DecisionResponse(
        session_id=session_id,
        outcome=decision.outcome,
        reason_code=decision.reason_code.value,
        reason=decision.reason,
        model_status=model_status,
        processing_ms=round(processing_ms, 3),
        challenge=ChallengeType(session_record.challenge_type),
        challenge_status=session_record.challenge_status,
    )


def _update_schedule_fields(
    schedule: CampusAccessSchedule, payload: CampusAccessScheduleRequest
) -> None:
    schedule.permitted_days = [day.value for day in payload.permitted_days]
    schedule.access_start_time = payload.access_start_time
    schedule.access_end_time = payload.access_end_time
    schedule.timezone = payload.timezone
    schedule.enabled = payload.enabled


@router.put(
    "/admin/campus-schedules/{subject_id}",
    response_model=CampusAccessScheduleResponse,
    dependencies=[Depends(require_campus_schedule_admin)],
)
def upsert_campus_schedule(
    subject_id: Annotated[str, Path(min_length=1, max_length=128)],
    payload: CampusAccessScheduleRequest,
    db: Session = Depends(get_db),
):
    enrollment_exists = db.scalar(
        select(Enrollment.id).where(Enrollment.subject_id == subject_id)
    )
    if enrollment_exists is None:
        raise HTTPException(status_code=404, detail="Enrolled user not found.")

    schedule = db.get(CampusAccessSchedule, subject_id)
    if schedule is None:
        schedule = CampusAccessSchedule(
            subject_id=subject_id,
            permitted_days=[],
            access_start_time=payload.access_start_time,
            access_end_time=payload.access_end_time,
            timezone=payload.timezone,
            enabled=payload.enabled,
        )
        db.add(schedule)
    _update_schedule_fields(schedule, payload)
    db.commit()
    db.refresh(schedule)
    return schedule


@router.get(
    "/admin/campus-schedules/{subject_id}",
    response_model=CampusAccessScheduleResponse,
    dependencies=[Depends(require_campus_schedule_admin)],
)
def get_campus_schedule(
    subject_id: Annotated[str, Path(min_length=1, max_length=128)],
    db: Session = Depends(get_db),
):
    schedule = db.get(CampusAccessSchedule, subject_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Campus access schedule not found.")
    return schedule


@router.patch(
    "/admin/campus-schedules/{subject_id}/status",
    response_model=CampusAccessScheduleResponse,
    dependencies=[Depends(require_campus_schedule_admin)],
)
def set_campus_schedule_enabled(
    subject_id: Annotated[str, Path(min_length=1, max_length=128)],
    payload: CampusScheduleEnabledRequest,
    db: Session = Depends(get_db),
):
    schedule = db.get(CampusAccessSchedule, subject_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Campus access schedule not found.")
    schedule.enabled = payload.enabled
    db.commit()
    db.refresh(schedule)
    return schedule


@router.get("/offline/status", response_model=OfflineStatusResponse)
def offline_status(request: Request):
    settings = request.app.state.settings
    provider = request.app.state.inference_provider
    verification_key_ready = False
    issuance_key_ready = False
    if settings.offline_signing_public_key:
        try:
            validate_public_key(settings.offline_signing_public_key)
            verification_key_ready = True
        except ValueError:
            pass
    if settings.offline_signing_private_key and settings.offline_signing_public_key:
        try:
            validate_key_pair(
                settings.offline_signing_private_key,
                settings.offline_signing_public_key,
            )
            issuance_key_ready = True
        except ValueError:
            pass
    return OfflineStatusResponse(
        enabled=settings.offline_enabled,
        offline_allow_enabled=settings.offline_allow_enabled,
        policy_verification_key_ready=verification_key_ready,
        policy_issuance_key_ready=issuance_key_ready,
        local_inference_provider=(
            ModelState.UNAVAILABLE.value
            if isinstance(provider, UnavailableInferenceProvider)
            else "CONFIGURED_UNVERIFIED"
        ),
        policy_ttl_seconds=settings.offline_policy_ttl_seconds,
        sync_configured=bool(
            settings.offline_sync_token and settings.offline_sync_backend_url
        ),
        backend_available=detect_backend_availability(
            settings.offline_backend_health_url,
            float(settings.offline_connectivity_timeout_seconds),
        ),
    )


@router.post(
    "/admin/offline/policies/{subject_id}",
    response_model=OfflinePolicyResponse,
    dependencies=[Depends(require_campus_schedule_admin)],
)
def publish_offline_policy(
    subject_id: Annotated[str, Path(min_length=1, max_length=128)],
    payload: OfflinePolicyPublishRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = request.app.state.settings
    if not settings.offline_enabled:
        raise HTTPException(status_code=503, detail="Offline mode is disabled.")
    if not settings.offline_signing_private_key or not settings.offline_signing_public_key:
        raise HTTPException(status_code=503, detail="Offline policy signing is not configured.")
    schedule = db.get(CampusAccessSchedule, subject_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Campus access schedule not found.")
    now = request.app.state.clock()
    if now.tzinfo is None or now.utcoffset() is None:
        raise HTTPException(status_code=503, detail="A timezone-aware server clock is required.")
    now = now.astimezone(timezone.utc)
    claims = OfflinePolicyClaims(
        policy_id=str(uuid.uuid4()),
        subject_id=subject_id,
        permitted_days=schedule.permitted_days,
        access_start_time=schedule.access_start_time,
        access_end_time=schedule.access_end_time,
        timezone=schedule.timezone,
        enabled=schedule.enabled,
        allow_offline=payload.allow_offline,
        issued_at=now,
        expires_at=now + timedelta(seconds=settings.offline_policy_ttl_seconds),
    )
    try:
        signed_policy = sign_policy(
            claims,
            settings.offline_signing_private_key,
            settings.offline_signing_public_key,
        )
        with request.app.state.offline_session_factory() as offline_db:
            store_signed_policy(
                offline_db, subject_id, signed_policy, now, settings
            )
    except ValueError as exc:
        logger.error("Offline policy could not be signed or cached (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Offline policy could not be issued.") from exc
    except SQLAlchemyError as exc:
        logger.error("Offline policy storage failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Offline policy could not be stored.") from exc
    return OfflinePolicyResponse(policy=signed_policy)


@router.put(
    "/offline/policies/{subject_id}",
    response_model=OfflinePolicyResponse,
    dependencies=[Depends(require_offline_sync_auth)],
)
def cache_offline_policy(
    subject_id: Annotated[str, Path(min_length=1, max_length=128)],
    policy: SignedOfflinePolicy,
    request: Request,
):
    try:
        with request.app.state.offline_session_factory() as offline_db:
            store_signed_policy(
                offline_db,
                subject_id,
                policy,
                request.app.state.clock(),
                request.app.state.settings,
            )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="Signed offline policy was rejected.") from exc
    except SQLAlchemyError as exc:
        logger.error("Offline policy cache failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Offline policy could not be cached.") from exc
    return OfflinePolicyResponse(policy=policy)


@router.get(
    "/offline/policies/{subject_id}",
    response_model=OfflinePolicyResponse,
    dependencies=[Depends(require_offline_sync_auth)],
)
def get_offline_policy(
    subject_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: Request,
):
    try:
        with request.app.state.offline_session_factory() as offline_db:
            policy = get_cached_policy(offline_db, subject_id)
            if policy is None:
                raise HTTPException(status_code=404, detail="No offline policy is cached.")
            verify_policy(policy, request.app.state.settings.offline_signing_public_key)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Cached offline policy integrity check failed.") from exc
    except SQLAlchemyError as exc:
        logger.error("Offline policy retrieval failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Offline policy is unavailable.") from exc
    return OfflinePolicyResponse(policy=policy)


@router.post(
    "/offline/events/sync",
    response_model=OfflineSyncResponse,
    dependencies=[Depends(require_offline_sync_auth)],
)
def receive_offline_events(
    batch: OfflineSyncBatch,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        accepted = receive_sync_batch(db, batch, request.app.state.clock())
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Offline event data conflicts with a prior event.") from exc
    return OfflineSyncResponse(accepted_event_ids=accepted)


@router.post(
    "/offline/sync",
    response_model=OfflineSyncStatusResponse,
    dependencies=[Depends(require_offline_sync_auth)],
)
def synchronize_pending_offline_events(request: Request):
    settings = request.app.state.settings
    try:
        return sync_pending_events(
            request.app.state.offline_session_factory,
            endpoint=settings.offline_sync_backend_url,
            token=settings.offline_sync_token,
            timeout_seconds=settings.offline_sync_timeout_seconds,
            now=request.app.state.clock(),
        )
    except SQLAlchemyError as exc:
        logger.error("Offline event queue access failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Offline event queue is unavailable.") from exc


@router.post(
    "/offline/verifications/{subject_id}",
    response_model=OfflineDecisionResponse,
)
async def verify_offline(
    subject_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: Request,
    media_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    now = request.app.state.clock()
    if now.tzinfo is not None and now.utcoffset() is not None:
        now = now.astimezone(timezone.utc)
    event_id = str(uuid.uuid4())
    media = await store_upload(
        media_file,
        request.app.state.settings.max_image_bytes,
        request.app.state.settings.max_video_bytes,
    )
    evaluation: OfflineEvaluation
    inference: VerificationInference
    try:
        enrollment = None
        try:
            enrollment_row = db.scalar(
                select(Enrollment).where(Enrollment.subject_id == subject_id)
            )
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Local enrollment lookup failed (%s)", type(exc).__name__)
            enrollment_row = None
        if enrollment_row is not None:
            try:
                enrollment = decrypt_embedding(
                    enrollment_row.encrypted_embedding,
                    enrollment_row.encryption_nonce,
                    request.app.state.settings.biometric_encryption_key,
                )
            except HTTPException as exc:
                logger.warning("Offline enrollment could not be loaded (%s)", exc.status_code)

        inference = await request.app.state.run_inference(
            _safe_verify, request, media, enrollment
        )
        if enrollment is None:
            inference = VerificationInference(
                inference.face_detection,
                inference.pad,
                ComponentOutput(
                    state=ModelState.UNAVAILABLE,
                    error_code="enrollment_not_found_or_unavailable",
                ),
                inference.temporal,
                inference.mask,
                inference.errors,
            )
        biometric_decision = decide(
            inference,
            request.app.state.settings,
            media_kind=media.media_kind,
        )
        policy = None
        policy_storage_error = False
        try:
            with request.app.state.offline_session_factory() as offline_db:
                policy = get_cached_policy(offline_db, subject_id)
        except (SQLAlchemyError, ValueError) as exc:
            logger.error("Offline policy retrieval failed (%s)", type(exc).__name__)
            policy_storage_error = True
        evaluation = apply_offline_policy(
            biometric_decision,
            subject_id,
            policy,
            now,
            request.app.state.settings,
        )
        if policy_storage_error and evaluation.decision.outcome == DecisionOutcome.ALLOW:
            evaluation = OfflineEvaluation(
                Decision(
                    DecisionOutcome.STEP_UP,
                    ReasonCode.OFFLINE_STORAGE_ERROR,
                    "The local offline policy store could not be read.",
                ),
                evaluation.policy_id,
            )
        try:
            record_offline_event(
                request.app.state.offline_session_factory,
                event_id=event_id,
                subject_id=subject_id,
                occurred_at=now,
                decision=evaluation.decision,
                policy_id=evaluation.policy_id,
            )
        except (SQLAlchemyError, ValueError) as exc:
            logger.error("Offline audit event could not be persisted (%s)", type(exc).__name__)
            if evaluation.decision.outcome == DecisionOutcome.ALLOW:
                evaluation = OfflineEvaluation(
                    Decision(
                        DecisionOutcome.STEP_UP,
                        ReasonCode.OFFLINE_STORAGE_ERROR,
                        "The offline audit event could not be stored.",
                    ),
                    evaluation.policy_id,
                )
        return OfflineDecisionResponse(
            event_id=event_id,
            subject_id=subject_id,
            outcome=evaluation.decision.outcome,
            reason_code=evaluation.decision.reason_code.value,
            reason=evaluation.decision.reason,
            policy_id=evaluation.policy_id,
            model_status=_status_payload(inference),
        )
    finally:
        media.cleanup()
        await media_file.close()


def _safe_verify(request: Request, media: TemporaryMedia, embedding) -> VerificationInference:
    try:
        result = request.app.state.inference_provider.verify(media.path, media.media_type, embedding)
        if not isinstance(result, VerificationInference):
            raise TypeError("Inference provider returned an invalid verification contract.")
        for component in (
            result.face_detection,
            result.pad,
            result.identity,
            result.temporal,
            result.mask,
        ):
            validate_component(component)
        if not isinstance(result.errors, tuple) or not all(isinstance(error, str) for error in result.errors):
            raise TypeError("Inference provider errors must be a tuple of strings.")
        face_provider = request.app.state.face_detection_provider
        if (
            media.media_kind == "image"
            and face_provider is not None
            and result.face_detection.state != ModelState.REAL
        ):
            frame = decode_image_file(media.path, media.media_type)
            face_output = face_provider.detect_and_align(frame)
            face_component = face_detection_component(face_output)
            validate_component(face_component)
            result = VerificationInference(
                face_component,
                result.pad,
                result.identity,
                result.temporal,
                result.mask,
                result.errors,
            )
        return result
    except Exception as exc:
        logger.error("Verification inference failed (%s)", type(exc).__name__)
        error = ComponentOutput(state=ModelState.ERROR, error_code="inference_error")
        return VerificationInference(error, error, error, error, error, errors=("inference_error",))


def _safe_enroll(request: Request, media: TemporaryMedia) -> EmbeddingOutput:
    try:
        result = request.app.state.inference_provider.create_embedding(media.path, media.media_type)
        if not isinstance(result, EmbeddingOutput) or not isinstance(result.state, ModelState):
            raise TypeError("Inference provider returned an invalid enrollment contract.")
        if result.state == ModelState.REAL and (
            not isinstance(result.model_name, str)
            or not result.model_name.strip()
            or not isinstance(result.model_version, str)
            or not result.model_version.strip()
            or not isinstance(result.embedding, tuple)
            or not result.embedding
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in result.embedding
            )
            or not isinstance(result.processing_time_ms, (int, float))
            or isinstance(result.processing_time_ms, bool)
            or not math.isfinite(result.processing_time_ms)
            or result.processing_time_ms < 0
        ):
            raise ValueError("Real enrollment output does not match the embedding contract.")
        return result
    except Exception as exc:
        logger.error("Enrollment inference failed (%s)", type(exc).__name__)
        return EmbeddingOutput(state=ModelState.ERROR, error_code="inference_error")


def _safe_challenge(request: Request, media: TemporaryMedia, challenge: str) -> ComponentOutput:
    try:
        result = request.app.state.inference_provider.validate_challenge(
            media.path, media.media_type, challenge
        )
        if not isinstance(result, ComponentOutput):
            raise TypeError("Inference provider returned an invalid challenge contract.")
        validate_component(result)
        return result
    except Exception as exc:
        logger.error("Challenge inference failed (%s)", type(exc).__name__)
        return ComponentOutput(state=ModelState.ERROR, error_code="inference_error")


@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(payload: SessionCreateRequest, request: Request, db: Session = Depends(get_db)):
    now = _utcnow()
    challenge = secrets.choice(CHALLENGES)
    record = VerificationSession(
        id=str(uuid.uuid4()),
        subject_id=payload.subject_id,
        status="created",
        challenge_type=challenge.value,
        challenge_expires_at=now + timedelta(seconds=request.app.state.settings.challenge_ttl_seconds),
        challenge_status="assigned",
        challenge_used=False,
        expires_at=now + timedelta(seconds=request.app.state.settings.session_ttl_seconds),
        created_at=now,
    )
    db.add(record)
    db.commit()
    return SessionResponse(
        session_id=record.id,
        status=record.status,
        challenge=challenge,
        challenge_expires_at=record.challenge_expires_at,
        expires_at=record.expires_at,
    )


@router.post("/enrollments", response_model=EnrollmentResponse, status_code=201)
async def create_enrollment(
    request: Request,
    subject_id: str = Form(min_length=1, max_length=128),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if db.scalar(select(Enrollment).where(Enrollment.subject_id == subject_id)) is not None:
        raise HTTPException(status_code=409, detail="An enrollment already exists for this subject.")
    media = await store_upload(image, request.app.state.settings.max_image_bytes, request.app.state.settings.max_video_bytes)
    try:
        if media.media_kind != "image":
            raise HTTPException(status_code=415, detail="Enrollment requires an image.")
        output = await request.app.state.run_inference(_safe_enroll, request, media)
        if output.state != ModelState.REAL or not output.embedding:
            raise HTTPException(
                status_code=503,
                detail=f"Real face embedding is unavailable ({output.state.value}).",
            )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in output.embedding
        ):
            raise HTTPException(status_code=422, detail="Inference returned an invalid face embedding.")
        ciphertext, nonce = encrypt_embedding(
            output.embedding, request.app.state.settings.biometric_encryption_key
        )
        row = Enrollment(
            id=str(uuid.uuid4()),
            subject_id=subject_id,
            encrypted_embedding=ciphertext,
            encryption_nonce=nonce,
            model_name=output.model_name or "unspecified-real-model",
            model_version=output.model_version or "unknown",
            created_at=_utcnow(),
        )
        db.add(row)
        db.commit()
        return EnrollmentResponse(
            subject_id=subject_id,
            status="enrolled",
            model_status=output.state.value,
            model_name=output.model_name or "unspecified-real-model",
            model_version=output.model_version or "unknown",
            processing_time_ms=output.processing_time_ms or 0,
        )
    finally:
        media.cleanup()
        await image.close()


@router.post("/verifications/{session_id}", response_model=DecisionResponse)
async def verify_session(
    session_id: str,
    request: Request,
    media_file: UploadFile = File(...),
    subject_id: str | None = Form(default=None, min_length=1, max_length=128),
    db: Session = Depends(get_db),
):
    record = _session_or_404(db, session_id)
    if record.verification_submitted_at is not None:
        raise HTTPException(status_code=409, detail="Verification media has already been submitted for this session.")
    effective_subject = subject_id or record.subject_id
    if record.subject_id and effective_subject != record.subject_id:
        raise HTTPException(status_code=409, detail="Subject does not match the session.")
    if effective_subject is not None:
        record.subject_id = effective_subject
    media = await store_upload(media_file, request.app.state.settings.max_image_bytes, request.app.state.settings.max_video_bytes)
    started = time.perf_counter()
    try:
        enrollment = None
        if effective_subject:
            enrollment_row = db.scalar(select(Enrollment).where(Enrollment.subject_id == effective_subject))
            if enrollment_row is not None:
                enrollment = decrypt_embedding(
                    enrollment_row.encrypted_embedding,
                    enrollment_row.encryption_nonce,
                    request.app.state.settings.biometric_encryption_key,
                )
        _reserve_verification(db, record, _utcnow())
        inference = await request.app.state.run_inference(_safe_verify, request, media, enrollment)
        if enrollment is None:
            inference = VerificationInference(
                inference.face_detection,
                inference.pad,
                ComponentOutput(state=ModelState.UNAVAILABLE, error_code="enrollment_not_found"),
                inference.temporal,
                inference.mask,
                inference.errors,
            )
        decision = decide(inference, request.app.state.settings, media_kind=media.media_kind)
        if decision.outcome == DecisionOutcome.ALLOW:
            schedule = (
                db.get(CampusAccessSchedule, effective_subject)
                if effective_subject is not None
                else None
            )
            decision = apply_campus_schedule(
                decision, schedule, request.app.state.clock()
            )
        processing_ms = (time.perf_counter() - started) * 1000
        result_row = _record_result(db, record, decision, inference, media.media_kind, processing_ms)
        return _decision_response(
            record.id,
            decision,
            json.loads(result_row.model_status_json),
            processing_ms,
            record,
        )
    finally:
        media.cleanup()
        await media_file.close()


@router.post("/challenges/{session_id}/response", response_model=DecisionResponse)
async def respond_to_challenge(
    session_id: str,
    request: Request,
    media_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    record = _session_or_404(db, session_id)
    if record.challenge_used:
        raise HTTPException(status_code=409, detail="This challenge has already been used.")
    if not record.verification_submitted_at:
        raise HTTPException(status_code=409, detail="Submit verification media before responding to the challenge.")
    previous = db.scalar(
        select(VerificationResult)
        .where(VerificationResult.session_id == session_id)
        .order_by(VerificationResult.created_at.desc())
    )
    if previous is None or previous.outcome != DecisionOutcome.STEP_UP.value:
        raise HTTPException(status_code=409, detail="Challenge is only accepted for a step-up decision.")
    now = _utcnow()
    if now >= _as_utc(record.challenge_expires_at):
        record.challenge_status = "expired"
        record.challenge_used = True
        db.commit()
        raise HTTPException(status_code=410, detail="Challenge has expired.")
    media = await store_upload(media_file, request.app.state.settings.max_image_bytes, request.app.state.settings.max_video_bytes)
    started = time.perf_counter()
    try:
        reservation_time = _utcnow()
        claim = db.execute(
            update(VerificationSession)
            .where(
                VerificationSession.id == record.id,
                VerificationSession.challenge_used.is_(False),
                VerificationSession.challenge_expires_at > reservation_time,
                VerificationSession.expires_at > reservation_time,
            )
            .values(challenge_used=True, challenge_status="submitted")
            .execution_options(synchronize_session=False)
        )
        if claim.rowcount != 1:
            db.rollback()
            db.refresh(record)
            if reservation_time >= _as_utc(record.expires_at):
                record.status = "expired"
                db.commit()
                raise HTTPException(status_code=410, detail="Verification session has expired.")
            if reservation_time >= _as_utc(record.challenge_expires_at):
                record.challenge_status = "expired"
                record.challenge_used = True
                db.commit()
                raise HTTPException(status_code=410, detail="Challenge has expired.")
            raise HTTPException(status_code=409, detail="This challenge has already been used.")
        db.commit()
        db.refresh(record)
        challenge_output = await request.app.state.run_inference(
            _safe_challenge, request, media, record.challenge_type
        )
        original = _deserialize_inference(previous.evidence_json)
        decision = decide(
            original,
            request.app.state.settings,
            media_kind=previous.media_kind,
            challenge_result=challenge_output,
            challenge_required=True,
        )
        if decision.outcome == DecisionOutcome.ALLOW:
            schedule = (
                db.get(CampusAccessSchedule, record.subject_id)
                if record.subject_id is not None
                else None
            )
            decision = apply_campus_schedule(
                decision, schedule, request.app.state.clock()
            )
        record.challenge_status = (
            "validated"
            if challenge_output.state == ModelState.REAL
            and challenge_output.label == "passed"
            else "unavailable"
            if challenge_output.state == ModelState.UNAVAILABLE
            else "failed"
        )
        evidence = VerificationInference(
            original.face_detection,
            original.pad,
            original.identity,
            original.temporal,
            original.mask,
            original.errors,
        )
        statuses = _status_payload(evidence)
        statuses["challenge"] = challenge_output.state.value
        processing_ms = (time.perf_counter() - started) * 1000
        row = VerificationResult(
            session_id=record.id,
            outcome=decision.outcome.value,
            reason_code=decision.reason_code.value,
            reason=decision.reason,
            model_status_json=json.dumps(statuses, separators=(",", ":")),
            evidence_json=_serialize_inference(evidence),
            media_kind=previous.media_kind,
            processing_ms=processing_ms,
            created_at=_utcnow(),
        )
        record.status = decision.outcome.value.lower()
        record.latest_outcome = decision.outcome.value
        record.latest_reason_code = decision.reason_code.value
        db.add(row)
        db.commit()
        return _decision_response(
            record.id, decision, statuses, processing_ms, record
        )
    finally:
        media.cleanup()
        await media_file.close()


@router.get("/verifications/{session_id}", response_model=VerificationStatusResponse)
def get_verification(session_id: str, db: Session = Depends(get_db)):
    record = db.get(VerificationSession, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Verification session not found.")
    results = db.scalars(
        select(VerificationResult)
        .where(VerificationResult.session_id == session_id)
        .order_by(VerificationResult.created_at)
    ).all()
    parsed = []
    for row in results:
        parsed.append(
            DecisionResponse(
                session_id=session_id,
                outcome=DecisionOutcome(row.outcome),
                reason_code=row.reason_code,
                reason=row.reason,
                model_status=json.loads(row.model_status_json),
                processing_ms=row.processing_ms,
                challenge=ChallengeType(record.challenge_type),
                challenge_status=record.challenge_status,
            )
        )
    return VerificationStatusResponse(
        session_id=record.id,
        status=record.status,
        latest_outcome=DecisionOutcome(record.latest_outcome) if record.latest_outcome else None,
        latest_reason_code=record.latest_reason_code,
        challenge=ChallengeType(record.challenge_type),
        challenge_status=record.challenge_status,
        challenge_expires_at=record.challenge_expires_at,
        expires_at=record.expires_at,
        results=parsed,
    )
