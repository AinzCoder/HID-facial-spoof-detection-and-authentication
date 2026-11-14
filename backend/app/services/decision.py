from dataclasses import dataclass
from enum import StrEnum
import math

from app.config import Settings
from app.schemas import DecisionOutcome
from app.services.inference import (
    ComponentOutput,
    ModelState,
    VerificationInference,
    validate_component,
)


class ReasonCode(StrEnum):
    CONFIRMED_SPOOF = "confirmed_spoof"
    FACE_NOT_DETECTED = "face_not_detected"
    IDENTITY_MISMATCH = "identity_mismatch"
    VERIFIED = "verified"
    INCOMPLETE_OR_UNAVAILABLE_MODELS = "incomplete_or_unavailable_models"
    UNCERTAIN_MODEL_RESULT = "uncertain_model_result"
    CHALLENGE_REQUIRED = "challenge_required"
    CHALLENGE_FAILED = "challenge_failed"
    CAMPUS_SCHEDULE_MISSING = "campus_schedule_missing"
    CAMPUS_SCHEDULE_DISABLED = "campus_schedule_disabled"
    OUTSIDE_ACCESS_SCHEDULE = "outside_access_schedule"
    CAMPUS_SCHEDULE_EVALUATION_ERROR = "campus_schedule_evaluation_error"
    OFFLINE_DISABLED = "offline_disabled"
    OFFLINE_ALLOW_DISABLED = "offline_allow_disabled"
    OFFLINE_POLICY_MISSING = "offline_policy_missing"
    OFFLINE_POLICY_INVALID = "offline_policy_invalid"
    OFFLINE_POLICY_EXPIRED = "offline_policy_expired"
    OFFLINE_POLICY_CLOCK_SKEW = "offline_policy_clock_skew"
    OFFLINE_POLICY_DISABLED = "offline_policy_disabled"
    OFFLINE_POLICY_NOT_PERMITTED = "offline_policy_not_permitted"
    OFFLINE_POLICY_SUBJECT_MISMATCH = "offline_policy_subject_mismatch"
    OFFLINE_SIGNING_KEY_UNAVAILABLE = "offline_signing_key_unavailable"
    OFFLINE_STORAGE_ERROR = "offline_storage_error"
    OFFLINE_CHALLENGE_FAILED = "offline_challenge_failed"


@dataclass(frozen=True)
class Decision:
    outcome: DecisionOutcome
    reason_code: ReasonCode
    reason: str


def _valid_score(output: ComponentOutput, minimum: float) -> bool:
    return (
        output.state == ModelState.REAL
        and isinstance(output.score, (int, float))
        and not isinstance(output.score, bool)
        and math.isfinite(output.score)
        and 0 <= output.score <= 1
        and output.score >= minimum
    )


def _confirmed_spoof(output: ComponentOutput, threshold: float, attack_label: str) -> bool:
    return (
        output.state == ModelState.REAL
        and output.label == attack_label
        and _valid_score(output, threshold)
    )


def decide(
    result: VerificationInference,
    settings: Settings,
    *,
    media_kind: str,
    challenge_result: ComponentOutput | None = None,
    challenge_required: bool = False,
) -> Decision:
    if media_kind not in {"image", "video"}:
        return Decision(
            DecisionOutcome.STEP_UP,
            ReasonCode.INCOMPLETE_OR_UNAVAILABLE_MODELS,
            "The media kind is invalid.",
        )
    if not isinstance(result, VerificationInference) or not isinstance(result.errors, tuple) or not all(
        isinstance(error, str) for error in result.errors
    ):
        return Decision(
            DecisionOutcome.STEP_UP,
            ReasonCode.INCOMPLETE_OR_UNAVAILABLE_MODELS,
            "The inference adapter returned malformed verification evidence.",
        )
    outputs = (
        result.face_detection,
        result.pad,
        result.identity,
        result.temporal,
        result.mask,
    ) + (() if challenge_result is None else (challenge_result,))
    try:
        for output in outputs:
            validate_component(output)
    except (TypeError, ValueError):
        return Decision(
            DecisionOutcome.STEP_UP,
            ReasonCode.INCOMPLETE_OR_UNAVAILABLE_MODELS,
            "The inference adapter returned malformed model output.",
        )

    if _confirmed_spoof(result.pad, settings.pad_spoof_threshold, "spoof") or _confirmed_spoof(
        result.mask, settings.pad_spoof_threshold, "mask_attack"
    ):
        return Decision(DecisionOutcome.DENY, ReasonCode.CONFIRMED_SPOOF, "A configured model reported a confirmed spoof.")
    if challenge_result is not None and _confirmed_spoof(
        challenge_result, settings.pad_spoof_threshold, "spoof"
    ):
        return Decision(DecisionOutcome.DENY, ReasonCode.CONFIRMED_SPOOF, "A configured model reported a confirmed spoof.")
    if media_kind == "video" and _confirmed_spoof(
        result.temporal, settings.pad_spoof_threshold, "replay"
    ):
        return Decision(DecisionOutcome.DENY, ReasonCode.CONFIRMED_SPOOF, "Temporal analysis reported a confirmed replay.")

    if result.face_detection.label == "no_face" and _valid_score(
        result.face_detection, settings.face_detection_threshold
    ):
        return Decision(DecisionOutcome.DENY, ReasonCode.FACE_NOT_DETECTED, "No face was detected by the configured model.")

    if (
        result.identity.state == ModelState.REAL
        and result.identity.label == "mismatch"
        and result.identity.matched is False
        and isinstance(result.identity.score, (int, float))
        and not isinstance(result.identity.score, bool)
        and math.isfinite(result.identity.score)
        and 0 <= result.identity.score <= 1
        and result.identity.score <= settings.identity_mismatch_threshold
    ):
        return Decision(DecisionOutcome.DENY, ReasonCode.IDENTITY_MISMATCH, "The identity model reported a mismatch.")

    if result.errors:
        return Decision(
            DecisionOutcome.STEP_UP,
            ReasonCode.INCOMPLETE_OR_UNAVAILABLE_MODELS,
            "The inference adapter reported an error.",
        )

    challenge_ok = (
        challenge_result is not None
        and challenge_result.label == "passed"
        and _valid_score(challenge_result, settings.challenge_pass_threshold)
    )
    model_outputs = (result.face_detection, result.pad, result.identity, result.mask)
    if any(output.state != ModelState.REAL for output in model_outputs):
        return Decision(
            DecisionOutcome.STEP_UP,
            ReasonCode.INCOMPLETE_OR_UNAVAILABLE_MODELS,
            "Required face, liveness, or identity model output is missing, mocked, or unavailable.",
        )

    face_ok = result.face_detection.label == "face" and _valid_score(
        result.face_detection, settings.face_detection_threshold
    )
    identity_ok = (
        result.identity.label == "matched"
        and result.identity.matched is True
        and _valid_score(result.identity, settings.identity_match_threshold)
    )
    pad_ok = (
        result.pad.label == "genuine"
        and _valid_score(result.pad, settings.pad_genuine_threshold)
    )
    mask_ok = result.mask.label == "no_mask" and _valid_score(
        result.mask, settings.mask_clear_threshold
    )
    temporal_ok = media_kind != "video" or (
        result.temporal.state == ModelState.REAL
        and result.temporal.label == "live"
        and _valid_score(result.temporal, settings.pad_genuine_threshold)
    )

    if challenge_required and not challenge_ok:
        return Decision(DecisionOutcome.STEP_UP, ReasonCode.CHALLENGE_REQUIRED, "A validated server-side challenge is required.")
    if face_ok and identity_ok and pad_ok and mask_ok and temporal_ok:
        return Decision(DecisionOutcome.ALLOW, ReasonCode.VERIFIED, "Required real model outputs passed configured thresholds.")
    return Decision(DecisionOutcome.STEP_UP, ReasonCode.UNCERTAIN_MODEL_RESULT, "Model outputs did not meet all configured verification requirements.")
