from dataclasses import dataclass, field
import math
import re
from typing import Protocol

from app.services.inference_contracts import (
    FaceDetectionResult,
    FacePADResult,
    ChallengeValidationResult,
    IdentityEmbeddingResult,
    IdentityVerificationResult,
    InferenceOutput,
    ModelStatus,
    ModelReasonCode,
    PhysicalMaskResult,
    VideoTemporalResult,
)


ModelState = ModelStatus


@dataclass(frozen=True)
class ComponentOutput:
    state: ModelState
    label: str | None = None
    score: float | None = None
    matched: bool | None = None
    model_name: str | None = None
    error_code: str | None = None
    model_version: str | None = None
    processing_time_ms: float | None = None
    reason_code: str | None = None
    attack_classification: str | None = None


@dataclass(frozen=True)
class EmbeddingOutput:
    state: ModelState
    embedding: tuple[float, ...] | None = None
    model_name: str | None = None
    error_code: str | None = None
    model_version: str | None = None
    processing_time_ms: float | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class VerificationInference:
    face_detection: ComponentOutput
    pad: ComponentOutput
    identity: ComponentOutput
    temporal: ComponentOutput
    mask: ComponentOutput
    errors: tuple[str, ...] = field(default_factory=tuple)

    def statuses(self) -> dict[str, str]:
        return {
            "face_detection": self.face_detection.state.value,
            "pad": self.pad.state.value,
            "identity": self.identity.state.value,
            "temporal": self.temporal.state.value,
            "mask": self.mask.state.value,
        }


def validate_component(output: ComponentOutput) -> None:
    if not isinstance(output, ComponentOutput) or not isinstance(output.state, ModelState):
        raise TypeError("Inference component does not match the output contract.")
    if output.score is not None and (
        not isinstance(output.score, (int, float))
        or isinstance(output.score, bool)
        or not math.isfinite(output.score)
        or not 0 <= output.score <= 1
    ):
        raise ValueError("Inference component score must be finite and between 0 and 1.")
    if output.state == ModelState.REAL and (
        not isinstance(output.model_name, str) or not output.model_name.strip()
    ):
        raise ValueError("Real inference output must identify its model.")
    if output.model_name is not None and not isinstance(output.model_name, str):
        raise TypeError("Inference model name must be a string.")
    if output.model_version is not None and not isinstance(output.model_version, str):
        raise TypeError("Inference model version must be a string.")
    if output.model_version is not None and (
        not output.model_version.strip() or len(output.model_version) > 128
    ):
        raise ValueError("Inference model version must be non-empty and at most 128 characters.")
    if output.processing_time_ms is not None and (
        not isinstance(output.processing_time_ms, (int, float))
        or isinstance(output.processing_time_ms, bool)
        or not math.isfinite(output.processing_time_ms)
        or output.processing_time_ms < 0
    ):
        raise ValueError("Inference processing time must be finite and non-negative.")
    if output.label is not None and not isinstance(output.label, str):
        raise TypeError("Inference component label must be a string.")
    if output.matched is not None and not isinstance(output.matched, bool):
        raise TypeError("Identity match output must be boolean.")
    if output.error_code is not None and (
        not isinstance(output.error_code, str)
        or re.fullmatch(r"[a-z0-9_]{1,64}", output.error_code) is None
    ):
        raise ValueError("Inference error code must be a non-sensitive code.")
    if output.reason_code is not None and (
        not isinstance(output.reason_code, str)
        or re.fullmatch(r"[a-z0-9_]{1,64}", output.reason_code) is None
    ):
        raise ValueError("Inference reason code must be a non-sensitive code.")
    if output.attack_classification is not None and (
        not isinstance(output.attack_classification, str)
        or re.fullmatch(r"[a-z0-9_]{1,64}", output.attack_classification) is None
    ):
        raise ValueError("Attack classification must be a non-sensitive code.")


def _validate_contract_output(output, result_type: type) -> None:
    if not isinstance(output, InferenceOutput):
        raise TypeError("AI module returned an object outside InferenceOutput.")
    if output.result is not None and not isinstance(output.result, result_type):
        raise TypeError(f"AI module result must be {result_type.__name__}.")


def _contract_component(
    output: InferenceOutput,
    *,
    label: str | None,
    score: float | None,
    matched: bool | None = None,
    attack_classification: str | None = None,
) -> ComponentOutput:
    unavailable_or_error = output.model_status in {ModelStatus.UNAVAILABLE, ModelStatus.ERROR}
    return ComponentOutput(
        state=ModelState(output.model_status.value),
        label=label,
        score=score,
        matched=matched,
        model_name=output.model_name,
        error_code=output.reason_code.value if unavailable_or_error else None,
        model_version=output.model_version,
        processing_time_ms=output.processing_time_ms,
        reason_code=output.reason_code.value,
        attack_classification=attack_classification,
    )


def aggregate_model_outputs(
    *,
    face_detection: InferenceOutput[FaceDetectionResult],
    pad: InferenceOutput[FacePADResult],
    identity: InferenceOutput[IdentityVerificationResult],
    mask: InferenceOutput[PhysicalMaskResult],
    media_kind: str,
    temporal: InferenceOutput[VideoTemporalResult] | None = None,
) -> VerificationInference:
    """Translate validated module contracts to the existing route/decision boundary."""
    if media_kind not in {"image", "video"}:
        raise ValueError("media_kind must be image or video.")
    _validate_contract_output(face_detection, FaceDetectionResult)
    _validate_contract_output(pad, FacePADResult)
    _validate_contract_output(identity, IdentityVerificationResult)
    _validate_contract_output(mask, PhysicalMaskResult)
    if temporal is not None:
        _validate_contract_output(temporal, VideoTemporalResult)

    face = face_detection.result
    face_label = (
        None
        if face is None
        else "no_face"
        if not face.face_detected
        else "face"
        if face.face_count == 1
        else "multiple_faces"
    )
    face_output = _contract_component(
        face_detection, label=face_label, score=face_detection.confidence
    )

    pad_result = pad.result
    if pad_result is None:
        pad_label, pad_score, attack_classification = None, pad.confidence, None
    elif pad_result.is_real is True:
        pad_label = "genuine"
        pad_score = 1 - pad_result.spoof_probability
        attack_classification = (
            pad_result.attack_classification.value if pad_result.attack_classification else None
        )
    elif pad_result.is_real is False:
        pad_label = "spoof"
        pad_score = pad_result.spoof_probability
        attack_classification = (
            pad_result.attack_classification.value if pad_result.attack_classification else None
        )
    else:
        pad_label, pad_score = "uncertain", pad.confidence
        attack_classification = (
            pad_result.attack_classification.value if pad_result.attack_classification else None
        )
    pad_output = _contract_component(
        pad,
        label=pad_label,
        score=pad_score,
        attack_classification=attack_classification,
    )

    identity_result = identity.result
    identity_output = _contract_component(
        identity,
        label=None if identity_result is None else "matched" if identity_result.matched else "mismatch",
        score=None if identity_result is None else identity_result.similarity_score,
        matched=None if identity_result is None else identity_result.matched,
    )

    if temporal is None:
        temporal_output = ComponentOutput(
            state=ModelState.UNAVAILABLE,
            model_name="video_temporal_analysis",
            model_version="not_requested",
            processing_time_ms=0,
            reason_code=ModelReasonCode.NOT_APPLICABLE.value,
        )
    else:
        temporal_result = temporal.result
        temporal_label = (
            None
            if temporal_result is None
            else "live"
            if temporal_result.is_live is True
            else "replay"
            if temporal_result.is_live is False
            else "uncertain"
        )
        temporal_score = (
            temporal.confidence
            if temporal_result is None or temporal_result.is_live is None
            else 1 - temporal_result.replay_probability
            if temporal_result.is_live
            else temporal_result.replay_probability
        )
        temporal_output = _contract_component(
            temporal, label=temporal_label, score=temporal_score
        )

    mask_result = mask.result
    mask_label = (
        None
        if mask_result is None or mask_result.mask_detected is None
        else "mask_attack"
        if mask_result.mask_detected
        else "no_mask"
    )
    mask_output = _contract_component(
        mask, label=mask_label, score=mask.confidence
    )

    outputs = (face_detection, pad, identity, mask) + (() if temporal is None else (temporal,))
    errors = tuple(
        output.reason_code.value
        for output in outputs
        if output.model_status == ModelStatus.ERROR
    )
    return VerificationInference(
        face_detection=face_output,
        pad=pad_output,
        identity=identity_output,
        temporal=temporal_output,
        mask=mask_output,
        errors=errors,
    )


def adapt_embedding_output(
    output: InferenceOutput[IdentityEmbeddingResult],
) -> EmbeddingOutput:
    """Map the typed enrollment contract to the existing encrypted-enrollment adapter."""
    _validate_contract_output(output, IdentityEmbeddingResult)
    result = output.result
    return EmbeddingOutput(
        state=ModelState(output.model_status.value),
        embedding=None if result is None else result.embedding,
        model_name=output.model_name,
        error_code=(
            output.reason_code.value
            if output.model_status in {ModelStatus.UNAVAILABLE, ModelStatus.ERROR}
            else None
        ),
        model_version=output.model_version,
        processing_time_ms=output.processing_time_ms,
        reason_code=output.reason_code.value,
    )


def adapt_challenge_output(
    output: InferenceOutput[ChallengeValidationResult],
) -> ComponentOutput:
    """Map typed server-side challenge validation to the existing challenge route."""
    _validate_contract_output(output, ChallengeValidationResult)
    result = output.result
    label = (
        None
        if result is None or result.passed is None
        else "passed"
        if result.passed
        else "failed"
    )
    unavailable_or_error = output.model_status in {ModelStatus.UNAVAILABLE, ModelStatus.ERROR}
    return ComponentOutput(
        state=ModelState(output.model_status.value),
        label=label,
        score=output.confidence,
        model_name=output.model_name,
        error_code=output.reason_code.value if unavailable_or_error else None,
        model_version=output.model_version,
        processing_time_ms=output.processing_time_ms,
        reason_code=output.reason_code.value,
    )


class InferenceProvider(Protocol):
    def create_embedding(self, media_path: str, media_type: str) -> EmbeddingOutput: ...

    def verify(
        self, media_path: str, media_type: str, enrollment_embedding: tuple[float, ...] | None
    ) -> VerificationInference: ...

    def validate_challenge(
        self, media_path: str, media_type: str, challenge: str
    ) -> ComponentOutput: ...


class UnavailableInferenceProvider:
    """Production default: reports absence instead of simulating model results."""

    def create_embedding(self, media_path: str, media_type: str) -> EmbeddingOutput:
        return EmbeddingOutput(state=ModelState.UNAVAILABLE, error_code="embedding_model_unavailable")

    def verify(
        self, media_path: str, media_type: str, enrollment_embedding: tuple[float, ...] | None
    ) -> VerificationInference:
        unavailable = ComponentOutput(state=ModelState.UNAVAILABLE, error_code="model_unavailable")
        return VerificationInference(
            face_detection=unavailable,
            pad=unavailable,
            identity=unavailable,
            temporal=unavailable,
            mask=unavailable,
        )

    def validate_challenge(
        self, media_path: str, media_type: str, challenge: str
    ) -> ComponentOutput:
        return ComponentOutput(
            state=ModelState.UNAVAILABLE, error_code="challenge_validator_unavailable"
        )
