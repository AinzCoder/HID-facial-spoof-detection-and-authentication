import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas import DecisionOutcome
from app.services.decision import decide
from app.services.inference import (
    ModelState,
    adapt_challenge_output,
    adapt_embedding_output,
    aggregate_model_outputs,
)
from app.services.inference_contracts import (
    AttackClass,
    BoundingBox,
    ChallengeValidationResult,
    DecodedFrame,
    FaceDetectionResult,
    FacePADResult,
    FaceLandmark,
    IdentityEmbeddingResult,
    IdentityVerificationResult,
    InferenceOutput,
    MaskAnalysisInput,
    ModelReasonCode,
    ModelStatus,
    PhysicalMaskResult,
    TimestampedFrame,
    VideoAnalysisInput,
    VideoTemporalResult,
)


MODEL_NAMES = {
    "face": "fake-face-detector",
    "pad": "fake-pad",
    "identity": "fake-arcface",
    "temporal": "fake-temporal",
    "mask": "fake-mask-detector",
}


def frame() -> DecodedFrame:
    return DecodedFrame(pixels=b"\x10\x20\x30" * 4, width=2, height=2)


def output(model_key, result, confidence=0.99, status=ModelStatus.REAL):
    return InferenceOutput(
        model_name=MODEL_NAMES[model_key],
        model_version="fake-1.0",
        model_status=status,
        processing_time_ms=1.25,
        confidence=confidence if status in {ModelStatus.REAL, ModelStatus.MOCK} else None,
        result=result if status in {ModelStatus.REAL, ModelStatus.MOCK} else None,
        reason_code=(
            ModelReasonCode.OK
            if status in {ModelStatus.REAL, ModelStatus.MOCK}
            else ModelReasonCode.MODEL_UNAVAILABLE
        ),
    )


class FakeAIProviders:
    """Contract fixtures only; none of these outputs represents a real model."""

    def __init__(self, scenario="genuine"):
        self.scenario = scenario
        self.source_frame = frame()
        self.crop = frame()

    def detect_and_align(self, image):
        return output(
            "face",
            FaceDetectionResult(
                face_detected=True,
                face_count=1,
                bounding_box=BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5),
                aligned_face_crop=self.crop,
                landmarks=(
                    FaceLandmark(name="left_eye", x=0.4, y=0.4),
                    FaceLandmark(name="right_eye", x=0.6, y=0.4),
                ),
            ),
        )

    def predict(self, aligned_face_crop):
        if self.scenario in {"printed_photo", "screen_replay"}:
            attack = (
                AttackClass.PRINTED_PHOTO
                if self.scenario == "printed_photo"
                else AttackClass.SCREEN_REPLAY
            )
            result = FacePADResult(
                is_real=False,
                spoof_probability=0.99,
                attack_classification=attack,
            )
            return output("pad", result)
        if self.scenario == "inconclusive":
            return output(
                "pad",
                FacePADResult(is_real=None, spoof_probability=0.5),
                confidence=0.5,
            )
        if self.scenario == "unavailable":
            return output("pad", None, status=ModelStatus.UNAVAILABLE)
        return output("pad", FacePADResult(is_real=True, spoof_probability=0.01))

    def enroll(self, aligned_face_crop):
        return output(
            "identity",
            IdentityEmbeddingResult(embedding=(0.12, -0.08, 0.41)),
            confidence=None,
        )

    def verify(self, aligned_face_crop, enrolled_embedding):
        mismatch = self.scenario == "identity_mismatch"
        similarity = 0.1 if mismatch else 0.99
        return output(
            "identity",
            IdentityVerificationResult(similarity_score=similarity, matched=not mismatch),
            confidence=similarity,
        )

    def analyze(self, video):
        replay = self.scenario == "video_replay"
        probability = 0.99 if replay else 0.01
        return output(
            "temporal",
            VideoTemporalResult(
                replay_probability=probability,
                is_live=not replay,
                reason="replayed sequence" if replay else "temporal liveness passed",
            ),
            confidence=probability if replay else 1 - probability,
        )

    def detect_mask(self, media):
        is_mask = self.scenario == "mask"
        return output(
            "mask",
            PhysicalMaskResult(mask_detected=is_mask),
            confidence=0.99,
        )


def run_fake_pipeline(scenario="genuine", media_kind="image"):
    modules = FakeAIProviders(scenario)
    detected = modules.detect_and_align(modules.source_frame)
    crop = detected.result.aligned_face_crop
    pad = modules.predict(crop)
    enrolled = adapt_embedding_output(modules.enroll(crop))
    identity = modules.verify(crop, enrolled.embedding)
    mask = modules.detect_mask(MaskAnalysisInput(aligned_face_crop=crop))
    temporal = None
    if media_kind == "video":
        temporal = modules.analyze(
            VideoAnalysisInput(
                frames=(
                    TimestampedFrame(timestamp_ms=0, frame=modules.source_frame),
                    TimestampedFrame(timestamp_ms=33, frame=modules.source_frame),
                )
            )
        )
    aggregate = aggregate_model_outputs(
        face_detection=detected,
        pad=pad,
        identity=identity,
        mask=mask,
        temporal=temporal,
        media_kind=media_kind,
    )
    decision = decide(aggregate, Settings(), media_kind=media_kind)
    return aggregate, decision


def test_genuine_face_contracts_and_identity_enrollment_map_to_allow():
    embedding = adapt_embedding_output(FakeAIProviders().enroll(frame()))
    assert embedding.state == ModelState.REAL
    assert embedding.embedding == (0.12, -0.08, 0.41)
    _, decision = run_fake_pipeline()
    assert decision.outcome == DecisionOutcome.ALLOW


@pytest.mark.parametrize("attack", ["printed_photo", "screen_replay"])
def test_pad_detects_photo_and_screen_attack_classes(attack):
    inference, decision = run_fake_pipeline(attack)
    assert inference.pad.attack_classification == attack
    assert decision.outcome == DecisionOutcome.DENY


def test_temporal_provider_detects_prerecorded_video_replay():
    inference, decision = run_fake_pipeline("video_replay", media_kind="video")
    assert inference.temporal.label == "replay"
    assert decision.outcome == DecisionOutcome.DENY


def test_dedicated_mask_provider_detects_physical_mask():
    inference, decision = run_fake_pipeline("mask")
    assert inference.mask.label == "mask_attack"
    assert decision.outcome == DecisionOutcome.DENY


def test_identity_mismatch_is_denied():
    _, decision = run_fake_pipeline("identity_mismatch")
    assert decision.outcome == DecisionOutcome.DENY


def test_inconclusive_model_output_steps_up():
    inference, decision = run_fake_pipeline("inconclusive")
    assert inference.pad.label == "uncertain"
    assert decision.outcome == DecisionOutcome.STEP_UP


def test_unavailable_model_steps_up():
    inference, decision = run_fake_pipeline("unavailable")
    assert inference.pad.state == ModelStatus.UNAVAILABLE
    assert decision.outcome == DecisionOutcome.STEP_UP


def test_required_mock_model_cannot_authorize():
    modules = FakeAIProviders()
    detection = modules.detect_and_align(modules.source_frame)
    crop = detection.result.aligned_face_crop
    mock_pad = output(
        "pad",
        FacePADResult(is_real=True, spoof_probability=0.01),
        status=ModelStatus.MOCK,
    )
    inference = aggregate_model_outputs(
        face_detection=detection,
        pad=mock_pad,
        identity=modules.verify(crop, (0.1, 0.2)),
        mask=modules.detect_mask(MaskAnalysisInput(aligned_face_crop=crop)),
        media_kind="image",
    )
    assert decide(inference, Settings(), media_kind="image").outcome != DecisionOutcome.ALLOW


@pytest.mark.parametrize("missing", ["face", "pad", "identity", "mask"])
def test_missing_required_real_module_never_authorizes(missing):
    modules = FakeAIProviders()
    detected = modules.detect_and_align(modules.source_frame)
    crop = detected.result.aligned_face_crop
    values = {
        "face_detection": detected,
        "pad": modules.predict(crop),
        "identity": modules.verify(crop, (0.1, 0.2)),
        "mask": modules.detect_mask(MaskAnalysisInput(aligned_face_crop=crop)),
    }
    key = "face_detection" if missing == "face" else missing
    values[key] = output(
        missing if missing != "face" else "face",
        None,
        status=ModelStatus.UNAVAILABLE,
    )
    inference = aggregate_model_outputs(**values, media_kind="image")
    assert decide(inference, Settings(), media_kind="image").outcome != DecisionOutcome.ALLOW


def test_video_without_temporal_result_never_authorizes():
    modules = FakeAIProviders()
    detected = modules.detect_and_align(modules.source_frame)
    crop = detected.result.aligned_face_crop
    inference = aggregate_model_outputs(
        face_detection=detected,
        pad=modules.predict(crop),
        identity=modules.verify(crop, (0.1, 0.2)),
        mask=modules.detect_mask(MaskAnalysisInput(aligned_face_crop=crop)),
        temporal=output("temporal", None, status=ModelStatus.UNAVAILABLE),
        media_kind="video",
    )
    assert decide(inference, Settings(), media_kind="video").outcome != DecisionOutcome.ALLOW


def test_malformed_model_envelopes_are_rejected():
    with pytest.raises(ValidationError):
        InferenceOutput(
            model_name="bad-model",
            model_version="test",
            model_status=ModelStatus.REAL,
            processing_time_ms=1,
            confidence=1.1,
            result=FacePADResult(is_real=True, spoof_probability=0.0),
            reason_code=ModelReasonCode.OK,
        )
    with pytest.raises(ValidationError):
        InferenceOutput[FacePADResult](
            model_name="bad-model",
            model_version="test",
            model_status=ModelStatus.REAL,
            processing_time_ms=1,
            confidence=0.99,
            result=None,
            reason_code=ModelReasonCode.OK,
        )


def test_malformed_provider_object_fails_aggregation():
    modules = FakeAIProviders()
    detected = modules.detect_and_align(modules.source_frame)
    crop = detected.result.aligned_face_crop
    with pytest.raises(TypeError, match="InferenceOutput"):
        aggregate_model_outputs(
            face_detection=detected,
            pad={"model_status": "REAL"},
            identity=modules.verify(crop, (0.1, 0.2)),
            mask=modules.detect_mask(MaskAnalysisInput(aligned_face_crop=crop)),
            media_kind="image",
        )


def test_challenge_contract_maps_to_existing_challenge_boundary():
    validated = InferenceOutput[ChallengeValidationResult](
        model_name="fake-challenge-validator",
        model_version="fake-1.0",
        model_status=ModelStatus.REAL,
        processing_time_ms=5.0,
        confidence=0.99,
        result=ChallengeValidationResult(passed=True),
        reason_code=ModelReasonCode.OK,
    )
    component = adapt_challenge_output(validated)
    assert component.label == "passed"
    assert component.state == ModelStatus.REAL


def test_video_frames_must_be_ordered_and_mask_input_must_have_a_source():
    with pytest.raises(ValidationError):
        VideoAnalysisInput(
            frames=(
                TimestampedFrame(timestamp_ms=20, frame=frame()),
                TimestampedFrame(timestamp_ms=10, frame=frame()),
            )
        )
    with pytest.raises(ValidationError):
        MaskAnalysisInput()
