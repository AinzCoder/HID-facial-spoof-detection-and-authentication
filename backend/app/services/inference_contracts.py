from enum import StrEnum
from math import isfinite
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelStatus(StrEnum):
    REAL = "REAL"
    MOCK = "MOCK"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class ModelReasonCode(StrEnum):
    OK = "ok"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    ATTACK_DETECTED = "attack_detected"
    IDENTITY_MISMATCH = "identity_mismatch"
    REPLAY_DETECTED = "replay_detected"
    MASK_DETECTED = "mask_detected"
    INCONCLUSIVE = "inconclusive"
    MODEL_UNAVAILABLE = "model_unavailable"
    INFERENCE_ERROR = "inference_error"
    NOT_APPLICABLE = "not_applicable"
    CHALLENGE_FAILED = "challenge_failed"


class DecodedFrame(BaseModel):
    """Contiguous uint8 RGB pixels; no NumPy dependency crosses this interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pixels: bytes = Field(repr=False)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: Literal[3] = 3
    color_space: Literal["RGB"] = "RGB"

    @model_validator(mode="after")
    def validate_pixel_buffer(self):
        if len(self.pixels) != self.width * self.height * self.channels:
            raise ValueError("RGB pixel buffer length must equal width * height * 3.")
        return self


class BoundingBox(BaseModel):
    """Face rectangle normalized to the source frame, with origin at top left."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("Normalized bounding box must stay inside the frame.")
        return self


class FaceLandmark(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class FaceDetectionResult(BaseModel):
    face_detected: bool
    face_count: int = Field(ge=0)
    bounding_box: BoundingBox | None = None
    aligned_face_crop: DecodedFrame | None = None
    landmarks: tuple[FaceLandmark, ...] = ()

    @model_validator(mode="after")
    def validate_face_payload(self):
        if self.face_detected != (self.face_count > 0):
            raise ValueError("face_detected must agree with face_count.")
        if self.face_detected and (
            self.bounding_box is None
            or self.aligned_face_crop is None
            or not self.landmarks
        ):
            raise ValueError("A detected face requires a box, aligned crop, and landmarks.")
        if not self.face_detected and (
            self.bounding_box is not None
            or self.aligned_face_crop is not None
            or self.landmarks
        ):
            raise ValueError("No-face output cannot contain face geometry or a crop.")
        return self


class AttackClass(StrEnum):
    PRINTED_PHOTO = "printed_photo"
    SCREEN_REPLAY = "screen_replay"
    PRERECORDED_VIDEO_REPLAY = "prerecorded_video_replay"
    PHYSICAL_MASK = "physical_mask"
    OTHER = "other"


class FacePADResult(BaseModel):
    is_real: bool | None
    spoof_probability: float = Field(ge=0, le=1)
    attack_classification: AttackClass | None = None


class IdentityEmbeddingResult(BaseModel):
    embedding: tuple[float, ...] = Field(min_length=1, repr=False)

    @model_validator(mode="after")
    def validate_embedding(self):
        if not all(isfinite(value) for value in self.embedding):
            raise ValueError("Embedding values must be finite.")
        return self


class IdentityVerificationResult(BaseModel):
    similarity_score: float = Field(ge=0, le=1)
    matched: bool


class TimestampedFrame(BaseModel):
    timestamp_ms: int = Field(ge=0)
    frame: DecodedFrame


class VideoAnalysisInput(BaseModel):
    frames: tuple[TimestampedFrame, ...] = Field(min_length=2, max_length=512)

    @model_validator(mode="after")
    def validate_frame_order(self):
        timestamps = [frame.timestamp_ms for frame in self.frames]
        if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
            raise ValueError("Video frames must be strictly ordered by timestamp.")
        return self


class VideoTemporalResult(BaseModel):
    replay_probability: float = Field(ge=0, le=1)
    is_live: bool | None
    reason: str = Field(min_length=1, max_length=256)


class MaskAnalysisInput(BaseModel):
    aligned_face_crop: DecodedFrame | None = None
    frames: tuple[TimestampedFrame, ...] = ()

    @model_validator(mode="after")
    def validate_input_source(self):
        if self.aligned_face_crop is None and not self.frames:
            raise ValueError("Mask analysis requires an aligned crop or video frames.")
        if self.frames:
            VideoAnalysisInput(frames=self.frames)
        return self


class PhysicalMaskResult(BaseModel):
    mask_detected: bool | None


class ChallengeValidationResult(BaseModel):
    passed: bool | None


ResultT = TypeVar("ResultT")


class InferenceOutput(BaseModel, Generic[ResultT]):
    """Shared validated output envelope for every independently supplied model."""

    model_name: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=128)
    model_status: ModelStatus
    processing_time_ms: float = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    result: ResultT | None
    reason_code: ModelReasonCode

    @model_validator(mode="after")
    def validate_status_payload(self):
        if not self.model_name.strip() or not self.model_version.strip():
            raise ValueError("model_name and model_version must be non-empty.")
        if self.model_status in {ModelStatus.REAL, ModelStatus.MOCK} and self.result is None:
            raise ValueError("REAL and MOCK outputs require a result payload.")
        if self.model_status in {ModelStatus.UNAVAILABLE, ModelStatus.ERROR}:
            if self.result is not None or self.confidence is not None:
                raise ValueError("UNAVAILABLE and ERROR outputs cannot claim a result or confidence.")
        if not isfinite(self.processing_time_ms):
            raise ValueError("processing_time_ms must be finite.")
        if self.confidence is not None and not isfinite(self.confidence):
            raise ValueError("confidence must be finite.")
        return self


class FaceDetectionProvider(Protocol):
    def detect_and_align(self, frame: DecodedFrame) -> InferenceOutput[FaceDetectionResult]: ...


class FaceAntiSpoofingProvider(Protocol):
    def predict(self, aligned_face_crop: DecodedFrame) -> InferenceOutput[FacePADResult]: ...


class FaceIdentityProvider(Protocol):
    def enroll(self, aligned_face_crop: DecodedFrame) -> InferenceOutput[IdentityEmbeddingResult]: ...

    def verify(
        self,
        aligned_face_crop: DecodedFrame,
        enrolled_embedding: tuple[float, ...],
    ) -> InferenceOutput[IdentityVerificationResult]: ...


class VideoTemporalAnalysisProvider(Protocol):
    def analyze(self, video: VideoAnalysisInput) -> InferenceOutput[VideoTemporalResult]: ...


class PhysicalMaskDetectionProvider(Protocol):
    def detect_mask(self, media: MaskAnalysisInput) -> InferenceOutput[PhysicalMaskResult]: ...


class ChallengeValidationProvider(Protocol):
    def validate(
        self, media: MaskAnalysisInput, challenge: str
    ) -> InferenceOutput[ChallengeValidationResult]: ...
