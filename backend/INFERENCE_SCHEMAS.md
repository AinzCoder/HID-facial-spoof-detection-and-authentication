# Inference Schemas

The validated types are implemented in `app/services/inference_contracts.py`. These are in-process Python/Pydantic contracts; image bytes are not public API JSON responses.

## Common envelope

```python
InferenceOutput[ResultType](
    model_name="pad-mini-fasnet",
    model_version="2.1.0",
    model_status=ModelStatus.REAL,
    processing_time_ms=12.4,
    confidence=0.98,
    result=FacePADResult(
        is_real=False,
        spoof_probability=0.98,
        attack_classification=AttackClass.PRINTED_PHOTO,
    ),
    reason_code=ModelReasonCode.ATTACK_DETECTED,
)
```

`model_name`, `model_version`, `model_status`, `processing_time_ms`, `result`, and `reason_code` are required. `confidence` may be `None` when the result type has no calibrated confidence (for example an embedding). `REAL` and `MOCK` require a result payload; `UNAVAILABLE` and `ERROR` require `result=None` and `confidence=None`. All numeric values must be finite and in their documented ranges.

Permitted `ModelStatus` values: `REAL`, `MOCK`, `UNAVAILABLE`, `ERROR`.

`ModelReasonCode` values: `ok`, `no_face`, `multiple_faces`, `attack_detected`, `identity_mismatch`, `replay_detected`, `mask_detected`, `inconclusive`, `model_unavailable`, `inference_error`, `not_applicable`, `challenge_failed`.

## Shared image/frame input

`DecodedFrame(pixels: bytes, width: int, height: int, channels=3, color_space="RGB")` stores a contiguous row-major RGB uint8 buffer. Its validator requires exactly `width * height * 3` bytes. Face boxes and landmarks use normalized `[0,1]` coordinates relative to the source frame.

```python
face_input = DecodedFrame(
    pixels=rgb_bytes,  # len(rgb_bytes) == width * height * 3
    width=640,
    height=480,
)
face_output = face_detector.detect_and_align(face_input)
```

## Face detection and preprocessing

`FaceDetectionResult` fields: `face_detected`, `face_count`, optional `bounding_box`, optional `aligned_face_crop`, and `landmarks: tuple[FaceLandmark, ...]`. If a face is detected, the box, crop, and landmarks are required. A no-face result has `face_count=0` and no geometry.

```python
face_output = InferenceOutput[FaceDetectionResult](
    model_name="face-landmarker",
    model_version="1.4",
    model_status=ModelStatus.REAL,
    processing_time_ms=8.2,
    confidence=0.997,
    result=FaceDetectionResult(
        face_detected=True,
        face_count=1,
        bounding_box=BoundingBox(x=0.2, y=0.1, width=0.4, height=0.6),
        aligned_face_crop=aligned_rgb_frame,
        landmarks=(
            FaceLandmark(name="left_eye", x=0.31, y=0.32),
            FaceLandmark(name="right_eye", x=0.49, y=0.32),
        ),
    ),
    reason_code=ModelReasonCode.OK,
)
```

`face_output` above is the return value for `face_input`; its aligned crop is the input to PAD, identity, and crop-based mask analysis.

## Face anti-spoofing (PAD)

`FacePADResult`: `is_real: bool | None`, `spoof_probability: float` in `[0,1]`, and optional `attack_classification: AttackClass`.

```python
pad_input = face_output.result.aligned_face_crop
pad_output = InferenceOutput[FacePADResult](
    model_name="pad-model",
    model_version="3.0",
    model_status=ModelStatus.REAL,
    processing_time_ms=16.1,
    confidence=0.98,
    result=FacePADResult(
        is_real=False,
        spoof_probability=0.98,
        attack_classification=AttackClass.SCREEN_REPLAY,
    ),
    reason_code=ModelReasonCode.ATTACK_DETECTED,
)
# In a real adapter: pad_output = pad_model.predict(pad_input)
```

`AttackClass` values are `printed_photo`, `screen_replay`, `prerecorded_video_replay`, `physical_mask`, and `other`. Classification is optional; emit it only when the model supports that class.

## Identity enrollment and verification

Enrollment returns `IdentityEmbeddingResult(embedding: tuple[float, ...])`; embeddings must be non-empty and finite. Verification returns `IdentityVerificationResult(similarity_score, matched)`. The backend compares the model score with configured policy thresholds as well as requiring a real module result.

```python
identity_enrollment_input = face_output.result.aligned_face_crop
enrollment_output = InferenceOutput[IdentityEmbeddingResult](
    model_name="arcface",
    model_version="r100-v1",
    model_status=ModelStatus.REAL,
    processing_time_ms=11.2,
    confidence=None,
    result=IdentityEmbeddingResult(embedding=tuple(embedding_values)),
    reason_code=ModelReasonCode.OK,
)

identity_verification_input = (
    face_output.result.aligned_face_crop,
    enrollment_output.result.embedding,
)
verification_output = InferenceOutput[IdentityVerificationResult](
    model_name="arcface",
    model_version="r100-v1",
    model_status=ModelStatus.REAL,
    processing_time_ms=9.7,
    confidence=0.94,
    result=IdentityVerificationResult(similarity_score=0.94, matched=True),
    reason_code=ModelReasonCode.OK,
)
# In a real adapter:
# verification_output = identity_model.verify(*identity_verification_input)
```

## Video temporal analysis

`VideoAnalysisInput.frames` is an ordered tuple of `TimestampedFrame(timestamp_ms, frame)`, at least two and at most 512 sampled frames. Timestamps must strictly increase. `VideoTemporalResult` contains `replay_probability`, `is_live: bool | None`, and a short reason.

```python
video_input = VideoAnalysisInput(
    frames=(
        TimestampedFrame(timestamp_ms=0, frame=frame_0),
        TimestampedFrame(timestamp_ms=33, frame=frame_1),
    )
)
video_output = InferenceOutput[VideoTemporalResult](
    model_name="temporal-pad",
    model_version="0.8.2",
    model_status=ModelStatus.REAL,
    processing_time_ms=83.0,
    confidence=0.96,
    result=VideoTemporalResult(
        replay_probability=0.96,
        is_live=False,
        reason="replayed sequence",
    ),
    reason_code=ModelReasonCode.REPLAY_DETECTED,
)
# In a real adapter: video_output = temporal_model.analyze(video_input)
```

## Physical-mask analysis

`MaskAnalysisInput` accepts an `aligned_face_crop`, ordered `frames`, or both. `PhysicalMaskResult.mask_detected` is `True`, `False`, or `None` when inconclusive. The envelope `confidence` is the detector's calibrated confidence for that result.

```python
mask_input = MaskAnalysisInput(aligned_face_crop=face_output.result.aligned_face_crop)
mask_output = InferenceOutput[PhysicalMaskResult](
    model_name="face-mask-detector",
    model_version="1.2",
    model_status=ModelStatus.REAL,
    processing_time_ms=14.5,
    confidence=0.97,
    result=PhysicalMaskResult(mask_detected=True),
    reason_code=ModelReasonCode.MASK_DETECTED,
)
# In a real adapter: mask_output = mask_model.detect_mask(mask_input)
```

This output must come from a dedicated validated mask detector or a PAD capability validated for masks. Do not equate “PAD ran” with “mask analysis ran.”
