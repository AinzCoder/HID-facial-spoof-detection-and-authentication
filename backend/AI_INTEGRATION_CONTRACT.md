# AI Integration Contract

This contract adds typed model-module interfaces without replacing the backend's current route/provider boundary. The canonical Python definitions live in `app/services/inference_contracts.py`; `app/services/inference.py` contains the compatibility adapter into the existing `InferenceProvider` and `VerificationInference` used by `app/routes/api.py` and `app/services/decision.py`.

## Status and output envelope

Every model call returns `InferenceOutput[ResultType]` with:

- `model_name`, `model_version`
- `model_status`: `REAL`, `MOCK`, `UNAVAILABLE`, or `ERROR`
- `processing_time_ms`
- `confidence`: a model-produced value in `[0, 1]`, or `None` when not meaningful
- `result`: the domain-specific typed result, or `None` only for `UNAVAILABLE`/`ERROR`
- `reason_code`: a `ModelReasonCode`

The Pydantic schema validates ranges and status/result consistency. A `REAL` label is an assertion by the adapter, not model validation by this backend. Never label a mock, a fallback, an uninitialized model, or a model-loading failure `REAL`. A `MOCK` result is test-only and the decision engine will not authorize it.

Scores have component-specific meaning. PAD `spoof_probability` and temporal `replay_probability` are attack probabilities. Identity `similarity_score` is similarity. The envelope `confidence` is the module's calibrated confidence for its result; do not copy a score into it unless that is what the model output represents.

## Component interfaces

All interfaces are synchronous `Protocol`s; the existing routes dispatch adapter work through a thread pool. Use the exact types from `inference_contracts.py`.

```python
class FaceDetectionProvider(Protocol):
    def detect_and_align(self, frame: DecodedFrame) -> InferenceOutput[FaceDetectionResult]: ...

class FaceAntiSpoofingProvider(Protocol):
    def predict(self, aligned_face_crop: DecodedFrame) -> InferenceOutput[FacePADResult]: ...

class FaceIdentityProvider(Protocol):
    def enroll(self, aligned_face_crop: DecodedFrame) -> InferenceOutput[IdentityEmbeddingResult]: ...
    def verify(
        self, aligned_face_crop: DecodedFrame, enrolled_embedding: tuple[float, ...]
    ) -> InferenceOutput[IdentityVerificationResult]: ...

class VideoTemporalAnalysisProvider(Protocol):
    def analyze(self, video: VideoAnalysisInput) -> InferenceOutput[VideoTemporalResult]: ...

class PhysicalMaskDetectionProvider(Protocol):
    def detect_mask(self, media: MaskAnalysisInput) -> InferenceOutput[PhysicalMaskResult]: ...
```

### Input and output requirements

- **Face detector/preprocessor:** consumes a decoded row-major RGB `DecodedFrame`. Returns face count, normalized frame-relative bounding box, aligned RGB crop, normalized landmarks, and envelope confidence. `face_count > 1` is not eligible for `ALLOW`.
- **PAD:** consumes only the aligned face crop. Returns `is_real`, `spoof_probability`, and optional `attack_classification` (`printed_photo`, `screen_replay`, `prerecorded_video_replay`, `physical_mask`, `other`). It must not infer a supported attack class it was not trained/validated to distinguish.
- **Identity:** `enroll(crop)` returns a finite embedding; `verify(crop, enrolled_embedding)` returns similarity and the module's match decision. Enrollments are encrypted by the backend; do not log or persist the embedding in an AI module.
- **Temporal analyzer:** consumes at least two decoded, timestamped frames in strictly increasing order. Return replay probability, `is_live`, and a brief non-sensitive reason. Sample/limit frames in the adapter; never trust client timestamps without applying server-side bounds/ordering.
- **Mask detector:** consumes either an aligned crop or ordered frames. It returns `mask_detected` (`None` means inconclusive) and envelope confidence. This must be a dedicated validated mask model or a validated PAD capability. A generic PAD model is not automatically a mask detector.

The optional `ChallengeValidationProvider` contract consumes validated challenge media and the server-assigned challenge. It must evaluate the media server-side; a client `completed` flag is not an output.
`adapt_challenge_output()` maps its typed result to the existing challenge route contract.

## Existing backend adapter

Implement the existing `InferenceProvider` to adapt file paths to decoded input, call the modules, then use:

```python
aggregate_model_outputs(
    face_detection=face_output,
    pad=pad_output,
    identity=identity_verification_output,
    mask=mask_output,
    temporal=temporal_output,  # None only for still-image input
    media_kind="image",        # or "video"
)
```

This preserves per-module status, name, version, runtime, reason code, and supported PAD attack classification in the audited component evidence. `adapt_embedding_output()` maps the identity enrollment result into the existing encrypted enrollment path. Malformed output raises an exception; the current route boundary records an inference error and returns `STEP_UP`.

## Data handling

`DecodedFrame.pixels` is raw in-process pixel data and must not be logged, serialized to audit records, or retained after inference. Persist only encrypted enrollment embeddings and non-biometric audit metadata. Model-specific logs must omit image bytes, crops, embeddings, secrets, and exception messages containing request data.
