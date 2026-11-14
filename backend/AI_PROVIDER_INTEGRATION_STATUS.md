# AI Provider Integration Status

**Audit scope:** read-only inspection of the backend source, tests, configuration, integration documentation, model/checkpoint locations, and the active virtual environment. No model was downloaded or initialized. The only file created for this audit is this report.

## 1. Executive Summary

The backend has a typed integration boundary, a fail-closed decision engine, upload handling, enrollment persistence, and test fixtures for synthetic provider behavior. It does **not** contain an operational AI provider or model implementation for any of the six audited components.

The default provider, `UnavailableInferenceProvider` in [app/services/inference.py](./app/services/inference.py), returns `UNAVAILABLE`. `create_app()` in [app/main.py](./app/main.py) uses it unless an `InferenceProvider` is injected. Tests inject fake providers, including fixtures that deliberately label synthetic values `REAL`; those values validate API and policy behavior only.

No model-loading code, model checkpoint, or configured model path was found in the application. The top-level `models/` directory and backend model-weight search were empty. The checked virtual environment does not have OpenCV, MediaPipe, PyTorch, ONNX Runtime, InsightFace, or TensorFlow installed. No real inference, model accuracy, or per-model latency was verified.

| Runtime classification | Count |
|---|---:|
| REAL | 0 |
| MOCK (runtime providers) | 0 |
| UNAVAILABLE | 6 |
| PARTIAL | 0 |

Test-only fake providers exist and are identified in the coverage section; they are not registered by the default application. Challenge session orchestration is partially implemented, but the **challenge-validation AI provider** is unavailable.

## 2. Overall Provider Status Table

| Provider | Contract Exists | Real Implementation | Model Loads | Real Inference | Tests | Status |
|---|---|---|---|---|---|---|
| Face detection and alignment | Yes: `FaceDetectionProvider` | No; protocol plus test fake | No loader or checkpoint found | No | Synthetic contract and decision tests | **UNAVAILABLE** |
| Face anti-spoofing / PAD | Yes: `FaceAntiSpoofingProvider` | No; protocol plus test fake | No loader or checkpoint found | No | Synthetic print/screen/inconclusive tests | **UNAVAILABLE** |
| Identity enrollment and verification | Yes: `FaceIdentityProvider` | No ArcFace provider; embedding encryption/storage exists | No loader or checkpoint found | No | Synthetic embedding/match fixtures | **UNAVAILABLE** |
| Video temporal analysis / replay | Yes: `VideoTemporalAnalysisProvider` | No; protocol plus test fake; media intake does not decode video | No loader or checkpoint found | No | Synthetic timestamped frames and replay fixture | **UNAVAILABLE** |
| Physical mask attack detection | Yes: `PhysicalMaskDetectionProvider` | No; protocol plus test fake | No loader or checkpoint found | No | Synthetic mask fixture and decision tests | **UNAVAILABLE** |
| Challenge validation / active liveness | Yes: `ChallengeValidationProvider` and composite adapter method | No real validator; API workflow exists | No loader or checkpoint found | No | Synthetic pass/failure and API workflow fixtures | **UNAVAILABLE** |

Here, “MOCK” refers to the test doubles described below, not an installed runtime service. No test fake is wired into the module-level application `app`.

## 3. Detailed Provider Audit

The canonical protocols and Pydantic result types are in [app/services/inference_contracts.py](./app/services/inference_contracts.py). The shared envelope is `InferenceOutput[ResultT]`, with `model_name`, `model_version`, `model_status`, `processing_time_ms`, optional `confidence`, `result`, and `reason_code`. It validates status/result consistency and numeric ranges; it does not load or validate a model.

### 3.1 Face Detection and Alignment

- **Contract/interface:** `FaceDetectionProvider.detect_and_align(frame: DecodedFrame) -> InferenceOutput[FaceDetectionResult]`.
- **Implementation:** The application contains no concrete face-detection class/function. `aggregate_model_outputs()` maps a typed result into the decision boundary; `UnavailableInferenceProvider.verify()` returns unavailable components. `FakeAIProviders.detect_and_align()` in [tests/test_inference_contracts.py](./tests/test_inference_contracts.py) is a test-only fake.
- **Status:** **UNAVAILABLE**.
- **Model name/version:** None for an actual model. Documentation names such as `face-landmarker` are illustrative. The test fake identifies itself as `fake-face-detector` with `fake-1.0`.
- **Weights/loading:** No face checkpoint or loader/path found. No initialization success can be demonstrated. CPU/GPU selection is **NOT VERIFIED**; neither device execution nor fallback is implemented here.
- **Input:** `DecodedFrame`: `pixels: bytes`, row-major contiguous RGB, `width > 0`, `height > 0`, `channels=3`, `color_space="RGB"`. The validator checks `len(pixels) == width * height * 3`. The contract does not fix a model input resolution or implement resize/normalization.
- **Output:** `FaceDetectionResult(face_detected: bool, face_count: int, bounding_box: BoundingBox | None, aligned_face_crop: DecodedFrame | None, landmarks: tuple[FaceLandmark, ...])`. A detected face must include a normalized `[0,1]` box, aligned RGB crop, and landmarks; a no-face result must omit those values. Envelope confidence is optional and bounded to `[0,1]`.
- **Inference execution:** None in application code. `media.py` validates image bytes using Pillow but does not detect, crop, align, or convert an upload into `DecodedFrame`.
- **Latency:** **NOT MEASURED — REAL MODEL NOT AVAILABLE**.
- **Test coverage:** `test_genuine_face_contracts_and_identity_enrollment_map_to_allow`, `test_missing_required_real_module_never_authorizes`, and malformed-contract tests use synthetic 2×2 frame bytes and `FakeAIProviders`. They verify mapping/policy/schema only.
- **Limitations:** No OpenCV/MediaPipe provider, no real face detector, no real crop/landmark output, and no multiple-face model behavior verified on real imagery.

### 3.2 Face Anti-Spoofing / PAD

- **Contract/interface:** `FaceAntiSpoofingProvider.predict(aligned_face_crop: DecodedFrame) -> InferenceOutput[FacePADResult]`.
- **Implementation:** No concrete PAD implementation or model call exists. `aggregate_model_outputs()` maps typed `is_real` and `spoof_probability` to `genuine`, `spoof`, or `uncertain`; it does not perform PAD. `FakeAIProviders.predict()` simulates synthetic scenarios.
- **Status:** **UNAVAILABLE**.
- **Model name/version:** None for an actual model. `pad-mini-fasnet`, `pad-model`, and their versions in `INFERENCE_SCHEMAS.md` are examples, not evidence that MiniFASNet or another PAD model is present. The test fake is named `fake-pad`.
- **Weights/loading:** No MiniFASNet/equivalent checkpoint, loader, or configured path found. Loading is not implemented or verified. CPU/GPU support is **NOT VERIFIED**.
- **Input:** One aligned RGB `DecodedFrame`; width and height are positive but the contract sets no fixed PAD resolution, crop normalization, channel scaling, or model-specific preprocessing.
- **Output:** `FacePADResult(is_real: bool | None, spoof_probability: float in [0,1], attack_classification: AttackClass | None)`. Supported enum labels are `printed_photo`, `screen_replay`, `prerecorded_video_replay`, `physical_mask`, and `other`; the enum is a contract, not evidence of trained classes. The adapter maps `is_real=True` to `genuine` with score `1 - spoof_probability`, `False` to `spoof` with score `spoof_probability`, and `None` to `uncertain`.
- **Inference execution:** None in application code. No real attack classifier is loaded or invoked.
- **Latency:** **NOT MEASURED — REAL MODEL NOT AVAILABLE**.
- **Test coverage:** `test_pad_detects_photo_and_screen_attack_classes`, `test_inconclusive_model_output_steps_up`, `test_unavailable_model_steps_up`, `test_required_mock_model_cannot_authorize`, plus decision/challenge tests use fake outputs. These are not print/screen accuracy tests.
- **Limitations:** No model, class calibration, dataset evaluation, supported-attack evidence, or reliability evidence. The code must not infer that PAD covers physical masks unless that capability is separately validated.

### 3.3 Face Identity Enrollment and Verification

- **Contract/interface:** `FaceIdentityProvider.enroll(aligned_face_crop: DecodedFrame) -> InferenceOutput[IdentityEmbeddingResult]` and `verify(aligned_face_crop: DecodedFrame, enrolled_embedding: tuple[float, ...]) -> InferenceOutput[IdentityVerificationResult]`.
- **Implementation:** No ArcFace implementation exists. `adapt_embedding_output()` and `aggregate_model_outputs()` adapt results but do not create embeddings or compare faces. The backend does implement encrypted storage of a supplied embedding in [app/services/security.py](./app/services/security.py), and the API stores model name/version in [app/database/models.py](./app/database/models.py).
- **Status:** **UNAVAILABLE** for ArcFace inference; **PARTIAL plumbing only** for encryption, storage, and decision consumption.
- **Model name/version:** None for an actual identity model. `arcface` / `r100-v1` in the schema documentation is illustrative. Test fixtures return synthetic vectors and similarities.
- **Weights/loading:** No ArcFace checkpoint, model loader, or path found. CPU/GPU execution is **NOT VERIFIED**.
- **Input:** Enrollment: one aligned RGB `DecodedFrame`. Verification: aligned RGB crop plus a non-empty tuple of enrolled embedding floats.
- **Output:** Enrollment `IdentityEmbeddingResult(embedding: tuple[float, ...])` requires finite non-empty values. Verification `IdentityVerificationResult(similarity_score: float in [0,1], matched: bool)`. The decision engine independently applies configured match/mismatch thresholds.
- **Inference execution:** No real embedding generation or similarity model call. `encrypt_embedding()`/`decrypt_embedding()` use AES-GCM to protect stored vectors; cryptography is not an identity model.
- **Latency:** **NOT MEASURED — REAL MODEL NOT AVAILABLE**.
- **Test coverage:** `test_genuine_face_contracts_and_identity_enrollment_map_to_allow`, `test_identity_mismatch_is_denied`, `test_real_model_contract_can_allow_through_api_and_skip_challenge`, `test_real_enrollment_contract_encrypts_embedding`, and `test_enrollment_unavailable_does_not_store_face_data` use synthetic fixture vectors, statuses, and matches.
- **Limitations:** No ArcFace architecture, checkpoint, embedding normalization, model-specific similarity metric, real identity data, calibration, or accuracy evaluation. The API requires `REAL` enrollment before persisting, but a test fixture can supply that status.

### 3.4 Video Temporal Analysis / Replay Detection

- **Contract/interface:** `VideoTemporalAnalysisProvider.analyze(video: VideoAnalysisInput) -> InferenceOutput[VideoTemporalResult]`.
- **Implementation:** No temporal model or concrete provider exists. `media.py` performs shallow MP4/WebM container/signature checks only. It does not decode frames. `FakeAIProviders.analyze()` simulates a replay flag from a test scenario.
- **Status:** **UNAVAILABLE**.
- **Model name/version:** None for an actual model. `temporal-pad` is an illustrative schema example; `fake-temporal` is test-only.
- **Weights/loading:** No temporal checkpoint or loader/path found. CPU/GPU support is **NOT VERIFIED**.
- **Input:** `VideoAnalysisInput.frames` is a tuple of 2–512 `TimestampedFrame(timestamp_ms: int >= 0, frame: DecodedFrame)`; timestamps must strictly increase. Frames are row-major RGB byte buffers. No decoder, sampler, frame-rate policy, or fixed frame dimensions are implemented.
- **Output:** `VideoTemporalResult(replay_probability: float in [0,1], is_live: bool | None, reason: str)`. The adapter maps this to `live`, `replay`, or `uncertain`. For video, the decision engine requires a REAL `live` result above the configured threshold.
- **Inference execution:** None. No uploaded video is decoded into the contract input by the current application.
- **Latency:** **NOT MEASURED — REAL MODEL NOT AVAILABLE**.
- **Test coverage:** `test_temporal_provider_detects_prerecorded_video_replay`, `test_video_without_temporal_result_never_authorizes`, `test_video_requires_real_successful_temporal_output`, and `test_video_frames_must_be_ordered_and_mask_input_must_have_a_source` use fake outputs or synthetic frames. `test_video_container_validation_and_media_type_mismatch` uses a synthetic MP4 box byte sequence, not a decodable video.
- **Limitations:** No frame decoder, sampling/temporal tracking, replay model, real video sample, or replay accuracy measurement.

### 3.5 Physical Mask Attack Detection

- **Contract/interface:** `PhysicalMaskDetectionProvider.detect_mask(media: MaskAnalysisInput) -> InferenceOutput[PhysicalMaskResult]`.
- **Implementation:** No concrete mask provider exists. The adapter and decision boundary treat mask analysis as its own component; PAD output is not automatically converted to mask output. `FakeAIProviders.detect_mask()` returns a scenario-controlled boolean.
- **Status:** **UNAVAILABLE**.
- **Model name/version:** None for an actual model. `face-mask-detector` is only an example name; `fake-mask-detector` is test-only.
- **Weights/loading:** No mask checkpoint, loader, or path found. CPU/GPU support is **NOT VERIFIED**.
- **Input:** `MaskAnalysisInput` accepts an aligned RGB `DecodedFrame`, ordered timestamped frames, or both. Its validator requires at least one source and checks ordered frames through `VideoAnalysisInput`.
- **Output:** `PhysicalMaskResult(mask_detected: bool | None)`; calibrated confidence is in the common envelope. `None` represents inconclusive.
- **Inference execution:** None. No dedicated mask model is present or loaded.
- **Latency:** **NOT MEASURED — REAL MODEL NOT AVAILABLE**.
- **Test coverage:** `test_dedicated_mask_provider_detects_physical_mask`, `test_confirmed_mask_and_video_replay_are_denied`, and `test_video_frames_must_be_ordered_and_mask_input_must_have_a_source` use test-only fake output and synthetic inputs.
- **Limitations:** No dedicated detector, no mask-validated PAD capability demonstrated, and no real mask data or calibration. The engine requires an independently supplied REAL `no_mask` result to allow.

### 3.6 Challenge Validation / Active Liveness

- **Contract/interface:** `ChallengeValidationProvider.validate(media: MaskAnalysisInput, challenge: str) -> InferenceOutput[ChallengeValidationResult]`; the existing composite adapter method is `InferenceProvider.validate_challenge(media_path, media_type, challenge) -> ComponentOutput`.
- **Implementation:** The API implements session challenge assignment, expiry, one-time submission, upload validation, and server-side invocation of the injected adapter (`respond_to_challenge()` and `_safe_challenge()` in [app/routes/api.py](./app/routes/api.py)). `adapt_challenge_output()` maps the typed result. No actual provider evaluates head motion, gaze, blink, or liveness.
- **Status:** **UNAVAILABLE** for AI validation; challenge/session workflow is **PARTIAL**.
- **Model name/version:** None for an actual validator. `fake-challenge-validator` is used by a contract test only.
- **Weights/loading:** No challenge model/checkpoint/loader found. CPU/GPU support is **NOT VERIFIED**.
- **Input:** Server-uploaded media plus the assigned challenge. The typed module contract accepts `MaskAnalysisInput` (crop and/or ordered frames) and a challenge string. Current application code does not decode or analyze challenge media itself. `CHALLENGES` in `app/routes/api.py` excludes `BLINK`; assigned values are head-left, head-right, and look-up.
- **Output:** `ChallengeValidationResult(passed: bool | None)` in the common envelope. The adapter maps it to `passed`/`failed`/no label. A client-supplied completion flag is not accepted.
- **Inference execution:** None in production code. The API invokes whatever provider is injected; the default returns unavailable. Test fixtures synthesize `passed`.
- **Latency:** **NOT MEASURED — REAL MODEL NOT AVAILABLE**.
- **Test coverage:** `test_challenge_contract_maps_to_existing_challenge_boundary`, `test_challenge_can_add_server_validated_liveness_but_not_override_spoof`, `test_server_validated_challenge_does_not_replace_inconclusive_pad`, `test_challenge_expiry_and_reuse`, and concurrency tests. Challenge-result tests use fake adapters; expiry/reuse tests validate the API workflow, not liveness accuracy.
- **Limitations:** No real server-side challenge validator, motion/gaze/blink estimator, or validated challenge media pipeline. Challenge success cannot substitute for PAD in current code.

## 4. Model Weights and Loading Verification

- No model checkpoint files were found in the top-level `models/` directory or elsewhere in the backend outside the virtual environment. The `models/` and `uploads/` directories were empty when inventoried.
- No application model initialization/loading function, checkpoint path setting, `load_state_dict`, ONNX session, or model-construction code was found in `backend/app`.
- `backend/requirements.txt` lists FastAPI, SQLAlchemy, Boto3, Pillow, cryptography, and related API dependencies. It does not declare `opencv-python`, `mediapipe`, `torch`, `torchvision`, `onnxruntime`, `insightface`, `facenet-pytorch`, `tensorflow`, or `mxnet`.
- A non-invasive module availability check in `backend/venv` found `cv2`, `mediapipe`, `torch`, `torchvision`, `onnxruntime`, `insightface`, `facenet_pytorch`, `tensorflow`, and `mxnet` unavailable; `PIL` is available.
- **Architecture/framework:** No actual architecture/framework is selected or present for the six providers. MiniFASNet, ArcFace, OpenCV, and MediaPipe are referenced as requirements/examples, not active implementations.
- **Checkpoint validation/load success:** Not applicable: no model loader or weights were found. No model loading was attempted.
- **CPU/GPU:** **NOT VERIFIED**. The backend has no model device policy, CUDA selection, accelerator configuration, or real model dependency.
- **Packaging blocker:** Root `.dockerignore` excludes `models/`; `backend/Dockerfile` installs `backend/requirements.txt` and copies only `backend/app`. A future local `models/` checkpoint would not be included by the current Docker context/rules without an approved packaging change.

The model/version strings, confidence values, inference runtimes, and outputs shown in `INFERENCE_SCHEMAS.md` are illustrative examples. Test-only names such as `fake-pad`, `fake-arcface`, and `fake-temporal` are not model identities.

## 5. Input and Output Contract Summary

| Component | Input contract | Output contract |
|---|---|---|
| Face detection/alignment | `DecodedFrame`: RGB bytes, positive width/height, exactly 3 channels; no fixed resolution or preprocessing implemented | `InferenceOutput[FaceDetectionResult]`: face presence/count, normalized box, aligned RGB crop, landmarks, envelope confidence |
| PAD | One aligned RGB `DecodedFrame`; no fixed resolution or normalization implemented | `InferenceOutput[FacePADResult]`: `is_real`, spoof probability, optional attack classification |
| Identity | Aligned RGB crop; verification also receives enrolled embedding tuple | Enrollment embedding or `similarity_score` and `matched` |
| Temporal | 2–512 ordered timestamped RGB frames | Replay probability, optional live result, reason |
| Physical mask | Aligned crop and/or ordered video frames | Optional mask result and envelope confidence |
| Challenge validation | Uploaded challenge media represented as crop/frames plus assigned challenge string | Optional `passed` result and envelope confidence |

All contracts are in-process Python/Pydantic types, not HTTP JSON schemas for raw frame data. The application upload route provides a temporary file path and detected MIME type to the composite adapter. `media.py` validates JPEG/PNG/WebP bytes and performs Pillow image verification/dimension checks (maximum 40 million pixels); MP4/WebM handling is structural/shallow and does not decode video. The adapter is responsible for media decoding and preprocessing, but no production adapter currently does so.

## 6. Inference Latency and Performance

For **each of the six AI providers**:

> **NOT MEASURED — REAL MODEL NOT AVAILABLE**

There are no measured real-model preprocessing, inference, or postprocessing latencies. `processing_time_ms` in schemas and synthetic fixtures is a field/value, not evidence of an inference benchmark. Values in `INFERENCE_SCHEMAS.md` are examples.

`README.md` documents a separate **no-model API baseline**: 30 sequential TestClient requests using a 92-byte, 16×16 PNG; temporary SQLite; Windows/Python 3.12.10; default `UNAVAILABLE` provider. Recorded API wall time was 10.695 ms minimum, 12.297 ms median, 18.591 ms nearest-rank p95, and 19.210 ms maximum. Recorded `processing_ms` was 3.571 ms minimum, 3.952 ms median, 7.532 ms nearest-rank p95, and 8.206 ms maximum. The API interval includes local processing and persistence as described in README; it does **not** measure biometric inference or establish the two-second target.

## 7. Test Coverage Matrix

The test files are [tests/test_inference_contracts.py](./tests/test_inference_contracts.py) and [tests/test_backend.py](./tests/test_backend.py). A recent complete-suite run recorded **51 passed, 1 upstream Starlette/httpx deprecation warning**. This audit did not rerun tests. All AI/provider test paths use test doubles or synthetic data; no real model inference test exists.

| Test category | Relevant tests | Data/provider used | What it establishes |
|---|---|---|---|
| Contract/schema validation | `test_malformed_model_envelopes_are_rejected`; `test_malformed_provider_object_fails_aggregation`; `test_video_frames_must_be_ordered_and_mask_input_must_have_a_source`; `test_challenge_contract_maps_to_existing_challenge_boundary` | Pydantic objects, 2×2 synthetic RGB bytes | Contract constraints, mapping, ordering, and malformed-response rejection |
| Fake-provider behavior | `test_genuine_face_contracts_and_identity_enrollment_map_to_allow`; `test_pad_detects_photo_and_screen_attack_classes`; `test_temporal_provider_detects_prerecorded_video_replay`; `test_dedicated_mask_provider_detects_physical_mask`; `test_identity_mismatch_is_denied`; `test_inconclusive_model_output_steps_up`; `test_unavailable_model_steps_up` | `FakeAIProviders`, synthetic frames/embeddings/results | Expected mapping/decision behavior for scenarios selected by test code |
| Decision-engine tests | `test_required_mock_model_cannot_authorize`; `test_missing_required_real_module_never_authorizes`; `test_video_without_temporal_result_never_authorizes`; `test_real_model_contract_outputs_can_allow_in_decision_engine_only`; `test_missing_or_mock_required_outputs_never_allow`; `test_challenge_can_add_server_validated_liveness_but_not_override_spoof`; `test_malformed_required_model_output_cannot_allow`; `test_video_requires_real_successful_temporal_output` | Synthetic `ComponentOutput` values; some fixtures intentionally label results `REAL` | Fail-closed policy logic, not the truth of any model output |
| API integration tests | `test_valid_image_and_mock_inference_never_allow`; `test_video_container_validation_and_media_type_mismatch`; `test_model_exception_is_recorded_as_error_and_steps_up`; `test_invalid_model_output_is_not_persisted_as_a_success`; `test_server_validated_challenge_does_not_replace_inconclusive_pad`; `test_real_model_contract_can_allow_through_api_and_skip_challenge`; `test_real_enrollment_contract_encrypts_embedding`; `test_enrollment_unavailable_does_not_store_face_data` | `FixtureProvider`; generated 16×16 solid PNG; synthetic MP4 container bytes; temporary test databases | API routing, persistence, error handling, workflow, and response policy with injected fixtures |
| Real model inference | None | No actual model weights/provider | Not tested |
| Accuracy / attack validation | None | No real faces, attack dataset, or held-out set | No accuracy, APCER/BPCER, threshold calibration, or class coverage claim |
| Real-model performance | None | No weights/device/inference | No AI latency result |

The fixture mode named `contract-real` and the fake contract outputs set `ModelStatus.REAL` to exercise the real-status decision path. Their values remain synthetic and do not represent real inference.

## 8. Decision Engine Integration Audit

The policy is implemented in [app/services/decision.py](./app/services/decision.py); API orchestration is in [app/routes/api.py](./app/routes/api.py).

The **exact final ALLOW predicate** is:

```python
if face_ok and identity_ok and pad_ok and mask_ok and temporal_ok:
    return Decision(DecisionOutcome.ALLOW, ReasonCode.VERIFIED, ...)
```

with these preceding conditions:

- Invalid `media_kind`, malformed evidence/components, and non-empty `result.errors` fail to `STEP_UP`.
- Required `face_detection`, `pad`, `identity`, and `mask` component statuses must all be `REAL`.
- `face_ok`: label `face` and score at/above `face_detection_threshold`.
- `identity_ok`: label `matched`, `matched is True`, and score at/above `identity_match_threshold`.
- `pad_ok`: label `genuine` and score at/above `pad_genuine_threshold`.
- `mask_ok`: label `no_mask` and score at/above `mask_clear_threshold`.
- `temporal_ok`: always true for image; for video, temporal status must be `REAL`, label `live`, and score at/above `pad_genuine_threshold`.
- If `challenge_required` is true, `challenge_ok` must also pass its REAL-state and score/label checks before the final predicate.

Audit findings:

| Check | Finding |
|---|---|
| PAD independently passes | **Yes.** `pad_ok` is a separate required term in the final predicate. Challenge success cannot replace it. |
| Challenge cannot override confirmed spoof | **Yes.** Confirmed REAL PAD spoof, mask attack, challenge spoof, or video replay returns `DENY` before challenge requirement/final allow. |
| Identity mismatch cannot ALLOW | **Yes.** A sufficiently confident real mismatch returns `DENY`; any other non-matching result fails `identity_ok` and cannot satisfy ALLOW. |
| Required unavailable/MOCK/ERROR output | **Yes.** Required statuses must be `REAL`; `result.errors` also yields `STEP_UP`. |
| Malformed output | **Yes for the validated boundary.** `validate_component()` rejects malformed values before decision; malformed provider results are converted to error by `_safe_verify()`. |
| Inconclusive output | **Yes.** Unknown/uncertain labels fail the required component predicates and result in `STEP_UP`. |
| Video temporal output | **Yes.** Video requires REAL temporal `live` with sufficient score; unavailable, mock, error, replay, or uncertain results cannot ALLOW. |
| Physical mask | **Yes as a separate required component.** A REAL `no_mask` result at threshold is required independently of PAD; a confirmed `mask_attack` denies. Generic PAD evidence alone does not populate this component. |

**Documentation mismatch:** `README.md` and `DECISION_MATRIX.md` still say that a separately validated challenge may satisfy/add to the PAD liveness condition. That wording is stale relative to current `decision.py`, which requires `pad_ok` independently even after a challenge passes. This report does not modify those files.

## 9. Missing Implementations and Blockers

1. All six providers need concrete implementations and model ownership; no operational provider is included.
2. No model weights, loader, framework dependency, model version, model packaging, or CPU/GPU execution path exists.
3. Root `models/` is empty and excluded by `.dockerignore`; the Dockerfile currently copies only `backend/app`.
4. The composite provider must decode uploaded images/videos into the documented RGB frame contracts. Current video intake does not decode frames.
5. Real enrollment and verification require an ArcFace-compatible model, embedding policy, and calibration; current encryption/storage does not supply these.
6. PAD, attack classification, temporal replay, mask detection, and challenge actions require independent validated models/pipelines and representative evaluation data. Interface enums are not evidence that classes are detectable.
7. Test fixtures set synthetic results to `REAL`; no test currently loads a checkpoint or evaluates real faces/attacks.
8. No real-model performance or accuracy results exist. The README baseline is an unavailable-provider API overhead measurement only.
9. Runtime integration can be injected, but health reports `CONFIGURED_UNVERIFIED` for an injected provider rather than checking that its models load or are operational.
10. In the legacy `ComponentOutput` boundary, model version and processing time are optional. The canonical `InferenceOutput` requires them, but an injected composite provider can bypass it unless it consistently uses the typed adapter path.
11. Documentation in README/decision matrix needs to be aligned with the actual requirement that PAD pass independently of challenge.
12. Operational deployment gaps remain separate from AI integration: no authentication/authorization, rate limiting, bounded model queue, external managed database, or completed production AWS setup is established by this audit.

## 10. Recommended Integration Order

1. **Member 1 — Face detection, preprocessing, alignment:** implement `FaceDetectionProvider`; own image/video decoding as needed for the provider boundary, RGB conversion, face count, crop alignment, landmarks, model loading status, and contract/model-specific tests. Verify exactly-one-face, malformed image, multi-face, and crop geometry on real samples.
2. **Member 2 — Passive PAD and attack classification:** implement `FaceAntiSpoofingProvider` against the aligned crop; document the exact attack classes it supports, calibrate probabilities/thresholds on held-out bona fide and attack data, and never identify unsupported attacks. Do not claim physical mask capability without separate validation.
3. **Member 3 — ArcFace identity, backend orchestration, decision integration:** implement `FaceIdentityProvider` enrollment/verification; compose the member providers behind the existing `InferenceProvider`; call `aggregate_model_outputs()`, `adapt_embedding_output()`, and `adapt_challenge_output()`; inject it via `create_app(settings, inference_provider=...)`. Keep the existing `decide()` as the sole policy engine; do not make a parallel decision. Validate embedding lifecycle, errors, model metadata, and the fail-closed boundary.

Separate workstreams still required:

- **Video replay:** implement safe video decoding, server-side frame sampling/timestamps, and `VideoTemporalAnalysisProvider`; evaluate real prerecorded/live sequences.
- **Physical mask:** implement a dedicated, validated `PhysicalMaskDetectionProvider` or a PAD capability independently validated for masks; maintain its separate decision evidence.
- **Server-side challenge:** implement a real `ChallengeValidationProvider` for assigned head-turn/look-up media, bind inference to the server-assigned challenge, and validate expiry/reuse/failure behavior. Do not treat route/workflow completion as liveness validation.

After model owners select dependencies/weights, update packaging deliberately (including `.dockerignore`/Docker build strategy), test loading on target CPU/GPU, validate against a held-out dataset, and measure per-component plus end-to-end p50/p95 under target hardware and concurrency. Do not adjust thresholds or claim supported attack classes before calibration evidence exists.

## 11. Final Readiness Assessment

| Readiness area | Assessment |
|---|---|
| Backend architecture readiness | **Good prototype integration architecture.** Modular API/provider boundary, typed contracts, media intake, persistence, and standalone decision engine exist. |
| AI provider integration readiness | **Contract-ready, not model-integrated.** Protocols and adapters are defined; six operational AI providers are unavailable. |
| Real-time inference readiness | **Not demonstrated.** No real model latency, model startup, device, throughput, or production-concurrency result. The under-2-second target is not established. |
| Security / decision-policy readiness | **Fail-closed prototype policy is implemented and covered by synthetic tests.** This does not certify models or constitute full production security; the docs mismatch and deployment controls remain. |
| Demonstrated detection accuracy | **None.** No real model weights, real inference, attack dataset evaluation, or accuracy metrics were found. |

**Bottom line:** this backend is a testable integration scaffold, not an operational biometric anti-spoof system. The application cannot demonstrate face detection, PAD, ArcFace identity, replay detection, physical-mask detection, or active-liveness validation until real providers, dependencies, and weights are supplied, loaded, and validated.
