# Team Integration Guide

The route layer is already injectable: `create_app(settings, inference_provider)` accepts the existing `InferenceProvider`. Do not change API routes or decision policy to integrate model internals. Each AI owner implements their focused protocol from `app/services/inference_contracts.py`; one orchestrator implements the existing provider interface and translates outputs with `aggregate_model_outputs()`, `adapt_embedding_output()`, and `adapt_challenge_output()`.

## Ownership and module contract

| Owner | Protocol | Implementation responsibility |
|---|---|---|
| Face detection/preprocessing | `FaceDetectionProvider` | Decode frame input, detect face count, return normalized box, aligned RGB crop, landmarks, and calibrated confidence. |
| PAD | `FaceAntiSpoofingProvider` | Analyze the aligned crop; return realness/spoof probability and only supported attack classes. State which attack classes are not covered. |
| Identity | `FaceIdentityProvider` | Enroll a crop to a finite embedding; compare a verification crop with the server-supplied enrolled embedding and return similarity plus match decision. |
| Video temporal | `VideoTemporalAnalysisProvider` | Analyze timestamp-ordered sampled frames; report replay probability, liveness, and a short non-sensitive reason. |
| Mask (separate capability) | `PhysicalMaskDetectionProvider` | Return calibrated mask result from an aligned crop and/or appropriate frames. Do not claim this is provided by an unvalidated PAD model. |

The same team member may own several protocols, but each returned output must identify its model and version independently.

## Pipeline and safe parallelism

1. Receive uploaded media through the existing route.
2. Validate content and decode it in the adapter. `media.py` validates images and shallowly checks supported video containers; it does **not** decode video frames.
3. Detect faces in each required frame.
4. Align/preprocess the selected face; require exactly one face for an allow-capable path.
5. Run PAD on the aligned crop.
6. Run mask analysis on the crop or frames.
7. Run identity comparison on the crop and decrypted enrolled embedding. Enrollment instead calls identity `enroll`.
8. For video, run temporal analysis on ordered, server-decoded frames.
9. Aggregate validated outputs with `aggregate_model_outputs`.
10. Apply `decide()`; do not compute a parallel/second decision in the model adapter.
11. Return `ALLOW`, `STEP_UP`, or `DENY` through the existing API.

Face detection/alignment precedes all crop-dependent models. After the crop and enrollment embedding are ready, PAD, mask, and identity inference can run concurrently if their runtimes are thread-safe and hardware capacity allows. Temporal inference can run alongside crop models after frame sampling if it consumes raw ordered frames; if it uses face tracks/landmarks, wait for face detection/tracking first. Keep inference in the adapter's own bounded worker/queue; the API currently uses a thread pool and does not limit concurrent model execution.

## Example orchestrator shape

The snippets below show wiring only. `decode_image`, `decode_video`, model construction, resource limits, and calibrated thresholds must be supplied by the team; no such model implementations are included here.

```python
class TeamInferenceAdapter:
    def __init__(self, face, pad, identity, temporal, mask):
        self.face = face
        self.pad = pad
        self.identity = identity
        self.temporal = temporal
        self.mask = mask

    def create_embedding(self, media_path, media_type):
        frame = decode_image(media_path, media_type)
        detected = self.face.detect_and_align(frame)
        require_real_single_face(detected)
        typed = self.identity.enroll(detected.result.aligned_face_crop)
        return adapt_embedding_output(typed)

    def verify(self, media_path, media_type, enrollment_embedding):
        frames, media_kind = decode_and_sample(media_path, media_type)
        detected = self.face.detect_and_align(frames[0])
        require_real_single_face(detected)
        crop = detected.result.aligned_face_crop
        pad = self.pad.predict(crop)
        mask = self.mask.detect_mask(MaskAnalysisInput(aligned_face_crop=crop))
        identity = self.identity.verify(crop, enrollment_embedding)
        temporal = self.temporal.analyze(make_ordered_video_input(frames)) if media_kind == "video" else None
        return aggregate_model_outputs(
            face_detection=detected, pad=pad, identity=identity, mask=mask,
            temporal=temporal, media_kind=media_kind,
        )
```

`require_real_single_face`, decoders, and `make_ordered_video_input` above are illustrative responsibilities, not existing backend functions. Do not copy example confidence constants into a real adapter.

## Connecting the adapter

1. Add model dependencies/weights under the model owners' approved packaging and licensing setup; do not put secrets or raw user media in the image.
2. Implement the protocols and run `pytest backend/tests/test_inference_contracts.py`. Add model-specific tests for decode failures, malformed frames, multiple faces, and calibrated class boundaries.
3. Implement `TeamInferenceAdapter` with the current `create_embedding`, `verify`, and `validate_challenge` methods. Use `aggregate_model_outputs()` for verify output, `adapt_embedding_output()` for enrollment, and `adapt_challenge_output()` for challenge validation. Only return `ModelStatus.REAL` after successful load and validated inference.
4. In the deployment-owned app factory, instantiate the adapter and pass it to `create_app(inference_provider=...)`. The default `app` in `app/main.py` intentionally remains unavailable until that wiring is approved.
5. Run all backend tests, validate on a held-out attack dataset, document the model name/version and calibration protocol, and benchmark the target hardware before changing decision thresholds or calling detection supported.

## Example fake-module integration test

`tests/test_inference_contracts.py` uses a fake provider stack to exercise genuine, print, screen, video replay, mask, mismatch, uncertain, unavailable, malformed, and missing-module cases. These are policy/contract tests only; fake `REAL` outputs are synthetic and provide no accuracy evidence. Keep fakes in test code and never register them as the application provider in a deployed environment.
