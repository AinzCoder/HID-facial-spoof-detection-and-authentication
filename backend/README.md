# HID Photo Spoof Detection Backend

This is a Python 3.12 / FastAPI prototype for session handling, media intake, audit persistence, and integration with separately developed face/PAD models. The default inference adapter intentionally reports `UNAVAILABLE`; the API does not claim that a model, liveness check, or challenge validator is running.

## Architecture and contracts

- `app/main.py` creates the FastAPI application, database, health endpoint, and error handlers.
- `app/routes/api.py` implements sessions, enrollment, verification, challenge response, and result retrieval.
- `app/database/models.py` persists session/challenge state, encrypted enrollment embeddings, and verification audit results.
- `app/services/media.py` enforces byte limits, declared MIME allowlists, content signatures, image decoding, basic MP4/WebM container checks, temporary-file creation, and cleanup. Filenames are ignored.
- `app/services/decision.py` is route-independent and fails closed.
- `app/services/inference_contracts.py` defines validated per-model schemas and protocols; `app/services/inference.py` adapts their outputs into the existing API/decision boundary and defines the unavailable production default.
- `app/services/security.py` encrypts embeddings with AES-256-GCM. Raw uploaded face media is never stored in the database.

The AI integration package includes the [AI integration contract](./AI_INTEGRATION_CONTRACT.md), [inference schemas](./INFERENCE_SCHEMAS.md), [decision matrix](./DECISION_MATRIX.md), and [team integration guide](./TEAM_INTEGRATION_GUIDE.md). AI owners implement the focused component protocols; a composite adapter implements the existing API boundary:

1. `create_embedding(media_path, media_type) -> EmbeddingOutput`. Only `REAL` outputs with a finite, non-empty vector can be enrolled.
2. `verify(media_path, media_type, enrollment_embedding) -> VerificationInference`. Use `aggregate_model_outputs()` to map the validated per-model `InferenceOutput` schemas into this existing route/decision boundary. Each typed output includes model name/version, status, processing time, confidence, result, and reason code.
3. `validate_challenge(media_path, media_type, challenge) -> ComponentOutput`. A real server-side challenge validator must return `label="passed"` and a model-produced score to satisfy step-up.

The adapter receives a generated temporary media path and the content-detected MIME type. It owns model loading, decoding/frame extraction, face preprocessing, score calibration, and its own GPU/CPU lifecycle. Returned scores must be validated and calibrated by the model team; the backend does not manufacture confidence values. For video, temporal output labeled `live` is additionally required for `ALLOW`.

Use these output labels: face detection `face`/`no_face`; PAD `genuine`/`spoof`/`uncertain`; identity `matched`/`mismatch` plus `matched=True|False` and a similarity score; temporal `live`/`replay`/`uncertain`; mask `no_mask`/`mask_attack`/`uncertain`; challenge `passed`/`failed`/`uncertain`. For identity, the score is similarity; for classifiers, it is the model's calibrated confidence for the supplied label. `no_face`, `replay`, `spoof`, and `mask_attack` must not be emitted as confirmed findings without their calibrated threshold score.

`REAL` is an adapter assertion, not a backend certification. Integration code must set it only for an actually loaded, validated model. Test fixtures never establish model performance.

## Member 1 face preprocessing integration

An opt-in `Member1FaceDetectionProvider` reuses the pipeline in
`HID-facial-spoof-detection-and-authentication/face_detection/app/`. It decodes
the backend's validated image upload to contiguous RGB pixels, passes an
already-RGB frame to `FacePreprocessingPipeline.process_frame()`, and requires
successful alignment and landmarks before returning a detected-face result.
The response contains the selected highest-confidence face when there are
multiple detections, along with the total face count; a multiple-face result
does not qualify as a normal single-face verification.

The Member 1 package is loaded under a private package name, and its internal
pipeline imports are relative. This keeps it separate from the backend's
`app` package while preserving the standalone `uvicorn app.main:app` API from
the Member 1 project.

Install the three required runtime dependencies through the backend
requirements. The provider is disabled by default; enable it with
`HID_FACE_DETECTION_ENABLED=true`. It never downloads model assets. Configure
both local files with `HID_FACE_DETECTION_MODEL_PATH` and
`HID_FACE_LANDMARKER_MODEL_PATH`, or provision them in
`%LOCALAPPDATA%\face_preprocessing\models\` using the expected filenames:
`blaze_face_short_range.tflite` and `face_landmarker.task`. Missing assets or
dependencies yield `UNAVAILABLE`; corrupt/unloadable assets or inference
failures yield `ERROR`. Successful outputs identify the exact two local model
files by a combined SHA-256 fingerprint. The pipeline uses MediaPipe's default
CPU execution path; no GPU delegate is configured.

Test-injected pipelines are always reported as `MOCK`. Model-backed integration
tests run only when both local weights are provisioned. This provider performs
face detection, landmark extraction, crop, and alignment only; it is not PAD,
identity verification, mask detection, or liveness inference. Enabling this
provider alone cannot produce `ALLOW`.

## Decision policy

- `ALLOW` requires real face detection, real identity match, real PAD output, and a real `no_mask` output (from a dedicated validated mask detector or validated PAD capability). PAD must independently be `genuine` above threshold; a separately validated challenge cannot substitute for PAD. Video also requires real temporal `live` output. Mock/unavailable/error outputs cannot satisfy a required component.
- A real PAD `spoof`, video temporal `replay`, or mask `mask_attack` output at or above the configured spoof threshold, a high-confidence real `no_face` output, or a sufficiently low-similarity real identity mismatch returns `DENY`.
- Missing, mocked, erroneous, or uncertain outputs return `STEP_UP`.
- Challenge responses are single-use, session-bound, expire server-side, and are validated only through the adapter. A challenge cannot override a confirmed spoof.

Campus scheduling is an additional authorization gate after biometric verification: `DENY` and `STEP_UP` are preserved; biometric `ALLOW` requires an enabled schedule for the enrolled subject and a current local day/time inside that schedule. PAD must pass independently; neither a schedule nor a challenge substitutes for biometric evidence. The schedule interval includes its start time and excludes its end time.

Threshold settings are `HID_PAD_GENUINE_THRESHOLD`, `HID_PAD_SPOOF_THRESHOLD`, `HID_FACE_DETECTION_THRESHOLD`, `HID_IDENTITY_MATCH_THRESHOLD`, `HID_IDENTITY_MISMATCH_THRESHOLD`, `HID_MASK_CLEAR_THRESHOLD`, and `HID_CHALLENGE_PASS_THRESHOLD`. Changing thresholds requires model-specific calibration; defaults are not validated performance claims.

## Campus access schedules

Schedule management is admin-only and protected by a shared bearer secret. Set `HID_CAMPUS_SCHEDULE_ADMIN_TOKEN` to a randomly generated secret with at least 32 characters through a secret manager or environment injection. If it is unset, schedule management endpoints return `503` and cannot be used. This is an isolated bootstrap control, not administrator identity, role-based access control, or a replacement for the project's missing general authentication. Use TLS, restrict secret distribution, rotate it, and integrate campus SSO/IAM authorization before production.

The schedule endpoints are:

- `PUT /v1/admin/campus-schedules/{subject_id}` — create or replace a schedule; the subject must already be enrolled.
- `GET /v1/admin/campus-schedules/{subject_id}` — retrieve the schedule.
- `PATCH /v1/admin/campus-schedules/{subject_id}/status` — enable or disable it with `{"enabled": false}`.

All require `Authorization: Bearer <HID_CAMPUS_SCHEDULE_ADMIN_TOKEN>`. Example request:

```http
PUT /v1/admin/campus-schedules/student-42
Authorization: Bearer <admin-secret>
Content-Type: application/json

{
  "permitted_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
  "access_start_time": "08:00:00",
  "access_end_time": "18:00:00",
  "timezone": "America/Los_Angeles",
  "enabled": true
}
```

Example response:

```json
{
  "subject_id": "student-42",
  "permitted_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
  "access_start_time": "08:00:00",
  "access_end_time": "18:00:00",
  "timezone": "America/Los_Angeles",
  "enabled": true
}
```

Days are lowercase weekday names. Times are local wall-clock values in the named IANA timezone; timezone offsets on the time fields are rejected. The start is inclusive and the end exclusive. Same-day windows only are supported (`access_end_time` must be later than `access_start_time`); overnight schedules are not supported. An absent, disabled, malformed, or out-of-window schedule fails safely with `STEP_UP` and an explicit reason code (`campus_schedule_missing`, `campus_schedule_disabled`, `campus_schedule_evaluation_error`, or `outside_access_schedule`). A biometric `DENY` always remains `DENY`.

The new `campus_access_schedules` table is created by the existing SQLAlchemy `create_all()` startup initialization for existing databases. This creates the additive table; the project still has no managed schema migration system. The `tzdata` dependency supplies IANA timezone rules on platforms such as Windows where the system timezone database may be absent.

## Offline mode (prototype)

Offline mode is a separate local verification path; a timeout on the online API is never treated as biometric success and does not automatically switch a request to offline verification. A trusted client can poll `GET /v1/offline/status` with `HID_OFFLINE_BACKEND_HEALTH_URL` configured. The status reports `backend_available` as `true`, `false`, or `null` when no health URL is configured. The client must explicitly route a request to the local offline service after deciding the backend is unavailable.

Offline verification reuses the configured local `InferenceProvider` and the existing `decide()` logic, then evaluates a signed cached policy through the same campus schedule gate. DENY and STEP_UP are preserved. Missing, invalid, expired, disabled, subject-mismatched, or non-offline policies fail to STEP_UP. Offline ALLOW is disabled by default; enabling it requires both `HID_OFFLINE_ENABLED=true` and `HID_OFFLINE_ALLOW_ENABLED=true`, plus a valid Ed25519 public key. The offline provider must independently return valid REAL face, identity, PAD, and mask outputs (and REAL temporal output for video). The default provider is UNAVAILABLE, so this project cannot issue a real biometric ALLOW without an independently integrated model. A failed or unavailable challenge never upgrades an offline decision. The offline endpoint does not accept a client-supplied challenge-completed flag or implement a separate offline challenge session; a request needing step-up remains STEP_UP.

### Policy signing and local cache

Policies are built from the enrolled subject's current campus schedule and include a unique policy ID, subject, permitted weekdays, local time window, timezone, enabled flag, explicit `allow_offline` flag, issue timestamp, and expiry timestamp. The admin endpoint signs the claims using Ed25519. Verifiers check the signature against the configured public key before use, reject expired policies without extending them, and reject rollback to an older cached policy. A later signed policy (including a disabled policy) replaces the cache only when it is not older than the existing one. While a device is disconnected, new schedule changes cannot reach it and its cached policy remains usable until expiry. Signatures prevent policy forgery but do not protect a local database from deletion or rollback by an attacker with filesystem access; secure the local host and rely on short policy TTLs.

Generate a key pair with the already-installed `cryptography` package:

```powershell
.\backend\venv\Scripts\python.exe -c "import base64; from cryptography.hazmat.primitives import serialization; from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey; k=Ed25519PrivateKey.generate(); enc=lambda b: base64.urlsafe_b64encode(b).decode().rstrip('='); print('HID_OFFLINE_SIGNING_PRIVATE_KEY=' + enc(k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()))); print('HID_OFFLINE_SIGNING_PUBLIC_KEY=' + enc(k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)))"
```

Treat the private key output as a secret: put it only in the trusted policy-issuing service's secret store/environment, never in source control or a client request. The issuing service also configures the matching public key. An offline verifier needs only `HID_OFFLINE_SIGNING_PUBLIC_KEY`; policy issuance requires both keys. The application validates configured key lengths and checks a configured key pair matches. Rotate keys through an approved provisioning process; old policies will no longer verify when a verifier is switched to a different public key.

Set at least:

```text
HID_OFFLINE_ENABLED=true
HID_OFFLINE_ALLOW_ENABLED=false
HID_OFFLINE_POLICY_TTL_SECONDS=86400
HID_OFFLINE_MAX_CLOCK_SKEW_SECONDS=300
HID_OFFLINE_SIGNING_PUBLIC_KEY=<base64url raw Ed25519 public key>
```

Set `HID_OFFLINE_ALLOW_ENABLED=true` only after local inference, keys, durable storage, and device operations are validated. `HID_OFFLINE_POLICY_TTL_SECONDS` is the maximum policy validity accepted and assigned by the issuer. The verifier needs only the public key; the policy issuer needs the private key and matching public key. By default, the local SQLite store is a sibling `<primary-database>_offline.db` when the primary database is SQLite, or `./hid_offline.db` when the primary database is remote; override with `HID_OFFLINE_DATABASE_URL` and mount its path on durable local storage. The existing enrollment database must also be locally available to perform identity verification while disconnected. No enrollment/embedding synchronization to offline devices is implemented.

Issue a policy from the trusted administrative service:

```http
POST /v1/admin/offline/policies/student-42
Authorization: Bearer <HID_CAMPUS_SCHEDULE_ADMIN_TOKEN>
Content-Type: application/json

{"allow_offline": true}
```

The response contains the signed claims and signature, but never a signing key. An offline node can import that signed envelope with `PUT /v1/offline/policies/student-42` and retrieve it with `GET /v1/offline/policies/student-42`; both cache endpoints require `Authorization: Bearer <HID_OFFLINE_SYNC_TOKEN>`. Only envelopes whose signature verifies with the trusted public key are cached. Expired envelopes are rejected at import and use.

### Offline verification and audit synchronization

Submit media to the local verifier; media is still content-validated and held only in temporary files:

```http
POST /v1/offline/verifications/student-42
Content-Type: multipart/form-data

media_file=<image-or-video>
```

Example response:

```json
{
  "event_id": "2a03dc95-58bf-4f4f-9f22-43984a862885",
  "subject_id": "student-42",
  "mode": "offline",
  "outcome": "STEP_UP",
  "reason_code": "offline_policy_missing",
  "reason": "No signed offline policy is cached.",
  "policy_id": null,
  "model_status": {
    "face_detection": "UNAVAILABLE",
    "pad": "UNAVAILABLE",
    "identity": "UNAVAILABLE",
    "temporal": "UNAVAILABLE",
    "mask": "UNAVAILABLE"
  }
}
```

Each accepted attempt is queued in the separate local database with only event ID, subject ID, UTC timestamp, mode, outcome, reason code, policy ID, and sync metadata. Uploaded images/videos, face embeddings, and inference evidence are not put in this queue. Event decision fields are immutable; only sync metadata changes. Local storage failure prevents an otherwise successful ALLOW.

When connectivity is restored, configure `HID_OFFLINE_SYNC_BACKEND_URL` to the authenticated server's `/v1/offline/events/sync` endpoint and `HID_OFFLINE_SYNC_TOKEN` to a secret of at least 32 characters. `HID_OFFLINE_BACKEND_HEALTH_URL` may point to the backend health endpoint for status probing. Trigger an outbound batch with:

```http
POST /v1/offline/sync
Authorization: Bearer <HID_OFFLINE_SYNC_TOKEN>
```

The receiver endpoint is `POST /v1/offline/events/sync` with the same bearer authentication. It deduplicates by event ID and rejects reuse of an ID with different content. A local event is marked synchronized only when its ID appears in the server's acceptance response; timeouts and unconfirmed responses leave it pending for retry. Synchronization does not rerun or alter old verification decisions.

The shared sync bearer token is a prototype control, not a trusted device identity, per-device credential, or production-ready authentication system. Use TLS, isolate token distribution, and replace it with device-bound mTLS or an approved workload identity before deployment. Health probing and outbound synchronization are explicit operations; no background retry scheduler or automatic online-to-offline fallback is included.

## API

- `GET /health`
- `POST /v1/sessions` — JSON: `{"subject_id":"..."}` (optional)
- `POST /v1/enrollments` — multipart: `subject_id`, `image`
- `POST /v1/verifications/{session_id}` — multipart: `media_file`, optional `subject_id`
- `POST /v1/challenges/{session_id}/response` — multipart: `media_file`
- `GET /v1/verifications/{session_id}`
- `PUT /v1/admin/campus-schedules/{subject_id}` — create/update schedule (admin bearer secret required)
- `GET /v1/admin/campus-schedules/{subject_id}` — read schedule (admin bearer secret required)
- `PATCH /v1/admin/campus-schedules/{subject_id}/status` — enable/disable schedule (admin bearer secret required)
- `GET /v1/offline/status` — report offline configuration and optional backend health probe
- `POST /v1/admin/offline/policies/{subject_id}` — issue a schedule-derived signed policy (admin bearer secret required)
- `PUT` / `GET /v1/offline/policies/{subject_id}` — import/retrieve a signed local policy (offline sync bearer required)
- `POST /v1/offline/verifications/{subject_id}` — run local inference and queue a minimal offline event
- `POST /v1/offline/sync` — send pending local events to the configured backend (offline sync bearer required)
- `POST /v1/offline/events/sync` — authenticated, idempotent server-side event receiver

FastAPI's OpenAPI schema is available at `/docs` when the service is running. Error payloads use `{"detail":"...","code":"..."}`. Result `processing_ms` measures backend wall time around inference/decision work after upload validation; it does not include upload transfer time.

## Local setup and tests (Windows PowerShell)

From the project root:

```powershell
.\backend\venv\Scripts\python.exe -m pip install -r .\backend\requirements-dev.txt
$env:PYTHONPATH = (Resolve-Path .\backend).Path
.\backend\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir .\backend --host 127.0.0.1 --port 8000
```

In another PowerShell window:

```powershell
$env:PYTHONPATH = (Resolve-Path .\backend).Path
.\backend\venv\Scripts\python.exe -m pytest .\backend\tests -q
```

The database defaults to `sqlite:///./hid_spoof.db`. Override with `HID_DATABASE_URL`. To enable enrollment storage, set `HID_BIOMETRIC_ENCRYPTION_KEY` to a base64url-encoded 32-byte key supplied through a secret manager/environment injection; never commit it. The API remains unable to enroll until a real embedding adapter is connected, even if a key is configured. Configure `HID_CAMPUS_SCHEDULE_ADMIN_TOKEN` as described above to enable schedule management.

## AWS Sandbox readiness

Configuration is environment-based; logging is stdout/stderr compatible with CloudWatch collection. The database URL can target a managed database, and the media adapter boundary can be extended for private S3 temporary objects with short retention, server-side encryption, TLS, and least-privilege IAM. Current media files are local temporary files, no S3 calls are made, and no AWS resources or credentials are provisioned. SQLite is suitable only for local prototype use.

For containerization, build from the repository root with the backend Dockerfile:

```powershell
docker build -f .\backend\Dockerfile -t hid-photo-spoof-backend .
```

Production deployment still needs an external database, secret injection/KMS policy, resource limits, non-root runtime, S3 lifecycle policy (if enabled), health/readiness policy, network controls, and an approved model image/service.

## Current gaps

- No real OpenCV/MediaPipe face detector, MiniFASNet/equivalent PAD, ArcFace identity model, temporal replay model, or mask model is included. Since `ALLOW` requires real outputs for the applicable face/PAD/identity/mask components, the default app cannot return `ALLOW`.
- No model-backed challenge validation is included. `blink` remains a schema option but is not assigned until a validated pipeline supports it; currently assigned head-turn/look-up challenges are also not validated by a real model.
- MP4/WebM intake checks structural signatures/container boxes but does not decode frames; a model adapter must perform robust media decode and temporal analysis. Deeper video validation should be supplied by a hardened decoder.
- The backend cannot establish the reliability of a physical-mask detector. Mask decisions require a separately validated model output; MiniFASNet alone is not represented as a mask detector.
- SQLite schema creation is prototype initialization, not a managed migration workflow. The new enrollment `model_version` column may need a migration for a database created by an earlier prototype. Enrollment re-enrollment, retention/deletion policy, rate limiting, user authentication/authorization, and deployment hardening remain outstanding. Campus schedule endpoints use only a shared admin bearer secret; they do not establish an authenticated administrator identity or per-user roles.
- The two-second target has not been guaranteed. Measure with the selected model, media sizes, deployment hardware, and production-like concurrency.

## Measured local baseline

On the development Windows/Python 3.12.10 environment, 30 sequential TestClient requests used a 92-byte 16x16 PNG, SQLite in a temporary directory, and the default `UNAVAILABLE` inference adapter. Uvicorn/model startup and remote network transfer were not part of this benchmark. API request wall time (including local multipart upload, decode/validation, and persistence) was 10.695 ms minimum, 12.297 ms median, 18.591 ms nearest-rank p95, and 19.210 ms maximum. The response's `processing_ms` (inference/decision/database interval after upload validation) was 3.571 ms minimum, 3.952 ms median, 7.532 ms nearest-rank p95, and 8.206 ms maximum. These are no-model prototype overhead measurements, not biometric inference performance or evidence that the two-second target is achieved with models.

## AI team integration steps

1. Follow [TEAM_INTEGRATION_GUIDE.md](./TEAM_INTEGRATION_GUIDE.md) and implement the typed protocols from [inference_contracts.py](./app/services/inference_contracts.py).
2. Implement the existing `InferenceProvider` in a separately owned adapter and use `aggregate_model_outputs()`; return `UNAVAILABLE` or `ERROR` rather than synthetic scores when loading/inference fails. Return `REAL` only for actually loaded and validated models.
3. Ensure `verify()` consumes the decrypted enrollment vector only in memory and returns identity similarity/match output. Supply `no_mask` only from a dedicated validated mask model or validated PAD capability.
4. Implement server-side challenge validation from uploaded media; do not accept a client completion flag. Keep `blink` unassigned until validated.
5. Test photo/screen/video/mask cases and thresholds on a representative validation set, then tune environment thresholds. Keep fake providers scoped to tests.
6. Measure decode/inference and end-to-end p50/p95 on target AWS hardware. Request approval before provisioning AWS resources.
