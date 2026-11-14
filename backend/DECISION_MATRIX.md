# Decision Matrix

This matrix describes the existing `app/services/decision.py` policy and the typed-output adapter in `app/services/inference.py`. Thresholds are configurable through the `HID_*_THRESHOLD` environment variables. Defaults below are prototype policy values, not calibrated accuracy claims.

## Required evidence

| Signal | Required for `ALLOW` | Confirmed result |
|---|---|---|
| Face detection | `REAL`, exactly one face, `face` confidence `>= HID_FACE_DETECTION_THRESHOLD` (default 0.8) | `no_face` confidence `>= 0.8` => `DENY`; multiple faces or weak/no confidence => `STEP_UP` |
| PAD | `REAL` output required; `genuine` score `1 - spoof_probability >= HID_PAD_GENUINE_THRESHOLD` (default 0.8), or server-validated challenge pass may satisfy liveness | `spoof` score `spoof_probability >= HID_PAD_SPOOF_THRESHOLD` (default 0.9) => `DENY` |
| Identity | `REAL`, `matched=True`, similarity `>= HID_IDENTITY_MATCH_THRESHOLD` (default 0.8) | `mismatch`, `matched=False`, similarity `<= HID_IDENTITY_MISMATCH_THRESHOLD` (default 0.5) => `DENY`; gap => `STEP_UP` |
| Mask | `REAL`, `no_mask` confidence `>= HID_MASK_CLEAR_THRESHOLD` (default 0.8) | `mask_attack` confidence `>= HID_PAD_SPOOF_THRESHOLD` (default 0.9) => `DENY` |
| Temporal (video only) | `REAL`, `live` score `1 - replay_probability >= HID_PAD_GENUINE_THRESHOLD` | `replay` score `replay_probability >= HID_PAD_SPOOF_THRESHOLD` => `DENY` |
| Challenge (when step-up is required) | A `REAL` server-side validation result `passed` at `HID_CHALLENGE_PASS_THRESHOLD` (default 0.8) | Failure/unavailable/uncertain => `STEP_UP`; cannot override any confirmed spoof |

Final `ALLOW` requires real face, PAD, identity, and mask outputs; PAD must support genuine liveness or a validated challenge must supplement it. Video additionally requires real temporal `live` output. The challenge cannot replace a missing/mock/unavailable PAD, identity, face, or mask module. All required outputs must be `REAL`; `MOCK`, `UNAVAILABLE`, `ERROR`, malformed, or missing outputs never satisfy allow conditions.

## Attack-specific policy

| Attack | Required evidence to confirm | Decision | If the evidence is absent or inconclusive |
|---|---|---|---|
| Printed photograph | Real PAD reports `is_real=False` and spoof probability at/above the spoof threshold. `printed_photo` class is recorded only if the PAD supports it. | `DENY` | `STEP_UP`; do not infer a print attack from identity or face detection alone. |
| Screen replay | Real PAD reports `is_real=False` and spoof probability at/above the spoof threshold. `screen_replay` class is optional and model-dependent. | `DENY` | `STEP_UP`; do not assume the print detector also detects displays. |
| Prerecorded video replay | For video, real temporal analysis reports `is_live=False` and replay probability at/above the spoof threshold. A PAD spoof result independently denies. | `DENY` | `STEP_UP`; no video `ALLOW` without real temporal `live` evidence. |
| Physical mask | A dedicated validated mask model or mask-validated PAD capability reports `mask_detected=True` at/above the spoof threshold. | `DENY` | `STEP_UP`; missing mask evidence blocks `ALLOW`. A generic PAD result is not automatically mask evidence. |

Exactly-threshold scores count as passing the relevant inclusive comparison. Scores in the uncertainty gaps, contradictory class/score outputs, uncertain labels, or unavailable models result in `STEP_UP`, not `ALLOW`. A confidently confirmed spoof takes precedence over identity success and challenge success.

## Output-to-decision mapping

`aggregate_model_outputs()` maps PAD `is_real` and spoof probability to the existing `genuine`/`spoof` labels, identity similarity to `matched`/`mismatch`, temporal replay probability to `live`/`replay`, and mask output to `no_mask`/`mask_attack`. The module status and model metadata are preserved in the resulting evidence. Unsupported classification is not invented.
