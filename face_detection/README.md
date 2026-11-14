# Face Anti-Spoofing Preprocessing Service

Local face detection and preprocessing for the Face Anti-Spoofing Detection
System. This project prepares images for a downstream classifier; it does not
perform spoof classification, face recognition, identity matching, or tracking.

## Requirements and installation

The project is tested with Python 3.13.15. Create and activate a virtual
environment from the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MediaPipe Tasks models are downloaded on first initialization and kept in the
local user cache. They are not downloaded per request and image uploads are not
sent to a remote service:

- Face Detector: [BlazeFace short-range model](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite)
- Face Landmarker: [Face Landmarker task model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- Both models are distributed under the Apache License 2.0.

## Start the API

Run locally:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The default maximum image upload is 10 MiB. Configure another positive byte
limit with `FACE_API_MAX_UPLOAD_BYTES` before starting the service. The API
validates the encoded image contents as JPEG or PNG; it does not trust the
filename or declared MIME type.

## Endpoints

### `GET /health`

Returns service and model readiness only:

```json
{
  "status": "ok",
  "models_initialized": true
}
```

### `POST /detect`

Send a multipart upload with the field name `file`:

```powershell
curl.exe -F "file=@portrait.jpg" http://127.0.0.1:8000/detect
```

The response contains the face count, image dimensions (`width`, `height`), and
per-face bounding boxes and detector confidence. It does not include landmarks
or face images.

### `POST /preprocess`

Send the same multipart upload format:

```powershell
curl.exe -F "file=@portrait.jpg" http://127.0.0.1:8000/preprocess
```

The response includes the image dimensions, quality score and flags, quality
metrics, and a `faces` array. Each face includes its detection index, bounding
box, confidence, normalized and pixel landmark coordinates, crop coordinates,
context-crop dimensions, resized-crop dimensions, and alignment status.
Alignment is disabled by default.

`resized_crop_base64` contains a base64-encoded PNG of the resized RGB face crop.
`crop_dimensions` and `resized_crop_dimensions` are expressed as width/height
objects. The unresized context crop is retained by the pipeline during
processing but is not included in the JSON response.

Example response shape (landmark arrays and base64 content shortened):

```json
{
  "face_count": 1,
  "image_dimensions": {"width": 820, "height": 1024},
  "quality_score": 100.0,
  "quality_flags": [],
  "quality_metrics": {
    "width": 820,
    "height": 1024,
    "blur_variance": 230.4,
    "is_blurry": false,
    "brightness_mean": 122.6,
    "is_underexposed": false,
    "is_overexposed": false,
    "contrast_stddev": 49.8,
    "is_low_contrast": false,
    "is_low_resolution": false,
    "faces": [
      {
        "detection_index": 0,
        "bounding_box": {"x": 283, "y": 115, "width": 234, "height": 234},
        "face_area_ratio": 0.0652,
        "is_small": false,
        "is_clipped": false,
        "pose": {
          "face_index": 0,
          "detection_index": 0,
          "roll_angle_degrees": -0.67,
          "yaw_offset_ratio": 0.04,
          "is_approximately_frontal": true
        }
      }
    ],
    "pose_indicators": [
      {
        "face_index": 0,
        "detection_index": 0,
        "roll_angle_degrees": -0.67,
        "yaw_offset_ratio": 0.04,
        "is_approximately_frontal": true
      }
    ]
  },
  "faces": [
    {
      "face_id": 0,
      "detection_index": 0,
      "bounding_box": {"x": 283, "y": 115, "width": 234, "height": 234},
      "confidence": 0.92,
      "landmarks": {"face_index": 0, "detection_index": 0, "landmarks": [], "regions": {}},
      "quality_metrics": {
        "detection_index": 0,
        "bounding_box": {"x": 283, "y": 115, "width": 234, "height": 234},
        "face_area_ratio": 0.0652,
        "is_small": false,
        "is_clipped": false,
        "pose": {
          "face_index": 0,
          "detection_index": 0,
          "roll_angle_degrees": -0.67,
          "yaw_offset_ratio": 0.04,
          "is_approximately_frontal": true
        }
      },
      "quality_flags": [],
      "crop_coordinates": {"x": 236, "y": 68, "width": 328, "height": 328},
      "crop_dimensions": {"width": 328, "height": 328},
      "resized_crop_dimensions": {"width": 224, "height": 224},
      "resized_crop_base64": "<base64 PNG>",
      "color_format": "RGB",
      "alignment_applied": false
    }
  ]
}
```

The values above illustrate response structure, not guaranteed model output.
Quality thresholds and scores are heuristics requiring calibration; they are
not anti-spoofing decisions.

## Member 2: decode model-ready crops

Decode the base64 PNG to an RGB `uint8` NumPy array:

```python
import base64
import io

import numpy as np
from PIL import Image

encoded = response_json["faces"][0]["resized_crop_base64"]
with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
    model_input = np.asarray(image.convert("RGB"), dtype=np.uint8)
```

`model_input` has shape `(height, width, 3)` and RGB channel order. Apply only
the normalization and tensor layout expected by the classifier; this API does
not normalize or otherwise enhance the crops.

## Privacy and deployment

The application does not persist uploads, crops, or biometric metadata itself.
Multipart upload handling may buffer request data in memory or a temporary
spool file managed by the web framework. Avoid collecting raw request bodies
in logs. Returned landmarks and face crops are biometric data; base64 is an
encoding, not encryption.

The API enables CORS only for the local frontend origins
`http://localhost:3000` and `http://127.0.0.1:3000`. Configure a comma-separated
explicit allowlist with `FACE_API_CORS_ORIGINS` when the trusted development
origin differs; wildcard origins are rejected. The sample service has no
authentication. Keep it bound to localhost for development. Before public
deployment, add appropriate authentication, HTTPS, access controls, retention
limits, and operational monitoring for your environment. Do not expose the
service publicly as-is.

## Live webcam preprocessing demo

Run the local OpenCV webcam preview:

```powershell
python -m app.live_preprocessing_demo
```

Press `Q` to quit. The preview shows the captured frame, bounding boxes,
confidence, frame index, approximate processed FPS, quality score, and warning
flags. A second window shows the first detected face's 224×224 RGB crop
converted for OpenCV display. With multiple faces, all boxes are drawn and the
first face in detection order is shown in the crop window.

Optional arguments:

```powershell
python -m app.live_preprocessing_demo --camera-index 1 --sample-every 2
python -m app.live_preprocessing_demo --show-landmarks
```

Frame sampling reduces preprocessing load; the displayed FPS is approximate
and reflects sampled/processed frames. No frames or crops are saved or
uploaded. The demo reuses one webcam capture stream and initializes its
MediaPipe models once for the run.

## Tests

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Frontend webcam integration

The Next.js frontend includes an opt-in webcam workflow on `/live-detection`.
Camera permission is requested only after clicking **Start Camera**. The page
captures one JPEG frame per second and sends it to the local `/preprocess` API;
the browser does not save frames or face crops. Image uploads elsewhere in the
frontend, result history, and anti-spoof classification are not implemented.

Requirements: Node.js 20.9 or newer and npm.

Start the backend in one terminal from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Set-Location .\frontend
npm install
Copy-Item .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000/live-detection`. The frontend reads the backend
URL from `NEXT_PUBLIC_API_BASE_URL` (default: `http://127.0.0.1:8000`). The API
allows `http://localhost:3000` and `http://127.0.0.1:3000` by default; configure
additional trusted origins with the comma-separated `FACE_API_CORS_ORIGINS`
environment variable. Wildcard CORS origins are not allowed.

Available routes:

- `/` — project dashboard
- `/live-detection` — live camera preview and preprocessing results
- `/image-analysis` — upload workflow placeholder (uploads are disabled)
- `/results` — empty state (no history is fabricated or saved)
- `/about` — system components, implementation status, and privacy boundary

Validation commands:

```powershell
npm run lint
npm run build
npm run test
```

The preview is mirrored for user familiarity; submitted frames remain in the
camera's original orientation, and the canvas overlay mirrors bounding boxes to
match the preview. Requests are limited to one in flight, time out after 15
seconds, and are aborted when the camera stops or the page unmounts. A physical
camera and browser permission are required for an end-to-end live capture.
The API has no authentication and must not be exposed publicly without
authentication, HTTPS, and appropriate access controls.
