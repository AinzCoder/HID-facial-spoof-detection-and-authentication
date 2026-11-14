import { afterEach, describe, expect, it, vi } from "vitest";

import { drawFaceOverlay } from "./face-overlay";
import {
  createFrameFormData,
  parsePreprocessResponse,
  preprocessUrl,
  type PreprocessResponse,
} from "./live-detection";

function responseWithFaces(
  faces: PreprocessResponse["faces"],
): PreprocessResponse {
  return {
    face_count: faces.length,
    image_dimensions: { width: 640, height: 480 },
    quality_score: 83.5,
    quality_flags: [],
    quality_metrics: { blur_variance: 190.25 },
    faces,
  };
}

describe("live detection API helpers", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("uses the configured base URL and appends the preprocessing endpoint", () => {
    expect(preprocessUrl("http://localhost:8000/")).toBe(
      "http://localhost:8000/preprocess",
    );
    expect(preprocessUrl(undefined)).toBe(
      "http://127.0.0.1:8000/preprocess",
    );
  });

  it("sends captured frames using the backend's multipart field name", () => {
    const formData = createFrameFormData(
      new Blob(["frame"], { type: "image/jpeg" }),
      4,
    );
    const file = formData.get("file");

    expect(file).toBeInstanceOf(File);
    expect((file as File).name).toBe("webcam-frame-4.jpg");
    expect(formData.has("image")).toBe(false);
  });

  it("accepts no-face and multi-face backend responses without changing indices", () => {
    expect(parsePreprocessResponse(responseWithFaces([])).face_count).toBe(0);

    const faces: PreprocessResponse["faces"] = [
      {
        face_id: 0,
        detection_index: 0,
        bounding_box: { x: 20, y: 30, width: 100, height: 120 },
        confidence: 0.91,
        quality_flags: [],
      },
      {
        face_id: 1,
        detection_index: 1,
        bounding_box: { x: 300, y: 40, width: 80, height: 95 },
        confidence: null,
        quality_flags: ["small_face"],
      },
    ];
    const parsed = parsePreprocessResponse(responseWithFaces(faces));

    expect(parsed.face_count).toBe(2);
    expect(parsed.faces.map((face) => face.detection_index)).toEqual([0, 1]);
    expect(parsed.faces[1].confidence).toBeNull();
  });

  it("keeps image encodings and detailed landmarks out of UI state", () => {
    const rawResponse = {
      ...responseWithFaces([
        {
          face_id: 0,
          detection_index: 0,
          bounding_box: { x: 10, y: 12, width: 40, height: 45 },
          confidence: 0.9,
          quality_flags: [],
        },
      ]),
      quality_metrics: { blur_variance: 70, faces: [{ pose: "detail" }] },
      faces: [
        {
          face_id: 0,
          detection_index: 0,
          bounding_box: { x: 10, y: 12, width: 40, height: 45 },
          confidence: 0.9,
          quality_flags: [],
          landmarks: { points: [1, 2, 3] },
          resized_crop_base64: "private-image-data",
        },
      ],
    };

    const parsed = parsePreprocessResponse(rawResponse);
    expect(parsed.quality_metrics).toEqual({ blur_variance: 70 });
    expect(parsed.faces[0]).not.toHaveProperty("landmarks");
    expect(parsed.faces[0]).not.toHaveProperty("resized_crop_base64");
  });

  it("rejects malformed backend responses", () => {
    expect(() => parsePreprocessResponse({ face_count: 2, faces: [] })).toThrow(
      "unexpected response",
    );
  });
});

describe("face overlay geometry", () => {
  it("scales and mirrors bounding boxes to match the preview", () => {
    const canvas = document.createElement("canvas");
    const drawing = {
      setTransform: vi.fn(),
      clearRect: vi.fn(),
      strokeRect: vi.fn(),
      fillRect: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn((text: string) => ({ width: text.length * 7 })),
    };
    vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
      width: 320,
      height: 240,
      top: 0,
      left: 0,
      right: 320,
      bottom: 240,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
    vi.spyOn(canvas, "getContext").mockReturnValue(
      drawing as unknown as CanvasRenderingContext2D,
    );
    vi.stubGlobal("devicePixelRatio", 1);

    drawFaceOverlay(
      canvas,
      { width: 640, height: 480 },
      [{ bounding_box: { x: 64, y: 40, width: 128, height: 120 } }],
      true,
    );

    expect(drawing.strokeRect).toHaveBeenCalledWith(224, 20, 64, 60);
    expect(drawing.fillText).toHaveBeenCalledWith("Face 1", expect.any(Number), expect.any(Number));
  });

  it("keeps labels inside the overlay at image boundaries", () => {
    const canvas = document.createElement("canvas");
    const drawing = {
      setTransform: vi.fn(),
      clearRect: vi.fn(),
      strokeRect: vi.fn(),
      fillRect: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn(() => ({ width: 42 })),
    };
    vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
      width: 100,
      height: 80,
      top: 0,
      left: 0,
      right: 100,
      bottom: 80,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
    vi.spyOn(canvas, "getContext").mockReturnValue(
      drawing as unknown as CanvasRenderingContext2D,
    );
    vi.stubGlobal("devicePixelRatio", 1);

    drawFaceOverlay(
      canvas,
      { width: 100, height: 80 },
      [{ bounding_box: { x: 80, y: 0, width: 20, height: 20 } }],
    );

    expect(drawing.fillRect).toHaveBeenCalledWith(0, 3, 56, 21);
  });
});
