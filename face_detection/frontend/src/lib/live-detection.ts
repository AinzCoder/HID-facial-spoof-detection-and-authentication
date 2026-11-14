export interface ImageDimensions {
  width: number;
  height: number;
}

export interface FaceBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PreprocessedFaceResponse {
  face_id: number;
  detection_index: number;
  bounding_box: FaceBoundingBox;
  confidence: number | null;
  quality_flags: string[];
}

export interface PreprocessResponse {
  face_count: number;
  image_dimensions: ImageDimensions;
  quality_score: number;
  quality_flags: string[];
  quality_metrics: {
    blur_variance: number;
    [key: string]: unknown;
  };
  faces: PreprocessedFaceResponse[];
}

export function preprocessUrl(baseUrl: string | undefined): string {
  return `${(baseUrl || "http://127.0.0.1:8000").replace(/\/+$/, "")}/preprocess`;
}

export function createFrameFormData(blob: Blob, frameIndex: number): FormData {
  const formData = new FormData();
  formData.append("file", blob, `webcam-frame-${frameIndex}.jpg`);
  return formData;
}

export function parsePreprocessResponse(payload: unknown): PreprocessResponse {
  if (!payload || typeof payload !== "object") {
    throw new Error("The backend returned an unexpected response.");
  }

  const response = payload as Partial<PreprocessResponse>;
  const faceCount = response.face_count;
  const qualityScore = response.quality_score;
  const dimensions = response.image_dimensions;
  const metrics = response.quality_metrics;
  if (
    typeof faceCount !== "number" ||
    !Number.isInteger(faceCount) ||
    faceCount < 0 ||
    !dimensions ||
    !Number.isFinite(dimensions.width) ||
    dimensions.width <= 0 ||
    !Number.isFinite(dimensions.height) ||
    dimensions.height <= 0 ||
    typeof qualityScore !== "number" ||
    !Number.isFinite(qualityScore) ||
    !Array.isArray(response.quality_flags) ||
    !metrics ||
    !Number.isFinite(metrics.blur_variance) ||
    !Array.isArray(response.faces) ||
    response.faces.length !== faceCount ||
    !response.quality_flags.every((flag) => typeof flag === "string")
  ) {
    throw new Error("The backend returned an unexpected response.");
  }

  for (const face of response.faces) {
    if (!face || typeof face !== "object") {
      throw new Error("The backend returned an unexpected response.");
    }
    const box = face.bounding_box;
    if (
      !Number.isInteger(face.face_id) ||
      !Number.isInteger(face.detection_index) ||
      !box ||
      ![box.x, box.y, box.width, box.height].every(Number.isFinite) ||
      box.width <= 0 ||
      box.height <= 0 ||
      (face.confidence !== null && !Number.isFinite(face.confidence)) ||
      !Array.isArray(face.quality_flags) ||
      !face.quality_flags.every((flag) => typeof flag === "string")
    ) {
      throw new Error("The backend returned an unexpected response.");
    }
  }

  return {
    face_count: faceCount,
    image_dimensions: {
      width: dimensions.width,
      height: dimensions.height,
    },
    quality_score: qualityScore,
    quality_flags: [...response.quality_flags],
    quality_metrics: { blur_variance: metrics.blur_variance },
    faces: response.faces.map((face) => ({
      face_id: face.face_id,
      detection_index: face.detection_index,
      bounding_box: {
        x: face.bounding_box.x,
        y: face.bounding_box.y,
        width: face.bounding_box.width,
        height: face.bounding_box.height,
      },
      confidence: face.confidence,
      quality_flags: [...face.quality_flags],
    })),
  };
}

export function formatQualityFlag(flag: string): string {
  return flag.replaceAll("_", " ");
}
