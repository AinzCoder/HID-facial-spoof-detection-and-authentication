import type {
  FaceBoundingBox,
  ImageDimensions,
} from "./live-detection";

export function drawFaceOverlay(
  canvas: HTMLCanvasElement,
  imageDimensions: ImageDimensions | null,
  faces: ReadonlyArray<{
    bounding_box: FaceBoundingBox;
  }>,
  mirrored = true,
): void {
  const bounds = canvas.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const width = bounds.width;
  const height = bounds.height;
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);

  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!imageDimensions || imageDimensions.width <= 0 || imageDimensions.height <= 0) {
    return;
  }

  const scale = Math.min(
    width / imageDimensions.width,
    height / imageDimensions.height,
  );
  const contentWidth = imageDimensions.width * scale;
  const contentHeight = imageDimensions.height * scale;
  const offsetX = (width - contentWidth) / 2;
  const offsetY = (height - contentHeight) / 2;

  context.lineWidth = 2;
  context.strokeStyle = "#44e2c0";
  context.fillStyle = "#44e2c0";
  context.font = "600 12px Arial, sans-serif";

  faces.forEach((face, index) => {
    const box = face.bounding_box;
    const left = Math.max(0, Math.min(imageDimensions.width, box.x));
    const top = Math.max(0, Math.min(imageDimensions.height, box.y));
    const right = Math.max(left, Math.min(imageDimensions.width, box.x + box.width));
    const bottom = Math.max(top, Math.min(imageDimensions.height, box.y + box.height));
    if (right <= left || bottom <= top) {
      return;
    }

    const mirroredLeft = mirrored
      ? imageDimensions.width - right
      : left;
    const x = offsetX + mirroredLeft * scale;
    const y = offsetY + top * scale;
    const boxWidth = (right - left) * scale;
    const boxHeight = (bottom - top) * scale;
    context.strokeRect(x, y, boxWidth, boxHeight);

    const label = `Face ${index + 1}`;
    const labelWidth = context.measureText(label).width + 14;
    const labelHeight = 21;
    const labelX = Math.max(0, Math.min(x, width - labelWidth));
    const labelY =
      y >= labelHeight + 3
        ? y - labelHeight - 3
        : Math.min(height - labelHeight, y + 3);
    context.fillRect(labelX, labelY, labelWidth, labelHeight);
    context.fillStyle = "#07111f";
    context.fillText(label, labelX + 7, labelY + 15);
    context.fillStyle = "#44e2c0";
  });
}
