import asyncio
import os
import struct
import tempfile
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


IMAGE_MIME_TYPES = {
    "image/jpeg": {b"\xff\xd8\xff"},
    "image/png": {b"\x89PNG\r\n\x1a\n"},
    "image/webp": {b"RIFF"},
}
VIDEO_MIME_TYPES = {"video/mp4", "video/webm"}


@dataclass
class TemporaryMedia:
    path: str
    media_type: str
    media_kind: str
    size: int

    def cleanup(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def _read_box(data: bytes, offset: int) -> tuple[int, bytes] | None:
    if len(data) < offset + 8:
        return None
    size = struct.unpack(">I", data[offset : offset + 4])[0]
    box_type = data[offset + 4 : offset + 8]
    if size == 1:
        if len(data) < offset + 16:
            return None
        size = struct.unpack(">Q", data[offset + 8 : offset + 16])[0]
    header_size = 16 if struct.unpack(">I", data[offset : offset + 4])[0] == 1 else 8
    if size < header_size:
        return None
    return size, box_type


def _detect_content(prefix: bytes, data: bytes) -> tuple[str, str] | None:
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "image"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "image"
    if prefix.startswith(b"RIFF") and len(prefix) >= 12 and prefix[8:12] == b"WEBP":
        return "image/webp", "image"
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
        offset = 0
        has_ftyp = False
        has_media_box = False
        while offset < len(data):
            box = _read_box(data, offset)
            if box is None:
                break
            box_size, box_type = box
            if box_type == b"ftyp":
                has_ftyp = True
            if box_type in {b"mdat", b"moov"}:
                has_media_box = True
            offset += box_size
        if has_ftyp and has_media_box and offset == len(data):
            return "video/mp4", "video"
        return None
    if prefix.startswith(b"\x1a\x45\xdf\xa3") and b"\x18\x53\x80\x67" in data[:4096]:
        return "video/webm", "video"
    return None


def _validate_image(path: str, media_type: str) -> None:
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(status_code=415, detail="Unsupported image format.")
            image.verify()
        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0 or image.width * image.height > 40_000_000:
                raise HTTPException(status_code=400, detail="Image dimensions are invalid or too large.")
            expected = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[media_type]
            if image.format != expected:
                raise HTTPException(status_code=415, detail="Image content does not match its detected format.")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Image data is malformed or truncated.") from exc


async def store_upload(upload: UploadFile, max_image_bytes: int, max_video_bytes: int) -> TemporaryMedia:
    declared_type = (upload.content_type or "").lower().split(";", 1)[0].strip()
    allowed_declared = set(IMAGE_MIME_TYPES) | VIDEO_MIME_TYPES
    if declared_type not in allowed_declared:
        raise HTTPException(status_code=415, detail="Unsupported or missing media content type.")
    declared_limit = max_image_bytes if declared_type in IMAGE_MIME_TYPES else max_video_bytes

    temporary = tempfile.NamedTemporaryFile(prefix="hid-media-", suffix=".bin", delete=False)
    size = 0
    try:
        with temporary:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > declared_limit:
                    raise HTTPException(status_code=413, detail="Upload exceeds the configured size limit.")
                temporary.write(chunk)
            temporary.flush()
        with open(temporary.name, "rb") as media_file:
            data = media_file.read()
        detected = _detect_content(data[:32], data)
        if detected is None:
            raise HTTPException(status_code=415, detail="Media content is not a supported, valid container.")
        media_type, media_kind = detected
        if media_type != declared_type:
            raise HTTPException(status_code=415, detail="Declared content type does not match media content.")
        max_bytes = max_image_bytes if media_kind == "image" else max_video_bytes
        if size > max_bytes:
            raise HTTPException(status_code=413, detail="Upload exceeds the configured size limit for this media type.")
        if media_kind == "image":
            await asyncio.to_thread(_validate_image, temporary.name, media_type)
        return TemporaryMedia(temporary.name, media_type, media_kind, size)
    except Exception:
        try:
            os.unlink(temporary.name)
        except FileNotFoundError:
            pass
        raise
    finally:
        await upload.close()
