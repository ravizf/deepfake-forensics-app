"""Basic media metadata helpers for prototype inspection."""

from __future__ import annotations

import os

from PIL import Image, UnidentifiedImageError

try:
    import exifread
except ImportError:  # pragma: no cover - optional dependency
    exifread = None


def extract_media_metadata(path, media_type="image", file_sha256=None):
    metadata = {
        "media_type": media_type,
        "file_sha256_present": bool(file_sha256),
        "exif_present": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "image_size": None,
        "warning": None,
    }

    if media_type != "image":
        metadata["warning"] = "EXIF inspection is limited for video files in this prototype."
        return metadata

    if not path or not os.path.exists(path):
        metadata["warning"] = "Source file was not available for metadata inspection."
        return metadata

    if exifread is not None:
        try:
            with open(path, "rb") as stream:
                tags = exifread.process_file(stream, details=False)
            if tags:
                metadata["exif_present"] = True
                metadata["camera_make"] = str(tags.get("Image Make") or metadata["camera_make"] or "")
                metadata["camera_model"] = str(
                    tags.get("Image Model") or metadata["camera_model"] or ""
                )
                metadata["software"] = str(tags.get("Image Software") or metadata["software"] or "")
        except Exception:
            pass

    try:
        with Image.open(path) as image:
            metadata["image_size"] = f"{image.width}x{image.height}"
            exif_data = image.getexif()
            if exif_data:
                metadata["exif_present"] = True
                metadata["camera_make"] = exif_data.get(271)
                metadata["camera_model"] = exif_data.get(272)
                metadata["software"] = exif_data.get(305)
    except (UnidentifiedImageError, OSError, ValueError):
        metadata["warning"] = "Image metadata could not be read reliably."

    return metadata
