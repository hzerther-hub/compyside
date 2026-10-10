"""Image import — PNG, JPEG, TIFF (and HEIC/SVG if their optional deps are installed).

Returns NumPy uint8 RGBA arrays so the rest of the pipeline never has to
worry about Pillow modes.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

PathLike = Union[str, Path]


def load_image(source: Union[PathLike, bytes]) -> np.ndarray:
    """Open ``source`` (path or raw bytes) and return RGBA uint8."""
    if isinstance(source, (str, Path)):
        with Image.open(source) as img:
            return _to_rgba(img)
    with Image.open(io.BytesIO(source)) as img:
        return _to_rgba(img)


def _to_rgba(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGBA":
        # Convert on a copy; do not mutate the caller's image.
        img = img.convert("RGBA")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full(arr.shape, 255, dtype=np.uint8)], axis=-1)
    return arr


def encode_png(rgba: np.ndarray) -> bytes:
    """Encode an RGBA uint8 array as 8-bit PNG bytes."""
    if rgba.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgba.ndim != 3 or rgba.shape[-1] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 array")
    img = Image.fromarray(rgba, mode="RGBA" if rgba.shape[-1] == 4 else "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def decode_png(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes to an HxWx3 or HxWx4 uint8 array.

    Promotes grayscale to RGB and adds an opaque alpha channel.
    """
    img = Image.open(io.BytesIO(png_bytes))
    img.load()
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    return arr.astype(np.uint8)


def encode_png_mask(gray: np.ndarray) -> bytes:
    """Encode an 8-bit grayscale array as PNG bytes."""
    if gray.dtype != np.uint8:
        raise TypeError("expected uint8")
    if gray.ndim != 2:
        raise ValueError("expected HxW grayscale array")
    img = Image.fromarray(gray, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def decode_png_mask(png_bytes: bytes) -> np.ndarray:
    """Decode a grayscale PNG (e.g. a layer mask) to uint8 HxW."""
    with Image.open(io.BytesIO(png_bytes)) as img:
        if img.mode != "L":
            img = img.convert("L")
        return np.asarray(img, dtype=np.uint8)
