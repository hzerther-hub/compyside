"""Image import/export tests."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from compositor.io.image_importer import (
    decode_png_mask,
    encode_png,
    encode_png_mask,
    load_image,
)


def test_roundtrip_png(tmp_path) -> None:
    arr = np.full((4, 4, 4), 123, dtype=np.uint8)
    arr[..., 3] = 255
    png = encode_png(arr)
    out_path = tmp_path / "out.png"
    out_path.write_bytes(png)
    loaded = load_image(out_path)
    np.testing.assert_array_equal(loaded, arr)


def test_load_from_bytes() -> None:
    img = Image.new("RGB", (3, 3), (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    arr = load_image(buf.getvalue())
    assert arr.shape == (3, 3, 4)
    assert arr[0, 0, 0] == 10


def test_mask_roundtrip() -> None:
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    png = encode_png_mask(gray)
    out = decode_png_mask(png)
    np.testing.assert_array_equal(out, gray)


def test_load_jpeg_is_rgba() -> None:
    img = Image.new("RGB", (2, 2), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    arr = load_image(buf.getvalue())
    assert arr.shape == (2, 2, 4)
    assert arr[0, 0, 3] == 255  # JPEG has no alpha; we fill 255.
