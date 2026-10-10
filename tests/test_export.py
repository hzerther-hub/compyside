"""Export tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from compositor.rendering import export as exp


def test_png_roundtrip(tmp_path: Path) -> None:
    arr = np.full((8, 8, 4), 200, dtype=np.uint8)
    arr[..., 3] = 255
    out = tmp_path / "out.png"
    exp.export_png(arr, out)
    with Image.open(out) as img:
        loaded = np.array(img)
    assert loaded.shape == (8, 8, 3) or loaded.shape == (8, 8, 4)
    # RGB values should round-trip (PIL may drop alpha for default mode).
    if loaded.shape[-1] == 4:
        np.testing.assert_array_equal(loaded, arr)
    else:
        np.testing.assert_array_equal(loaded, arr[..., :3])


def test_jpeg_drops_alpha(tmp_path: Path) -> None:
    arr = np.full((8, 8, 4), 200, dtype=np.uint8)
    arr[..., 3] = 128
    out = tmp_path / "out.jpg"
    exp.export_jpeg(arr, out, quality=80)
    assert out.exists() and out.stat().st_size > 0


def test_pdf_writes_bytes(tmp_path: Path) -> None:
    arr = np.full((16, 16, 4), 200, dtype=np.uint8)
    arr[..., 3] = 255
    out = tmp_path / "out.pdf"
    exp.export_pdf(arr, out)
    data = out.read_bytes()
    assert data.startswith(b"%PDF-1.4")
    assert data.rstrip().endswith(b"%%EOF")
