"""Export — PNG, JPEG, single-page PDF."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def export_png(image: np.ndarray, path: PathLike) -> None:
    if image.dtype != np.uint8:
        raise TypeError("expected uint8")
    arr = np.ascontiguousarray(image[..., :3] if image.shape[-1] == 4 else image)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    img = Image.fromarray(arr, mode=mode)
    img.save(str(path), format="PNG", optimize=False)


def export_jpeg(image: np.ndarray, path: PathLike, quality: int = 90) -> None:
    """JPEG export — alpha is composited over white."""
    if image.dtype != np.uint8:
        raise TypeError("expected uint8")
    rgb = image[..., :3]
    if image.shape[-1] == 4:
        a = image[..., 3:4].astype(np.float32) / 255.0
        white = np.full_like(rgb, 255)
        rgb = (rgb.astype(np.float32) * a + white * (1 - a)).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(str(path), format="JPEG", quality=quality, optimize=True)


def export_pdf(image: np.ndarray, path: PathLike, page_size: tuple = (595, 842)) -> None:
    """Single-page PDF export — embeds the image at ``page_size`` in points."""
    if image.dtype != np.uint8:
        raise TypeError("expected uint8")
    # We embed the PNG bytes inside the PDF via a tiny hand-written writer.
    png_bytes = _png_bytes(image)
    _write_pdf(path, png_bytes, page_size)


def _png_bytes(image: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(image[..., :3] if image.shape[-1] == 4 else image)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_pdf(path: PathLike, png_bytes: bytes, page_size: tuple) -> None:
    """Minimal PDF writer that embeds a single PNG page."""
    width_pt, height_pt = page_size
    h, w = 1, 1
    from PIL import Image as _Im

    with _Im.open(io.BytesIO(png_bytes)) as img:
        w, h = img.size
    scale = min(width_pt / w, height_pt / h)
    draw_w = w * scale
    draw_h = h * scale
    x = (width_pt - draw_w) / 2
    y = (height_pt - draw_h) / 2

    objs: list[bytes] = []

    def _add(obj_bytes: bytes) -> int:
        objs.append(obj_bytes)
        return len(objs)

    catalog_id = _add(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = _add(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    page_id = _add(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_pt} {height_pt}] "
        f"/Resources << /XObject << /Im1 5 0 R >> >> /Contents 4 0 R >>".encode()
    )
    content_id = _add(
        f"q\n{scale} 0 0 {scale} {x} {y} cm\n/Im1 Do\nQ".encode()
    )
    image_obj_id = _add(
        b"<< /Type /XObject /Subtype /Image /Width "
        + str(w).encode() + b" /Height " + str(h).encode()
        + b" /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length "
        + b"TBD" + b" >>\nstream\n" + png_bytes + b"\nendstream"
    )
    # Patch the /Length for the image stream object.
    objs[image_obj_id - 1] = (
        f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(png_bytes)} >>\nstream\n".encode()
    ) + png_bytes + b"\nendstream"

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode())
        out.write(obj)
        out.write(b"\nendobj\n")
    xref_pos = out.tell()
    out.write(b"xref\n0 " + str(len(objs) + 1).encode() + b"\n")
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(b"trailer\n<< /Size " + str(len(objs) + 1).encode() + b" /Root 1 0 R >>\nstartxref\n")
    out.write(str(xref_pos).encode())
    out.write(b"\n%%EOF")
    Path(path).write_bytes(out.getvalue())
