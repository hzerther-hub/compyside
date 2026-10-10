"""Selection masks — marquee, lasso, wand.

Each function returns a uint8 HxW mask (255 inside the selection, 0 outside).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from . import pixel_ops as po


def rectangle_mask(canvas_shape: Tuple[int, int], rect: Tuple[int, int, int, int]) -> np.ndarray:
    """rect = (x0, y0, x1, y1) in pixels; clipped to canvas."""
    h, w = canvas_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = rect
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def ellipse_mask(canvas_shape: Tuple[int, int], rect: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = canvas_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return mask
    cx = (x0 + x1 - 1) / 2.0
    cy = (y0 + y1 - 1) / 2.0
    rx = max(1, (x1 - x0) / 2.0)
    ry = max(1, (y1 - y0) / 2.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.where(t <= 1.0, 255, 0).astype(np.uint8)
    return mask


def lasso_mask(canvas_shape: Tuple[int, int], points: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Fill a closed polygon defined by ``points``."""
    h, w = canvas_shape
    if len(points) < 3:
        return np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    inside = np.zeros((h, w), dtype=bool)
    n = len(points)
    for i in range(n):
        xa, ya = points[i]
        xb, yb = points[(i + 1) % n]
        cond1 = (ya <= yy) & (yy < yb)
        cond2 = (yb <= yy) & (yy < ya)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = yb - ya
            # Avoid both /0 and the `max(neg, 1e-6) → 1e-6` sign-flip trap.
            safe_denom = np.where(np.abs(denom) < 1e-6, 1.0, denom).astype(np.float32)
            x_int = xa + (yy - ya) * (xb - xa) / safe_denom
        crossing = (cond1 | cond2) & (xx < x_int)
        inside = inside ^ crossing
    return np.where(inside, 255, 0).astype(np.uint8)


def wand_mask(image: np.ndarray, seed: Tuple[int, int], tolerance: int = 32) -> np.ndarray:
    """Flood-fill from ``seed`` over pixels within ``tolerance`` of the seed colour.

    ``image`` is HxWx3 or HxWx4 uint8.
    """
    h, w = image.shape[:2]
    rgb = image[..., :3] if image.ndim == 3 else image
    sx, sy = seed
    sx = max(0, min(w - 1, sx))
    sy = max(0, min(h - 1, sy))
    seed_color = rgb[sy, sx].astype(np.int32)
    diff = np.abs(rgb.astype(np.int32) - seed_color).max(axis=-1)
    mask = (diff <= tolerance).astype(np.uint8) * 255
    return mask


def feather(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return mask
    blurred = po.gaussian_blur(mask[..., None], radius)[..., 0]
    return blurred.astype(np.uint8)


def expand(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    return np.clip(mask.astype(np.int32) + pixels * 255, 0, 255).astype(np.uint8)


def contract(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    out = mask.astype(np.int32) - pixels * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int32) - b.astype(np.int32)).astype(np.uint8)


def intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)
