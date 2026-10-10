"""Brush engine — stamping, smoothing, hardness falloff.

The brush produces RGBA "dabs" at requested positions and blits them onto
a target RGBA canvas with the configured blend mode (paint or erase).

Smoothing is a simple input-easing filter: it stores the last few raw
positions and emits a smoothed point. Real Compositor does more (velocity-
aware smoothing) but this is adequate for an MVP.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class BrushSettings:
    size: int = 24
    hardness: float = 0.8  # 0..1 — fraction of radius that's fully opaque
    opacity: float = 1.0  # 0..1
    spacing: float = 0.1  # fraction of size
    smoothing: float = 0.5  # 0..1 — higher = smoother stroke
    flow: float = 1.0
    mode: str = "paint"  # "paint" | "erase"
    color: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class BrushState:
    """Mutable state for an in-progress stroke."""

    last_position: Optional[Tuple[float, float]] = None
    raw_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))
    smoothed_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))


def make_stamp(settings: BrushSettings) -> np.ndarray:
    """Pre-render a single brush dab as an RGBA uint8 stamp.

    The stamp is square (size × size) with a radial falloff:
    - inside (hardness * radius) → fully opaque
    - outside → linear ramp to zero
    The alpha channel carries the falloff; RGB is the brush color.
    """
    s = max(1, int(settings.size))
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    cx = cy = (s - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = s / 2.0
    inner = max(0.0, settings.hardness) * radius
    # Normalised radius 0..1.
    t = np.clip((r - inner) / max(radius - inner, 1e-6), 0.0, 1.0)
    alpha = ((1.0 - t) * 255.0 * settings.opacity * settings.flow).astype(np.uint8)
    stamp = np.zeros((s, s, 4), dtype=np.uint8)
    stamp[..., 0] = settings.color[0]
    stamp[..., 1] = settings.color[1]
    stamp[..., 2] = settings.color[2]
    stamp[..., 3] = alpha
    return stamp


def stamp_distance_step(settings: BrushSettings) -> float:
    return max(1.0, settings.size * max(0.01, settings.spacing))


def smoothed_point(state: BrushState, target: Tuple[float, float], smoothing: float) -> Tuple[float, float]:
    """Exponential smoothing of the incoming point stream."""
    state.raw_history.append(target)
    if state.last_position is None:
        state.last_position = target
        state.smoothed_history.append(target)
        return target
    if smoothing <= 0:
        state.last_position = target
        state.smoothed_history.append(target)
        return target
    lx, ly = state.last_position
    nx = lx + (target[0] - lx) * (1.0 - smoothing)
    ny = ly + (target[1] - ly) * (1.0 - smoothing)
    state.last_position = (nx, ny)
    state.smoothed_history.append(state.last_position)
    return state.last_position


def apply_dab(
    canvas: np.ndarray,
    stamp: np.ndarray,
    center: Tuple[float, float],
    mode: str = "paint",
) -> None:
    """Blit ``stamp`` onto ``canvas`` at ``center``. In-place."""
    h_can, w_can = canvas.shape[:2]
    h, w = stamp.shape[:2]
    cx, cy = center
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1 = x0 + w
    y1 = y0 + h
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    src_x1 = w - max(0, x1 - w_can)
    src_y1 = h - max(0, y1 - h_can)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    if mode == "paint":
        src_rgb = stamp[src_y0:src_y1, src_x0:src_x1, :3].astype(np.float32)
        src_a = stamp[src_y0:src_y1, src_x0:src_x1, 3:4].astype(np.float32) / 255.0
        dst = canvas[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
        out_rgb = src_rgb * src_a + dst[..., :3] * (1.0 - src_a)
        out_a = np.maximum(dst[..., 3:4], (src_a * 255.0).astype(np.float32))
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, 3:4] = np.clip(out_a, 0, 255).astype(np.uint8)
    elif mode == "erase":
        src_a = stamp[src_y0:src_y1, src_x0:src_x1, 3:4].astype(np.float32) / 255.0
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, 3:4] = np.clip(
            canvas[dst_y0:dst_y1, dst_x0:dst_x1, 3:4].astype(np.float32) * (1.0 - src_a),
            0, 255,
        ).astype(np.uint8)


def stroke_segment(
    canvas: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    settings: BrushSettings,
    state: BrushState,
    stamp: np.ndarray,
) -> None:
    """Stamp a brush stroke from ``start`` to ``end`` onto ``canvas``."""
    step = stamp_distance_step(settings)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        smoothed = smoothed_point(state, end, settings.smoothing)
        apply_dab(canvas, stamp, smoothed, settings.mode)
        return
    n = max(1, int(math.ceil(dist / step)))
    for i in range(n + 1):
        t = i / n
        target = (start[0] + dx * t, start[1] + dy * t)
        smoothed = smoothed_point(state, target, settings.smoothing)
        apply_dab(canvas, stamp, smoothed, settings.mode)
