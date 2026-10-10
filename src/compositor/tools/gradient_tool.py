"""Gradient tool — fills a linear or radial gradient on the active layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from .base import Tool, ToolContext


@dataclass
class GradientStop:
    position: float  # 0..1
    color: Tuple[int, int, int, int]


@dataclass
class GradientSettings:
    kind: str = "linear"  # "linear" | "radial"
    stops: list = None  # type: ignore[assignment]
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if self.stops is None:
            self.stops = [
                GradientStop(0.0, (0, 0, 0, 255)),
                GradientStop(1.0, (255, 255, 255, 255)),
            ]


def _render_gradient(shape: Tuple[int, int], settings: GradientSettings) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if settings.kind == "radial":
        cx, cy = w / 2.0, h / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_r = math.sqrt(cx * cx + cy * cy)
        t = np.clip(r / max_r, 0.0, 1.0)
    else:
        t = xx / max(w - 1, 1)
    # Linear blend across stops.
    stops = sorted(settings.stops, key=lambda s: s.position)
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for i in range(len(stops) - 1):
        s0 = stops[i]
        s1 = stops[i + 1]
        mask = (t >= s0.position) & (t <= s1.position)
        if not mask.any():
            continue
        local = np.clip((t - s0.position) / max(s1.position - s0.position, 1e-6), 0.0, 1.0)
        c0 = np.array(s0.color, dtype=np.float32)
        c1 = np.array(s1.color, dtype=np.float32)
        rgba += mask[..., None] * (c0 * (1.0 - local[..., None]) + c1 * local[..., None])
    rgba *= settings.opacity
    out = np.clip(rgba, 0, 255).astype(np.uint8)
    return out


class GradientTool(Tool):
    name = "gradient"
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = GradientSettings()
        self._start: Optional[Tuple[float, float]] = None

    def set_stops(self, stops) -> None:
        self.settings.stops = stops

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._start = (cx, cy)

    def mouse_release(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None:
            return
        cache = getattr(context.session, "_image_cache", {})
        img = cache.get(layer.id)
        if img is None:
            return
        # Simple: produce a gradient the size of the canvas, then mask to the layer's alpha.
        grad = _render_gradient((context.canvas_height, context.canvas_width), self.settings)
        # Blend over existing pixels using the gradient's alpha.
        sa = grad[..., 3:4].astype(np.float32) / 255.0
        img[..., :3] = (grad[..., :3].astype(np.float32) * sa + img[..., :3].astype(np.float32) * (1.0 - sa)).astype(np.uint8)
        cache[layer.id] = img
        context.request_repaint()
        self._start = None


import math  # noqa: E402
