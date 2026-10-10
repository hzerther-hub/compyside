"""Shape tool — rectangles, rounded rectangles, ellipses, lines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from .base import Tool, ToolContext


@dataclass
class ShapeSettings:
    kind: str = "rectangle"  # rectangle | rounded | ellipse | line
    fill: Tuple[int, int, int, int] = (255, 255, 255, 255)
    stroke: Optional[Tuple[int, int, int, int]] = None
    stroke_width: int = 0
    corner_radius: int = 8


class ShapeTool(Tool):
    name = "shape"
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = ShapeSettings()
        self._start: Optional[Tuple[float, float]] = None
        self._current: Optional[Tuple[float, float]] = None

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._start = (cx, cy)
        self._current = (cx, cy)

    def mouse_move(self, context: ToolContext, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._current = (cx, cy)
        context.request_repaint()

    def mouse_release(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._start is None or self._current is None:
            return
        x0, y0 = self._start
        x1, y1 = self._current
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        if x1 <= x0 or y1 <= y0:
            self._start = self._current = None
            return
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None:
            self._start = self._current = None
            return
        cache = getattr(context.session, "_image_cache", {})
        img = cache.get(layer.id)
        if img is None:
            self._start = self._current = None
            return
        mask = self._render_mask((context.canvas_height, context.canvas_width), (x0, y0, x1, y1))
        # Composite over existing alpha.
        out_alpha = np.maximum(img[..., 3], mask).astype(np.uint8)
        fill = np.array(self.settings.fill, dtype=np.uint8).reshape(1, 1, 4)
        composite = (mask[..., None].astype(np.float32) / 255.0) * fill + \
                    (1 - mask[..., None].astype(np.float32) / 255.0) * img
        if self.settings.stroke is not None and self.settings.stroke_width > 0:
            outline = self._stroke_mask((context.canvas_height, context.canvas_width), (x0, y0, x1, y1))
            stroke_color = np.array(self.settings.stroke, dtype=np.uint8).reshape(1, 1, 4)
            composite = np.where(outline[..., None] > 0, stroke_color, composite.astype(np.uint8))
            out_alpha = np.maximum(out_alpha, outline).astype(np.uint8)
        img[..., :3] = composite[..., :3].astype(np.uint8)
        img[..., 3] = out_alpha
        cache[layer.id] = img
        context.request_repaint()
        self._start = self._current = None

    def _render_mask(self, shape: Tuple[int, int], rect: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = shape
        x0, y0, x1, y1 = rect
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.settings.kind == "rectangle":
            mask[y0:y1, x0:x1] = 255
        elif self.settings.kind == "ellipse":
            cx, cy = (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0
            rx = max(1, (x1 - x0) / 2.0)
            ry = max(1, (y1 - y0) / 2.0)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            t = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            mask = np.where(t <= 1.0, 255, 0).astype(np.uint8)
        elif self.settings.kind == "rounded":
            mask[y0:y1, x0:x1] = 255
            r = min(self.settings.corner_radius, (x1 - x0) // 2, (y1 - y0) // 2)
            for cx, cy in ((x0 + r, y0 + r), (x1 - r - 1, y0 + r), (x0 + r, y1 - r - 1), (x1 - r - 1, y1 - r - 1)):
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
                mask = np.where(inside, 255, mask)
        elif self.settings.kind == "line":
            # Bresenham line, then dilate by stroke_width.
            mask = self._line_mask(shape, (x0, y0), (x1, y1), self.settings.stroke_width or 1)
        return mask

    def _line_mask(self, shape: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int], width: int) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0 = a
        x1, y1 = b
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps, dtype=np.int32)
        ys = np.linspace(y0, y1, steps, dtype=np.int32)
        for x, y in zip(xs, ys):
            yy, xx = np.mgrid[max(0, y - width):min(h, y + width + 1), max(0, x - width):min(w, x + width + 1)]
            mask[yy, xx] = 255
        return mask

    def _stroke_mask(self, shape: Tuple[int, int], rect: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        x0, y0, x1, y1 = rect
        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)
        outer[y0:y1, x0:x1] = 255
        sw = self.settings.stroke_width
        inner[max(0, y0 + sw):min(h, y1 - sw), max(0, x0 + sw):min(w, x1 - sw)] = 255
        return np.maximum(outer - inner, 0).astype(np.uint8)
