"""Crop tool — drag a rectangle, then commit to resize the document."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from .base import Tool, ToolContext


class CropTool(Tool):
    name = "crop"
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._start: Optional[Tuple[float, float]] = None
        self._current: Optional[Tuple[float, float]] = None

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._start = (cx, cy)
        self._current = (cx, cy)
        context.request_repaint()

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
        x0 = max(0, min(context.canvas_width, x0))
        x1 = max(0, min(context.canvas_width, x1))
        y0 = max(0, min(context.canvas_height, y0))
        y1 = max(0, min(context.canvas_height, y1))
        # Crop: resize document, translate layers, clip their pixels.
        new_w = x1 - x0
        new_h = y1 - y0
        if new_w <= 0 or new_h <= 0:
            self._start = self._current = None
            return
        _crop_document(context, x0, y0, new_w, new_h)
        context.session.width = new_w
        context.session.height = new_h
        context.canvas_width = new_w
        context.canvas_height = new_h
        context.request_reload()
        self._start = self._current = None


def _crop_document(context: ToolContext, x0: int, y0: int, new_w: int, new_h: int) -> None:
    """Shift all layers by (-x0, -y0) and clip their pixels to the new bounds."""
    from ..document.layers import Transform
    from ..rendering.layer_renderer import transform_matrix, warp_into_canvas
    from ..rendering.blend_modes import apply_blend

    cache = getattr(context.session, "_image_cache", {})
    for layer in context.session.layers:
        if layer.transform.origin is not None:
            ox, oy = layer.transform.origin
            layer.transform = Transform(
                origin=(ox - x0, oy - y0),
                size=layer.transform.size,
                rotation=layer.transform.rotation,
                flip_x=layer.transform.flip_x,
                flip_y=layer.transform.flip_y,
                sampling=layer.transform.sampling,
            )
        img = cache.get(layer.id)
        if img is None:
            continue
        matrix = transform_matrix(layer.transform)
        new_img = warp_into_canvas(img, matrix, (new_h, new_w))
        cache[layer.id] = new_img
