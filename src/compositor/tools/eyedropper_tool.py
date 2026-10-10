"""Eyedropper — sample a colour from the canvas."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from .base import Tool, ToolContext


class EyedropperTool(Tool):
    name = "eyedropper"
    cursor = Qt.CursorShape.CrossCursor
    last_color = None

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        ix, iy = int(round(cx)), int(round(cy))
        cache = getattr(context.session, "_image_cache", {})
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None:
            return
        img = cache.get(layer.id)
        if img is None or not (0 <= ix < img.shape[1] and 0 <= iy < img.shape[0]):
            return
        rgb = img[iy, ix, :3].tolist()
        self.last_color = tuple(int(v) for v in rgb)
