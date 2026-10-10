"""Brush and eraser tools — paint / erase on the active layer."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from ..rendering import brush as brush_engine
from .base import Tool, ToolContext


class BrushTool(Tool):
    name = "brush"
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = brush_engine.BrushSettings()
        self._stamp: Optional[np.ndarray] = None
        self._state = brush_engine.BrushState()

    def set_settings(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
        self._stamp = brush_engine.make_stamp(self.settings)

    def activate(self, context: ToolContext) -> None:
        self._stamp = brush_engine.make_stamp(self.settings)
        self._state = brush_engine.BrushState()

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        canvas = self._active_pixels(context)
        if canvas is None:
            return
        if self._stamp is None:
            self._stamp = brush_engine.make_stamp(self.settings)
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._state = brush_engine.BrushState(last_position=(cx, cy))
        brush_engine.apply_dab(canvas, self._stamp, (cx, cy), self.settings.mode)
        self._commit_pixels(context, canvas)

    def mouse_move(self, context: ToolContext, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        canvas = self._active_pixels(context)
        if canvas is None or self._stamp is None:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        prev = self._state.last_position or (cx, cy)
        brush_engine.stroke_segment(canvas, prev, (cx, cy), self.settings, self._state, self._stamp)
        self._commit_pixels(context, canvas)

    def _active_pixels(self, context: ToolContext) -> Optional[np.ndarray]:
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None:
            return None
        image = getattr(context.session, "_image_cache", {}).get(layer.id)
        if image is None:
            return None
        return image

    def _commit_pixels(self, context: ToolContext, canvas: np.ndarray) -> None:
        cache = getattr(context.session, "_image_cache", {})
        cache[context.active_layer_id] = canvas
        context.request_repaint()


class EraserTool(BrushTool):
    name = "eraser"
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings.mode = "erase"
