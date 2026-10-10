"""Object Select tool — rectangle + cv2 GrabCut."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from ..rendering.object_select import object_select
from .base import Tool, ToolContext


class ObjectSelectTool(Tool):
    name = "object-select"
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

    def mouse_move(self, context: ToolContext, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._current = (cx, cy)
        context.request_repaint()

    def mouse_release(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._current = (cx, cy)
        x0, y0 = self._start
        x1, y1 = self._current
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        if x1 <= x0 or y1 <= y0:
            self._start = self._current = None
            return
        # Build a composite to give GrabCut the rendered pixels.
        from ..rendering.layer_renderer import composite_document, CompositorOptions
        cache = getattr(context.session, "_image_cache", {})
        composite = composite_document(
            context.session,
            CompositorOptions(
                canvas_width=context.canvas_width,
                canvas_height=context.canvas_height,
                image_loader=cache.get,
                mask_loader=getattr(context.session, "_mask_cache", {}).get,
            ),
        )
        try:
            mask = object_select(composite, (x0, y0, x1, y1))
        except RuntimeError:
            mask = None
        if mask is not None:
            context.session._selection_mask = mask  # type: ignore[attr-defined]
        self._start = self._current = None
        context.request_repaint()
