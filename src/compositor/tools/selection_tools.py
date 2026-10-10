"""Selection tools — marquee (rect / ellipse), freehand and polygonal lasso, wand."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from ..rendering import selections as sel
from .base import Tool, ToolContext


class _SelectionTool(Tool):
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
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._start is None or self._current is None:
            return
        mask = self._build_mask(context)
        if mask is None:
            return
        self._apply_mask(context, mask)
        self._start = self._current = None
        context.request_repaint()

    def _build_mask(self, context: ToolContext) -> Optional[np.ndarray]:
        return None

    def _apply_mask(self, context: ToolContext, mask: np.ndarray) -> None:
        existing = getattr(context.session, "_selection_mask", None)
        mode = "replace"
        if event_is_shift(event := None) if False else False:  # placeholder, real logic via modifiers
            mode = "add"
        if mode == "replace" or existing is None:
            context.session._selection_mask = mask
        elif mode == "add":
            context.session._selection_mask = sel.union(existing, mask)
        elif mode == "subtract":
            context.session._selection_mask = sel.difference(existing, mask)


def event_is_shift(event):
    return bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)


class RectangleMarqueeTool(_SelectionTool):
    name = "marquee-rect"
    cursor = Qt.CursorShape.CrossCursor

    def _build_mask(self, context: ToolContext) -> np.ndarray:
        x0, y0 = self._start
        x1, y1 = self._current
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        return sel.rectangle_mask((context.canvas_height, context.canvas_width), (x0, y0, x1, y1))


class EllipseMarqueeTool(_SelectionTool):
    name = "marquee-ellipse"
    cursor = Qt.CursorShape.CrossCursor

    def _build_mask(self, context: ToolContext) -> np.ndarray:
        x0, y0 = self._start
        x1, y1 = self._current
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        return sel.ellipse_mask((context.canvas_height, context.canvas_width), (x0, y0, x1, y1))


class LassoTool(_SelectionTool):
    """Freehand + polygonal lasso.

    Default mode is freehand — drag to draw a continuous outline. Hold
    Alt (or set ``polygonal`` to True) to switch to polygonal mode, where
    each click adds a vertex; Enter or double-click closes the polygon,
    Escape cancels.
    """

    name = "lasso"
    cursor = Qt.CursorShape.CrossCursor
    _points: List[Tuple[int, int]]

    def __init__(self, parent=None, polygonal: bool = False):
        super().__init__(parent)
        self._points = []
        self._polygonal = bool(polygonal)
        # Active state for the polygon: building between clicks.
        self._building = False

    def set_polygonal(self, enabled: bool) -> None:
        """Switch between freehand and polygonal mode."""
        self._polygonal = bool(enabled)
        # If we drop out of polygon mode mid-build, commit what's there.
        if not self._polygonal and self._building and len(self._points) >= 3:
            self._commit_polygon(context=None)  # type: ignore[arg-type]
        self._points = []
        self._building = False

    @property
    def polygonal(self) -> bool:
        return self._polygonal

    def _commit_polygon(self, context: Optional[ToolContext]) -> None:
        if len(self._points) < 3:
            return
        if context is None:
            return
        mask = sel.lasso_mask((context.canvas_height, context.canvas_width), self._points)
        self._apply_mask(context, mask)

    # -- freehand mode (default) -----------------------------------------

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if self._polygonal:
            self._polygon_press(context, event)
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._points = [(int(round(cx)), int(round(cy)))]

    def mouse_move(self, context: ToolContext, event: QMouseEvent) -> None:
        if self._polygonal:
            # Polygon mode only redraws the rubber-band to the cursor.
            context.request_repaint()
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._points.append((int(round(cx)), int(round(cy))))
        context.request_repaint()

    def mouse_release(self, context: ToolContext, event: QMouseEvent) -> None:
        if self._polygonal:
            return  # polygon mode commits via Enter / double-click only.
        if self._points:
            mask = sel.lasso_mask((context.canvas_height, context.canvas_width), self._points)
            self._apply_mask(context, mask)
        self._points = []
        context.request_repaint()

    # -- polygon mode -----------------------------------------------------

    def _polygon_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            # Right-click cancels the in-progress polygon.
            self._points = []
            self._building = False
            context.request_repaint()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        pt = (int(round(cx)), int(round(cy)))
        if not self._building:
            self._points = [pt]
            self._building = True
        else:
            self._points.append(pt)
        context.request_repaint()

    def mouse_double_click(self, context: ToolContext, event: QMouseEvent) -> None:
        if not self._polygonal:
            return
        self._commit_polygon(context)
        self._points = []
        self._building = False
        context.request_repaint()

    def key_press(self, context: ToolContext, event: QKeyEvent) -> None:
        if not self._polygonal:
            return
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._commit_polygon(context)
            self._points = []
            self._building = False
            context.request_repaint()
            event.accept()
        elif key == Qt.Key.Key_Escape:
            self._points = []
            self._building = False
            context.request_repaint()
            event.accept()

    # -- overlay access for the canvas paint ------------------------------

    def active_outline(self) -> Optional[List[Tuple[int, int]]]:
        """Return the in-progress outline for the canvas to draw over."""
        return list(self._points) if self._points else None

    @property
    def building_polygon(self) -> bool:
        return self._polygonal and self._building


class PolygonalLassoTool(LassoTool):
    """Lasso variant that defaults to polygon mode."""

    name = "polygon-lasso"
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent, polygonal=True)


class WandTool(Tool):
    name = "wand"
    cursor = Qt.CursorShape.CrossCursor

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        ix, iy = int(round(cx)), int(round(cy))
        # Composite the document so the wand sees the result, not just the active layer.
        from ..rendering.layer_renderer import composite_document, CompositorOptions

        cache = getattr(context.session, "_image_cache", {})
        mask_loader = getattr(context.session, "_mask_cache", {}).get
        canvas = composite_document(
            context.session,
            CompositorOptions(
                canvas_width=context.canvas_width,
                canvas_height=context.canvas_height,
                image_loader=cache.get,
                mask_loader=mask_loader,
            ),
        )
        mask = sel.wand_mask(canvas, (ix, iy), tolerance=32)
        context.session._selection_mask = mask
        context.request_repaint()
