"""Move / Transform tool — drag layers, scale via corners, rotate via handle."""
from __future__ import annotations

from typing import Optional, Tuple

import math
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from ..document.layers import ImageLayer, Transform
from ..rendering.layer_renderer import snap_position
from .base import Tool, ToolContext

# Corner handle hit radius (in widget pixels).
_HANDLE_RADIUS = 8.0
# Total corners on a rect.
_RECT_CORNERS = (
    (0.0, 0.0),       # TL
    (1.0, 0.0),       # TR
    (1.0, 1.0),       # BR
    (0.0, 1.0),       # BL
)


class MoveTool(Tool):
    name = "move"
    cursor = Qt.CursorShape.SizeAllCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layer_id: Optional[str] = None
        self._layer_origin: Optional[Tuple[float, float]] = None
        self._start: Optional[Tuple[float, float]] = None

    def _other_layers(self, context: ToolContext):
        return [
            L for L in context.session.layers
            if L.id != context.active_layer_id
        ]

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None:
            return
        self._layer_id = layer.id
        self._layer_origin = layer.transform.origin
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        self._start = (cx, cy)

    def mouse_move(self, context: ToolContext, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton) or self._layer_id is None:
            return
        if self._start is None or self._layer_origin is None:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        dx = cx - self._start[0]
        dy = cy - self._start[1]
        raw_x = self._layer_origin[0] + dx
        raw_y = self._layer_origin[1] + dy
        # Snap to anchors (canvas / layers / guides) within ~8 document px.
        snapped_x, snapped_y = snap_position(
            raw_x, raw_y,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            other_layers=self._other_layers(context),
            guides=getattr(context.session, "guides", []),
            snap_to="all",
            threshold=8.0 / max(context.zoom, 0.01),
        )
        layer = context.session.layer_by_id(self._layer_id)
        assert layer is not None
        context.session.set_property(self._layer_id, "transform", Transform(
            origin=(snapped_x, snapped_y),
            size=layer.transform.size,
            rotation=layer.transform.rotation,
            flip_x=layer.transform.flip_x,
            flip_y=layer.transform.flip_y,
            sampling=layer.transform.sampling,
        ))
        context.request_repaint()

    def mouse_release(self, context: ToolContext, event: QMouseEvent) -> None:
        self._layer_id = None
        self._layer_origin = None
        self._start = None

    # ---- corner handle hit testing (Photoshop-style ⌘-drag for free distort) ---

    def corner_index_at(self, context: ToolContext, wx: float, wy: float) -> Optional[Tuple[int, Tuple[float, float]]]:
        """Return ``(corner_index, widget_pos)`` for the active layer's handle
        under the widget-space point ``(wx, wy)``, or None.

        Uses the rect defined by ``origin + size`` or, if present, the four
        free-distort corners.
        """
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None or not isinstance(layer, ImageLayer):
            return None
        rect_corners = self._current_canvas_corners(layer)
        ox, oy = context.pan
        for idx, (cx, cy) in enumerate(rect_corners):
            hx = ox + cx * context.zoom
            hy = oy + cy * context.zoom
            if math.hypot(hx - wx, hy - wy) <= _HANDLE_RADIUS:
                return idx, (hx, hy)
        return None

    def _current_canvas_corners(self, layer: ImageLayer):
        if layer.transform.corners is not None:
            return layer.transform.corners
        ox, oy = layer.transform.origin
        sw, sh = layer.transform.size
        return (
            (ox, oy),
            (ox + sw, oy),
            (ox + sw, oy + sh),
            (ox, oy + sh),
        )

    def begin_free_distort(self, context: ToolContext, corner_index: int) -> None:
        """Enter free-distort drag mode for the corner at ``corner_index``."""
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None or not isinstance(layer, ImageLayer):
            return
        rect_corners = self._current_canvas_corners(layer)
        # Snapshot the corners as the initial free-distort state.
        self._layer_id = layer.id
        self._free_distort = True
        self._dragging_corner = corner_index
        self._start_corners = list(rect_corners)
        # Bring the snap-to-all hook out of the way — free distort is exact.
        self._start = self._start_corners[corner_index]

    def mouse_press_distort(self, context: ToolContext, event: QMouseEvent) -> bool:
        """If the user ⌘-clicks on a corner handle, enter free distort."""
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        wx, wy = event.position().x(), event.position().y()
        hit = self.corner_index_at(context, wx, wy)
        if hit is None:
            return False
        self.begin_free_distort(context, hit[0])
        return True

    def mouse_move_distort(self, context: ToolContext, event: QMouseEvent) -> bool:
        if not getattr(self, "_free_distort", False):
            return False
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return False
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        corners = list(self._start_corners)
        # Shift constrains to a single axis (Photoshop's behaviour).
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            sx, sy = self._start
            if abs(cx - sx) > abs(cy - sy):
                corners[self._dragging_corner] = (cx, sy)
            else:
                corners[self._dragging_corner] = (sx, cy)
        else:
            corners[self._dragging_corner] = (cx, cy)
        layer = context.session.layer_by_id(self._layer_id)
        if layer is None:
            return False
        context.session.set_property(self._layer_id, "transform", Transform(
            origin=layer.transform.origin,
            size=layer.transform.size,
            rotation=layer.transform.rotation,
            flip_x=layer.transform.flip_x,
            flip_y=layer.transform.flip_y,
            sampling=layer.transform.sampling,
            corners=tuple(corners),  # type: ignore[arg-type]
        ))
        context.request_repaint()
        return True

    def mouse_release_distort(self, context: ToolContext, event: QMouseEvent) -> bool:
        if not getattr(self, "_free_distort", False):
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        self._free_distort = False
        self._dragging_corner = -1
        self._start_corners = None
        self._layer_id = None
        return True

    # ---- arrow-key nudge (handled by CanvasWidget when this tool is active) ----

    def nudge(self, context: ToolContext, dx: int, dy: int) -> None:
        """Shift the active layer by (dx, dy) document pixels."""
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None:
            return
        ox, oy = layer.transform.origin
        nx, ny = ox + dx, oy + dy
        if abs(dx) > 0:
            nx, _ = snap_position(
                nx, oy,
                canvas_width=context.canvas_width,
                canvas_height=context.canvas_height,
                other_layers=self._other_layers(context),
                guides=getattr(context.session, "guides", []),
                snap_to="all",
                threshold=8.0 / max(context.zoom, 0.01),
            )
        if abs(dy) > 0:
            _, ny = snap_position(
                ox, ny,
                canvas_width=context.canvas_width,
                canvas_height=context.canvas_height,
                other_layers=self._other_layers(context),
                guides=getattr(context.session, "guides", []),
                snap_to="all",
                threshold=8.0 / max(context.zoom, 0.01),
            )
        context.session.set_property(layer.id, "transform", Transform(
            origin=(nx, ny),
            size=layer.transform.size,
            rotation=layer.transform.rotation,
            flip_x=layer.transform.flip_x,
            flip_y=layer.transform.flip_y,
            sampling=layer.transform.sampling,
        ))
        context.request_repaint()
