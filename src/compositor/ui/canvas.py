"""Canvas widget — the document viewport.

A ``QWidget`` that paints the composited document via ``QPainter`` + a
``QImage`` backed by a NumPy RGBA buffer. Supports pan/zoom, a grid,
rulers overlay, transform handles for the active layer, a marching-ants
selection outline, the brush preview circle, and the navigator minimap.

All painting is CPU-side; the GPU canvas is a later optimisation.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from ..document.editor_session import EditorSession
from ..document.layers import AnyLayer, ImageLayer, AdjustmentLayer, GroupLayer
from ..rendering.layer_renderer import composite_document, CompositorOptions
from ..tools.base import ToolContext


class CanvasWidget(QWidget):
    """The main editing surface."""

    document_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.session: Optional[EditorSession] = None
        self._image_cache: Dict[str, np.ndarray] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}

        # View transform.
        self._zoom = 1.0
        self._pan = (0.0, 0.0)

        # Active tool + overlays.
        self._active_tool = None
        self._active_tool_name = ""
        self._show_grid = False
        self._grid_spacing = 32
        self._show_rulers = True
        self._show_navigator = True
        self._brush_preview_pos: Optional[Tuple[float, float]] = None

        # Guide interaction state.
        # ``_dragging_guide`` is the guide id being dragged (None = no drag).
        self._dragging_guide: Optional[str] = None

        # Background checker pattern (transparency indicator).
        self._checker = self._build_checker(16)
        self._placeholder_text = self._placeholder_for(None)

        # Mipmap cache for zoomed-out compositing.
        from ..rendering.cache import MipmapCache
        self._mip_cache = MipmapCache(max_entries=16)

    # ---- public API ------------------------------------------------------

    def set_session(self, session: Optional[EditorSession]) -> None:
        self.session = session
        # Sync the image / mask caches from the session so that the canvas
        # sees every layer's pixels without each tool re-loading them.
        self._image_cache = dict(getattr(session, "_image_cache", {})) if session is not None else {}
        self._mask_cache = dict(getattr(session, "_mask_cache", {})) if session is not None else {}
        # New session → different layers → mip cache is stale.
        self.invalidate_mip_cache()
        if session is not None:
            self._zoom = self._fit_zoom()
            cx = (self.width() - session.width * self._zoom) / 2.0
            cy = (self.height() - session.height * self._zoom) / 2.0
            self._pan = (cx, cy)
        # Refresh the placeholder text in case the language has switched.
        self._placeholder_text = self._placeholder_for(session)
        self.update()

    def _placeholder_for(self, session: Optional[EditorSession]) -> str:
        from .. import i18n

        if session is None:
            return i18n.tr("no_project")
        return i18n.tr(
            "project_loaded"
        ).format(w=session.width, h=session.height, n=len(session.layers))

    def sync_caches_from_session(self) -> None:
        """Re-pull the latest in-memory image / mask caches from the session."""
        if self.session is None:
            return
        self._image_cache = dict(getattr(self.session, "_image_cache", {}))
        self._mask_cache = dict(getattr(self.session, "_mask_cache", {}))
        self.update()

    def set_active_tool(self, tool_name: str, tool) -> None:
        if self._active_tool is not None:
            ctx = self._build_context()
            self._active_tool.deactivate(ctx)
        self._active_tool = tool
        self._active_tool_name = tool_name
        self.setCursor(tool.cursor if tool else Qt.CursorShape.ArrowCursor)
        if tool is not None:
            tool.activate(self._build_context())
        self.update()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.05, min(32.0, zoom))
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.25)

    def fit_to_window(self) -> None:
        if self.session is None:
            return
        self._zoom = self._fit_zoom()
        cx = (self.width() - self.session.width * self._zoom) / 2.0
        cy = (self.height() - self.session.height * self._zoom) / 2.0
        self._pan = (cx, cy)
        self.update()

    def toggle_grid(self, enabled: Optional[bool] = None) -> None:
        self._show_grid = not self._show_grid if enabled is None else enabled
        self.update()

    def toggle_rulers(self, enabled: Optional[bool] = None) -> None:
        self._show_rulers = not self._show_rulers if enabled is None else enabled
        self.update()

    # ---- event handlers --------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 1.0 / 1.25
        # Zoom around the cursor.
        old_zoom = self._zoom
        new_zoom = max(0.05, min(32.0, old_zoom * factor))
        cursor = event.position()
        # Adjust pan so the canvas point under the cursor stays put.
        dx = (cursor.x() - self._pan[0]) * (new_zoom / old_zoom - 1.0)
        dy = (cursor.y() - self._pan[1]) * (new_zoom / old_zoom - 1.0)
        self._pan = (self._pan[0] - dx, self._pan[1] - dy)
        self._zoom = new_zoom
        self.update()

    RULER_THICKNESS = 16

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.session is None or self._active_tool is None:
            return
        # Middle-click pan, regardless of tool.
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_anchor = (event.position().x(), event.position().y(), self._pan[0], self._pan[1])
            return
        # Drag-from-ruler: create a new guide.
        if event.button() == Qt.MouseButton.LeftButton and self._show_rulers:
            x, y = event.position().x(), event.position().y()
            if y < self.RULER_THICKNESS and x >= self.RULER_THICKNESS:
                self._dragging_guide = self._create_guide("h", self._to_doc_y(y))
                return
            if x < self.RULER_THICKNESS and y >= self.RULER_THICKNESS:
                self._dragging_guide = self._create_guide("v", self._to_doc_x(x))
                return
        # Free-distort hook (move tool only): ⌘-click a corner handle.
        if hasattr(self._active_tool, "mouse_press_distort"):
            if self._active_tool.mouse_press_distort(self._build_context(), event):  # type: ignore[attr-defined]
                return
        self._active_tool.mouse_press(self._build_context(), event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if getattr(self, "_panning", False):
            ax, ay, apx, apy = self._pan_anchor
            self._pan = (apx + event.position().x() - ax, apy + event.position().y() - ay)
            self.update()
            return
        # Drag the active guide.
        if self._dragging_guide and self.session is not None:
            guide = self.session.guide_by_id(self._dragging_guide)
            if guide is not None:
                if guide.orientation == "h":
                    pos = self._to_doc_y(event.position().y())
                else:
                    pos = self._to_doc_x(event.position().x())
                self.session.move_guide(self._dragging_guide, pos)
                self.update()
                return
        # Free-distort drag hook.
        if self._active_tool is not None and hasattr(self._active_tool, "mouse_move_distort"):
            if self._active_tool.mouse_move_distort(self._build_context(), event):  # type: ignore[attr-defined]
                self.update()
                return
        self._brush_preview_pos = self.to_canvas_coords(event.position().x(), event.position().y())
        if self.session is None or self._active_tool is None:
            self.update()
            return
        self._active_tool.mouse_move(self._build_context(), event)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if getattr(self, "_panning", False) and event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            return
        if self._dragging_guide and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_guide = None
            self.update()
            return
        if self._active_tool is not None and hasattr(self._active_tool, "mouse_release_distort"):
            if self._active_tool.mouse_release_distort(self._build_context(), event):  # type: ignore[attr-defined]
                self.update()
                return
        if self.session is None or self._active_tool is None:
            return
        self._active_tool.mouse_release(self._build_context(), event)
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._active_tool is None or self.session is None:
            return
        # Arrow-key nudge for the move tool (active layer).
        from PySide6.QtCore import QEvent
        if (self._active_tool_name == "move"
                and event.type() == QEvent.Type.KeyPress):
            key = event.key()
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            dx, dy = 0, 0
            if key == Qt.Key.Key_Left:
                dx = -step
            elif key == Qt.Key.Key_Right:
                dx = step
            elif key == Qt.Key.Key_Up:
                dy = -step
            elif key == Qt.Key.Key_Down:
                dy = step
            if dx or dy:
                if hasattr(self._active_tool, "nudge"):
                    self._active_tool.nudge(self._build_context(), dx, dy)
                    return
        self._active_tool.key_press(self._build_context(), event)
        self.update()

    # ---- painting --------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#222"))

        if self.session is None:
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder_text)
            return

        canvas_h, canvas_w = self.session.height, self.session.width
        zoom = self._zoom
        pan = self._pan
        x0 = int(round(pan[0]))
        y0 = int(round(pan[1]))
        w = int(round(canvas_w * zoom))
        h = int(round(canvas_h * zoom))

        # Checker background so transparent areas show.
        painter.fillRect(QRect(x0, y0, w, h), self._checker)

        # Composite (with mipmap cache for zoomed-out views).
        composite = self._composite_for_zoom()
        qimg = self._numpy_to_qimage(composite)
        painter.drawImage(QRectF(x0, y0, w, h), qimg)

        # Document border.
        painter.setPen(QPen(QColor("#666"), 1))
        painter.drawRect(QRect(x0, y0, w, h))

        # Grid.
        if self._show_grid and zoom >= 0.5:
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
            step = self._grid_spacing
            x = x0
            while x < x0 + w:
                painter.drawLine(x, y0, x, y0 + h)
                x += int(round(step * zoom))
            y = y0
            while y < y0 + h:
                painter.drawLine(x0, y, x0 + w, y)
                y += int(round(step * zoom))

        # Pixel grid — at high zoom, draw a 1-px line at every document pixel
        # boundary so you can see the exact pixel grid.
        if zoom >= 8.0:
            painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
            for px in range(canvas_w + 1):
                x = x0 + int(round(px * zoom))
                painter.drawLine(x, y0, x, y0 + h)
            for py in range(canvas_h + 1):
                y = y0 + int(round(py * zoom))
                painter.drawLine(x0, y, x0 + w, y)

        # Selection outline (marching ants).
        mask = getattr(self.session, "_selection_mask", None)
        if mask is not None:
            self._paint_selection_outline(painter, mask, x0, y0, w, h)

        # Transform handles on the active layer.
        if self.session.active_layer_id is not None:
            self._paint_transform_handles(painter, self.session.active_layer_id, x0, y0, w, h)

        # Brush preview circle.
        if self._brush_preview_pos and self._active_tool_name in ("brush", "eraser", "blur", "heal", "clone-stamp"):
            self._paint_brush_preview(painter)

        # Rulers (left + top).
        if self._show_rulers:
            self._paint_rulers(painter, x0, y0, w, h)

        # Guides.
        if self.session is not None:
            self._paint_guides(painter, x0, y0, w, h)

        # Navigator minimap.
        if self._show_navigator and zoom > 2.5:
            self._paint_navigator(painter, composite)

    # ---- helpers ---------------------------------------------------------

    def _composite(self) -> np.ndarray:
        if self.session is None:
            return np.zeros((1, 1, 4), dtype=np.uint8)
        opts = CompositorOptions(
            canvas_width=self.session.width,
            canvas_height=self.session.height,
            image_loader=self._image_cache.get,
            mask_loader=self._mask_cache.get,
        )
        return composite_document(self.session, opts)

    def _composite_for_zoom(self) -> np.ndarray:
        """Return a composite, picking the right mipmap level for the current zoom."""
        full = self._composite()
        if self._zoom >= 0.5 or not hasattr(self, "_mip_cache"):
            return full
        from ..rendering.cache import mip_factor_for_zoom, mip_image
        factor = mip_factor_for_zoom(self._zoom)
        if factor == 1:
            return full
        sig = self._mip_signature()
        cached = self._mip_cache.get(sig, factor)
        if cached is not None:
            return cached
        downsampled = mip_image(full, factor)
        self._mip_cache.put(sig, factor, downsampled)
        return downsampled

    def _mip_signature(self) -> tuple:
        """Stable hash of which layer versions contribute to the composite."""
        if self.session is None:
            return (("none", 0),)
        items = []
        cache = getattr(self.session, "_image_cache", {})
        for layer in self.session.layers:
            version = int(getattr(layer, "transform", 0).__hash__()) if hasattr(layer.transform, "__hash__") else 0
            present = layer.id in cache
            items.append((layer.id, int(present) * 1000 + version & 0xFF))
        return tuple(items)

    def invalidate_mip_cache(self) -> None:
        if hasattr(self, "_mip_cache"):
            self._mip_cache.invalidate_all()

    def _build_context(self) -> ToolContext:
        return ToolContext(
            session=self.session,
            canvas_width=self.session.width if self.session else 0,
            canvas_height=self.session.height if self.session else 0,
            zoom=self._zoom,
            pan=self._pan,
            selection=getattr(self.session, "_selection_mask", None) if self.session else None,
            active_layer_id=self.session.active_layer_id if self.session else None,
            request_repaint=self.update,
            request_reload=lambda: self.document_changed.emit(),
        )

    def to_canvas_coords(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self._pan[0]) / self._zoom, (y - self._pan[1]) / self._zoom)

    def _to_doc_x(self, widget_x: float) -> float:
        return (widget_x - self._pan[0]) / max(self._zoom, 0.01)

    def _to_doc_y(self, widget_y: float) -> float:
        return (widget_y - self._pan[1]) / max(self._zoom, 0.01)

    def _create_guide(self, orientation: str, position: float) -> str:
        """Add a guide to the session, return its id."""
        from ..document.guides import Guide, new_guide_id
        gid = new_guide_id()
        self.session.add_guide(Guide(id=gid, orientation=orientation, position=position))
        return gid

    def _paint_guides(self, painter: QPainter, x0: int, y0: int, w: int, h: int) -> None:
        """Draw all visible guides."""
        if self.session is None:
            return
        pen = QPen(QColor(0, 150, 255, 200), 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        for g in self.session.guides:
            if not g.is_visible:
                continue
            if g.orientation == "h":
                y_widget = self._pan[1] + g.position * self._zoom
                painter.drawLine(x0, int(y_widget), x0 + w, int(y_widget))
            else:
                x_widget = self._pan[0] + g.position * self._zoom
                painter.drawLine(int(x_widget), y0, int(x_widget), y0 + h)

    def _fit_zoom(self) -> float:
        if self.session is None:
            return 1.0
        return min(self.width() / self.session.width, self.height() / self.session.height) * 0.9

    def _numpy_to_qimage(self, array: np.ndarray) -> QImage:
        if array.shape[-1] == 4:
            h, w, _ = array.shape
            return QImage(array.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        h, w, _ = array.shape
        return QImage(array.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()

    def _build_checker(self, tile: int) -> QPixmap:
        pix = QPixmap(tile * 2, tile * 2)
        pix.fill(QColor("#444"))
        painter = QPainter(pix)
        painter.fillRect(0, 0, tile, tile, QColor("#666"))
        painter.fillRect(tile, tile, tile, tile, QColor("#666"))
        painter.end()
        return pix

    def _paint_selection_outline(self, painter: QPainter, mask: np.ndarray, x0: int, y0: int, w: int, h: int) -> None:
        # Find the bounding rect of the mask (simple box for performance).
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        sx0 = x0 + int(round(x_min * self._zoom))
        sy0 = y0 + int(round(y_min * self._zoom))
        sx1 = x0 + int(round((x_max + 1) * self._zoom))
        sy1 = y0 + int.round((y_max + 1) * self._zoom) if hasattr(int, "round") else y0 + int(round((y_max + 1) * self._zoom))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRect(sx0, sy0, sx1 - sx0, sy1 - sy0))

    def _paint_transform_handles(self, painter: QPainter, layer_id: str, x0: int, y0: int, w: int, h: int) -> None:
        layer = self.session.layer_by_id(layer_id)
        if layer is None or isinstance(layer, (GroupLayer, AdjustmentLayer)):
            return
        ox, oy = layer.transform.origin
        sw, sh = layer.transform.size
        # For simplicity, ignore rotation when drawing handles.
        rect_x = x0 + int(round(ox * self._zoom))
        rect_y = y0 + int(round(oy * self._zoom))
        rect_w = int(round(sw * self._zoom))
        rect_h = int(round(sh * self._zoom))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawRect(QRect(rect_x, rect_y, rect_w, rect_h))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        for hx, hy in (
            (rect_x, rect_y),
            (rect_x + rect_w, rect_y),
            (rect_x, rect_y + rect_h),
            (rect_x + rect_w, rect_y + rect_h),
        ):
            painter.drawRect(QRect(hx - 4, hy - 4, 8, 8))

    def _paint_brush_preview(self, painter: QPainter) -> None:
        if self._brush_preview_pos is None or self._active_tool is None:
            return
        from ..rendering import brush as brush_engine
        size = getattr(self._active_tool.settings, "size", 24)
        cx, cy = self._brush_preview_pos
        wx = self._pan[0] + cx * self._zoom
        wy = self._pan[1] + cy * self._zoom
        radius = size * self._zoom / 2.0
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
        painter.drawEllipse(QPointF(wx, wy), radius, radius)

    def _paint_rulers(self, painter: QPainter, x0: int, y0: int, w: int, h: int) -> None:
        painter.fillRect(0, 0, self.width(), 16, QColor("#333"))
        painter.fillRect(0, 0, 16, self.height(), QColor("#333"))
        painter.setPen(QColor("#aaa"))
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)
        # Top ruler.
        step = max(1, int(50 / self._zoom))
        x = x0
        doc_x = 0
        while x < x0 + w:
            painter.drawLine(x, 16, x, 10)
            painter.drawText(x + 1, 9, str(doc_x))
            x += int(round(step * self._zoom))
            doc_x += step
        # Left ruler.
        y = y0
        doc_y = 0
        while y < y0 + h:
            painter.drawLine(16, y, 10, y)
            painter.drawText(1, y - 1, str(doc_y))
            y += int(round(step * self._zoom))
            doc_y += step

    def _paint_navigator(self, painter: QPainter, composite: np.ndarray) -> None:
        size = 120
        margin = 8
        x = self.width() - size - margin
        y = margin
        painter.fillRect(QRect(x, y, size, size), QColor("#111"))
        painter.setPen(QColor("#888"))
        painter.drawRect(QRect(x, y, size, size))
        # Render a downsample.
        small = self._downsample(composite, (64, 64))
        qimg = self._numpy_to_qimage(small)
        painter.drawImage(QRectF(x + 2, y + 2, size - 4, size - 4), qimg)
        # Viewport rectangle.
        if self.session is None:
            return
        vx = (self._pan[0] - x - 2) / (size - 4) * self.session.width / self._zoom
        vy = (self._pan[1] - y - 2) / (size - 4) * self.session.height / self._zoom
        vw = self.width() / (size - 4) * self.session.width / self._zoom
        vh = self.height() / (size - 4) * self.session.height / self._zoom
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawRect(QRect(x + 2 + int(round(vx * (size - 4) / self.session.width * self._zoom)),
                               y + 2 + int(round(vy * (size - 4) / self.session.height * self._zoom)),
                               int(round(vw * (size - 4) / self.session.width * self._zoom)),
                               int(round(vh * (size - 4) / self.session.height * self._zoom))))

    def _downsample(self, img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        except ImportError:
            h, w, _ = img.shape
            ys = np.linspace(0, h - 1, shape[0]).astype(np.int32)
            xs = np.linspace(0, w - 1, shape[1]).astype(np.int32)
            return img[ys][:, xs]
