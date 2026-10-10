"""Clone stamp tool — copy pixels from a sampled source offset to the cursor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from .base import Tool, ToolContext


@dataclass
class CloneSettings:
    size: int = 32
    hardness: float = 0.8
    opacity: float = 1.0
    aligned: bool = True
    sample_layer_id: Optional[str] = None  # None = composite all layers
    source_offset: Tuple[int, int] = (0, 0)
    has_source: bool = False


class CloneTool(Tool):
    name = "clone-stamp"
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = CloneSettings()
        self._last_target: Optional[Tuple[float, float]] = None

    def set_sample_source(self, x: float, y: float) -> None:
        """Alt-click sets the sample source relative to current cursor."""
        self.settings.source_offset = (int(round(x)), int(round(y)))
        self.settings.has_source = True

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton or event.modifiers() & Qt.KeyboardModifier.AltModifier:
            cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
            self.set_sample_source(cx, cy)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._stroke(context, event.position().x(), event.position().y())

    def mouse_move(self, context: ToolContext, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        self._stroke(context, event.position().x(), event.position().y())

    def _stroke(self, context: ToolContext, x: float, y: float) -> None:
        if not self.settings.has_source:
            return
        cx, cy = self.to_canvas_coords(context, x, y)
        sx = cx - self.settings.source_offset[0]
        sy = cy - self.settings.source_offset[1]
        ix0, iy0 = int(round(cx - self.settings.size / 2)), int(round(cy - self.settings.size / 2))
        ix1, iy1 = ix0 + self.settings.size, iy0 + self.settings.size
        sx0, sy0 = int(round(sx - self.settings.size / 2)), int(round(sy - self.settings.size / 2))
        sx1, sy1 = sx0 + self.settings.size, sy0 + self.settings.size

        if self.settings.sample_layer_id is None:
            # Sample from the composite.
            from ..rendering.layer_renderer import composite_document, CompositorOptions
            cache = getattr(context.session, "_image_cache", {})
            mask_loader = getattr(context.session, "_mask_cache", {}).get
            source = composite_document(
                context.session,
                CompositorOptions(
                    canvas_width=context.canvas_width,
                    canvas_height=context.canvas_height,
                    image_loader=cache.get,
                    mask_loader=mask_loader,
                ),
            )
        else:
            source = getattr(context.session, "_image_cache", {}).get(self.settings.sample_layer_id)

        target_layer = context.session.layer_by_id(context.active_layer_id)
        if target_layer is None or source is None:
            return
        cache = getattr(context.session, "_image_cache", {})
        target = cache.get(target_layer.id)
        if target is None:
            return

        # Clip source/target rects to canvas.
        h_can, w_can = context.canvas_height, context.canvas_width
        sx0c, sy0c = max(0, sx0), max(0, sy0)
        sx1c, sy1c = min(w_can, sx1), min(h_can, sy1)
        ix0c, iy0c = max(0, ix0), max(0, iy0)
        ix1c, iy1c = min(w_can, ix1), min(h_can, iy1)
        if ix1c <= ix0c or iy1c <= iy0c:
            return
        if sx1c <= sx0c or sy1c <= sy0c:
            return
        # Adjust sample rect to same size as target rect.
        dx = ix0c - ix0
        dy = iy0c - iy0
        sx0c += dx
        sy0c += dy
        sx1c = sx0c + (ix1c - ix0c)
        sy1c = sy0c + (iy1c - iy0c)
        if sx1c > w_can or sy1c > h_can:
            return
        sample = source[sy0c:sy1c, sx0c:sx1c, :3].astype(np.float32)
        # Build a circular mask.
        s = min(ix1c - ix0c, iy1c - iy0c)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        cx2, cy2 = s / 2.0, s / 2.0
        r = np.sqrt((xx - cx2) ** 2 + (yy - cy2) ** 2)
        inner = self.settings.hardness * s / 2.0
        t = np.clip((r - inner) / max(s / 2.0 - inner, 1e-6), 0.0, 1.0)
        stamp_alpha = ((1.0 - t) * 255.0 * self.settings.opacity).astype(np.float32) / 255.0
        # Composite into target.
        tgt = target[iy0c:iy1c, ix0c:ix1c, :3].astype(np.float32)
        out = sample * stamp_alpha[..., None] + tgt * (1.0 - stamp_alpha[..., None])
        target[iy0c:iy1c, ix0c:ix1c, :3] = np.clip(out, 0, 255).astype(np.uint8)
        target[iy0c:iy1c, ix0c:ix1c, 3] = np.clip(
            np.maximum(target[iy0c:iy1c, ix0c:ix1c, 3].astype(np.float32),
                       stamp_alpha[..., None] * 255.0),
            0, 255,
        ).astype(np.uint8)
        cache[target_layer.id] = target
        context.request_repaint()
