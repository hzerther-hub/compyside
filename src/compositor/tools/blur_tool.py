"""Real spot-heal brush — patch-based texture synthesis.

For each dab:
  1. Sample a source patch from a small offset away from the destination.
  2. Copy that source patch into the destination region.
  3. Blend the seam using a feathered alpha to avoid hard edges.

This is a simplified content-aware heal — the original Swift version does
gradient-guided patch blending with poisson blending; this version is
close enough to look right in practice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from ..rendering import pixel_ops as po
from .base import Tool, ToolContext


@dataclass
class HealSettings:
    size: int = 24
    strength: float = 0.7


class BlurTool(Tool):
    name = "blur"
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = HealSettings()

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._stamp(context, event.position().x(), event.position().y())

    def mouse_move(self, context: ToolContext, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        self._stamp(context, event.position().x(), event.position().y())

    def _stamp(self, context: ToolContext, x: float, y: float) -> None:
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None:
            return
        cache = getattr(context.session, "_image_cache", {})
        img = cache.get(layer.id)
        if img is None:
            return
        cx, cy = self.to_canvas_coords(context, x, y)
        ix0, iy0 = int(round(cx - self.settings.size / 2)), int(round(cy - self.settings.size / 2))
        ix1, iy1 = ix0 + self.settings.size, iy0 + self.settings.size
        h_can, w_can = context.canvas_height, context.canvas_width
        ix0c, iy0c = max(0, ix0), max(0, iy0)
        ix1c, iy1c = min(w_can, ix1), min(h_can, iy1)
        if ix1c <= ix0c or iy1c <= iy0c:
            return
        patch = img[iy0c:iy1c, ix0c:ix1c].copy()
        blurred = po.gaussian_blur(patch, sigma=self.settings.size * 0.2)
        out = (patch.astype(np.float32) * (1 - self.settings.strength) +
               blurred.astype(np.float32) * self.settings.strength).astype(np.uint8)
        img[iy0c:iy1c, ix0c:ix1c] = out
        cache[layer.id] = img
        context.request_repaint()


class HealTool(Tool):
    """Patch-based spot heal — samples a small offset region as the source.

    Alt-click to set the sample offset (defaults to a small downward
    offset so the dab blends with the area just below).
    """

    name = "heal"
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = HealSettings()
        self._sample_offset = (0, 16)  # dx, dy in document px
        self._has_sample = False

    def set_sample_source(self, x: float, y: float) -> None:
        """Treat the current cursor as the source; the dab will copy from here."""
        self._sample_offset = (int(round(x)), int(round(y)))
        self._has_sample = True

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
            self.set_sample_source(cx, cy)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._stamp(context, event.position().x(), event.position().y())

    def mouse_move(self, context: ToolContext, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        self._stamp(context, event.position().x(), event.position().y())

    def _stamp(self, context: ToolContext, x: float, y: float) -> None:
        layer = context.session.layer_by_id(context.active_layer_id)
        if layer is None:
            return
        cache = getattr(context.session, "_image_cache", {})
        img = cache.get(layer.id)
        if img is None:
            return
        cx, cy = self.to_canvas_coords(context, x, y)
        h_can, w_can = context.canvas_height, context.canvas_width
        s = self.settings.size
        # Destination rect.
        dx0, dy0 = int(round(cx - s / 2)), int(round(cy - s / 2))
        dx1, dy1 = dx0 + s, dy0 + s
        # Source rect: offset from destination by sample_offset.
        ox, oy = self._sample_offset
        sx0, sy0 = dx0 + ox, dy0 + oy
        sx1, sy1 = sx0 + s, sy0 + s

        def clip(a0, a1, lim):
            return max(0, min(lim, a0)), max(0, min(lim, a1))

        dx0c, dy0c = clip(dx0, dy0, w_can), clip(dx0, dy0, h_can) if False else (clip(dx0, dy0, w_can)[0], clip(dy0, dy0, h_can)[0])
        # Simpler clip:
        dx0c, dy0c, dx1c, dy1c = max(0, dx0), max(0, dy0), min(w_can, dx1), min(h_can, dy1)
        sx0c, sy0c, sx1c, sy1c = max(0, sx0), max(0, sy0), min(w_can, sx1), min(h_can, sy1)
        if dx1c <= dx0c or dy1c <= dy0c or sx1c <= sx0c or sy1c <= sy0c:
            return
        # Make the two rects the same shape.
        w = min(dx1c - dx0c, sx1c - sx0c)
        h = min(dy1c - dy0c, sy1c - sy0c)
        if w <= 0 or h <= 0:
            return
        dest = img[dy0c:dy0c + h, dx0c:dx0c + w].astype(np.float32)
        src = img[sy0c:sy0c + h, sx0c:sx0c + w].astype(np.float32)
        # Feather alpha at the borders to hide seams.
        feather = max(2, s // 8)
        alpha = np.ones((h, w), dtype=np.float32)
        for i in range(feather):
            t = (i + 1) / feather
            alpha[i, :] = np.minimum(alpha[i, :], t)
            alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], t)
            alpha[:, i] = np.minimum(alpha[:, i], t)
            alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], t)
        alpha = alpha[..., None] * self.settings.strength  # HxWx1
        blended = (src * alpha + dest * (1.0 - alpha))
        img[dy0c:dy0c + h, dx0c:dx0c + w] = np.clip(blended, 0, 255).astype(np.uint8)
        cache[layer.id] = img
        context.request_repaint()
