"""Layer effects — drop shadow, inner glow, outer glow, color overlay, stroke.

Each effect takes the layer's RGBA pixels and returns the new RGBA pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from . import pixel_ops as po


@dataclass
class DropShadow:
    color: tuple = (0, 0, 0, 255)
    opacity: float = 0.75
    angle: float = 120.0
    distance: float = 5.0
    blur: float = 5.0
    spread: int = 0


@dataclass
class InnerShadow:
    color: tuple = (0, 0, 0, 255)
    opacity: float = 0.75
    angle: float = 120.0
    distance: float = 5.0
    blur: float = 5.0


@dataclass
class OuterGlow:
    color: tuple = (255, 255, 255, 255)
    opacity: float = 0.75
    blur: float = 10.0
    spread: int = 0


@dataclass
class InnerGlow:
    color: tuple = (255, 255, 255, 255)
    opacity: float = 0.75
    blur: float = 10.0


@dataclass
class ColorOverlay:
    color: tuple = (255, 0, 0, 255)
    opacity: float = 1.0


@dataclass
class Stroke:
    color: tuple = (0, 0, 0, 255)
    size: int = 1
    position: str = "outside"  # "inside" | "center" | "outside"


def apply_drop_shadow(layer: np.ndarray, effect: DropShadow, canvas_shape: tuple) -> np.ndarray:
    """Produce a composited image of ``layer`` plus its drop shadow on a transparent canvas.

    The output canvas matches ``canvas_shape`` (H, W). The shadow is drawn
    first (offset, blurred, coloured), then the layer is drawn on top.
    """
    h_can, w_can = canvas_shape
    out = np.zeros((h_can, w_can, 4), dtype=np.uint8)
    if layer.shape[-1] != 4:
        layer = np.concatenate([layer, np.full(layer.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)

    # Build the shadow buffer: opaque in the layer's alpha, then blurred.
    shadow_alpha = layer[..., 3]
    if effect.blur > 0:
        shadow_alpha = po.gaussian_blur(shadow_alpha[..., None], effect.blur)[..., 0]
    if effect.spread > 0:
        shadow_alpha = _dilate(shadow_alpha, effect.spread)
    shadow_alpha = (shadow_alpha.astype(np.float32) * effect.opacity).astype(np.uint8)

    color = np.array(effect.color[:3], dtype=np.uint8)
    shadow = np.zeros((h_can, w_can, 4), dtype=np.uint8)
    dx = int(round(math.cos(math.radians(effect.angle)) * effect.distance))
    dy = int(round(-math.sin(math.radians(effect.angle)) * effect.distance))
    sh_h, sh_w = shadow_alpha.shape
    # Place shadow at offset.
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(sh_w, w_can - dx)
    src_y1 = min(sh_h, h_can - dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = max(0, dx)
        dst_y0 = max(0, dy)
        h_slice = src_y1 - src_y0
        w_slice = src_x1 - src_x0
        shadow[dst_y0 : dst_y0 + h_slice, dst_x0 : dst_x0 + w_slice, :3] = color
        shadow[dst_y0 : dst_y0 + h_slice, dst_x0 : dst_x0 + w_slice, 3] = shadow_alpha[src_y0:src_y1, src_x0:src_x1]

    # Layer on top, centred at (0,0) in the canvas — caller decides offset.
    lh, lw = layer.shape[:2]
    cy0 = max(0, (h_can - lh) // 2)
    cx0 = max(0, (w_can - lw) // 2)
    cy1 = min(h_can, cy0 + lh)
    cx1 = min(w_can, cx0 + lw)
    if cy1 > cy0 and cx1 > cx0:
        out[cy0:cy1, cx0:cx1, :] = layer[: cy1 - cy0, : cx1 - cx0, :]

    out = _alpha_over(shadow, out)
    return out


def apply_color_overlay(layer: np.ndarray, effect: ColorOverlay) -> np.ndarray:
    if layer.shape[-1] != 4:
        layer = np.concatenate([layer, np.full(layer.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    color = np.array(effect.color, dtype=np.uint8).reshape(1, 1, 4)
    alpha = layer[..., 3:4].astype(np.float32) / 255.0 * effect.opacity
    rgb = (1 - alpha) * layer[..., :3] + alpha * color[..., :3]
    return np.concatenate([rgb.astype(np.uint8), layer[..., 3:4]], axis=-1)


def apply_outer_glow(layer: np.ndarray, effect: OuterGlow, canvas_shape: tuple) -> np.ndarray:
    """Outer glow: blurred alpha of the layer, tinted by ``effect.color``."""
    h_can, w_can = canvas_shape
    if layer.shape[-1] != 4:
        layer = np.concatenate([layer, np.full(layer.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    glow_alpha = layer[..., 3]
    if effect.blur > 0:
        glow_alpha = po.gaussian_blur(glow_alpha[..., None], effect.blur)[..., 0]
    if effect.spread > 0:
        glow_alpha = _dilate(glow_alpha, effect.spread)
    glow_alpha = (glow_alpha.astype(np.float32) * effect.opacity).astype(np.uint8)

    color = np.array(effect.color[:3], dtype=np.uint8)
    glow = np.zeros((h_can, w_can, 4), dtype=np.uint8)
    lh, lw = glow_alpha.shape
    cy0 = max(0, (h_can - lh) // 2)
    cx0 = max(0, (w_can - lw) // 2)
    cy1 = min(h_can, cy0 + lh)
    cx1 = min(w_can, cx0 + lw)
    if cy1 > cy0 and cx1 > cx0:
        glow[cy0:cy1, cx0:cx1, :3] = color
        glow[cy0:cy1, cx0:cx1, 3] = glow_alpha[: cy1 - cy0, : cx1 - cx0]
    out = np.zeros((h_can, w_can, 4), dtype=np.uint8)
    lh, lw = layer.shape[:2]
    cy0 = max(0, (h_can - lh) // 2)
    cx0 = max(0, (w_can - lw) // 2)
    cy1 = min(h_can, cy0 + lh)
    cx1 = min(w_can, cx0 + lw)
    if cy1 > cy0 and cx1 > cx0:
        out[cy0:cy1, cx0:cx1, :] = layer[: cy1 - cy0, : cx1 - cx0, :]
    return _alpha_over(glow, out)


def apply_inner_glow(layer: np.ndarray, effect: InnerGlow) -> np.ndarray:
    """Inner glow: blurred alpha of the inverted layer, tinted, masked back to the layer."""
    if layer.shape[-1] != 4:
        layer = np.concatenate([layer, np.full(layer.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    inv_alpha = 255 - layer[..., 3]
    glow_alpha = inv_alpha
    if effect.blur > 0:
        glow_alpha = po.gaussian_blur(glow_alpha[..., None], effect.blur)[..., 0]
    glow_alpha = (glow_alpha.astype(np.float32) * effect.opacity).astype(np.uint8)
    color = np.array(effect.color[:3], dtype=np.uint8)
    glow = np.zeros_like(layer)
    glow[..., :3] = color
    glow[..., 3] = glow_alpha
    # Mask glow to the inside of the layer (where alpha was inverted and is now filled).
    inside = layer[..., 3] > 0
    glow[..., 3] = np.where(inside, glow[..., 3], 0)
    return _alpha_over(glow, layer)


def apply_stroke(layer: np.ndarray, effect: Stroke) -> np.ndarray:
    """Outline the layer's alpha with ``effect.color``."""
    if layer.shape[-1] != 4:
        layer = np.concatenate([layer, np.full(layer.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    if effect.size <= 0:
        return layer
    if effect.position == "inside":
        outline = _erode(layer[..., 3], effect.size)
        outline = layer[..., 3] - outline
    elif effect.position == "outside":
        outline = _dilate(layer[..., 3], effect.size)
        outline = outline - layer[..., 3]
    else:
        inside = _erode(layer[..., 3], effect.size // 2)
        outside = _dilate(layer[..., 3], effect.size - effect.size // 2)
        outline = outside - inside
        outline = np.maximum(outline - layer[..., 3], layer[..., 3] - inside)
    color = np.array(effect.color, dtype=np.uint8).reshape(1, 1, 4)
    overlay = np.zeros_like(layer)
    overlay[..., :] = color
    overlay[..., 3] = np.minimum(outline, 255)
    return _alpha_over(overlay, layer)


# ---------------------------------------------------------------------------
# Morphology helpers
# ---------------------------------------------------------------------------


def _dilate(alpha: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return alpha
    out = alpha.copy()
    for _ in range(radius):
        shifted = np.pad(out, ((0, 1), (0, 1)), mode="constant")
        out = np.maximum(out, shifted[:-1, :-1])
        out = np.maximum(out, shifted[:-1, 1:])
        out = np.maximum(out, shifted[1:, :-1])
        out = np.maximum(out, shifted[1:, 1:])
    return out


def _erode(alpha: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return alpha
    out = alpha.copy()
    for _ in range(radius):
        shifted = np.pad(out, ((1, 0), (1, 0)), mode="constant", constant_values=255)
        out = np.minimum(out, shifted[1:, 1:])
        out = np.minimum(out, shifted[1:, :-1])
        out = np.minimum(out, shifted[:-1, 1:])
        out = np.minimum(out, shifted[:-1, :-1])
    return out


def _alpha_over(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Straight-alpha source-over."""
    sa = src[..., 3:4].astype(np.float32) / 255.0
    da = dst[..., 3:4].astype(np.float32) / 255.0
    out_a = sa + da * (1.0 - sa)
    safe = np.where(out_a == 0, 1.0, out_a)
    out_rgb = (src[..., :3].astype(np.float32) * sa + dst[..., :3].astype(np.float32) * da * (1.0 - sa)) / safe
    return np.concatenate([np.clip(out_rgb, 0, 255).astype(np.uint8), np.clip(out_a * 255.0, 0, 255).astype(np.uint8)], axis=-1)


# ---------------------------------------------------------------------------
# Effect dispatch (loads from manifest's layer record when present)
# ---------------------------------------------------------------------------


def apply_effects(layer: np.ndarray, effects: Dict[str, Any], canvas_shape: tuple) -> np.ndarray:
    """Apply a dict of effects (keyed by lowercase name) in order."""
    out = layer
    for key, params in effects.items():
        if key == "dropShadow" and params.get("enabled", True):
            ds = DropShadow(**{k: v for k, v in params.items() if k != "enabled"})
            out = apply_drop_shadow(out, ds, canvas_shape)
        elif key == "outerGlow" and params.get("enabled", True):
            og = OuterGlow(**{k: v for k, v in params.items() if k != "enabled"})
            out = apply_outer_glow(out, og, canvas_shape)
        elif key == "innerGlow" and params.get("enabled", True):
            ig = InnerGlow(**{k: v for k, v in params.items() if k != "enabled"})
            out = apply_inner_glow(out, ig)
        elif key == "colorOverlay" and params.get("enabled", True):
            co = ColorOverlay(**{k: v for k, v in params.items() if k != "enabled"})
            out = apply_color_overlay(out, co)
        elif key == "stroke" and params.get("enabled", True):
            st = Stroke(**{k: v for k, v in params.items() if k != "enabled"})
            out = apply_stroke(out, st)
    return out
