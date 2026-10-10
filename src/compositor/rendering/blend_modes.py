"""Photoshop-style blend modes, vectorised in NumPy.

All functions take two ``uint8`` arrays of shape ``(H, W, 4)`` (RGBA) and
return a new ``uint8`` array of the same shape. Alpha is composited using
the standard "source over" rule; the blend function operates on the RGB
channels only.

The math follows the W3C "Compositing and Blending Level 1" spec, which is
what Photoshop documents are based on.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ..format.manifest import BLEND_MODES

BlendFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float(rgb: np.ndarray) -> np.ndarray:
    """uint8 [0..255] -> float [0..1]."""
    return rgb.astype(np.float32) / 255.0


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    """float [0..1] (possibly out of range) -> uint8, clamped."""
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _split_rgba(img: np.ndarray):
    return img[..., 0], img[..., 1], img[..., 2], img[..., 3]


def _stack_rgba(r: np.ndarray, g: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.stack([r, g, b, a], axis=-1).astype(np.uint8)


def _composite_over(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Standard source-over with straight alpha. Inputs are uint8 RGBA."""
    s_r, s_g, s_b, s_a = _split_rgba(src)
    d_r, d_g, d_b, d_a = _split_rgba(dst)
    sa = s_a.astype(np.float32) / 255.0
    da = d_a.astype(np.float32) / 255.0
    out_a = sa + da * (1.0 - sa)
    # Avoid 0/0; when out_a is 0 the multiplication is moot.
    safe_out_a = np.where(out_a == 0, 1.0, out_a)
    out_r = (s_r.astype(np.float32) * sa + d_r.astype(np.float32) * da * (1.0 - sa)) / safe_out_a
    out_g = (s_g.astype(np.float32) * sa + d_g.astype(np.float32) * da * (1.0 - sa)) / safe_out_a
    out_b = (s_b.astype(np.float32) * sa + d_b.astype(np.float32) * da * (1.0 - sa)) / safe_out_a
    return _stack_rgba(
        np.clip(out_r, 0, 255).astype(np.uint8),
        np.clip(out_g, 0, 255).astype(np.uint8),
        np.clip(out_b, 0, 255).astype(np.uint8),
        np.clip(out_a * 255.0 + 0.5, 0, 255).astype(np.uint8),
    )


# ---------------------------------------------------------------------------
# Individual blend modes (operate on float RGB in [0,1]; return float RGB)
# ---------------------------------------------------------------------------


def normal(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return S


def darken(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.minimum(B, S)


def multiply(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return B * S


def color_burn(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - np.where(S == 0, 1.0, (1.0 - B) / S)
    return np.where(S == 0, 0.0, out)


def linear_burn(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.maximum(B + S - 1.0, 0.0)


def lighten(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.maximum(B, S)


def screen(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return B + S - B * S


def color_dodge(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(S == 1, 1.0, B / (1.0 - S))
    return np.where(S == 1, 1.0, out)


def linear_dodge_add(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.minimum(B + S, 1.0)


def overlay(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Overlay = multiply on darker halves, screen on lighter halves."""
    return np.where(
        B < 0.5,
        2.0 * B * S,
        1.0 - 2.0 * (1.0 - B) * (1.0 - S),
    )


def soft_light(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    """W3C soft-light (the formula Photoshop uses)."""
    return np.where(
        S <= 0.5,
        B - (1.0 - 2.0 * S) * B * (1.0 - B),
        B + (2.0 * S - 1.0) * (_soft_light_d(B) - B),
    )


def _soft_light_d(B: np.ndarray) -> np.ndarray:
    return np.where(
        B <= 0.25,
        ((16.0 * B - 12.0) * B + 4.0) * B,
        np.sqrt(B),
    )


def hard_light(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Hard light = overlay with B/S swapped."""
    return overlay(S, B)


def vivid_light(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.where(
        S < 0.5,
        color_burn(B, 2.0 * S),
        color_dodge(B, 2.0 * (S - 0.5)),
    )


def linear_light(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.where(
        S < 0.5,
        linear_burn(B, 2.0 * S),
        linear_dodge_add(B, 2.0 * (S - 0.5)),
    )


def pin_light(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.where(
        S < 0.5,
        np.minimum(B, 2.0 * S),
        np.maximum(B, 2.0 * (S - 0.5)),
    )


def hard_mix(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.where(vivid_light(B, S) < 1.0, 0.0, 1.0)


def difference(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.abs(B - S)


def exclusion(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return B + S - 2.0 * B * S


def subtract(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.maximum(B - S, 0.0)


def divide(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(S == 0, 1.0, np.minimum(B / S, 1.0))


# Hue / Saturation / Color / Luminosity — operate in HSL. We use a small
# inline converter; for production work this would route through colour-science.


def _rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.max(rgb, axis=-1)
    min_c = np.min(rgb, axis=-1)
    l = (max_c + min_c) / 2.0
    s = np.zeros_like(l)
    h = np.zeros_like(l)

    nonzero = max_c != min_c
    diff = max_c - min_c
    safe_diff = np.where(nonzero, diff, 1.0)

    # Saturation
    s = np.where(
        l < 0.5,
        diff / (max_c + min_c + 1e-12),
        diff / (2.0 - max_c - min_c + 1e-12),
    )
    s = np.where(nonzero, s, 0.0)

    # Hue
    rmax = (max_c == r) & nonzero
    gmax = (max_c == g) & nonzero & ~rmax
    bmax = (max_c == b) & nonzero & ~rmax & ~gmax

    h = np.where(
        rmax,
        (g - b) / safe_diff + (g < b) * 6.0,
        h,
    )
    h = np.where(
        gmax,
        (b - r) / safe_diff + 2.0,
        h,
    )
    h = np.where(
        bmax,
        (r - g) / safe_diff + 4.0,
        h,
    )
    h = h / 6.0
    h = np.where(nonzero, h, 0.0)
    return np.stack([h, s, l], axis=-1)


def _hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    out = np.zeros_like(hsl)

    nonzero = s != 0
    q = np.where(l < 0.5, l * (1.0 + s), l + s - l * s)
    p = 2.0 * l - q

    def _hue_to_rgb(t: np.ndarray) -> np.ndarray:
        t = np.where(t < 0, t + 1.0, t)
        t = np.where(t > 1, t - 1.0, t)
        out = np.where(
            t < 1.0 / 6.0,
            p + (q - p) * 6.0 * t,
            t,
        )
        out = np.where(
            (t >= 1.0 / 6.0) & (t < 0.5),
            q,
            out,
        )
        out = np.where(
            (t >= 0.5) & (t < 2.0 / 3.0),
            p + (q - p) * (2.0 / 3.0 - t) * 6.0,
            out,
        )
        return np.where(
            t >= 2.0 / 3.0,
            p,
            out,
        )

    out[..., 0] = np.where(nonzero, _hue_to_rgb(h + 1.0 / 3.0), l)
    out[..., 1] = np.where(nonzero, _hue_to_rgb(h), l)
    out[..., 2] = np.where(nonzero, _hue_to_rgb(h - 1.0 / 3.0), l)
    return out


def hue_blend(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    hsl_s = _rgb_to_hsl(S)
    hsl_b = _rgb_to_hsl(B)
    return _hsl_to_rgb(np.stack([hsl_s[..., 0], hsl_b[..., 1], hsl_b[..., 2]], axis=-1))


def saturation_blend(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    hsl_s = _rgb_to_hsl(S)
    hsl_b = _rgb_to_hsl(B)
    return _hsl_to_rgb(np.stack([hsl_b[..., 0], hsl_s[..., 1], hsl_b[..., 2]], axis=-1))


def color_blend(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    hsl_s = _rgb_to_hsl(S)
    hsl_b = _rgb_to_hsl(B)
    return _hsl_to_rgb(np.stack([hsl_s[..., 0], hsl_s[..., 1], hsl_b[..., 2]], axis=-1))


def luminosity_blend(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    hsl_s = _rgb_to_hsl(S)
    hsl_b = _rgb_to_hsl(B)
    return _hsl_to_rgb(np.stack([hsl_b[..., 0], hsl_b[..., 1], hsl_s[..., 2]], axis=-1))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BLEND_MODE_FUNCTIONS: Dict[str, BlendFn] = {
    "Normal": normal,
    "Darken": darken,
    "Multiply": multiply,
    "Color Burn": color_burn,
    "Linear Burn": linear_burn,
    "Lighten": lighten,
    "Screen": screen,
    "Color Dodge": color_dodge,
    "Linear Dodge (Add)": linear_dodge_add,
    "Overlay": overlay,
    "Soft Light": soft_light,
    "Hard Light": hard_light,
    "Vivid Light": vivid_light,
    "Linear Light": linear_light,
    "Pin Light": pin_light,
    "Hard Mix": hard_mix,
    "Difference": difference,
    "Exclusion": exclusion,
    "Subtract": subtract,
    "Divide": divide,
    "Hue": hue_blend,
    "Saturation": saturation_blend,
    "Color": color_blend,
    "Luminosity": luminosity_blend,
}


def apply_blend(backdrop: np.ndarray, source: np.ndarray, mode: str) -> np.ndarray:
    """Apply ``mode`` over ``backdrop`` with ``source`` on top.

    Inputs are uint8 RGBA, shape ``(H, W, 4)``. Output is uint8 RGBA of the
    same shape. Alpha is composited with straight-alpha source-over.
    """
    if mode not in BLEND_MODE_FUNCTIONS:
        raise ValueError(f"unknown blend mode: {mode!r}")
    if mode == "Normal":
        return _composite_over(source, backdrop)
    if backdrop.shape != source.shape:
        raise ValueError(f"shape mismatch: backdrop {backdrop.shape}, source {source.shape}")
    if backdrop.shape[-1] not in (3, 4) or source.shape[-1] not in (3, 4):
        raise ValueError("inputs must be RGB or RGBA")
    # Ensure RGBA so the alpha math at the end has a channel.
    if backdrop.shape[-1] == 3:
        backdrop = np.concatenate([backdrop, np.full(backdrop.shape[:-1] + (1,), 255, dtype=np.uint8)], axis=-1)
    if source.shape[-1] == 3:
        source = np.concatenate([source, np.full(source.shape[:-1] + (1,), 255, dtype=np.uint8)], axis=-1)
    fn = BLEND_MODE_FUNCTIONS[mode]
    Bf = _to_float(backdrop[..., :3])
    Sf = _to_float(source[..., :3])
    blended = _to_uint8(fn(Bf, Sf))
    return _composite_over(
        np.concatenate([blended, source[..., 3:4]], axis=-1),
        backdrop,
    )


def all_blend_modes() -> list[str]:
    """Return the canonical list of blend-mode spellings."""
    return list(BLEND_MODES)
