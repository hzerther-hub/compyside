"""Adjustment-layer and filter application.

Adjustments are applied to a NumPy RGBA image in place via a fresh array.
The dispatcher uses the adjustment kind to pick the right routine.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..document.layers import AdjustmentKind
from . import pixel_ops as po


def apply_adjustment(image: np.ndarray, kind: AdjustmentKind, settings: Dict[str, Any]) -> np.ndarray:
    """Apply ``kind`` to ``image`` (HxWx4 uint8) and return the result."""
    if image.dtype != np.uint8:
        raise TypeError("expected uint8")
    if image.shape[-1] not in (3, 4):
        raise ValueError("expected RGB or RGBA")

    if kind == AdjustmentKind.HUE_SATURATION:
        return po.hue_saturation_lightness(
            image,
            hue_shift=float(settings.get("hue", 0.0)),
            sat_shift=float(settings.get("saturation", 0.0)),
            light_shift=float(settings.get("lightness", 0.0)),
            colorize=bool(settings.get("colorize", False)),
        )
    if kind == AdjustmentKind.LEVELS:
        return _apply_levels(image, settings)
    if kind == AdjustmentKind.CURVES:
        return _apply_curves(image, settings)
    if kind == AdjustmentKind.EXPOSURE:
        return po.exposure(image, stops=float(settings.get("exposure", 0.0)))
    if kind == AdjustmentKind.INVERT:
        return po.invert(image)
    if kind == AdjustmentKind.BLACK_WHITE:
        return _apply_black_white(image, settings)
    if kind == AdjustmentKind.COLOR_BALANCE:
        return _apply_color_balance(image, settings)
    if kind == AdjustmentKind.GRADIENT_MAP:
        return _apply_gradient_map(image, settings)
    if kind == AdjustmentKind.GRAIN:
        return _apply_grain(image, settings)
    if kind == AdjustmentKind.GAUSSIAN_BLUR:
        return po.gaussian_blur(image, sigma=float(settings.get("radius", 5.0)))
    if kind == AdjustmentKind.MOTION_BLUR:
        return _apply_motion_blur(image, settings)
    if kind == AdjustmentKind.ADD_NOISE:
        return po.add_noise(
            image,
            amount=float(settings.get("amount", 0.1)),
            distribution=str(settings.get("distribution", "uniform")),
            seed=settings.get("seed"),
        )
    if kind == AdjustmentKind.VIGNETTE:
        vs = settings.get("vignetteSettings", settings)
        return po.vignette(
            image,
            amount=float(vs.get("amount", 0.5)),
            midpoint=float(vs.get("midpoint", 0.5)),
            feather=float(vs.get("feather", 0.5)),
            color=tuple(vs.get("color", (0, 0, 0))),
        )
    if kind == AdjustmentKind.BLOOM:
        bs = settings.get("bloomSettings", settings)
        return po.bloom(
            image,
            threshold=float(bs.get("threshold", 0.7)),
            intensity=float(bs.get("intensity", 1.0)),
            radius=float(bs.get("radius", 15.0)),
        )
    if kind == AdjustmentKind.DITHER:
        ds = settings.get("ditherSettings", settings)
        return po.bayer_dither(
            image,
            levels=int(ds.get("levels", 2)),
            strength=float(ds.get("strength", 1.0)),
        )
    if kind == AdjustmentKind.SCANLINES:
        ss = settings.get("scanlinesSettings", settings)
        return po.scanlines(
            image,
            spacing=int(ss.get("spacing", 2)),
            intensity=float(ss.get("intensity", 0.5)),
        )
    if kind == AdjustmentKind.TONAL_CONTRAST:
        ts = settings.get("tonalContrastSettings", settings)
        return po.tonal_contrast(
            image,
            amount=float(ts.get("amount", 0.5)),
            radius=float(ts.get("radius", 3.0)),
        )
    if kind == AdjustmentKind.LENS_CORRECTION:
        ls = settings.get("lensCorrectionSettings", settings)
        return po.lens_correct(
            image,
            strength=float(ls.get("strength", 0.0)),
            cx=float(ls.get("cx", 0.5)),
            cy=float(ls.get("cy", 0.5)),
        )
    if kind == AdjustmentKind.REMOVE_BACKGROUND:
        rs = settings.get("removeBackgroundSettings", settings)
        return po.remove_background(image, margin=float(rs.get("margin", 0.1)))
    raise ValueError(f"unsupported adjustment kind: {kind!r}")


# ---------------------------------------------------------------------------
# Per-kind helpers
# ---------------------------------------------------------------------------


def _apply_levels(image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    levels = settings.get("levels")
    if not levels or "ranges" not in levels:
        return image
    ranges_data = levels["ranges"]
    if len(ranges_data) < 4:
        return image
    ranges = [po.ChannelRange(**_camel_to_snake(r)) for r in ranges_data[:4]]
    return po.levels_image(image, ranges)


def _camel_to_snake(d: Dict[str, Any]) -> Dict[str, Any]:
    """Map manifest camelCase keys (``outputBlack``) to dataclass snake_case
    (``output_black``) without renaming ``black`` / ``white`` / ``gamma``.
    """
    mapping = {"outputBlack": "output_black", "outputWhite": "output_white"}
    return {mapping.get(k, k): v for k, v in d.items()}


def _apply_curves(image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    curves = settings.get("curves")
    if not curves or "channels" not in curves:
        return image
    channels = curves["channels"]
    if len(channels) < 4:
        return image
    luts: list[np.ndarray] = []
    for ch in channels[:4]:
        luts.append(_build_curve_lut(ch))
    return po.apply_curves_image(image, luts)


def _build_curve_lut(points: list) -> np.ndarray:
    """Interpolate a 256-entry LUT from control points [{x, y}]."""
    lut = np.zeros(256, dtype=np.uint8)
    pts = sorted(points, key=lambda p: p["x"])
    if not pts:
        lut[:] = np.arange(256, dtype=np.uint8)
        return lut
    # Ensure endpoints.
    if pts[0]["x"] > 0:
        pts = [{"x": 0, "y": 0}] + pts
    if pts[-1]["x"] < 255:
        pts = pts + [{"x": 255, "y": 255}]
    # Walk the piecewise-linear interpolation.
    for i in range(len(pts) - 1):
        x0 = int(pts[i]["x"])
        x1 = int(pts[i + 1]["x"])
        y0 = float(pts[i]["y"])
        y1 = float(pts[i + 1]["y"])
        if x1 == x0:
            continue
        xs = np.arange(x0, x1 + 1)
        ys = y0 + (xs - x0) * (y1 - y0) / (x1 - x0)
        lut[x0 : x1 + 1] = np.clip(ys, 0, 255).astype(np.uint8)
    return lut


def _apply_black_white(image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    """Quick luminance-only B&W; finer controls deferred to Phase 2."""
    lum = po.luminance(image[..., :3])[..., None]
    if image.shape[-1] == 4:
        lum = np.concatenate([lum, lum, lum, image[..., 3:4]], axis=-1)
    else:
        lum = np.concatenate([lum, lum, lum], axis=-1)
    return np.clip(lum, 0, 255).astype(np.uint8)


def _apply_color_balance(image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    cb = settings.get("colorBalanceSettings", {})
    rgb = image[..., :3].astype(np.float32)
    shadow = np.array([
        float(cb.get("shadowCyanRed", 0)),
        float(cb.get("shadowMagentaGreen", 0)),
        float(cb.get("shadowYellowBlue", 0)),
    ])
    mid = np.array([
        float(cb.get("midCyanRed", 0)),
        float(cb.get("midMagentaGreen", 0)),
        float(cb.get("midYellowBlue", 0)),
    ])
    high = np.array([
        float(cb.get("highlightCyanRed", 0)),
        float(cb.get("highlightMagentaGreen", 0)),
        float(cb.get("highlightYellowBlue", 0)),
    ])
    lum = po.luminance(image[..., :3]) / 255.0
    weights = np.stack([1.0 - lum, lum * 2 * (1.0 - lum), lum * lum], axis=-1)  # (H, W, 3)
    weights = weights / weights.sum(axis=-1, keepdims=True).clip(1e-6)
    correction = weights @ np.stack([shadow, mid, high], axis=0)  # (H, W, 3)
    rgb = rgb + correction[..., None, :] * (255.0 / 100.0) if False else rgb + correction * (255.0 / 100.0)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if image.shape[-1] == 4:
        rgb = np.concatenate([rgb, image[..., 3:4]], axis=-1)
    return rgb


def _apply_gradient_map(image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    """Map luminance through a 2-stop gradient (shadows/highlights)."""
    gm = settings.get("gradientMapSettings", {})
    shadows = np.array(gm.get("shadows", [0, 0, 0]), dtype=np.float32)
    highlights = np.array(gm.get("highlights", [255, 255, 255]), dtype=np.float32)
    if bool(gm.get("reversed", False)):
        shadows, highlights = highlights, shadows
    lum = po.luminance(image[..., :3]) / 255.0
    t = lum[..., None]
    mapped = shadows * (1.0 - t) + highlights * t
    mapped = np.clip(mapped, 0, 255).astype(np.uint8)
    if image.shape[-1] == 4:
        mapped = np.concatenate([mapped, image[..., 3:4]], axis=-1)
    return mapped


def _apply_grain(image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    grain = settings.get("grainSettings", {})
    amount = float(grain.get("amount", 0.1))
    size = float(grain.get("size", 1.0))
    # "size" > 1 means larger grain → blur the noise before adding.
    if size > 1.0:
        noise = po.add_noise(np.zeros_like(image), amount=amount, seed=grain.get("seed"))
        noise = po.gaussian_blur(noise, sigma=size)
        return np.clip(image.astype(np.float32) + noise.astype(np.float32) - 128, 0, 255).astype(np.uint8)
    return po.add_noise(image, amount=amount, seed=grain.get("seed"))


def _apply_motion_blur(image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    mb = settings.get("motionBlurSettings", {})
    distance = float(mb.get("distance", 10.0))
    angle = float(mb.get("angle", 0.0))
    if distance <= 0:
        return image
    # Build a 1D motion-blur kernel along (cos a, sin a).
    radius = max(1, int(math.ceil(distance)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))
    for i, t in enumerate(coords):
        x = int(round(t * cos_a))
        y = int(round(t * sin_a))
        if -radius <= y < radius + 1 and -radius <= x < radius + 1:
            kernel[y + radius, x + radius] = 1.0
    if kernel.sum() == 0:
        return image
    kernel /= kernel.sum()
    return _convolve2d(image, kernel)


import math  # noqa: E402  (used by helpers above)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Naive 2D convolution. Adequate for small kernels (motion blur)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect")
    out = np.zeros_like(image)
    for j in range(out.shape[0]):
        for i in range(out.shape[1]):
            window = padded[j : j + kh, i : i + kw, :]
            out[j, i, :] = (window * kernel[..., None]).sum(axis=(0, 1))
    return np.clip(out, 0, 255).astype(np.uint8)
