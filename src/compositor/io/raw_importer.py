"""Camera RAW import via rawpy (LibRaw)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def is_raw(path: str | Path) -> bool:
    """Best-effort detection of camera RAW formats."""
    suffixes = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw", ".x3f"}
    return Path(path).suffix.lower() in suffixes


def import_raw(path: str | Path) -> np.ndarray:
    """Open a RAW file and return the demosaiced image as RGBA uint8.

    Falls back to ``None`` if rawpy isn't installed.
    """
    try:
        import rawpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "rawpy not installed. Install with: pip install rawpy"
        ) from exc

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=16,
        )
    rgb8 = (rgb16.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    if rgb8.ndim == 3:
        alpha = np.full(rgb8.shape[:2] + (1,), 255, dtype=np.uint8)
        rgb8 = np.concatenate([rgb8, alpha], axis=-1)
    return rgb8


def develop(
    image: np.ndarray,
    *,
    exposure: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    temperature: float = 0.0,
    tint: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    whites: float = 0.0,
    blacks: float = 0.0,
    clarity: float = 0.0,
    vibrance: float = 0.0,
    sharpening_amount: float = 0.0,
    noise_reduction: float = 0.0,
    # T3-22 additions — Photoshop-Camera-Raw panels:
    dehaze: float = 0.0,
    curve_darks: float = 0.0,
    curve_lights: float = 0.0,
    shadow_hue: float = 220.0,
    shadow_sat: float = 0.5,
    highlight_hue: float = 50.0,
    highlight_sat: float = 0.5,
    balance: float = 0.0,
    sharpening_radius: float = 1.0,
    lens_strength: float = 0.0,
    lens_cx: float = 0.5,
    lens_cy: float = 0.5,
    rotate: float = 0.0,
    aspect: float = 1.0,
    scale: float = 1.0,
    **mixer: float,
) -> np.ndarray:
    """Apply a Photoshop-Camera-Raw-style develop pass to an RGBA uint8 image.

    Each control is a small NumPy transformation; together they cover the
    most-used sliders. Complex demosaic / geometry is left to the rawpy
    import step.
    """
    rgb = image[..., :3].astype(np.float32)
    a = image[..., 3:4]

    # Exposure (stops).
    if exposure:
        rgb = rgb * (2.0 ** exposure)
    # Temperature / tint — adjust warm/cool and magenta/green balance.
    if temperature:
        rgb[..., 0] += temperature * 8  # R
        rgb[..., 2] -= temperature * 8  # B
    if tint:
        rgb[..., 1] += tint * 6  # G
        rgb[..., 0] -= tint * 3
        rgb[..., 2] -= tint * 3
    # Contrast around 50%.
    if contrast != 1.0:
        rgb = (rgb - 127.5) * contrast + 127.5
    # Highlights / shadows lift.
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if highlights:
        rgb += (np.clip(lum - 127, 0, 128) / 128.0)[..., None] * highlights * 60.0
    if shadows:
        rgb += (np.clip(127 - lum, 0, 128) / 128.0)[..., None] * shadows * 60.0
    if whites:
        rgb += (np.clip(lum - 200, 0, 55) / 55.0)[..., None] * whites * 60.0
    if blacks:
        rgb += (np.clip(50 - lum, 0, 50) / 50.0)[..., None] * blacks * 60.0
    # Saturation / vibrance.
    if saturation != 1.0:
        gray = rgb.mean(axis=-1, keepdims=True)
        rgb = gray + (rgb - gray) * saturation
    if vibrance:
        sat = 1.0 - rgb.max(axis=-1, keepdims=True) / 255.0
        rgb += (rgb - rgb.mean(axis=-1, keepdims=True)) * vibrance * sat
    # Clarity — local contrast.
    if clarity:
        from ..rendering import pixel_ops as po
        blurred = po.gaussian_blur(rgb.astype(np.uint8), sigma=20).astype(np.float32)
        rgb += (rgb - blurred) * clarity * 0.5
    # Dehaze — boosts mid-contrast (similar to clarity but wider radius).
    if dehaze:
        from ..rendering import pixel_ops as po
        blurred = po.gaussian_blur(rgb.astype(np.uint8), sigma=40).astype(np.float32)
        rgb += (rgb - blurred) * dehaze * 0.3
    # Curves — darks/lights pivots around the 0 / 255 anchors.
    if curve_darks:
        # Lift or crush the bottom 25% of the tonal range.
        rgb = np.where(rgb < 64, rgb + curve_darks * 0.4, rgb)
    if curve_lights:
        rgb = np.where(rgb > 192, rgb + curve_lights * 0.4, rgb)
    # Split toning — colourise shadows and highlights.
    if shadow_sat > 0 or highlight_sat > 0:
        sh_hue_rad = np.radians(shadow_hue)
        hi_hue_rad = np.radians(highlight_hue)
        sh_rgb = np.array([
            np.cos(sh_hue_rad) * 255 * shadow_sat,
            np.cos(sh_hue_rad - 2 * np.pi / 3) * 255 * shadow_sat,
            np.cos(sh_hue_rad - 4 * np.pi / 3) * 255 * shadow_sat,
        ])
        hi_rgb = np.array([
            np.cos(hi_hue_rad) * 255 * highlight_sat,
            np.cos(hi_hue_rad - 2 * np.pi / 3) * 255 * highlight_sat,
            np.cos(hi_hue_rad - 4 * np.pi / 3) * 255 * highlight_sat,
        ])
        shadow_weight = np.clip(1.0 - lum / 128.0 + balance * 0.5, 0.0, 1.0)[..., None]
        high_weight = np.clip((lum - 128) / 128.0 - balance * 0.5, 0.0, 1.0)[..., None]
        rgb = rgb + sh_rgb * shadow_weight + hi_rgb * high_weight
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if sharpening_amount:
        from ..rendering import pixel_ops as po
        blurred = po.gaussian_blur(rgb, sigma=sharpening_radius)
        rgb = np.clip(rgb.astype(np.float32) + (rgb - blurred).astype(np.float32) * sharpening_amount, 0, 255).astype(np.uint8)
    if noise_reduction:
        from ..rendering import pixel_ops as po
        rgb = po.gaussian_blur(rgb, sigma=noise_reduction)
    # Optics — lens correction (barrel / pincushion).
    if lens_strength:
        from ..rendering.pixel_ops import lens_correct
        rgb = lens_correct(rgb, strength=lens_strength, cx=lens_cx, cy=lens_cy)
    # Geometry — rotate/aspect/scale produce a canvas-resize + warp. For
    # simplicity we apply them as a quick cv2 warp when available; fall
    # back to no-op otherwise (the geometry preview is then approximate).
    if abs(rotate) > 0.01 or abs(scale - 1.0) > 0.01 or abs(aspect - 1.0) > 0.01:
        try:
            import cv2  # type: ignore
            h, w = rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate, scale)
            M[0, 0] *= aspect
            rgb = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        except ImportError:
            pass
    return np.concatenate([rgb, a], axis=-1)
