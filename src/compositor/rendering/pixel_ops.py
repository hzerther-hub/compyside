"""Pixel operations — NumPy translations of the C kernels in Compositor/Rendering/.

These cover the simple per-channel adjustments that don't need to be in a
shader. The intent is parity with the Swift app's output, so the same
input bytes go through the same math.

For now we ship the most-used building blocks:

- Levels (input/output black/white, gamma per channel)
- Curves (1D LUT per channel)
- Hue / Saturation / Lightness shifts
- Exposure (stops)
- Invert, brightness/contrast, posterise
- Gaussian blur (separable, fixed kernel)
- Add noise (uniform / Gaussian)

The full parity battery lives in ``tests/test_pixel_ops.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Levels / Curves / Hue-Sat primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelRange:
    """Photoshop Levels input/output ranges."""

    black: float = 0.0
    gamma: float = 1.0
    white: float = 255.0
    output_black: float = 0.0
    output_white: float = 255.0


def levels_channel(channel: np.ndarray, r: ChannelRange) -> np.ndarray:
    """Apply Photoshop-style levels to a single uint8 channel."""
    if channel.dtype != np.uint8:
        raise TypeError("expected uint8")
    out = channel.astype(np.float32)
    in_range = max(r.white - r.black, 1e-6)
    # Input mapping: bring [black..white] -> [0..255]
    out = (out - r.black) * (255.0 / in_range)
    # Gamma
    np.clip(out, 0, 255, out=out)
    out = (out / 255.0) ** (1.0 / r.gamma) * 255.0
    # Output mapping
    out = (out / 255.0) * (r.output_white - r.output_black) + r.output_black
    return np.clip(out, 0, 255).astype(np.uint8)


def levels_image(rgb: np.ndarray, ranges: Sequence[ChannelRange]) -> np.ndarray:
    """Apply per-channel levels. ``ranges`` has length 4: RGB + luminance (master).

    The master (index 0) is applied first, then the per-channel overrides.
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.shape[-1] not in (3, 4):
        raise ValueError("expected RGB or RGBA")
    if len(ranges) != 4:
        raise ValueError("expected 4 ranges: master, R, G, B")
    master, rR, rG, rB = ranges
    out = rgb.copy()
    for i, rng in enumerate((rR, rG, rB)):
        out[..., i] = levels_channel(out[..., i], master)
        out[..., i] = levels_channel(out[..., i], rng)
    return out


def apply_curves_lut(channel: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 256-entry 1D LUT (uint8 -> uint8) to a channel."""
    if lut.shape != (256,):
        raise ValueError("LUT must be length 256")
    if lut.dtype != np.uint8:
        lut = lut.astype(np.uint8)
    return lut[channel]


def apply_curves_image(rgb: np.ndarray, luts: Sequence[np.ndarray]) -> np.ndarray:
    """Per-channel 1D LUTs. ``luts`` length 4: master, R, G, B (master unused)."""
    if len(luts) != 4:
        raise ValueError("expected 4 LUTs: master, R, G, B")
    out = rgb.copy()
    for i, lut in enumerate(luts[1:]):
        out[..., i] = apply_curves_lut(out[..., i], lut)
    return out


# ---------------------------------------------------------------------------
# Hue / Saturation / Lightness
# ---------------------------------------------------------------------------


def _rgb_to_hsv_u8(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> HSV float (h in [0,1], s/v in [0,1]).

    Prefers OpenCV's ``cvtColor`` which is ~50× faster than the NumPy
    cascade on large images. Falls back to a pure-NumPy implementation
    when cv2 isn't installed.
    """
    try:
        import cv2  # type: ignore
        # cv2 uses H ∈ [0, 180] (scaled ×2) for 8-bit, but FULL variant
        # returns H ∈ [0, 256] which is close enough to a [0, 1] scale ×255.
        hsv_u8 = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2HSV_FULL)
        h = hsv_u8[..., 0].astype(np.float32) / 255.0
        s = hsv_u8[..., 1].astype(np.float32) / 255.0
        v = hsv_u8[..., 2].astype(np.float32) / 255.0
        return np.stack([h, s, v], axis=-1)
    except ImportError:
        pass
    r = rgb[..., 0].astype(np.float32) / 255.0
    g = rgb[..., 1].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    diff = maxc - minc
    s = np.where(maxc > 1e-6, diff / np.maximum(maxc, 1e-6), 0.0)
    chan = np.argmax(rgb[..., :3], axis=-1)  # 0=R, 1=G, 2=B
    safe_diff = np.where(diff > 1e-6, diff, 1.0)
    h_r = ((g - b) / safe_diff + 6.0) % 6.0
    h_g = ((b - r) / safe_diff) + 2.0
    h_b = ((r - g) / safe_diff) + 4.0
    h_by_chan = np.stack([h_r, h_g, h_b], axis=-1) / 6.0
    h = np.take_along_axis(h_by_chan, chan[..., None], axis=-1).squeeze(-1)
    h = np.where(diff > 1e-6, h, 0.0)
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb_u8(hsv: np.ndarray) -> np.ndarray:
    """HSV float (0..1) -> RGB float (0..1). Same cv2-fast-path preference."""
    try:
        import cv2  # type: ignore
        h_u8 = np.clip(hsv[..., 0] * 255.0, 0, 255).astype(np.uint8)
        s_u8 = np.clip(hsv[..., 1] * 255.0, 0, 255).astype(np.uint8)
        v_u8 = np.clip(hsv[..., 2] * 255.0, 0, 255).astype(np.uint8)
        hsv_u8 = np.stack([h_u8, s_u8, v_u8], axis=-1)
        return cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB_FULL).astype(np.float32) / 255.0
    except ImportError:
        pass
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = (h * 6.0) - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    table = np.stack([v, q, p, p, t, v], axis=-1)
    r = np.take_along_axis(table, i[..., None], axis=-1).squeeze(-1)
    table = np.stack([t, v, v, q, p, p], axis=-1)
    g = np.take_along_axis(table, i[..., None], axis=-1).squeeze(-1)
    table = np.stack([p, p, t, v, v, q], axis=-1)
    b = np.take_along_axis(table, i[..., None], axis=-1).squeeze(-1)
    return np.stack([r, g, b], axis=-1)


def hue_saturation_lightness(
    rgb: np.ndarray,
    hue_shift: float = 0.0,
    sat_shift: float = 0.0,
    light_shift: float = 0.0,
    colorize: bool = False,
) -> np.ndarray:
    """Apply Photoshop-style hue/saturation/lightness.

    ``hue_shift`` is in degrees ([-180, 180]); ``sat_shift`` and
    ``light_shift`` are percentages ([-100, 100]). ``colorize`` replaces
    lightness with the saturation, blending toward fully colored.
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.shape[-1] not in (3, 4):
        raise ValueError("expected RGB or RGBA")
    hsv = _rgb_to_hsv_u8(rgb[..., :3])
    h = (hsv[..., 0] + hue_shift / 360.0) % 1.0
    s = np.clip(hsv[..., 1] * (1.0 + sat_shift / 100.0), 0.0, 1.0)
    v = np.clip(hsv[..., 2] * (1.0 + light_shift / 100.0), 0.0, 1.0)
    if colorize:
        v = np.where(s == 0, v, (v + s) / 2.0)
        s = np.where(hsv[..., 1] == 0, s, 1.0)
    new_hsv = np.stack([h, s, v], axis=-1)
    new_rgb = np.clip(_hsv_to_rgb_u8(new_hsv) * 255.0, 0, 255).astype(np.uint8)
    if rgb.shape[-1] == 4:
        new_rgb = np.concatenate([new_rgb, rgb[..., 3:4]], axis=-1)
    return new_rgb


# ---------------------------------------------------------------------------
# Exposure / Invert / Brightness-Contrast / Posterise
# ---------------------------------------------------------------------------


def exposure(rgb: np.ndarray, stops: float) -> np.ndarray:
    """Apply exposure compensation (in stops). Operates on linear RGB."""
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    factor = 2.0 ** stops
    out = rgb.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def invert(rgb: np.ndarray) -> np.ndarray:
    """Invert the RGB channels. Alpha is preserved."""
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.shape[-1] not in (3, 4):
        raise ValueError("expected RGB or RGBA")
    out = 255 - rgb[..., :3]
    if rgb.shape[-1] == 4:
        out = np.concatenate([out, rgb[..., 3:4]], axis=-1)
    return out.astype(np.uint8)


def brightness_contrast(rgb: np.ndarray, brightness: float = 0.0, contrast: float = 0.0) -> np.ndarray:
    """Brightness/contrast in [-100, 100]. Photoshop-style contrast pivots on 127.5."""
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    out = rgb.astype(np.float32) + brightness * (255.0 / 100.0)
    out = (out - 127.5) * (1.0 + contrast / 100.0) + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def posterise(rgb: np.ndarray, levels: int = 4) -> np.ndarray:
    if levels < 2:
        raise ValueError("levels must be >= 2")
    out = (rgb.astype(np.float32) / 255.0)
    out = np.floor(out * levels) / levels
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gaussian blur (separable)
# ---------------------------------------------------------------------------


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def gaussian_blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur. Returns uint8 of the same shape.

    Implementation uses ``np.lib.stride_tricks.sliding_window_view`` to
    vectorise the convolution — the kernel applies to every pixel in a
    single einsum-style reduction. Falls back to a small Python loop for
    very large kernels where the windows themselves would exceed RAM.
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.ndim != 3:
        raise ValueError("expected HxWxC array")
    if sigma <= 0:
        return rgb.copy()
    k = gaussian_kernel_1d(sigma)
    pad = len(k) // 2
    h, w, c = rgb.shape

    # Use scipy.signal.convolve if available — its separable path is even
    # faster than the vectorised fallback for moderate kernels.
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        # gaussian_filter takes a per-axis sigma and supports uint8 in/out via
        # truncate trick; we map the desired σ to the integer truncation.
        return gaussian_filter(rgb, sigma=(sigma, sigma, 0), mode="reflect", truncate=4.0).astype(np.uint8)
    except ImportError:
        pass

    # Fallback: full-vectorised separable convolution via stride_tricks.
    # Window size stays well under RAM because we only materialise two
    # sliding windows at a time.
    if pad <= 64:  # sane upper bound for the fallback
        padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        # Horizontal: build (H, W, K) view over the padded array.
        h_view = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=(1, len(k), 1), axis=(0, 1, 2)
        )[:, :w, 0, :, 0]  # (H, W, K)
        temp_h = (h_view * k.astype(np.float32)).sum(axis=-1)
        # Vertical: same trick on the temp buffer.
        v_view = np.lib.stride_tricks.sliding_window_view(
            temp_h, window_shape=(len(k), 1), axis=(0, 1)
        )[:h, :, :]  # (H, W, K)
        out = (v_view * k.astype(np.float32)).sum(axis=-1)
        return np.clip(out, 0, 255).astype(np.uint8)

    # Last-resort: tiny kernels only.
    out = rgb.astype(np.float32).copy()
    for _ in range(2):  # two passes
        for axis in (0, 1):
            window_shape = [1, 1]
            window_shape[axis] = len(k)
            view = np.lib.stride_tricks.sliding_window_view(
                np.pad(out, [(pad, pad), (pad, pad), (0, 0)], mode="reflect"),
                window_shape=tuple(window_shape + [1]),
                axis=(0, 1, 2),
            )[:, :, 0, :, 0]
            out = (view * k.astype(np.float32)).sum(axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def add_noise(rgb: np.ndarray, amount: float = 0.1, distribution: str = "uniform", seed: int | None = None) -> np.ndarray:
    """amount in [0, 1]. ``distribution`` is ``"uniform"`` or ``"gaussian"``."""
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        noise = rng.uniform(-amount, amount, size=rgb.shape).astype(np.float32) * 127.0
    elif distribution == "gaussian":
        noise = rng.normal(0.0, amount, size=rgb.shape).astype(np.float32) * 127.0
    else:
        raise ValueError("distribution must be 'uniform' or 'gaussian'")
    out = rgb.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Vignette / Bloom / Dither / Scanlines / Tonal Contrast
# ---------------------------------------------------------------------------


def vignette(
    rgb: np.ndarray,
    amount: float = 0.5,
    midpoint: float = 0.5,
    feather: float = 0.5,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Multiply the image by a radial falloff toward ``color`` at the corners.

    ``amount`` in [-1, 1]: positive darkens corners, negative brightens them.
    ``midpoint`` in [0, 1]: distance at which the falloff begins (0 = centre).
    ``feather`` in [0, 1]: width of the falloff band.
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 array")
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return rgb.copy()
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_dist = float(np.hypot(cy, cx))
    if max_dist <= 0:
        return rgb.copy()
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.hypot(xs - cx, ys - cy) / max_dist  # [0, 1]
    # Smooth falloff band between (midpoint, midpoint + feather).
    edge = midpoint + max(feather, 1e-3)
    t = np.clip((dist - midpoint) / max(edge - midpoint, 1e-3), 0.0, 1.0)
    falloff = 1.0 - amount * t * t * (3.0 - 2.0 * t)  # smoothstep
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = rgb.astype(np.float32).copy()
    out[..., :3] = out[..., :3] * falloff[..., None] + color_arr * (1.0 - falloff[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def bloom(rgb: np.ndarray, threshold: float = 0.7, intensity: float = 1.0, radius: float = 15.0) -> np.ndarray:
    """Extract highlights above ``threshold``, blur, and add back scaled by ``intensity``.

    ``threshold`` in [0, 1] acts on luminance; ``intensity`` is the add-back gain;
    ``radius`` is the Gaussian blur sigma in pixels.
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 array")
    lum = luminance(rgb[..., :3]) / 255.0
    # Soft knee around threshold.
    soft = np.clip((lum - threshold) / max(1.0 - threshold, 1e-3), 0.0, 1.0)
    highlight_f = soft[..., None] * rgb[..., :3].astype(np.float32)
    # Round-trip through uint8 because the Gaussian kernel expects it.
    highlight_u8 = np.clip(highlight_f, 0, 255).astype(np.uint8)
    blurred = gaussian_blur(highlight_u8, sigma=radius).astype(np.float32)
    out = rgb.astype(np.float32).copy()
    out[..., :3] = out[..., :3] + blurred * intensity
    return np.clip(out, 0, 255).astype(np.uint8)


def bayer_dither(rgb: np.ndarray, levels: int = 2, strength: float = 1.0) -> np.ndarray:
    """Ordered (Bayer) dither to ``levels`` output levels per channel.

    ``levels`` >= 2 is the number of output steps. ``strength`` in [0, 1]
    scales how much dither is applied (0 = plain quantisation).
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if levels < 2:
        raise ValueError("levels must be >= 2")
    # 8x8 Bayer matrix normalised to [-0.5, 0.5].
    base = np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    )
    base = (base + 0.5) / 64.0 - 0.5  # in [-0.5, 0.5)
    step = 255.0 / (levels - 1)
    h, w = rgb.shape[:2]
    tile = np.tile(base, ((h + 7) // 8, (w + 7) // 8))[:h, :w]
    threshold = tile * step * strength
    shifted = rgb.astype(np.float32) + threshold[..., None]
    quant = np.round(shifted / step) * step
    return np.clip(quant, 0, 255).astype(np.uint8)


def scanlines(rgb: np.ndarray, spacing: int = 2, intensity: float = 0.5) -> np.ndarray:
    """Darken every Nth row by ``intensity`` (0 = none, 1 = full black).

    ``spacing`` is the row period; rows 1, 1+spacing, 1+2*spacing, … are
    darkened. ``intensity`` is the darkening amount.
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if spacing < 1:
        raise ValueError("spacing must be >= 1")
    out = rgb.astype(np.float32).copy()
    out[1::spacing, :, :3] *= 1.0 - intensity
    return np.clip(out, 0, 255).astype(np.uint8)


def tonal_contrast(rgb: np.ndarray, amount: float = 0.5, radius: float = 3.0) -> np.ndarray:
    """Local contrast enhancement.

    Subtract a Gaussian-blurred luminance field from the original to isolate
    the high-frequency tonal detail, then amplify it by ``amount``.
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 array")
    if radius <= 0:
        return rgb.copy()
    lum = luminance(rgb[..., :3])[..., None].astype(np.float32)
    # Pack luminance into R=G=B so the existing Gaussian treats it as a colour
    # image, then drop the alpha on the way out.
    stacked = np.repeat(lum, 3, axis=-1).astype(np.float32)
    stacked_u8 = np.clip(stacked, 0, 255).astype(np.uint8)
    blurred_stack = gaussian_blur(stacked_u8, sigma=radius).astype(np.float32)
    base_blur = blurred_stack  # HxWx3
    base = stacked  # HxWx3 (luminance tiled to 3 channels)
    detail = base - base_blur
    out = rgb.astype(np.float32).copy()
    out[..., :3] = out[..., :3] + detail * amount
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# AI upscale (Real-ESRGAN with Lanczos fallback)
# ---------------------------------------------------------------------------


def upscale(rgb: np.ndarray, factor: float = 2.0, method: str = "auto") -> np.ndarray:
    """Upscale an RGBA uint8 image by ``factor``.

    ``method`` selects the algorithm:

    * ``"auto"`` (default) — try Real-ESRGAN (``realesrgan-ncnn-py``)
      if installed, fall back to PIL Lanczos.
    * ``"realesrgan"`` — force Real-ESRGAN; raises when missing.
    * ``"lanczos"`` — PIL high-quality resampling.

    Returns an RGBA uint8 array.
    """
    import numpy as _np
    if rgb.dtype != _np.uint8:
        raise TypeError("expected uint8")
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 array")
    h, w = rgb.shape[:2]
    nw, nh = int(round(w * factor)), int(round(h * factor))
    if nw == 0 or nh == 0:
        return rgb.copy()

    chosen = method
    if chosen == "auto":
        try:
            import realesrgan_ncnn_py  # noqa: F401
            chosen = "realesrgan"
        except ImportError:
            chosen = "lanczos"

    if chosen == "realesrgan":
        try:
            import realesrgan_ncnn_py  # type: ignore
            from PIL import Image
            upsampler = realesrgan_ncnn_py.Realesrgan(gpuid=0)
            pil = Image.fromarray(rgb[..., :3])
            out = upsampler.process(pil)
            out_arr = _np.asarray(out, dtype=_np.uint8)
            if rgb.shape[-1] == 4:
                from PIL import Image as PILImage
                alpha_pil = PILImage.fromarray(rgb[..., 3], mode="L").resize(
                    (nw, nh), PILImage.Resampling.LANCZOS
                )
                return _np.concatenate(
                    [out_arr, _np.array(alpha_pil)[..., None]], axis=-1
                )
            return out_arr
        except ImportError as exc:
            raise RuntimeError(
                "realesrgan-ncnn-py is not installed. Run `pip install realesrgan-ncnn-py`."
            ) from exc

    # Lanczos fallback (PIL).
    from PIL import Image
    pil = Image.fromarray(rgb[..., :3], mode="RGB").resize(
        (nw, nh), Image.Resampling.LANCZOS
    )
    out_rgb = _np.asarray(pil, dtype=_np.uint8)
    if rgb.shape[-1] == 4:
        alpha_pil = Image.fromarray(rgb[..., 3], mode="L").resize(
            (nw, nh), Image.Resampling.LANCZOS
        )
        return _np.concatenate([out_rgb, _np.array(alpha_pil)[..., None]], axis=-1)
    return out_rgb


# ---------------------------------------------------------------------------
# Lens correction (barrel / pincushion)
# ---------------------------------------------------------------------------


def lens_correct(
    rgb: np.ndarray,
    strength: float = 0.0,
    cx: float = 0.5,
    cy: float = 0.5,
) -> np.ndarray:
    """Barrel (positive) / pincushion (negative) distortion.

    ``strength`` in [-1, 1] controls magnitude. ``cx`` / ``cy`` are the
    optical centre as a fraction of the image size (default = centre).
    Uses OpenCV's ``remap`` if available, otherwise a NumPy grid_sample.
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 array")
    h, w = rgb.shape[:2]
    if h == 0 or w == 0 or strength == 0.0:
        return rgb.copy()
    cy_pix = float(cy) * h
    cx_pix = float(cx) * w
    # Build normalised (u, v) relative to the optical centre, scaled so that
    # the furthest corner has r ≈ 1.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx = (xs - cx_pix) / max(cx_pix, w - cx_pix, 1.0)
    ry = (ys - cy_pix) / max(cy_pix, h - cy_pix, 1.0)
    r2 = rx * rx + ry * ry
    # Apply the barrel/pincushion mapping: r' = r * (1 + k * r^2).
    k = float(strength)
    factor = 1.0 + k * r2
    src_x = (cx_pix + rx * factor * max(cx_pix, w - cx_pix, 1.0)).astype(np.float32)
    src_y = (cy_pix + ry * factor * max(cy_pix, h - cy_pix, 1.0)).astype(np.float32)
    try:
        import cv2  # type: ignore
        map_x = src_x
        map_y = src_y
        # cv2.remap needs the channels split if RGBA.
        if rgb.shape[-1] == 4:
            channels = [
                cv2.remap(rgb[..., c], map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
                for c in range(4)
            ]
            out = np.stack(channels, axis=-1)
        else:
            channels = [
                cv2.remap(rgb[..., c], map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
                for c in range(3)
            ]
            out = np.stack(channels, axis=-1)
        return out
    except ImportError:
        # Pure NumPy fallback via bilinear interpolation.
        return _bilinear_remap(rgb, src_x, src_y)


def _bilinear_remap(rgb: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    """Bilinear sample from ``rgb`` at fractional (src_x, src_y) coords.

    Out-of-range samples reflect at the image boundary (Photoshop-style).
    """
    h, w = rgb.shape[:2]
    src_x = np.clip(src_x, 0, w - 1.0001)
    src_y = np.clip(src_y, 0, h - 1.0001)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = src_x - x0
    wy = src_y - y0
    out = np.empty_like(rgb)
    one_minus_wx = 1 - wx
    one_minus_wy = 1 - wy
    for c in range(rgb.shape[-1]):
        tl = rgb[y0, x0, c].astype(np.float32)
        tr = rgb[y0, x1, c].astype(np.float32)
        bl = rgb[y1, x0, c].astype(np.float32)
        br = rgb[y1, x1, c].astype(np.float32)
        top = tl * one_minus_wx + tr * wx
        bot = bl * one_minus_wx + br * wx
        out[..., c] = top * one_minus_wy + bot * wy
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Remove Background (foreground extraction)
# ---------------------------------------------------------------------------


def remove_background(rgb: np.ndarray, margin: float = 0.1, method: str = "auto") -> np.ndarray:
    """Extract the foreground and return an RGBA uint8 image.

    ``method`` selects the algorithm:

    * ``"auto"`` (default) — try rembg (U²-Net) if installed, fall back to
      cv2 GrabCut, fall back to the corner-flood heuristic.
    * ``"rembg"`` — force rembg; raises :class:`AIBackendUnavailable`
      when the package is missing.
    * ``"grabcut"`` — cv2 GrabCut with the given ``margin``.
    * ``"flood"`` — pure-NumPy corner-flood heuristic.

    Returns RGBA with the new alpha.
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 array")
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return rgb.copy()

    # Pick the method.
    chosen = method
    if chosen == "auto":
        from . import ai_backends
        if ai_backends.is_rembg_available():
            chosen = "rembg"
        else:
            chosen = "grabcut"

    fg: np.ndarray
    if chosen == "rembg":
        from . import ai_backends
        out = ai_backends.rembg_remove_background(rgb)
        # out is RGBA already; either replace alpha (if input had one) or
        # return as-is.
        if rgb.shape[-1] == 4:
            out = out.copy()
            out[..., 3] = np.minimum(out[..., 3], rgb[..., 3])
            return out
        return out

    if chosen == "grabcut":
        side = max(2, int(round(min(h, w) * max(0.0, min(margin, 0.45)))))
        rect = (side, side, max(2, w - 2 * side), max(2, h - 2 * side))
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            import cv2  # type: ignore
            cv2.grabCut(
                rgb[..., :3],
                mask,
                rect,
                np.zeros((1, 65), dtype=np.float64),
                np.zeros((1, 65), dtype=np.float64),
                3,
                cv2.GC_INIT_WITH_RECT,
            )
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            fg = _corner_flood_mask(rgb[..., :3])
    elif chosen == "flood":
        fg = _corner_flood_mask(rgb[..., :3])
    else:
        raise ValueError(f"unknown method: {chosen!r}")

    # Compose alpha.
    if rgb.shape[-1] == 4:
        out = rgb.copy()
        out[..., 3] = (out[..., 3].astype(np.uint16) * fg.astype(np.uint16) // 255).astype(np.uint8)
        return out
    return np.concatenate([rgb[..., :3], fg[..., None]], axis=-1)


def _corner_flood_mask(rgb: np.ndarray, tol: float = 0.18) -> np.ndarray:
    """Pure-NumPy fallback: mark anything similar to the corners as background."""
    h, w = rgb.shape[:2]
    samples = np.concatenate([
        rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]
    ], axis=0).astype(np.float32) / 255.0
    mean = samples.mean(axis=0)
    std = samples.std(axis=0).max() + 1e-6
    img = rgb.astype(np.float32) / 255.0
    dist = np.linalg.norm(img - mean, axis=-1)
    fg = np.where(dist > tol * (1.0 + std), 255, 0).astype(np.uint8)
    return fg


# ---------------------------------------------------------------------------
# Content-Aware Fill (cv2.inpaint)
# ---------------------------------------------------------------------------


def content_aware_fill(
    rgb: np.ndarray,
    mask: np.ndarray,
    radius: float = 5.0,
    method: str = "auto",
) -> np.ndarray:
    """Inpaint the pixels under ``mask``.

    ``method`` selects the algorithm:

    * ``"auto"`` (default) — try LaMa (deep-learning inpainting) if
      installed, fall back to cv2.inpaint Telea, fall back to NumPy
      iterative diffusion.
    * ``"lama"`` — force LaMa; raises :class:`AIBackendUnavailable` if
      the package is missing.
    * ``"telea"`` — cv2 INPAINT_TELEA (fast, classical).
    * ``"ns"`` — cv2 INPAINT_NS (Navier-Stokes, slower, smoother).
    * ``"numpy"`` — pure-NumPy iterative diffusion.

    Returns RGBA uint8 (preserves the input alpha when present).
    """
    if rgb.dtype != np.uint8:
        raise TypeError("expected uint8")
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 array")
    h, w = rgb.shape[:2]
    if mask.shape != (h, w):
        raise ValueError("mask shape must match image")
    if not (mask > 0).any():
        return rgb.copy()

    chosen = method
    if chosen == "auto":
        from . import ai_backends
        if ai_backends.is_lama_available():
            chosen = "lama"
        else:
            chosen = "telea"

    if chosen == "lama":
        from . import ai_backends
        out = ai_backends.lama_inpaint(rgb, mask)
        if rgb.shape[-1] == 4:
            out = out.copy()
            original_alpha = rgb[..., 3]
            out[..., 3] = np.where(mask > 0, 255, original_alpha).astype(np.uint8)
        return out

    if chosen in ("telea", "ns"):
        try:
            import cv2  # type: ignore
            flag = cv2.INPAINT_TELEA if chosen == "telea" else cv2.INPAINT_NS
            if rgb.shape[-1] == 4:
                filled = rgb.copy()
                for c in range(3):
                    filled[..., c] = cv2.inpaint(rgb[..., c], mask, radius, flag)
                original_alpha = rgb[..., 3]
                filled[..., 3] = np.where(mask > 0, 255, original_alpha).astype(np.uint8)
                return filled
            filled = rgb.copy()
            for c in range(3):
                filled[..., c] = cv2.inpaint(rgb[..., c], mask, radius, flag)
            return filled
        except ImportError:
            return _numpy_inpaint(rgb, mask, radius)
    if chosen == "numpy":
        return _numpy_inpaint(rgb, mask, radius)
    raise ValueError(f"unknown method: {chosen!r}")


def _numpy_inpaint(rgb: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
    """Naive NumPy inpainting fallback.

    Iteratively dilates the source region into the masked region by averaging
    over a small disc. Adequate for small holes when OpenCV is unavailable.
    """
    h, w = rgb.shape[:2]
    out = rgb.copy().astype(np.float32)
    cur_mask = mask.copy().astype(bool)
    side = max(1, int(round(radius)))
    # Run a handful of iterations to propagate surrounding pixels into the hole.
    for _ in range(side * 2 + 1):
        if not cur_mask.any():
            break
        # Build a window average over the unmasked neighbourhood.
        pad = side
        padded = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        mask_padded = np.pad(~cur_mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)
        # For each masked pixel, average unmasked neighbours.
        ys, xs = np.where(cur_mask)
        if len(ys) == 0:
            break
        avg = np.zeros((len(ys), rgb.shape[-1]), dtype=np.float32)
        count = np.zeros(len(ys), dtype=np.float32)
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                py = ys + pad + dy
                px = xs + pad + dx
                valid = mask_padded[py, px]
                avg[valid] += padded[py[valid], px[valid], :]
                count[valid] += 1.0
        count = np.maximum(count, 1.0)
        avg = avg / count[..., None]
        out[ys, xs, :] = avg
        # Shrink the mask by one pixel ring so we propagate outward.
        cur_mask[:] = False
        # Use scipy-free erosion via numpy slicing — invert via slice trick.
        cur_mask[1:-1, 1:-1] = mask[2:, 2:] | mask[2:, 1:-1] | mask[2:, :-2] \
            | mask[1:-1, 2:] | mask[1:-1, :-2] \
            | mask[:-2, 2:] | mask[:-2, 1:-1] | mask[:-2, :-2]
        cur_mask &= mask > 0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers used by adjustments module
# ---------------------------------------------------------------------------


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of an RGBA uint8 image."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b)
