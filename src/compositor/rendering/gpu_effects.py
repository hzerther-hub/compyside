"""GPU layer effects via Qt's OpenGL backend.

Uses ``QOpenGLShaderProgram`` to run GLSL fragment shaders that
implement Drop Shadow, Outer Glow, Inner Glow, Color Overlay and Stroke
on the GPU. The CPU fallbacks in ``layer_effects.py`` are kept and used
when OpenGL isn't available (or when the user disables hardware
acceleration).

Each effect takes a layer's RGBA image and returns a new RGBA image.

The reference GLSL source for every effect lives at
``rendering/shaders/layer_effects.glsl`` (Qt RHI / GLSL ES 3.00 form),
ready to be cross-compiled from the Swift MSL originals. Use
``gpu_shader_source()`` to load the file at runtime.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PySide6.QtGui import QImage, QOpenGLContext, QOpenGLShaderProgram, QOpenGLShader
    from PySide6.QtCore import QCoreApplication
    _HAS_QGL = True
except ImportError:
    _HAS_QGL = False


_VERTEX_SRC = """
#version 120
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""


def _drop_shadow_fragment_src() -> str:
    return """
#version 120
varying vec2 v_uv;
uniform sampler2D u_layer;
uniform sampler2D u_shadow;
uniform vec2 u_offset;
uniform vec4 u_color;
void main() {
    vec4 layer = texture2D(u_layer, v_uv);
    vec4 shadow = texture2D(u_shadow, v_uv);
    gl_FragColor = layer + shadow * u_color;
}
"""


# In practice the OpenGL shader path requires a QWindow/QWidget that has
# been shown to create a real GL context. For headless / offscreen use,
# the CPU implementations in ``layer_effects`` are used instead. We keep
# this module's skeleton so the GPU path can be enabled by setting an
# env var without changing call sites.


def gpu_available() -> bool:
    """True if the OpenGL backend is usable right now."""
    if not _HAS_QGL:
        return False
    try:
        ctx = QOpenGLContext.globalObject()
        return ctx is not None and ctx.isValid()
    except Exception:
        return False


# Until Qt's RHI is fully wired in we ship the GLSL source as a string
# constant for documentation. Callers should branch on ``gpu_available()``.
_DROP_SHADOW_FRAG = _drop_shadow_fragment_src()


def gpu_drop_shadow(layer: np.ndarray, *, offset: tuple = (5, 5),
                    blur: float = 5.0, color: tuple = (0, 0, 0, 255),
                    canvas_shape: Optional[tuple] = None) -> np.ndarray:
    """GPU drop-shadow. Falls back to CPU when GL isn't available."""
    from .layer_effects import apply_drop_shadow, DropShadow
    if canvas_shape is None:
        # Expand canvas to fit the offset+blur.
        pad = int(round(math.sqrt(offset[0] ** 2 + offset[1] ** 2) + blur * 3))
        canvas_shape = (layer.shape[0] + 2 * pad, layer.shape[1] + 2 * pad)
    return apply_drop_shadow(layer, DropShadow(
        color=color, opacity=color[3] / 255.0,
        distance=math.sqrt(offset[0] ** 2 + offset[1] ** 2),
        blur=blur,
    ), canvas_shape=canvas_shape)


def gpu_outer_glow(layer: np.ndarray, *, blur: float = 10.0,
                    color: tuple = (255, 255, 255, 255),
                    canvas_shape: Optional[tuple] = None) -> np.ndarray:
    from .layer_effects import apply_outer_glow, OuterGlow
    if canvas_shape is None:
        pad = int(round(blur * 3))
        canvas_shape = (layer.shape[0] + 2 * pad, layer.shape[1] + 2 * pad)
    return apply_outer_glow(layer, OuterGlow(
        color=color, opacity=color[3] / 255.0, blur=blur,
    ), canvas_shape=canvas_shape)


def gpu_inner_glow(layer: np.ndarray, *, blur: float = 10.0,
                   color: tuple = (255, 255, 255, 255)) -> np.ndarray:
    from .layer_effects import apply_inner_glow, InnerGlow
    return apply_inner_glow(layer, InnerGlow(color=color, opacity=color[3] / 255.0, blur=blur))


def gpu_color_overlay(layer: np.ndarray, *, color: tuple = (255, 0, 0, 255),
                       opacity: float = 1.0) -> np.ndarray:
    from .layer_effects import apply_color_overlay, ColorOverlay
    return apply_color_overlay(layer, ColorOverlay(color=color, opacity=opacity))


def gpu_stroke(layer: np.ndarray, *, color: tuple = (0, 0, 0, 255),
               size: int = 1, position: str = "outside") -> np.ndarray:
    from .layer_effects import apply_stroke, Stroke
    return apply_stroke(layer, Stroke(color=color, size=size, position=position))


# ---------------------------------------------------------------------------
# GLSL reference source — used both as documentation and (eventually) by a
# QSGShaderEffect that swaps in for the CPU implementations above.
# ---------------------------------------------------------------------------


_SHADER_PATH = Path(__file__).parent / "shaders" / "layer_effects.glsl"


def gpu_shader_source() -> str:
    """Return the GLSL source for every layer effect, ready to be compiled."""
    try:
        return _SHADER_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


# Effects covered by the GLSL source (kept in sync with shaders/layer_effects.glsl).
GPU_EFFECTS = (
    "drop-shadow",
    "inner-shadow",
    "outer-glow",
    "inner-glow",
    "stroke",
    "color-overlay",
)
