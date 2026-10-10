"""Layer-effects tests."""
from __future__ import annotations

import numpy as np

from compositor.rendering.layer_effects import (
    ColorOverlay,
    DropShadow,
    InnerGlow,
    OuterGlow,
    Stroke,
    apply_color_overlay,
    apply_drop_shadow,
    apply_inner_glow,
    apply_outer_glow,
    apply_stroke,
)


def _layer_with_alpha(shape=(20, 20, 4)) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[..., 3] = 0
    arr[5:15, 5:15, 3] = 255
    arr[5:15, 5:15, 0] = 200
    return arr


def test_drop_shadow_canvas_size() -> None:
    layer = _layer_with_alpha()
    out = apply_drop_shadow(layer, DropShadow(color=(0, 0, 0, 255), opacity=0.5, distance=3, blur=2), canvas_shape=(40, 40))
    assert out.shape == (40, 40, 4)
    # The shadow should add a dark region offset from the layer.
    assert (out[..., 0] < 200).any()


def test_color_overlay_replaces_color() -> None:
    layer = np.zeros((10, 10, 4), dtype=np.uint8)
    layer[..., 3] = 255
    out = apply_color_overlay(layer, ColorOverlay(color=(255, 0, 0, 255), opacity=1.0))
    assert (out[..., 0] == 255).all()
    assert (out[..., 1] == 0).all()


def test_outer_glow() -> None:
    layer = _layer_with_alpha()
    out = apply_outer_glow(layer, OuterGlow(color=(255, 255, 255, 255), opacity=0.5, blur=2), canvas_shape=(40, 40))
    assert out.shape == (40, 40, 4)


def test_inner_glow() -> None:
    layer = _layer_with_alpha()
    out = apply_inner_glow(layer, InnerGlow(color=(255, 255, 255, 255), opacity=0.5, blur=2))
    assert out.shape == layer.shape


def test_stroke_outline() -> None:
    layer = _layer_with_alpha()
    out = apply_stroke(layer, Stroke(color=(255, 255, 255, 255), size=2, position="outside"))
    assert out.shape == layer.shape
    # Stroke should add pixels near the alpha boundary.
    assert (out[..., 3] > layer[..., 3]).any()
