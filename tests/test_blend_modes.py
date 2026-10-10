"""Blend-mode tests against Photoshop reference behaviour.

We don't have a Photoshop here, but the W3C compositing spec gives exact
formulas; we verify known edge cases and confirm the registry covers all
24 documented modes.
"""
from __future__ import annotations

import numpy as np
import pytest

from compositor.format.manifest import BLEND_MODES
from compositor.rendering import blend_modes as bm


def _canvas(value: int = 128) -> np.ndarray:
    arr = np.full((4, 4, 4), value, dtype=np.uint8)
    arr[..., 3] = 255
    return arr


def test_registry_covers_all_modes() -> None:
    for mode in BLEND_MODES:
        assert mode in bm.BLEND_MODE_FUNCTIONS, f"missing implementation for {mode}"


def test_normal_is_source_over() -> None:
    bg = _canvas(50)
    fg = _canvas(200)
    out = bm.apply_blend(bg, fg, "Normal")
    # Alpha 255 + source-over: result should be ~ source.
    np.testing.assert_allclose(out[..., :3], 200, atol=1)


def test_multiply_darkens() -> None:
    bg = _canvas(200)
    fg = _canvas(200)
    out = bm.apply_blend(bg, fg, "Multiply")
    # Multiply 200/255 * 200/255 ≈ 0.616 * 255 ≈ 157
    assert 140 < out[0, 0, 0] < 170


def test_screen_lightens() -> None:
    bg = _canvas(80)
    fg = _canvas(80)
    out = bm.apply_blend(bg, fg, "Screen")
    # Screen: 1 - (1-80/255) * (1-80/255) ≈ 1 - 0.478 ≈ 0.522 → 133
    assert 120 < out[0, 0, 0] < 145


def test_difference_is_absolute() -> None:
    bg = _canvas(100)
    fg = _canvas(50)
    out = bm.apply_blend(bg, fg, "Difference")
    assert out[0, 0, 0] == 50  # |100 - 50|


def test_linear_burn_clamps_to_zero() -> None:
    bg = _canvas(30)
    fg = _canvas(30)
    out = bm.apply_blend(bg, fg, "Linear Burn")
    # max(30 + 30 - 255, 0) = 0
    assert out[0, 0, 0] == 0


def test_linear_dodge_clamps_to_255() -> None:
    bg = _canvas(200)
    fg = _canvas(200)
    out = bm.apply_blend(bg, fg, "Linear Dodge (Add)")
    assert out[0, 0, 0] == 255


def test_color_dodge_pure_white() -> None:
    bg = np.full((2, 2, 4), 100, dtype=np.uint8)
    bg[..., 3] = 255
    fg = np.full((2, 2, 4), 255, dtype=np.uint8)
    fg[..., 3] = 255
    out = bm.apply_blend(bg, fg, "Color Dodge")
    assert (out[..., :3] == 255).all()


def test_overlay_pivots_on_backdrop() -> None:
    bg = np.full((2, 2, 4), 64, dtype=np.uint8)  # dark backdrop
    bg[..., 3] = 255
    fg = np.full((2, 2, 4), 200, dtype=np.uint8)
    fg[..., 3] = 255
    out = bm.apply_blend(bg, fg, "Overlay")
    # On a dark backdrop, overlay = 2*B*S = 2*(64/255)*(200/255) ≈ 0.394 → ~100.
    # That's darker than the source (200) but not as dark as the backdrop (64).
    assert out[0, 0, 0] < 200
    assert out[0, 0, 0] > 64


def test_unknown_mode_raises() -> None:
    bg = _canvas()
    fg = _canvas()
    with pytest.raises(ValueError):
        bm.apply_blend(bg, fg, "Made Up Mode")


def test_shape_mismatch_raises() -> None:
    bg = np.zeros((4, 4, 4), dtype=np.uint8)
    fg = np.zeros((4, 5, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        bm.apply_blend(bg, fg, "Multiply")
