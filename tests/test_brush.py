"""Brush engine tests."""
from __future__ import annotations

import numpy as np

from compositor.rendering import brush as brush_engine


def test_make_stamp_shape() -> None:
    settings = brush_engine.BrushSettings(size=20, hardness=0.7, opacity=1.0, color=(255, 0, 0))
    stamp = brush_engine.make_stamp(settings)
    assert stamp.shape == (20, 20, 4)
    assert (stamp[..., 0] == 255).all()
    assert (stamp[..., 1] == 0).all()
    # Center alpha should be high.
    assert stamp[10, 10, 3] > 200


def test_apply_dab_paint() -> None:
    canvas = np.zeros((20, 20, 4), dtype=np.uint8)
    stamp = brush_engine.make_stamp(brush_engine.BrushSettings(size=8, color=(255, 255, 255)))
    brush_engine.apply_dab(canvas, stamp, (10, 10), mode="paint")
    assert (canvas[..., :3] == 255).any()


def test_apply_dab_erase() -> None:
    canvas = np.full((20, 20, 4), 200, dtype=np.uint8)
    canvas[..., 3] = 255
    stamp = brush_engine.make_stamp(brush_engine.BrushSettings(size=8, color=(0, 0, 0)))
    brush_engine.apply_dab(canvas, stamp, (10, 10), mode="erase")
    assert (canvas[..., 3] < 255).any()


def test_stroke_segment() -> None:
    canvas = np.zeros((20, 20, 4), dtype=np.uint8)
    settings = brush_engine.BrushSettings(size=4, color=(255, 255, 255), spacing=0.2)
    stamp = brush_engine.make_stamp(settings)
    state = brush_engine.BrushState()
    brush_engine.stroke_segment(canvas, (2, 2), (18, 18), settings, state, stamp)
    # Some painted pixels along the diagonal.
    assert (canvas[..., 0] == 255).any()
