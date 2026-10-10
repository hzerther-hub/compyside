"""Performance regression tests — fail if hot paths exceed a budget.

These are gated by env var so they don't slow down regular ``pytest`` runs.
Set ``COMPY_PERF=1`` to enable:

    COMPY_PERF=1 pytest tests/test_benchmarks.py

Budgets are deliberately generous (10× what we measured) so flaky timing
doesn't trip CI. Tighten as hardware catches up.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("COMPY_PERF") != "1",
    reason="set COMPY_PERF=1 to enable performance regression tests",
)


def _time(fn, *, iterations: int = 3) -> float:
    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    return min(durations)


def test_gaussian_blur_512_budget() -> None:
    from compositor.rendering.pixel_ops import gaussian_blur
    arr = (np.random.rand(512, 512, 4) * 255).astype(np.uint8)
    arr[..., 3] = 255
    elapsed = _time(lambda: gaussian_blur(arr, sigma=5.0))
    # 512² blur should fit in < 200 ms on a modern laptop.
    assert elapsed < 0.2, f"gaussian_blur took {elapsed:.3f}s"


def test_hue_sat_512_budget() -> None:
    from compositor.rendering.pixel_ops import hue_saturation_lightness
    arr = (np.random.rand(512, 512, 4) * 255).astype(np.uint8)
    arr[..., 3] = 255
    elapsed = _time(lambda: hue_saturation_lightness(arr, hue_shift=30.0))
    assert elapsed < 0.15, f"hue_sat took {elapsed:.3f}s"


def test_composite_4layers_512_budget() -> None:
    from compositor.rendering.layer_renderer import composite_document, CompositorOptions
    from compositor.document.editor_session import EditorSession
    from compositor.document.layers import ImageLayer, Transform, new_id
    rng = np.random.default_rng(0)
    cache = {}
    session = EditorSession(width=512, height=512)
    for _ in range(4):
        lid = new_id()
        cache[lid] = (rng.random((512, 512, 4)) * 255).astype(np.uint8)
        cache[lid][..., 3] = 255
        session.add_layer(ImageLayer(id=lid, name="L", transform=Transform(size=(512, 512)), image_file=f"{lid}.png"))
    opts = CompositorOptions(canvas_width=512, canvas_height=512, image_loader=cache.get)
    elapsed = _time(lambda: composite_document(session, opts))
    assert elapsed < 0.4, f"composite took {elapsed:.3f}s"
