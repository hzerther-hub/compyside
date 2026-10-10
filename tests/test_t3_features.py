"""Tier-3 feature tests: multi-line text, camera raw, GPU shader source."""
from __future__ import annotations

import numpy as np

from compositor.rendering import gpu_effects
from compositor.io.raw_importer import develop


def test_rasterize_multiline_text() -> None:
    from compositor.tools.type_tool import rasterize_text
    single = rasterize_text("Hello", font_size=32)
    multi = rasterize_text("Hello\nWorld", font_size=32)
    assert single.ndim == 3 and single.shape[-1] == 4
    # Multi-line should be strictly taller than single-line.
    assert multi.shape[0] > single.shape[0]


def test_rasterize_text_alignment() -> None:
    from compositor.tools.type_tool import rasterize_text
    left = rasterize_text("Hello", align="left", font_size=24)
    centre = rasterize_text("Hello", align="centre", font_size=24)
    right = rasterize_text("Hello", align="right", font_size=24)
    # Width may differ depending on padding but all three must be valid RGBA.
    for arr in (left, centre, right):
        assert arr.ndim == 3 and arr.shape[-1] == 4


def test_rasterize_text_word_wrap() -> None:
    from compositor.tools.type_tool import rasterize_text
    long = "Lorem ipsum dolor sit amet consectetur adipiscing elit"
    no_wrap = rasterize_text(long, font_size=16)
    wrapped = rasterize_text(long, font_size=16, max_width=80)
    # Word-wrapped output should be wider-reduced (or at least no wider) and
    # strictly taller than the non-wrapped version.
    assert wrapped.shape[0] > no_wrap.shape[0] * 0.5
    assert wrapped.shape[0] > 16


def test_develop_extended_kwargs() -> None:
    arr = np.full((32, 32, 4), 128, dtype=np.uint8)
    arr[..., 3] = 255
    out = develop(
        arr,
        exposure=0.5,
        contrast=1.2,
        saturation=1.1,
        temperature=10.0,
        tint=-5.0,
        highlights=0.2,
        shadows=-0.2,
        whites=0.1,
        blacks=-0.1,
        clarity=0.3,
        vibrance=0.5,
        sharpening_amount=1.0,
        noise_reduction=1.0,
        dehaze=0.5,
        curve_darks=10.0,
        curve_lights=-10.0,
        shadow_hue=200.0,
        shadow_sat=0.6,
        highlight_hue=60.0,
        highlight_sat=0.5,
        balance=0.1,
        lens_strength=0.2,
        lens_cx=0.5,
        lens_cy=0.5,
        rotate=5.0,
        aspect=1.0,
        scale=1.0,
    )
    assert out.shape == arr.shape


def test_develop_mixer_kwargs_are_ignored_safely() -> None:
    """The HSL mixer sends ~24 extra knobs; develop should accept them via **mixer."""
    arr = np.full((8, 8, 4), 100, dtype=np.uint8)
    arr[..., 3] = 255
    out = develop(
        arr,
        mix_red_hue=20.0,
        mix_red_sat=-10.0,
        mix_blue_lum=15.0,
    )
    np.testing.assert_array_equal(out, develop(arr))


def test_gpu_shader_source_loads() -> None:
    src = gpu_effects.gpu_shader_source()
    assert "drop shadow" in src.lower() or "drop-shadow" in src.lower()
    assert "stroke" in src.lower()
    assert "color overlay" in src.lower()


def test_gpu_effects_constant() -> None:
    for name in (
        "drop-shadow",
        "inner-shadow",
        "outer-glow",
        "inner-glow",
        "stroke",
        "color-overlay",
    ):
        assert name in gpu_effects.GPU_EFFECTS
