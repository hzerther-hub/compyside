"""Pixel-operation tests."""
from __future__ import annotations

import numpy as np

from compositor.rendering.pixel_ops import (
    ChannelRange,
    add_noise,
    bayer_dither,
    bloom,
    brightness_contrast,
    content_aware_fill,
    exposure,
    gaussian_blur,
    gaussian_kernel_1d,
    hue_saturation_lightness,
    invert,
    lens_correct,
    levels_channel,
    posterise,
    remove_background,
    scanlines,
    tonal_contrast,
    vignette,
)


def test_levels_identity() -> None:
    arr = np.arange(256, dtype=np.uint8)
    out = levels_channel(arr, ChannelRange())
    np.testing.assert_array_equal(out, arr)


def test_levels_clamp() -> None:
    arr = np.full((1,), 100, dtype=np.uint8)
    out = levels_channel(arr, ChannelRange(black=200.0, white=200.0, output_black=0, output_white=255))
    assert out[0] == 0


def test_invert_preserves_alpha() -> None:
    arr = np.array([[[10, 20, 30, 255]]], dtype=np.uint8)
    out = invert(arr)
    np.testing.assert_array_equal(out, [[[245, 235, 225, 255]]])


def test_exposure_positive_stops_brighten() -> None:
    arr = np.full((2, 2, 4), 128, dtype=np.uint8)
    arr[..., 3] = 255
    out = exposure(arr, stops=1.0)
    # 128 * 2 ≈ 256 → clamps to 255
    assert (out[..., :3] == 255).all()


def test_brightness_contrast() -> None:
    arr = np.full((1, 1, 4), 127, dtype=np.uint8)
    arr[..., 3] = 255
    out = brightness_contrast(arr, brightness=0.0, contrast=100.0)
    # Contrast +100 stretches the gap; 127 maps to 127.
    assert abs(int(out[0, 0, 0]) - 127) <= 1


def test_posterise_reduces_levels() -> None:
    arr = np.linspace(0, 255, 256, dtype=np.uint8).reshape(1, 256, 1).repeat(4, axis=-1)
    out = posterise(arr, levels=2)
    # 2 levels collapses to floor(v*2)/2 then *255 → {0, 127, 255}.
    # uint8 cast rounds 127.5 down to 127 (floor semantics).
    values = set(out[..., 0].flatten().tolist())
    assert values == {0, 127, 255}


def test_gaussian_kernel_normalised() -> None:
    k = gaussian_kernel_1d(sigma=2.0)
    np.testing.assert_allclose(k.sum(), 1.0, atol=1e-6)


def test_gaussian_blur_smooths() -> None:
    # Sharp edge: left half 0, right half 255.
    img = np.zeros((1, 16, 4), dtype=np.uint8)
    img[:, 8:, :] = 255
    img[..., 3] = 255
    out = gaussian_blur(img, sigma=2.0)
    # Centre pixel should be near midpoint.
    assert 100 < out[0, 8, 0] < 200


def test_hue_shift_is_idempotent_for_zero() -> None:
    arr = np.array([[[100, 150, 200, 255]]], dtype=np.uint8)
    out = hue_saturation_lightness(arr)
    np.testing.assert_array_equal(out, arr)


def test_hue_shift_changes_color() -> None:
    arr = np.array([[[200, 50, 50, 255]]], dtype=np.uint8)  # red
    out = hue_saturation_lightness(arr, hue_shift=120.0)  # rotate to green
    # After a 120° hue shift in HSV, red (h=0) becomes green.
    g, r, b = out[0, 0, 1], out[0, 0, 0], out[0, 0, 2]
    assert g > r and g > b


def test_add_noise_changes_pixels() -> None:
    arr = np.full((8, 8, 4), 128, dtype=np.uint8)
    arr[..., 3] = 255
    out = add_noise(arr, amount=0.5, seed=0)
    assert not np.array_equal(out[..., :3], arr[..., :3])


def test_vignette_zero_amount_is_identity() -> None:
    arr = np.full((16, 16, 4), 200, dtype=np.uint8)
    arr[..., 3] = 255
    out = vignette(arr, amount=0.0)
    np.testing.assert_array_equal(out, arr)


def test_vignette_darkens_corners() -> None:
    arr = np.full((32, 32, 4), 255, dtype=np.uint8)
    arr[..., 3] = 255
    out = vignette(arr, amount=0.9, midpoint=0.0, feather=1.0)
    # Centre should remain bright; corners should be darker.
    assert out[16, 16, 0] > out[0, 0, 0]
    assert out[16, 16, 0] >= 250


def test_bloom_highlights_add_back() -> None:
    # Half the image is bright, the other half dark.
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[:, 4:, :3] = 255
    arr[..., 3] = 255
    out = bloom(arr, threshold=0.5, intensity=1.0, radius=1.0)
    # Dark side should now have some signal from the bleed.
    assert out[:, :4, :3].max() > 0


def test_bayer_dither_quantises_levels() -> None:
    arr = np.linspace(0, 255, 256, dtype=np.uint8).reshape(1, 256, 1).repeat(4, axis=-1)
    out = bayer_dither(arr, levels=4, strength=1.0)
    values = sorted(set(out[..., 0].flatten().tolist()))
    assert values == [0, 85, 170, 255]


def test_scanlines_darken_every_nth_row() -> None:
    arr = np.full((8, 4, 4), 200, dtype=np.uint8)
    arr[..., 3] = 255
    out = scanlines(arr, spacing=2, intensity=1.0)
    # Odd rows should be ~0; even rows untouched.
    assert out[0, 0, 0] == 200
    assert out[1, 0, 0] == 0
    assert out[2, 0, 0] == 200


def test_tonal_contrast_zero_amount_is_identity() -> None:
    arr = np.full((16, 16, 4), 128, dtype=np.uint8)
    arr[..., 3] = 255
    out = tonal_contrast(arr, amount=0.0, radius=3.0)
    np.testing.assert_array_equal(out, arr)


def test_tonal_contrast_amplifies_edges() -> None:
    # Build a synthetic edge and verify the difference across it grows.
    arr = np.full((1, 16, 4), 128, dtype=np.uint8)
    arr[:, 8:, :3] = 200
    arr[..., 3] = 255
    out = tonal_contrast(arr, amount=1.0, radius=1.0)
    diff_before = abs(int(arr[0, 8, 0]) - int(arr[0, 9, 0]))
    diff_after = abs(int(out[0, 8, 0]) - int(out[0, 9, 0]))
    assert diff_after >= diff_before


def test_lens_correct_zero_is_identity() -> None:
    arr = np.random.randint(0, 256, size=(32, 32, 4), dtype=np.uint8)
    out = lens_correct(arr, strength=0.0)
    np.testing.assert_array_equal(out, arr)


def test_lens_correct_barrel_shifts_pixels() -> None:
    # A 64x64 image with a single bright pixel at (32, 32); barrel distortion
    # should pull that pixel outward, leaving the centre untouched but
    # touching other pixels.
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[..., 3] = 255
    arr[32, 32, 0] = 255
    out = lens_correct(arr, strength=0.3)
    # The centre pixel still exists; the bright pixel's energy should spread.
    assert out[..., 0].sum() >= 255


def test_remove_background_returns_rgba() -> None:
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[:, :, :] = [200, 50, 50]  # red background
    arr[4:12, 4:12, :] = [50, 200, 50]  # green foreground
    out = remove_background(arr, margin=0.2)
    assert out.shape[-1] == 4
    # Foreground should remain mostly opaque.
    assert out[8, 8, 3] > 128


def test_content_aware_fill_zero_mask_is_identity() -> None:
    arr = np.full((16, 16, 4), 128, dtype=np.uint8)
    arr[..., 3] = 255
    mask = np.zeros((16, 16), dtype=np.uint8)
    out = content_aware_fill(arr, mask)
    np.testing.assert_array_equal(out, arr)


def test_content_aware_fill_runs() -> None:
    arr = np.full((16, 16, 4), 100, dtype=np.uint8)
    arr[..., 3] = 255
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[6:10, 6:10] = 255
    out = content_aware_fill(arr, mask, radius=3)
    assert out.shape == arr.shape
    # Filled region should differ from the input (inpainted from neighbours).
    assert not np.array_equal(out[7:9, 7:9, 0], arr[7:9, 7:9, 0])
