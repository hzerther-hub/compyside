"""Mipmap cache tests."""
from __future__ import annotations

import numpy as np

from compositor.rendering.cache import (
    MipmapCache,
    mip_factor_for_zoom,
    mip_image,
)


def test_mip_factor_thresholds() -> None:
    assert mip_factor_for_zoom(1.0) == 1
    assert mip_factor_for_zoom(0.5) == 1
    assert mip_factor_for_zoom(0.4) == 2
    assert mip_factor_for_zoom(0.2) == 4
    assert mip_factor_for_zoom(0.1) == 8


def test_mip_image_halves_dimensions() -> None:
    arr = np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)
    half = mip_image(arr, 2)
    assert half.shape == (32, 32, 4)
    quarter = mip_image(arr, 4)
    assert quarter.shape == (16, 16, 4)


def test_mip_image_factor_one_is_identity() -> None:
    arr = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)
    out = mip_image(arr, 1)
    np.testing.assert_array_equal(out, arr)


def test_mip_cache_put_get_roundtrip() -> None:
    cache = MipmapCache(max_entries=4)
    sig = (("layer-1", 1), ("layer-2", 2))
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    cache.put(sig, 2, arr)
    out = cache.get(sig, 2)
    assert out is not None
    np.testing.assert_array_equal(out, arr)


def test_mip_cache_eviction() -> None:
    cache = MipmapCache(max_entries=2)
    cache.put((("a", 1),), 1, np.zeros((4, 4, 4), dtype=np.uint8))
    cache.put((("b", 1),), 1, np.zeros((4, 4, 4), dtype=np.uint8))
    cache.put((("c", 1),), 1, np.zeros((4, 4, 4), dtype=np.uint8))  # evicts "a"
    assert cache.get((("a", 1),), 1) is None
    assert cache.get((("b", 1),), 1) is not None
    assert cache.get((("c", 1),), 1) is not None


def test_mip_cache_invalidate() -> None:
    cache = MipmapCache()
    cache.put((("a", 1),), 2, np.zeros((4, 4, 4), dtype=np.uint8))
    cache.invalidate_all()
    assert cache.get((("a", 1),), 2) is None


def test_layer_signature_stable() -> None:
    sig_a = MipmapCache.layer_signature({"a": 1, "b": 2})
    sig_b = MipmapCache.layer_signature({"b": 2, "a": 1})
    assert sig_a == sig_b
