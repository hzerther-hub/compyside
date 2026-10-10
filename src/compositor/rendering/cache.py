"""Mipmap cache for layer compositing.

The full-size composite can be expensive on 4K canvases; the canvas
only ever needs a particular mip level for a given zoom. This module
keeps a small LRU cache of pre-rendered tiles at fixed power-of-two
scales so re-paints at the same zoom are O(1).

Today the cache is a single flat mipmap per document. A future iteration
will split the canvas into 256×256 tiles (matching the Swift app's
``TiledLayerRenderer``).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np


def _downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Halve the H/W of ``image`` by ``factor`` using area-averaging."""
    h, w = image.shape[:2]
    nh = max(1, h // factor)
    nw = max(1, w // factor)
    try:
        import cv2  # type: ignore
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    except ImportError:
        # Pick every Nth pixel — adequate for low-res previews.
        ys = np.linspace(0, h - 1, nh).astype(np.int32)
        xs = np.linspace(0, w - 1, nw).astype(np.int32)
        return image[ys][:, xs]


class MipmapCache:
    """LRU cache of mip levels for a single composite.

    Stores the most recently used ``max_entries`` mip levels, keyed by
    the layer-id set + zoom factor. A composite is invalidated by calling
    ``invalidate_all()`` whenever a layer's pixels or transform change.
    """

    MAX_LEVEL = 4  # 1×, 1/2, 1/4, 1/8

    def __init__(self, max_entries: int = 16):
        self._cache: "OrderedDict[Tuple, np.ndarray]" = OrderedDict()
        self._max_entries = max_entries

    def _key(self, layer_signature: Tuple[Tuple[str, int], ...], level: int) -> Tuple:
        return ("mip", level, layer_signature)

    def get(self, layer_signature: Tuple[Tuple[str, int], ...], level: int) -> Optional[np.ndarray]:
        key = self._key(layer_signature, level)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, layer_signature: Tuple[Tuple[str, int], ...], level: int, image: np.ndarray) -> None:
        key = self._key(layer_signature, level)
        self._cache[key] = image
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def invalidate_all(self) -> None:
        self._cache.clear()

    @staticmethod
    def layer_signature(layer_id_to_version: dict) -> Tuple[Tuple[str, int], ...]:
        """Stable, hashable signature of which layer versions contribute."""
        return tuple(sorted(layer_id_to_version.items()))


def mip_factor_for_zoom(zoom: float) -> int:
    """Return the integer mip factor (1, 2, 4 or 8) for a given zoom level.

    At zoom ≥ 0.5 the user sees the canvas at half-resolution or better, so
    the full-resolution composite is appropriate. Below that we serve a
    pre-downsampled mip.
    """
    if zoom >= 0.5:
        return 1
    if zoom >= 0.25:
        return 2
    if zoom >= 0.125:
        return 4
    return 8


def mip_image(full: np.ndarray, factor: int) -> np.ndarray:
    """Return a pre-downsampled copy at the given factor (clamped to [1, 8])."""
    factor = max(1, min(MipmapCache.MAX_LEVEL, factor))
    if factor == 1:
        return full
    return _downsample(full, factor)
