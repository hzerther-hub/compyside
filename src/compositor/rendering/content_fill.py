"""Content-Aware Fill — cv2.inpaint bridge.

``cv2.inpaint`` does patch-based inpainting: it propagates texture and
gradient information from the area surrounding the mask into the masked
region. Two algorithms are shipped:

- ``telea``: Alexandru Telea's "Fast Marching Method" — fast, good for
  thin masks.
- ``ns``: Bertalmio et al. "Navier-Stokes" — fluid-dynamics based, better
  for larger filled regions.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def content_aware_fill(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    algorithm: Literal["telea", "ns"] = "telea",
    radius: int = 3,
) -> np.ndarray:
    """Fill the masked region of ``image`` in-place.

    ``mask`` is a uint8 HxW array (255 = fill here, 0 = keep).
    ``algorithm`` is one of ``"telea"`` or ``"ns"``.
    """
    if not _HAS_CV2:
        raise RuntimeError(
            "OpenCV (cv2) is required for content-aware fill. "
            "Install via: pip install opencv-python"
        )
    if image.dtype != np.uint8:
        raise TypeError("expected uint8 image")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    flag = cv2.INPAINT_TELEA if algorithm == "telea" else cv2.INPAINT_NS
    alpha = None
    work = image
    if image.ndim == 3:
        if image.shape[-1] == 4:
            alpha = image[..., 3:4]
            work = image[..., :3]
    out = cv2.inpaint(work, mask, radius, flag)
    if alpha is not None:
        out = np.concatenate([out, alpha], axis=-1)
    return out
