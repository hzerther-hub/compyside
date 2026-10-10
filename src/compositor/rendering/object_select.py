"""Object Select — rectangle + GrabCut (or SAM) foreground extraction.

Workflow:
  1. User drags a rectangle (or clicks a point) on the canvas.
  2. ``method="grabcut"`` (default) runs cv2 GrabCut with that rectangle.
  3. ``method="sam"`` (when ``segment-anything`` is installed) uses Meta
     Segment Anything with the rectangle's centre as the prompt point.
  4. ``method="rembg"`` uses U²-Net for a click anywhere on the subject.

The returned mask becomes the active selection. Each algorithm needs a
different setup cost: GrabCut is bundled; rembg is a one-line
``pip install rembg``; SAM is a 350 MB checkpoint download.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def object_select(
    image: np.ndarray,
    rect: Tuple[int, int, int, int],
    *,
    iterations: int = 5,
    method: str = "auto",
    sam_checkpoint: Optional[str] = None,
) -> np.ndarray:
    """Return a uint8 mask (255 = foreground, 0 = background).

    ``rect = (x0, y0, x1, y1)`` is the seed rectangle in document pixels.
    ``method`` selects the algorithm; ``"auto"`` picks the first
    available backend in priority order ``sam`` → ``rembg`` → ``grabcut``.
    """
    if image.dtype != np.uint8:
        raise TypeError("expected uint8 image")
    h, w = image.shape[:2]
    x0, y0, x1, y1 = rect
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), dtype=np.uint8)

    chosen = method
    if chosen == "auto":
        from . import ai_backends
        if ai_backends.is_sam_available():
            chosen = "sam"
        elif ai_backends.is_rembg_available():
            chosen = "rembg"
        else:
            chosen = "grabcut"

    if chosen == "sam":
        from . import ai_backends
        cx = int(round((x0 + x1) / 2))
        cy = int(round((y0 + y1) / 2))
        return ai_backends.sam_segment(image, (cx, cy), checkpoint=sam_checkpoint)

    if chosen == "rembg":
        from . import ai_backends
        rgb = image[..., :3] if image.ndim == 3 and image.shape[-1] == 4 else image
        rgba = ai_backends.rembg_remove_background(rgb)
        if rgba.ndim == 3 and rgba.shape[-1] == 4:
            return rgba[..., 3]
        return np.zeros((h, w), dtype=np.uint8)

    if chosen == "grabcut":
        if not _HAS_CV2:
            raise RuntimeError(
                "OpenCV (cv2) is required for GrabCut object select. "
                "Install via: pip install opencv-python"
            )
        if image.ndim == 2:
            work = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            work = image[..., :3] if image.shape[-1] == 4 else image
        mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            work,
            mask,
            (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
        return np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    raise ValueError(f"unknown method: {chosen!r}")
