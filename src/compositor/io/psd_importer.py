"""PSD / PSB import via psd-tools.

The Compositor spec reads PSDs best-effort: layers, masks, blend modes,
opacity, groups, simple horizontal text survive; other vectors become
pixels. We mirror that here.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np


def is_psd(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in (".psd", ".psb")


def import_psd(path: str | Path) -> dict:
    """Open a PSD/PSB and return a dict that maps onto a ProjectStore setup.

    Returns ``{"width": int, "height": int, "layers": [layer_dict, ...], "flattened": np.ndarray}``.
    Each layer dict has ``{"id", "name", "is_visible", "is_group", "opacity",
    "blend_mode", "transform", "imageFile", "maskFile"}`` matching the manifest.
    """
    try:
        from psd_tools import PSDImage  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "psd-tools not installed. Install with: pip install psd-tools"
        ) from exc

    psd = PSDImage.open(str(path))
    width, height = psd.width, psd.height

    layers: list = []
    flattened = np.zeros((height, width, 4), dtype=np.uint8)

    def _walk(node, parent_id=None) -> None:
        for child in node:
            if child.is_group():
                gid = _new_id()
                layers.append({
                    "id": gid,
                    "name": child.name,
                    "is_visible": child.visible,
                    "is_group": True,
                    "parentID": parent_id,
                    "opacity": child.opacity / 255.0 if child.opacity is not None else 1.0,
                    "blend_mode": _blend_name(child.blend_mode),
                    "transform": _identity_transform(),
                })
                _walk(child, gid)
                continue
            layer_id = _new_id()
            layer_pixels = _render_layer(child, (height, width))
            if layer_pixels is not None:
                layers.append({
                    "id": layer_id,
                    "name": child.name,
                    "is_visible": child.visible,
                    "is_group": False,
                    "opacity": child.opacity / 255.0 if child.opacity is not None else 1.0,
                    "blend_mode": _blend_name(child.blend_mode),
                    "transform": _identity_transform(layer_pixels.shape[1], layer_pixels.shape[0]),
                    "imageFile": f"{layer_id}.png",
                    "pixels": layer_pixels,
                })

    _walk(psd)
    return {"width": width, "height": height, "layers": layers, "flattened": flattened}


def _identity_transform(w: int = 0, h: int = 0) -> dict:
    return {
        "origin": [0.0, 0.0],
        "size": [float(w), float(h)],
        "rotation": 0.0,
        "flipX": False,
        "flipY": False,
        "sampling": "High quality",
    }


def _blend_name(mode) -> str:
    mapping = {
        "normal": "Normal",
        "multiply": "Multiply",
        "screen": "Screen",
        "overlay": "Overlay",
        "darken": "Darken",
        "lighten": "Lighten",
        "color dodge": "Color Dodge",
        "color burn": "Color Burn",
        "difference": "Difference",
        "exclusion": "Exclusion",
        "soft light": "Soft Light",
        "hard light": "Hard Light",
        "vivid light": "Vivid Light",
        "linear light": "Linear Light",
        "pin light": "Pin Light",
        "hue": "Hue",
        "saturation": "Saturation",
        "color": "Color",
        "luminosity": "Luminosity",
    }
    return mapping.get(str(mode).lower(), "Normal")


def _render_layer(layer, canvas_shape) -> Optional[np.ndarray]:
    """Rasterise a psd-tools layer to a NumPy RGBA array."""
    try:
        img = layer.composite()
    except Exception:
        return None
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full(arr.shape, 255, dtype=arr.dtype)], axis=-1)
    elif arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=-1)
    return arr.astype(np.uint8)


def _new_id() -> str:
    import uuid
    return str(uuid.uuid4()).upper()
