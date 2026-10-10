"""PSD / PSB import via psd-tools.

The Compositor spec reads PSDs best-effort. What survives:

* groups (including nesting), per-layer visibility, opacity and blend mode;
* pixel layers, positioned by their real canvas offset;
* user masks, re-expressed as Compositor mask PNGs (white = visible);
* the adjustment layers Compositor has a matching kind for;
* the document's flattened composite, as a preview.

What does not: vector text (rasterised), smart objects and shape layers
(rasterised), clipping groups (flattened), and adjustment kinds Compositor
does not implement. Every one of those is recorded in the ``report`` list so
the UI can tell the user what changed instead of silently degrading.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# psd-tools BlendMode member name -> the exact spelling Compositor's manifest
# uses. Keyed by the member name rather than its 4-byte value because the
# value is a bytes literal and ``str()`` on it round-trips badly.
_BLEND_MODES = {
    "PASS_THROUGH": "Normal",
    "NORMAL": "Normal",
    "DISSOLVE": "Normal",
    "DARKEN": "Darken",
    "MULTIPLY": "Multiply",
    "COLOR_BURN": "Color Burn",
    "LINEAR_BURN": "Linear Burn",
    "DARKER_COLOR": "Darken",
    "LIGHTEN": "Lighten",
    "SCREEN": "Screen",
    "COLOR_DODGE": "Color Dodge",
    "LINEAR_DODGE": "Linear Dodge (Add)",
    "LIGHTER_COLOR": "Lighten",
    "OVERLAY": "Overlay",
    "SOFT_LIGHT": "Soft Light",
    "HARD_LIGHT": "Hard Light",
    "VIVID_LIGHT": "Vivid Light",
    "LINEAR_LIGHT": "Linear Light",
    "PIN_LIGHT": "Pin Light",
    "HARD_MIX": "Hard Mix",
    "DIFFERENCE": "Difference",
    "EXCLUSION": "Exclusion",
    "SUBTRACT": "Subtract",
    "DIVIDE": "Divide",
    "HUE": "Hue",
    "SATURATION": "Saturation",
    "COLOR": "Color",
    "LUMINOSITY": "Luminosity",
}

# psd-tools adjustment class name -> (Compositor kind, settings builder).
# Anything not listed here is rasterised or dropped with a report entry.
_ADJUSTMENT_KINDS = {
    "HueSaturation": "Hue/Saturation",
    "LevelRecord": "Levels",
    "Curves": "Curves",
    "Exposure": "Exposure",
    "GradientMap": "Gradient Map",
    "Invert": "Invert",
    "BlackAndWhite": "Black & White",
    "ColorBalance": "Color Balance",
}

# Layer kinds that carry no pixel data of their own and get rasterised.
_RASTERISED_KINDS = {
    "type": "text layer rasterised",
    "smartobject": "smart object rasterised",
    "shape": "shape layer rasterised",
}


def is_psd(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in (".psd", ".psb")


def import_psd(path: str | Path) -> Dict[str, Any]:
    """Open a PSD/PSB and return a dict that maps onto a ProjectStore setup.

    Returns ``{"width", "height", "layers", "flattened", "report", "pixels",
    "masks"}``. ``layers`` are manifest-ready layer dicts; ``pixels`` and
    ``masks`` map layer id -> uint8 array and must be written to the project
    store separately, since the manifest only names the files.
    """
    try:
        from psd_tools import PSDImage  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "psd-tools not installed. Install with: pip install psd-tools"
        ) from exc

    psd = PSDImage.open(str(path))
    width, height = int(psd.width), int(psd.height)

    report: List[Dict[str, str]] = []
    layers: List[Dict[str, Any]] = []
    pixel_maps: Dict[str, np.ndarray] = {}
    mask_maps: Dict[str, np.ndarray] = {}

    def _walk(node, parent_id: Optional[str]) -> None:
        # Compositor stores layers bottom-to-top; psd-tools iterates the same
        # way, so a straight append keeps the ordering.
        for child in node:
            layer_id = _new_id()
            # Manifest keys are camelCase — the same spelling the schema and
            # ``Layer.from_dict`` use.
            common = {
                "id": layer_id,
                "name": child.name or "Layer",
                "isVisible": bool(child.visible),
                "opacity": _opacity(child.opacity),
                "blendMode": _blend_name(child),
            }
            if parent_id is not None:
                common["parentID"] = parent_id

            if child.is_group():
                if _blend_member(child) != "NORMAL":
                    report.append(_note(child, "group blend mode flattened to Normal"))
                layers.append(dict(common, isGroup=True, transform=_identity_transform()))
                _walk(child, layer_id)
                continue

            pending_mask = None
            if child.has_mask():
                pending_mask = _layer_mask(child)
                if pending_mask is not None:
                    common["maskFile"] = f"{layer_id}.mask.png"
                    common["maskEnabled"] = not _mask_disabled(child)
                    if not common["maskEnabled"]:
                        report.append(_note(child, "mask imported but disabled in the PSD"))

            adjustment = _adjustment_entry(child)
            if adjustment is not None:
                kind, settings, note = adjustment
                entry = dict(
                    common,
                    isGroup=False,
                    transform=_identity_transform(width, height),
                    adjustment={"kind": kind, **settings},
                )
                if pending_mask is not None:
                    # Compositor's renderer masks image layers but not
                    # adjustment layers, so the file would sit there unused.
                    entry.pop("maskFile", None)
                    entry.pop("maskEnabled", None)
                    report.append(
                        _note(child, "mask dropped — Compositor cannot mask an adjustment layer yet")
                    )
                layers.append(entry)
                report.append(_note(child, note))
                continue

            pixels = _render_pixels(child, (height, width))
            if pixels is None:
                report.append(_note(child, "layer had no renderable pixels and was skipped"))
                continue

            kind_note = _RASTERISED_KINDS.get(child.kind)
            if kind_note:
                report.append(_note(child, kind_note))

            ph, pw = pixels.shape[:2]
            layers.append(
                dict(
                    common,
                    isGroup=False,
                    transform=_transform(child, pw, ph),
                    imageFile=f"{layer_id}.png",
                )
            )
            pixel_maps[layer_id] = pixels
            if pending_mask is not None:
                mask_maps[layer_id] = pending_mask

    _walk(psd, None)

    # A PSD with nothing importable is a failure, not an empty document.
    if not layers:
        raise RuntimeError("no importable layers found in this PSD")

    return {
        "width": width,
        "height": height,
        "layers": layers,
        "flattened": _flattened(psd, (height, width)),
        "report": report,
        "pixels": pixel_maps,
        "masks": mask_maps,
    }


# ---------------------------------------------------------------------------
# Layer helpers
# ---------------------------------------------------------------------------


def _identity_transform(w: int = 0, h: int = 0) -> Dict[str, Any]:
    return {
        "origin": [0.0, 0.0],
        "size": [float(w), float(h)],
        "rotation": 0.0,
        "flipX": False,
        "flipY": False,
        "sampling": "High quality",
    }


def _transform(layer, w: int, h: int) -> Dict[str, Any]:
    """Place a layer at its real canvas offset."""
    try:
        left, top = int(layer.offset[0]), int(layer.offset[1])
    except Exception:  # noqa: BLE001 - psd-tools raises on degenerate records
        left, top = 0, 0
    out = _identity_transform(w, h)
    out["origin"] = [float(left), float(top)]
    return out


def _opacity(value) -> float:
    if value is None:
        return 1.0
    return max(0.0, min(1.0, float(value) / 255.0))


def _blend_member(layer) -> str:
    """The psd-tools BlendMode member name, or ``"NORMAL"`` when absent."""
    mode = getattr(layer, "blend_mode", None)
    name = getattr(mode, "name", None)
    if isinstance(name, str) and name:
        return name
    return "NORMAL"


def _blend_name(layer) -> str:
    return _BLEND_MODES.get(_blend_member(layer), "Normal")


def _is_adjustment(layer) -> bool:
    try:
        from psd_tools.api.adjustments import AdjustmentLayer  # type: ignore

        return isinstance(layer, AdjustmentLayer)
    except Exception:  # noqa: BLE001
        return False


def _adjustment_entry(layer) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """Map a psd-tools adjustment layer onto a Compositor adjustment layer.

    Returns ``(kind, settings, report_note)`` or ``None`` when the layer
    should be rasterised instead.
    """
    if not _is_adjustment(layer):
        return None
    kind = _ADJUSTMENT_KINDS.get(type(layer).__name__)
    if kind is None:
        return None
    settings = _adjustment_settings(type(layer).__name__, layer)
    note = f"adjustment layer imported as {kind}"
    if not settings:
        note += " (parameters reset to defaults)"
    return kind, settings, note


def _adjustment_settings(name: str, layer) -> Dict[str, Any]:
    """Carry across the parameters Compositor models the same way."""
    try:
        if name == "HueSaturation":
            # Photoshop stores hue in degrees and the other two as percentages,
            # which is exactly what Compositor's adjustment expects.
            return {
                "hue": float(getattr(layer, "hue", 0.0)),
                "saturation": float(getattr(layer, "saturation", 0.0)),
                "lightness": float(getattr(layer, "lightness", 0.0)),
                "colorize": bool(getattr(layer, "colorize", False)),
            }
        if name == "Exposure":
            # Compositor models exposure as stops only; Photoshop's separate
            # black-point offset has no counterpart.
            return {"exposure": float(getattr(layer, "exposure", 0.0))}
    except Exception:  # noqa: BLE001 - descriptor shapes vary by PSD version
        return {}
    return {}


def _mask_disabled(layer) -> bool:
    try:
        return bool(layer.mask.disabled)
    except Exception:  # noqa: BLE001
        return False


def _layer_mask(layer) -> Optional[np.ndarray]:
    """Render the user mask as a Compositor mask: layer-sized, white = visible.

    PSD masks are white-to-hide and usually cover only part of the layer's
    bbox; everything outside the mask bbox takes the mask's background
    colour, and Compositor wants the opposite polarity, so both are flipped
    here rather than at draw time.
    """
    try:
        pil = layer.mask.topil()
    except Exception:  # noqa: BLE001 - a broken mask must not kill the import
        return None
    if pil is None:
        return None

    left, top, right, bottom = (int(v) for v in layer.bbox)
    layer_w, layer_h = right - left, bottom - top
    mask_w, mask_h = int(pil.width), int(pil.height)
    if mask_w <= 0 or mask_h <= 0 or layer_w <= 0 or layer_h <= 0:
        return None

    # Compositor's polarity: 255 = fully visible.
    background = 0 if _mask_background_is_white(layer) else 255
    out = np.full((layer_h, layer_w), background, dtype=np.uint8)

    values = np.asarray(pil.convert("L"), dtype=np.uint8)
    if values.shape[:2] != (mask_h, mask_w):  # pragma: no cover - defensive
        return None

    # The mask rectangle is in document coordinates; convert it to an offset
    # inside the layer so it lines up with the layer's pixels.
    mask_left, mask_top = int(layer.mask.left), int(layer.mask.top)
    off_x, off_y = mask_left - left, mask_top - top
    x0, y0 = max(0, off_x), max(0, off_y)
    x1 = min(mask_w, off_x + mask_w)
    y1 = min(mask_h, off_y + mask_h)
    if x1 > x0 and y1 > y0:
        region = values[y0:y1, x0:x1]
        out[off_y + y0 : off_y + y1, off_x + x0 : off_x + x1] = 255 - region
    return out


def _mask_background_is_white(layer) -> bool:
    try:
        return int(layer.mask.background_color) == 255
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_pixels(layer, canvas_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Rasterise a layer to RGBA uint8 at its own bbox size.

    Prefers the raw pixel data so the user mask stays editable; falls back to
    a forced composite for shape, text and smart-object layers, which have no
    raw pixels of their own.
    """
    arr = None
    try:
        arr = layer.numpy()
    except Exception:  # noqa: BLE001
        arr = None
    if arr is not None:
        arr = _to_rgba_uint8(arr)
        if arr is not None and arr.size:
            return arr

    for kwargs in ({"force": True}, {}):
        try:
            img = layer.composite(**kwargs)
        except Exception:  # noqa: BLE001
            continue
        if img is None:
            continue
        arr = _to_rgba_uint8(np.asarray(img))
        if arr is not None and arr.size:
            return arr
    return None


def _to_rgba_uint8(arr: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            arr = np.clip(arr * 255.0, 0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None].repeat(3, axis=-1)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 1:
        arr = arr.repeat(3, axis=-1)
    if arr.shape[-1] == 2:  # grey + alpha
        arr = np.concatenate([arr[..., :1].repeat(3, axis=-1), arr[..., 1:]], axis=-1)
    if arr.shape[-1] == 3:
        arr = np.concatenate(
            [arr, np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1
        )
    return arr if arr.shape[-1] == 4 else None


def _flattened(psd, canvas_shape: Tuple[int, int]) -> np.ndarray:
    """The document's own flattened composite, transparent on failure."""
    height, width = canvas_shape
    for getter in (lambda: psd.composite(), lambda: psd.topil()):
        try:
            arr = getter()
        except Exception:  # noqa: BLE001
            continue
        if arr is None:
            continue
        arr = _to_rgba_uint8(np.asarray(arr))
        if arr is not None and arr.size:
            return arr
    return np.zeros((height, width, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def _note(layer, message: str) -> Dict[str, str]:
    return {"name": layer.name or "Layer", "kind": str(getattr(layer, "kind", "?")), "message": message}


def _new_id() -> str:
    return str(uuid.uuid4()).upper()
