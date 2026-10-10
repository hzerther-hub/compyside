"""Layer compositing — flatten a layer stack into a single NumPy canvas.

Pure NumPy. No Qt. Takes an EditorSession and produces an RGBA uint8 image
of the document. Honours transforms (translation/scale/rotation/flip),
opacity, blend modes, layer masks, clipping masks, adjustment layers,
group opacity (v8+), and folder masks (v6+).

This is the workhorse of the canvas — every paint event flows through
``composite_document``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..document.editor_session import EditorSession
from ..document.layers import (
    AdjustmentKind,
    AdjustmentLayer,
    AnyLayer,
    BlendMode,
    GroupLayer,
    ImageLayer,
    Transform,
)
from .adjustments import apply_adjustment
from .blend_modes import apply_blend


# ---------------------------------------------------------------------------
# Affine transforms
# ---------------------------------------------------------------------------


def snap_position(
    x: float,
    y: float,
    *,
    canvas_width: int,
    canvas_height: int,
    other_layers: Sequence,
    guides: Sequence,
    snap_to: str = "all",
    threshold: float = 8.0,
) -> tuple:
    """Snap ``(x, y)`` to the nearest anchor within ``threshold`` pixels.

    Anchors are collected from:
      - canvas edges / midlines (if ``snap_to`` includes ``"canvas"``)
      - every other layer's bounding box (4 corners + 4 edge midpoints + center)
        (if ``snap_to`` includes ``"layers"``)
      - every visible guide (if ``snap_to`` includes ``"guides"``)

    X and Y are snapped independently — a layer can snap to a canvas edge
    on one axis and a layer edge on the other.
    """
    x_candidates = []
    y_candidates = []
    if "canvas" in snap_to or snap_to == "all":
        x_candidates += [0.0, canvas_width / 2.0, float(canvas_width)]
        y_candidates += [0.0, canvas_height / 2.0, float(canvas_height)]
    if ("layers" in snap_to or snap_to == "all") and other_layers:
        for L in other_layers:
            ox, oy = L.transform.origin
            w, h = L.transform.size
            x_candidates += [ox, ox + w / 2.0, ox + w]
            y_candidates += [oy, oy + h / 2.0, oy + h]
    if ("guides" in snap_to or snap_to == "all") and guides:
        for g in guides:
            if not g.is_visible:
                continue
            if g.orientation == "v":
                x_candidates.append(g.position)
            else:
                y_candidates.append(g.position)

    def snap_axis(value: float, candidates: list) -> float:
        if not candidates:
            return value
        nearest = min(candidates, key=lambda c: abs(c - value))
        if abs(nearest - value) <= threshold:
            return float(nearest)
        return value

    return snap_axis(x, x_candidates), snap_axis(y, y_candidates)


def transform_matrix(transform: Transform) -> np.ndarray:
    """3x3 affine that maps *source* pixels to the layer's draw rect.

    For a layer drawn at ``origin`` with ``size`` matching its source pixels
    (the common case), this is identity.

    OpenCV's ``warpAffine`` expects a 2x3 affine that maps destination
    pixels back to source pixels, so the inverse of this matrix is what
    actually gets passed to ``cv2``.
    """
    ox, oy = transform.origin
    w, h = transform.size
    sx_scale = -1.0 if transform.flip_x else 1.0
    sy_scale = -1.0 if transform.flip_y else 1.0
    theta = math.radians(transform.rotation)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Forward map (source → canvas) is rotation → flip → translate.
    # We deliberately do NOT bake ``w`` / ``h`` into the matrix — those are
    # handled by the canvas size, not by the layer transform. ``size`` here
    # only affects where the layer is *anchored*.
    m = np.eye(3)
    m[0, 0] = cos_t * sx_scale
    m[0, 1] = -sin_t * sy_scale
    m[1, 0] = sin_t * sx_scale
    m[1, 1] = cos_t * sy_scale
    m[0, 2] = ox
    m[1, 2] = oy
    return m


def warp_into_canvas(src: np.ndarray, matrix: np.ndarray, canvas_shape: Tuple[int, int]) -> np.ndarray:
    """Warp ``src`` (HxWxC uint8) into a canvas of ``canvas_shape`` using ``matrix``.

    ``matrix`` is the 3x3 forward map (source → canvas). We invert it for
    OpenCV, which expects the destination → source affine.
    """
    h_can, w_can = canvas_shape
    try:
        import cv2  # type: ignore

        # Destination → source affine.
        inv = np.linalg.inv(matrix[:2, :2])
        offset = matrix[:2, 2]
        M = np.zeros((2, 3), dtype=np.float64)
        M[:, :2] = inv
        M[:, 2] = -inv @ offset

        # M maps destination pixels back to source pixels, so it must run
        # with WARP_INVERSE_MAP — without the flag OpenCV inverts it again
        # and every translated layer (origin ≠ 0) renders as transparent.
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        if src.ndim == 3:
            return cv2.warpAffine(
                src,
                M,
                (w_can, h_can),
                flags=flags,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return cv2.warpAffine(
            src,
            M,
            (w_can, h_can),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except ImportError:
        pass
    inv = np.linalg.inv(matrix[:2, :2])
    offset = matrix[:2, 2]
    ys, xs = np.mgrid[0:h_can, 0:w_can].astype(np.float32)
    coords = np.stack([xs.ravel(), ys.ravel()], axis=0)
    src_coords = inv @ coords - inv @ offset[:, None]
    src_x = src_coords[0].reshape(h_can, w_can)
    src_y = src_coords[1].reshape(h_can, w_can)
    return _sample(src, src_x, src_y).astype(np.uint8)


def _sample(src: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``src`` at float coordinates (x, y)."""
    h, w = src.shape[:2]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (x - x0).astype(np.float32)[..., None]
    fy = (y - y0).astype(np.float32)[..., None]
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    if src.ndim == 3:
        a = src[y0c, x0c]
        b = src[y0c, x1c]
        c = src[y1c, x0c]
        d = src[y1c, x1c]
    else:
        a = src[y0c, x0c][..., None]
        b = src[y0c, x1c][..., None]
        c = src[y1c, x0c][..., None]
        d = src[y1c, x1c][..., None]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy)


def warp_into_canvas_corners(
    src: np.ndarray,
    canvas_shape: Tuple[int, int],
    corners: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Bilinearly warp the rectangular ``src`` onto the canvas at four corners.

    ``corners`` is (TL, TR, BR, BL) in canvas coordinates. Areas outside the
    convex hull of the four corners are filled with zeros. This implements
    Photoshop-style free distort.
    """
    h_can, w_can = canvas_shape
    if len(corners) != 4:
        raise ValueError("expected 4 corners (TL, TR, BR, BL)")
    tl, tr, br, bl = (np.array(c, dtype=np.float32) for c in corners)
    h_src, w_src = src.shape[:2]
    if h_src == 0 or w_src == 0:
        return np.zeros((h_can, w_can) + ((src.shape[-1],) if src.ndim == 3 else ()), dtype=np.uint8)

    # Build a per-pixel mapping from canvas → source. For a bilinear warp the
    # inverse is also bilinear: parametrize the quad with (u, v) ∈ [0,1]² and
    # invert numerically.
    try:
        import cv2  # type: ignore
        # Direct mapping via perspective warp: source rect → quadrilateral.
        src_quad = np.array(
            [[0, 0], [w_src - 1, 0], [w_src - 1, h_src - 1], [0, h_src - 1]],
            dtype=np.float32,
        )
        dst_quad = np.array([tl, tr, br, bl], dtype=np.float32)
        # We want the inverse (canvas → source), so swap src/dst.
        M = cv2.getPerspectiveTransform(dst_quad, src_quad)
        return cv2.warpPerspective(
            src, M, (w_can, h_can),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except ImportError:
        pass

    # Pure NumPy fallback: solve per-pixel via two-pass bilinear interpolation
    # of the corner coords.
    ys, xs = np.mgrid[0:h_can, 0:w_can].astype(np.float32)
    # First estimate u, v assuming an affine mapping, then refine with a
    # closed-form bilinear inverse via Newton's method (single iteration is
    # usually enough since the quad is close to a rectangle).
    src_corners = np.array(
        [[0, 0], [w_src - 1, 0], [w_src - 1, h_src - 1], [0, h_src - 1]],
        dtype=np.float32,
    )
    dst_corners = np.array([tl, tr, br, bl], dtype=np.float32)
    # Linear affine approx as the initial guess.
    A = np.zeros((2, 2), dtype=np.float32)
    b_off = np.zeros(2, dtype=np.float32)
    # Least-squares affine from 4 src→dst pairs.
    for sx, sy, dx, dy in zip(src_corners[:, 0], src_corners[:, 1], dst_corners[:, 0], dst_corners[:, 1]):
        A += np.outer([dx, dy], [sx, sy])
        b_off += np.array([dx, dy], dtype=np.float32)
    A /= 4
    b_off /= 4
    sx_aff = A[0, 0] * xs + A[0, 1] * ys + b_off[0]
    sy_aff = A[1, 0] * xs + A[1, 1] * ys + b_off[1]
    # If we have cv2 missing, just use the affine approx (good enough for
    # mild free-distort).
    return _sample(src, sx_aff, sy_aff).astype(np.uint8)


# ---------------------------------------------------------------------------
# Layer fetcher — abstract over the on-disk PNGs.
# ---------------------------------------------------------------------------


LayerImageLoader = Callable[[str], Optional[np.ndarray]]
LayerMaskLoader = Callable[[str], Optional[np.ndarray]]


def _empty_image(w: int, h: int) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    return arr


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


@dataclass
class CompositorOptions:
    canvas_width: int
    canvas_height: int
    image_loader: Optional[LayerImageLoader] = None
    mask_loader: Optional[LayerMaskLoader] = None


def composite_document(
    session: EditorSession,
    options: CompositorOptions,
    *,
    selection: Optional[np.ndarray] = None,
    visible_layers: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Render the document into an HxWx4 uint8 canvas."""
    out = np.zeros((options.canvas_height, options.canvas_width, 4), dtype=np.uint8)
    visible_set = set(visible_layers) if visible_layers is not None else None
    # Each group is a pass-through; we recursively render its children with
    # the group's accumulated opacity.
    _composite_recursive(
        out,
        session.layers,
        parent_id=None,
        group_opacity=1.0,
        options=options,
        selection=selection,
        visible_set=visible_set,
    )
    return out


def merge_layers_pixels(
    session: EditorSession,
    cache: Dict[str, np.ndarray],
    bottom: ImageLayer,
    top: ImageLayer,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Flatten ``bottom`` + ``top`` into baked document-space pixels.

    Renders the pair exactly as they appear (transforms, masks, opacity)
    but with Normal blending — the result becomes one plain image layer,
    matching Photoshop's merge-down semantics. Returns
    ``(pixels, origin, size)`` for the crop that tightly bounds the merged
    content; a fully transparent result keeps the full canvas extent.
    """
    import copy as _copy
    from types import SimpleNamespace

    def _flat(layer: ImageLayer) -> ImageLayer:
        clone = _copy.copy(layer)
        clone.blend_mode = BlendMode.NORMAL
        clone.opacity = 1.0
        clone.is_visible = True
        return clone

    pair = SimpleNamespace(layers=[_flat(bottom), _flat(top)])
    merged = composite_document(pair, CompositorOptions(  # type: ignore[arg-type]
        canvas_width=session.width,
        canvas_height=session.height,
        image_loader=cache.get,
        mask_loader=getattr(session, "_mask_cache", {}).get,
    ))

    alpha = merged[..., 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return merged, (0.0, 0.0), (float(session.width), float(session.height))
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y0, y1 = int(y_indices[0]), int(y_indices[-1])
    x0, x1 = int(x_indices[0]), int(x_indices[-1])
    pixels = merged[y0 : y1 + 1, x0 : x1 + 1]
    return pixels, (float(x0), float(y0)), (float(x1 - x0 + 1), float(y1 - y0 + 1))


def _composite_recursive(
    canvas: np.ndarray,
    layers: Sequence[AnyLayer],
    parent_id: Optional[str],
    group_opacity: float,
    options: CompositorOptions,
    selection: Optional[np.ndarray],
    visible_set: Optional[set],
) -> None:
    """Walk ``layers`` in order; each layer draws into ``canvas`` over what came before."""
    for layer in layers:
        if layer.parent_id != parent_id:
            continue
        if not layer.is_visible:
            continue
        if visible_set is not None and layer.id not in visible_set:
            continue
        if isinstance(layer, GroupLayer):
            # If the group carries a mask, render the children into a temp
            # buffer first so we can mask the group's composite contribution
            # as a single layer (folder mask, v6+).
            has_group_mask = bool(layer.mask_file) and bool(layer.mask_enabled)
            if has_group_mask and options.mask_loader is not None:
                _composite_group_with_mask(
                    canvas,
                    layers,
                    layer,
                    group_opacity,
                    options,
                    selection,
                    visible_set,
                )
                continue
            # Pass-through: render children, multiplying by the group's own opacity (v8+).
            _composite_recursive(
                canvas,
                layers,
                layer.id,
                group_opacity * layer.opacity,
                options,
                selection,
                visible_set,
            )
            continue
        if isinstance(layer, AdjustmentLayer):
            _apply_adjustment_to_canvas(canvas, layer, selection=selection)
            continue
        # Image layer.
        _composite_image_layer(
            canvas,
            layer,
            group_opacity=group_opacity,
            options=options,
            selection=selection,
        )


def _composite_group_with_mask(
    canvas: np.ndarray,
    layers: Sequence[AnyLayer],
    group: "GroupLayer",
    group_opacity: float,
    options: CompositorOptions,
    selection: Optional[np.ndarray],
    visible_set: Optional[set],
) -> None:
    """Composite a group with its mask applied to the children's combined output."""
    # 1. Snapshot the canvas before the group renders.
    before = canvas.copy()
    # 2. Render the children into the canvas.
    _composite_recursive(
        canvas,
        layers,
        group.id,
        group_opacity * group.opacity,
        options,
        selection,
        visible_set,
    )
    # 3. Compute the group's contribution = (after - before), premultiplied.
    after = canvas
    group_layer = after.astype(np.int32) - before.astype(np.int32)
    group_layer = np.clip(group_layer, 0, 255).astype(np.uint8)
    # 4. Load the mask and warp it into canvas coordinates.
    mask_src = options.mask_loader(group.id) if options.mask_loader is not None else None  # type: ignore[arg-type]
    if mask_src is None:
        return
    if mask_src.ndim == 3:
        mask_src = mask_src[..., 0]
    matrix = transform_matrix(group.transform)
    mask_warped = warp_into_canvas(
        mask_src.astype(np.float32) / 255.0, matrix, canvas.shape[:2]
    )
    # 5. Multiply the group's contribution alpha by the mask and add back.
    group_layer = before + (group_layer.astype(np.float32) * mask_warped[..., None]).astype(np.uint8)
    np.copyto(canvas, np.clip(group_layer, 0, 255).astype(np.uint8))


def _composite_image_layer(
    canvas: np.ndarray,
    layer: ImageLayer,
    *,
    group_opacity: float,
    options: CompositorOptions,
    selection: Optional[np.ndarray],
) -> None:
    if options.image_loader is None:
        return
    src = options.image_loader(layer.id)
    if src is None:
        return
    if src.ndim == 2:
        # Mask-style image (shouldn't happen for imageFile, but tolerate).
        src = np.stack([src, src, src, np.full(src.shape, 255, dtype=np.uint8)], axis=-1)
    if src.shape[-1] == 3:
        src = np.concatenate([src, np.full(src.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)

    if layer.transform.corners is not None:
        # Free-distort path: warp the rectangular source onto the 4 corners.
        warped = warp_into_canvas_corners(src, canvas.shape[:2], layer.transform.corners)
    else:
        matrix = transform_matrix(layer.transform)
        warped = warp_into_canvas(src, matrix, canvas.shape[:2])

    mask = _resolve_mask(layer, options)
    if mask is not None:
        if layer.transform.corners is not None:
            mask_warped = warp_into_canvas_corners(
                mask.astype(np.float32) / 255.0,
                canvas.shape[:2],
                layer.transform.corners,
            )
        else:
            matrix = transform_matrix(layer.transform)
            mask_warped = warp_into_canvas(mask.astype(np.float32) / 255.0, matrix, canvas.shape[:2])
        warped[..., 3] = (warped[..., 3].astype(np.float32) * mask_warped).astype(np.uint8)

    effective_opacity = layer.opacity * group_opacity
    if effective_opacity < 1.0:
        warped[..., 3] = (warped[..., 3].astype(np.float32) * effective_opacity).astype(np.uint8)

    if selection is not None:
        # Restrict the layer's draw region to the selection mask.
        warped[..., 3] = (warped[..., 3].astype(np.float32) * (selection.astype(np.float32) / 255.0)).astype(np.uint8)

    blended = apply_blend(canvas, warped, layer.blend_mode.value)
    canvas[...] = blended


def _resolve_mask(layer: ImageLayer, options: CompositorOptions) -> Optional[np.ndarray]:
    if layer.mask_file is None or not layer.mask_enabled:
        return None
    if options.mask_loader is None:
        return None
    mask = options.mask_loader(layer.id)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


# ---------------------------------------------------------------------------
# Adjustment layer application
# ---------------------------------------------------------------------------


def _apply_adjustment_to_canvas(
    canvas: np.ndarray,
    layer: AdjustmentLayer,
    *,
    selection: Optional[np.ndarray],
) -> None:
    if not layer.is_visible:
        return
    if layer.opacity <= 0:
        return
    mask = _resolve_adjustment_mask(layer)
    if selection is not None and mask is not None:
        mask = ((mask.astype(np.float32) / 255.0) * (selection.astype(np.float32) / 255.0) * 255.0).astype(np.uint8)
    elif selection is not None:
        mask = selection
    affected = apply_adjustment(canvas, layer.kind, layer.settings)
    if mask is None:
        canvas[...] = affected
    else:
        m = mask.astype(np.float32) / 255.0
        orig = canvas.astype(np.float32)
        new = affected.astype(np.float32)
        canvas[...] = (orig * (1 - m[..., None]) + new * m[..., None]).astype(np.uint8)


def _resolve_adjustment_mask(layer: AdjustmentLayer) -> Optional[np.ndarray]:
    # AdjustmentLayer has the same mask fields as Layer, but our base class
    # only exposes them through duck typing.
    mask_file = getattr(layer, "mask_file", None)
    mask_enabled = getattr(layer, "mask_enabled", False)
    if not mask_file or not mask_enabled:
        return None
    # The actual mask bytes must be provided by the renderer; we ask via
    # a global hook that the GUI layer registers.
    loader = _ADJUSTMENT_MASK_LOADERS.get(layer.id)
    if loader is None:
        return None
    mask = loader()
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


_ADJUSTMENT_MASK_LOADERS: Dict[str, Callable[[], Optional[np.ndarray]]] = {}


def register_adjustment_mask_loader(layer_id: str, loader: Callable[[], Optional[np.ndarray]]) -> None:
    _ADJUSTMENT_MASK_LOADERS[layer_id] = loader


def unregister_adjustment_mask_loader(layer_id: str) -> None:
    _ADJUSTMENT_MASK_LOADERS.pop(layer_id, None)
