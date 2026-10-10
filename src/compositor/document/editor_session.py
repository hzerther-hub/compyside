"""EditorSession — top-level document state and undo/redo.

Pure Python. No Qt, no NumPy. Holds the layer stack, the active layer id,
the canvas size, and a command-stack undo/redo history.

The command protocol is intentionally tiny: a "do" callable mutates the
session and returns the matching "undo" callable. ``undo()`` runs the undo
and parks the original do on the redo stack; ``redo()`` re-runs the do to
get a fresh undo closure.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .layers import (
    AdjustmentLayer,
    AnyLayer,
    GroupLayer,
    ImageLayer,
    Transform,
    layers_from_manifest,
    layers_to_manifest,
    new_id,
)
from .guides import Guide, new_guide_id


DoFn = Callable[["EditorSession"], Callable[["EditorSession"], None]]


@dataclass
class EditorSession:
    width: int
    height: int
    color_space: str = "sRGB"
    document_id: str = field(default_factory=new_id)
    resolution: float = 72.0
    layers: List[AnyLayer] = field(default_factory=list)
    guides: List[Guide] = field(default_factory=list)
    active_layer_id: Optional[str] = None

    # (do_fn, undo_fn) pairs, oldest at index 0.
    _undo: List[Any] = field(default_factory=list)
    _redo: List[Any] = field(default_factory=list)

    # ---- layer access ---------------------------------------------------

    def layer_by_id(self, layer_id: str) -> Optional[AnyLayer]:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def layer_index(self, layer_id: str) -> int:
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                return i
        raise KeyError(layer_id)

    def children_of(self, parent_id: Optional[str]) -> List[AnyLayer]:
        return [layer for layer in self.layers if layer.parent_id == parent_id]

    # ---- commands -------------------------------------------------------

    def _run(self, do_fn: DoFn) -> None:
        """Run ``do_fn(session)`` and record the resulting undo pair."""
        undo_fn = do_fn(self)
        self._undo.append((do_fn, undo_fn))
        self._redo.clear()

    def add_layer(self, layer: AnyLayer, *, at_index: Optional[int] = None) -> None:
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            idx = at_index if at_index is not None else len(s.layers)
            s.layers.insert(idx, layer)
            s.active_layer_id = layer.id

            def undo(s2: "EditorSession") -> None:
                popped = s2.layers.pop(idx)
                if s2.active_layer_id == popped.id:
                    s2.active_layer_id = None

            return undo

        self._run(do)

    def remove_layer(self, layer_id: str) -> None:
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            idx = s.layer_index(layer_id)
            removed = s.layers.pop(idx)
            prev_active = s.active_layer_id
            s.active_layer_id = None

            def undo(s2: "EditorSession") -> None:
                s2.layers.insert(idx, removed)
                s2.active_layer_id = prev_active or removed.id

            return undo

        self._run(do)

    def set_property(self, layer_id: str, attr: str, value: Any) -> None:
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            layer = s.layer_by_id(layer_id)
            assert layer is not None, f"unknown layer {layer_id!r}"
            previous = getattr(layer, attr)
            setattr(layer, attr, value)

            def undo(s2: "EditorSession") -> None:
                layer2 = s2.layer_by_id(layer_id)
                assert layer2 is not None
                setattr(layer2, attr, previous)

            return undo

        self._run(do)

    def reorder_layer(self, layer_id: str, new_index: int) -> None:
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            old_index = s.layer_index(layer_id)
            layer = s.layers.pop(old_index)
            new_index_clamped = max(0, min(new_index, len(s.layers)))
            s.layers.insert(new_index_clamped, layer)

            def undo(s2: "EditorSession") -> None:
                cur = s2.layer_index(layer_id)
                moved = s2.layers.pop(cur)
                s2.layers.insert(old_index, moved)

            return undo

        self._run(do)

    def merge_down(
        self,
        top_id: str,
        replacement: AnyLayer,
        *,
        new_pixels: Any = None,
        pixel_cache: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Replace the layer directly below ``top_id`` (same parent) with
        ``replacement`` and remove ``top_id``.

        When ``pixel_cache`` (the app's ``_image_cache``) and ``new_pixels``
        are given, the merged pixels are stored under ``replacement.id`` and
        undo restores the previous entries so a post-undo render is exact.
        """
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            top_idx = s.layer_index(top_id)
            below_idx = next(
                i for i in range(top_idx - 1, -1, -1)
                if s.layers[i].parent_id == s.layers[top_idx].parent_id
            )
            prev_top = s.layers[top_idx]
            prev_below = s.layers[below_idx]
            prev_active = s.active_layer_id
            old_pixels: Dict[str, Any] = {}
            if pixel_cache is not None and new_pixels is not None:
                old_pixels[prev_below.id] = pixel_cache.get(prev_below.id)
                old_pixels[prev_top.id] = pixel_cache.get(prev_top.id)
                pixel_cache[replacement.id] = new_pixels
            s.layers[below_idx] = replacement
            del s.layers[top_idx]
            s.active_layer_id = replacement.id

            def undo(s2: "EditorSession") -> None:
                s2.layers[below_idx] = prev_below
                s2.layers.insert(top_idx, prev_top)
                s2.active_layer_id = prev_active
                if pixel_cache is not None:
                    for lid, px in old_pixels.items():
                        if px is None:
                            pixel_cache.pop(lid, None)
                        else:
                            pixel_cache[lid] = px

            return undo

        self._run(do)

    # ---- guides ---------------------------------------------------------

    def guide_by_id(self, guide_id: str) -> Optional[Guide]:
        for g in self.guides:
            if g.id == guide_id:
                return g
        return None

    def add_guide(self, guide: Guide) -> None:
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            s.guides.append(guide)

            def undo(s2: "EditorSession") -> None:
                s2.guides[:] = [g for g in s2.guides if g.id != guide.id]

            return undo

        self._run(do)

    def move_guide(self, guide_id: str, position: float) -> None:
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            g = s.guide_by_id(guide_id)
            assert g is not None, f"unknown guide {guide_id!r}"
            old_position = g.position
            # Guide is frozen, replace it with a new one.
            new_g = Guide(id=g.id, orientation=g.orientation, position=position, is_visible=g.is_visible)
            idx = next(i for i, x in enumerate(s.guides) if x.id == guide_id)
            s.guides[idx] = new_g

            def undo(s2: "EditorSession") -> None:
                g2 = s2.guide_by_id(guide_id)
                assert g2 is not None
                restored = Guide(id=g2.id, orientation=g2.orientation,
                                 position=old_position, is_visible=g2.is_visible)
                idx2 = next(i for i, x in enumerate(s2.guides) if x.id == guide_id)
                s2.guides[idx2] = restored

            return undo

        self._run(do)

    def remove_guide(self, guide_id: str) -> None:
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            removed = next((g for g in s.guides if g.id == guide_id), None)
            assert removed is not None, f"unknown guide {guide_id!r}"
            s.guides[:] = [g for g in s.guides if g.id != guide_id]

            def undo(s2: "EditorSession") -> None:
                if removed is not None:
                    s2.guides.append(removed)

            return undo

        self._run(do)

    # ---- undo/redo ------------------------------------------------------

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        do_fn, undo_fn = self._undo.pop()
        undo_fn(self)
        self._redo.append((do_fn, undo_fn))
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        do_fn, _ = self._redo.pop()
        new_undo = do_fn(self)
        self._undo.append((do_fn, new_undo))
        return True

    # ---- manifest interop -----------------------------------------------

    def to_manifest_layers(self) -> List[Dict[str, Any]]:
        return layers_to_manifest(self.layers)

    def to_manifest_guides(self) -> List[Dict[str, Any]]:
        return [g.to_dict() for g in self.guides]

    def load_manifest_layers(self, layers_json: Sequence[Dict[str, Any]]) -> None:
        self.layers = layers_from_manifest(layers_json)

    def load_manifest_guides(self, guides_json: Sequence[Dict[str, Any]]) -> None:
        self.guides = [Guide.from_dict(g) for g in guides_json]

    # ---- canvas ops -----------------------------------------------------

    def resize_canvas(self, new_width: int, new_height: int) -> None:
        """Change the canvas size. Existing layer pixels are not resampled."""
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            old_w, old_h = s.width, s.height
            s.width = new_width
            s.height = new_height

            def undo(s2: "EditorSession") -> None:
                s2.width = old_w
                s2.height = old_h

            return undo

        self._run(do)

    def scale_image(self, new_width: int, new_height: int) -> None:
        """Resize the canvas and resample every ImageLayer's pixels to fit.

        The pixel resampling happens in the app layer (it has PIL); this
        method only updates the canvas size with undo support.
        """
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            old_w, old_h = s.width, s.height
            s.width = new_width
            s.height = new_height

            def undo(s2: "EditorSession") -> None:
                s2.width = old_w
                s2.height = old_h

            return undo

        self._run(do)

    def trim_document(self, x0: int, y0: int, new_width: int, new_height: int) -> None:
        """Crop the canvas to ``(x0, y0, new_width, new_height)``.

        Every layer's placement shifts by ``(-x0, -y0)`` so content stays
        put relative to the new origin; pixels are not resampled.
        """
        def do(s: "EditorSession") -> Callable[["EditorSession"], None]:
            old_w, old_h = s.width, s.height
            old_transforms = [(layer.id, layer.transform) for layer in s.layers]
            s.width = new_width
            s.height = new_height
            for layer in s.layers:
                t = layer.transform
                layer.transform = Transform(
                    origin=(t.origin[0] - x0, t.origin[1] - y0),
                    size=t.size,
                    rotation=t.rotation,
                    flip_x=t.flip_x,
                    flip_y=t.flip_y,
                    sampling=t.sampling,
                    corners=t.corners,
                )

            def undo(s2: "EditorSession") -> None:
                s2.width = old_w
                s2.height = old_h
                for lid, t in old_transforms:
                    layer2 = s2.layer_by_id(lid)
                    if layer2 is not None:
                        layer2.transform = t

            return undo

        self._run(do)
