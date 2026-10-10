"""Layer data classes.

These mirror the manifest schema but expose a typed Python API. Conversions
to/from the on-disk dict live in ``layers_from_manifest`` and
``layers_to_manifest``.

The model deliberately avoids Qt — this lets us run unit tests headless and
swap the renderer without disturbing the document.
"""
from __future__ import annotations

import copy
import enum
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..format.manifest import ADJUSTMENT_KINDS, BLEND_MODES, SAMPLINGS


class Sampling(str, enum.Enum):
    HIGH_QUALITY = "High quality"
    SMOOTH = "Smooth"
    NEAREST = "Nearest"


class BlendMode(str, enum.Enum):
    NORMAL = "Normal"
    DARKEN = "Darken"
    MULTIPLY = "Multiply"
    COLOR_BURN = "Color Burn"
    LINEAR_BURN = "Linear Burn"
    LIGHTEN = "Lighten"
    SCREEN = "Screen"
    COLOR_DODGE = "Color Dodge"
    LINEAR_DODGE_ADD = "Linear Dodge (Add)"
    OVERLAY = "Overlay"
    SOFT_LIGHT = "Soft Light"
    HARD_LIGHT = "Hard Light"
    VIVID_LIGHT = "Vivid Light"
    LINEAR_LIGHT = "Linear Light"
    PIN_LIGHT = "Pin Light"
    HARD_MIX = "Hard Mix"
    DIFFERENCE = "Difference"
    EXCLUSION = "Exclusion"
    SUBTRACT = "Subtract"
    DIVIDE = "Divide"
    HUE = "Hue"
    SATURATION = "Saturation"
    COLOR = "Color"
    LUMINOSITY = "Luminosity"


class AdjustmentKind(str, enum.Enum):
    HUE_SATURATION = "Hue/Saturation"
    LEVELS = "Levels"
    CURVES = "Curves"
    EXPOSURE = "Exposure"
    GRADIENT_MAP = "Gradient Map"
    GRAIN = "Grain"
    INVERT = "Invert"
    BLACK_WHITE = "Black & White"
    COLOR_BALANCE = "Color Balance"
    GAUSSIAN_BLUR = "Gaussian Blur"
    MOTION_BLUR = "Motion Blur"
    ADD_NOISE = "Add Noise"
    VIGNETTE = "Vignette"
    BLOOM = "Bloom"
    DITHER = "Dither"
    SCANLINES = "Scanlines"
    TONAL_CONTRAST = "Tonal Contrast"
    LENS_CORRECTION = "Lens Correction"
    REMOVE_BACKGROUND = "Remove Background"


def new_id() -> str:
    """Generate a layer UUID in Compositor's uppercase form."""
    return str(uuid.uuid4()).upper()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    """Layer placement in document pixels.

    Origin is the top-left corner; size is the drawn extent; rotation is
    clockwise degrees; flips apply before rotation. Sampling controls how
    the source pixels are resampled to ``size``.

    ``corners`` (optional, 4-tuple of (x, y)) overrides the rect drawn by
    origin+size — when set, the layer is bilinearly warped from the
    rectangular source extent to the four arbitrary corners. Order is
    top-left, top-right, bottom-right, bottom-left. Photoshop-style
    free-distort writes this; the rest of the time it's ``None``.
    """

    origin: Tuple[float, float] = (0.0, 0.0)
    size: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    sampling: Sampling = Sampling.HIGH_QUALITY
    corners: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "origin": [self.origin[0], self.origin[1]],
            "size": [self.size[0], self.size[1]],
            "rotation": self.rotation,
            "flipX": self.flip_x,
            "flipY": self.flip_y,
            "sampling": self.sampling.value,
        }
        if self.corners is not None:
            out["corners"] = [[c[0], c[1]] for c in self.corners]
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transform":
        origin = tuple(data["origin"])
        size = tuple(data["size"])
        corners = data.get("corners")
        if corners is not None:
            corners = tuple(tuple(c) for c in corners)  # type: ignore[assignment]
        return cls(
            origin=(float(origin[0]), float(origin[1])),
            size=(float(size[0]), float(size[1])),
            rotation=float(data.get("rotation", 0.0)),
            flip_x=bool(data.get("flipX", False)),
            flip_y=bool(data.get("flipY", False)),
            sampling=Sampling(data.get("sampling", Sampling.HIGH_QUALITY.value)),
            corners=corners,
        )


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


@dataclass
class Layer:
    """Common fields for every layer kind."""

    id: str
    name: str
    is_visible: bool = True
    parent_id: Optional[str] = None
    opacity: float = 1.0
    blend_mode: BlendMode = BlendMode.NORMAL
    transform: Transform = field(default_factory=Transform)
    mask_file: Optional[str] = None
    mask_enabled: bool = False
    mask_source_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "isVisible": self.is_visible,
            "isGroup": False,
            "opacity": self.opacity,
            "blendMode": self.blend_mode.value,
            "transform": self.transform.to_dict(),
        }
        if self.parent_id is not None:
            out["parentID"] = self.parent_id
        if self.mask_file is not None:
            out["maskFile"] = self.mask_file
        if self.mask_enabled:
            out["maskEnabled"] = True
        if self.mask_source_id is not None:
            out["maskSourceID"] = self.mask_source_id
        return out

    @classmethod
    def _from_dict_common(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            id=data["id"],
            name=data["name"],
            is_visible=bool(data.get("isVisible", True)),
            parent_id=data.get("parentID"),
            opacity=float(data.get("opacity", 1.0)),
            blend_mode=BlendMode(data.get("blendMode", BlendMode.NORMAL.value)),
            transform=Transform.from_dict(data["transform"]),
            mask_file=data.get("maskFile"),
            mask_enabled=bool(data.get("maskEnabled", False)),
            mask_source_id=data.get("maskSourceID"),
        )


@dataclass
class ImageLayer(Layer):
    """A regular image layer — pixels stored as a separate PNG."""

    image_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        if self.image_file is not None:
            out["imageFile"] = self.image_file
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageLayer":
        return cls(image_file=data.get("imageFile"), **Layer._from_dict_common(data))


@dataclass
class GroupLayer(Layer):
    """A folder of layers. No image, can carry its own opacity (v8+)."""

    is_group: bool = field(default=True, init=False)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out["isGroup"] = True
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupLayer":
        return cls(**Layer._from_dict_common(data))


@dataclass
class AdjustmentLayer(Layer):
    """An adjustment layer. Affects everything composited below it within its folder."""

    kind: AdjustmentKind = AdjustmentKind.HUE_SATURATION
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out["adjustment"] = {"kind": self.kind.value, **self.settings}
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdjustmentLayer":
        adj = data.get("adjustment", {})
        kind = AdjustmentKind(adj.get("kind", AdjustmentKind.HUE_SATURATION.value))
        settings = {k: v for k, v in adj.items() if k != "kind"}
        return cls(kind=kind, settings=settings, **Layer._from_dict_common(data))


AnyLayer = Union[ImageLayer, GroupLayer, AdjustmentLayer]


# ---------------------------------------------------------------------------
# Manifest <-> layer conversion
# ---------------------------------------------------------------------------


def layer_from_dict(data: Dict[str, Any]) -> AnyLayer:
    """Pick the right subclass based on the dict's fields."""
    if "adjustment" in data:
        return AdjustmentLayer.from_dict(data)
    if data.get("isGroup"):
        return GroupLayer.from_dict(data)
    return ImageLayer.from_dict(data)


def layers_from_manifest(layers_json: Sequence[Dict[str, Any]]) -> List[AnyLayer]:
    """Convert a manifest's ``layers`` array (bottom-to-top) to typed objects."""
    return [layer_from_dict(copy.deepcopy(d)) for d in layers_json]


def layers_to_manifest(layers: Sequence[AnyLayer]) -> List[Dict[str, Any]]:
    """Convert typed layers back into the dict form expected by the manifest."""
    return [layer.to_dict() for layer in layers]
