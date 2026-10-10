"""Manifest reader/writer for Compositor .comp project files.

The manifest is JSON. Versions are additive: v2 adds optional `parentID`/`isGroup`,
v3 adds `opacity`/`blendMode`, v4 adds `maskFile`/`maskEnabled`, v5 adds `maskSourceID`,
v6 allows masks on groups, v7 adds adjustment layers, v8 allows folder opacity,
v9 adds three more adjustment kinds, v10 reserved, v11 current.

See docs/project-format.md (Compositor repo) for the authoritative spec.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

CURRENT_VERSION = 11
MIN_SUPPORTED_VERSION = 1
IDENTIFIER = "com.compositor.project"

# 24 blend mode spellings, exact as Compositor names them.
BLEND_MODES = (
    "Normal",
    "Darken",
    "Multiply",
    "Color Burn",
    "Linear Burn",
    "Lighten",
    "Screen",
    "Color Dodge",
    "Linear Dodge (Add)",
    "Overlay",
    "Soft Light",
    "Hard Light",
    "Vivid Light",
    "Linear Light",
    "Pin Light",
    "Hard Mix",
    "Difference",
    "Exclusion",
    "Subtract",
    "Divide",
    "Hue",
    "Saturation",
    "Color",
    "Luminosity",
)

ADJUSTMENT_KINDS = (
    "Hue/Saturation",
    "Levels",
    "Curves",
    "Exposure",
    "Gradient Map",
    "Grain",
    "Invert",
    "Black & White",
    "Color Balance",
    "Gaussian Blur",
    "Motion Blur",
    "Add Noise",
    "Vignette",
    "Bloom",
    "Dither",
    "Scanlines",
    "Tonal Contrast",
    "Lens Correction",
    "Remove Background",
)

SAMPLINGS = ("High quality", "Smooth", "Nearest")


class ManifestError(ValueError):
    """Raised when a manifest is invalid or unreadable."""


# JSON Schema for the document. Required fields are version-tagged so an older
# manifest remains valid for the version it declares.
SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["format", "version", "colorSpace", "documentID", "width", "height", "layers"],
    "properties": {
        "format": {"const": IDENTIFIER},
        "version": {"type": "integer", "minimum": MIN_SUPPORTED_VERSION, "maximum": CURRENT_VERSION},
        "colorSpace": {"type": "string"},
        "documentID": {"type": "string", "pattern": "^[0-9A-Fa-f-]+$"},
        "width": {"type": "integer", "minimum": 1, "maximum": 30000},
        "height": {"type": "integer", "minimum": 1, "maximum": 30000},
        "resolution": {"type": "number", "minimum": 1, "maximum": 9600},
        "activeLayerID": {"type": "string"},
        "layers": {
            "type": "array",
            "maxItems": 10000,
            "items": {"$ref": "#/$defs/layer"},
        },
        "guides": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "orientation", "position"],
                "properties": {
                    "id": {"type": "string"},
                    "orientation": {"enum": ["h", "v"]},
                    "position": {"type": "number"},
                    "isVisible": {"type": "boolean"},
                },
            },
        },
    },
    "$defs": {
        "transform": {
            "type": "object",
            "required": ["origin", "size", "rotation", "flipX", "flipY", "sampling"],
            "properties": {
                "origin": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "size": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "rotation": {"type": "number"},
                "flipX": {"type": "boolean"},
                "flipY": {"type": "boolean"},
                "sampling": {"enum": list(SAMPLINGS)},
                "corners": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
            },
        },
        "adjustment": {
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {"enum": list(ADJUSTMENT_KINDS)},
                "hue": {"type": "number"},
                "saturation": {"type": "number"},
                "lightness": {"type": "number"},
                "colorize": {"type": "boolean"},
                "hsvSettings": {"type": "object"},
                "levels": {"type": "object"},
                "curves": {"type": "object"},
                "exposureSettings": {"type": "object"},
                "gradientMapSettings": {"type": "object"},
                "grainSettings": {"type": "object"},
                "blackWhiteSettings": {"type": "object"},
                "colorBalanceSettings": {"type": "object"},
                "gaussianBlurSettings": {"type": "object"},
                "motionBlurSettings": {"type": "object"},
                "addNoiseSettings": {"type": "object"},
                "vignetteSettings": {"type": "object"},
                "bloomSettings": {"type": "object"},
                "ditherSettings": {"type": "object"},
                "scanlinesSettings": {"type": "object"},
                "tonalContrastSettings": {"type": "object"},
                "lensCorrectionSettings": {"type": "object"},
                "removeBackgroundSettings": {"type": "object"},
            },
        },
        "layer": {
            "type": "object",
            "required": ["id", "name", "isVisible", "isGroup", "transform"],
            "properties": {
                "id": {"type": "string", "pattern": "^[0-9A-Fa-f-]+$"},
                "name": {"type": "string"},
                "isVisible": {"type": "boolean"},
                "isGroup": {"type": "boolean"},
                "parentID": {"type": "string"},
                "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                "blendMode": {"enum": list(BLEND_MODES)},
                "transform": {"$ref": "#/$defs/transform"},
                "imageFile": {"type": "string"},
                "maskFile": {"type": "string"},
                "maskEnabled": {"type": "boolean"},
                "maskSourceID": {"type": "string"},
                "adjustment": {"$ref": "#/$defs/adjustment"},
                "text": {"type": "object"},
            },
        },
    },
}


@dataclass
class Manifest:
    """In-memory representation of a .comp manifest.

    Layers are stored bottom-to-top (the order in the JSON array).
    """

    version: int
    color_space: str
    document_id: str
    width: int
    height: int
    layers: List[Dict[str, Any]] = field(default_factory=list)
    guides: List[Dict[str, Any]] = field(default_factory=list)
    guides: List[Dict[str, Any]] = field(default_factory=list)
    resolution: float = 72.0
    active_layer_id: Optional[str] = None

    # ---- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "format": IDENTIFIER,
            "version": self.version,
            "colorSpace": self.color_space,
            "documentID": self.document_id,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "layers": [copy.deepcopy(layer) for layer in self.layers],
            "guides": [copy.deepcopy(g) for g in self.guides],
        }
        if self.active_layer_id is not None:
            out["activeLayerID"] = self.active_layer_id
        return out

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # ---- queries --------------------------------------------------------

    def layer_ids(self) -> List[str]:
        return [layer["id"] for layer in self.layers]

    def find_layer(self, layer_id: str) -> Optional[Dict[str, Any]]:
        for layer in self.layers:
            if layer["id"] == layer_id:
                return layer
        return None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path) -> Manifest:
    """Read a manifest.json from disk. Raises ManifestError on any failure."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc

    validate_manifest(data)
    return _dict_to_manifest(data)


def save_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write a manifest.json. Atomic on POSIX (rename)."""
    path = Path(path)
    payload = manifest.to_dict()
    validate_manifest(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def validate_manifest(data: Dict[str, Any]) -> None:
    """Validate a manifest dict against the JSON Schema. Raises ManifestError."""
    try:
        jsonschema.validate(instance=data, schema=SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ManifestError(f"manifest failed validation: {exc.message}") from exc

    version = data.get("version", 0)
    if version < MIN_SUPPORTED_VERSION or version > CURRENT_VERSION:
        raise ManifestError(
            f"unsupported version {version}; supported: {MIN_SUPPORTED_VERSION}-{CURRENT_VERSION}"
        )

    # Cross-cutting rules from the spec — keep these here so they're enforced
    # regardless of which version wrote the file.
    layer_ids = [layer["id"] for layer in data["layers"]]
    if len(set(layer_ids)) != len(layer_ids):
        raise ManifestError("duplicate layer ids")

    # Image-bearing groups are illegal. Adjustment layers cannot carry an image.
    for layer in data["layers"]:
        if layer.get("isGroup") and layer.get("imageFile"):
            raise ManifestError(f"group layer {layer['id']} has imageFile")
        if layer.get("adjustment") and layer.get("imageFile"):
            raise ManifestError(
                f"adjustment layer {layer['id']} cannot also carry imageFile"
            )

        # v2+ group rules: parentID only if version >= 2.
        if version < 2 and "parentID" in layer:
            raise ManifestError(f"version {version} cannot contain parentID")

        # v3+ appearance rules: non-default opacity/blendMode require v3+.
        if version < 3:
            if layer.get("opacity", 1.0) != 1.0:
                raise ManifestError(f"version {version} cannot contain opacity")
            if layer.get("blendMode", "Normal") != "Normal":
                raise ManifestError(f"version {version} cannot contain blendMode")

        # v4+ mask on leaf only (groups allowed in v6+).
        if version < 4 and ("maskFile" in layer or "maskEnabled" in layer):
            raise ManifestError(f"version {version} cannot contain mask metadata")
        if 4 <= version < 6 and layer.get("isGroup") and "maskFile" in layer:
            raise ManifestError(f"version {version} cannot give a group a mask")

        # v7+ adjustment layers.
        if version < 7 and "adjustment" in layer:
            raise ManifestError(f"version {version} cannot contain adjustment layers")

    # Nesting depth limit (64 ancestors).
    children: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for layer in data["layers"]:
        parent = layer.get("parentID") if version >= 2 else None
        children.setdefault(parent, []).append(layer)
    _check_nesting(children, max_depth=64)

    # Live-mask cycles (v5+).
    if version >= 5:
        _check_mask_chains(data["layers"], max_chain=256)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _dict_to_manifest(data: Dict[str, Any]) -> Manifest:
    return Manifest(
        version=int(data["version"]),
        color_space=str(data["colorSpace"]),
        document_id=str(data["documentID"]),
        width=int(data["width"]),
        height=int(data["height"]),
        resolution=float(data.get("resolution", 72.0)),
        layers=[copy.deepcopy(layer) for layer in data["layers"]],
        guides=[copy.deepcopy(g) for g in data.get("guides", [])],
        active_layer_id=data.get("activeLayerID"),
    )


def _check_nesting(children: Dict[Optional[str], List[Dict[str, Any]]], max_depth: int) -> None:
    """Reject nesting deeper than max_depth ancestors; reject cycles."""
    seen: Dict[int, int] = {}

    def walk(parent_id: Optional[str], depth: int) -> None:
        if depth > max_depth:
            raise ManifestError(f"nesting exceeds {max_depth} levels")
        for child in children.get(parent_id, []):
            cid = id(child)
            if cid in seen:
                raise ManifestError("cycle in layer parent chain")
            seen[cid] = depth
            walk(child["id"], depth + 1)

    walk(None, 0)


def _check_mask_chains(layers: List[Dict[str, Any]], max_chain: int) -> None:
    by_id = {layer["id"]: layer for layer in layers}
    for layer in layers:
        source_id = layer.get("maskSourceID")
        if not source_id:
            continue
        # Self-link.
        if source_id == layer["id"]:
            raise ManifestError(f"layer {layer['id']} has self-referencing maskSourceID")
        # Walk the chain.
        seen: set[str] = set()
        current = source_id
        steps = 0
        while current is not None:
            steps += 1
            if steps > max_chain:
                raise ManifestError(f"maskSourceID chain exceeds {max_chain} nodes")
            if current in seen:
                raise ManifestError("cycle in maskSourceID chain")
            seen.add(current)
            node = by_id.get(current)
            if node is None:
                raise ManifestError(f"maskSourceID {current!r} references missing layer")
            if node.get("isGroup"):
                raise ManifestError(f"maskSourceID {current!r} points to a group")
            current = node.get("maskSourceID")
