"""Manifest round-trip and validation tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compositor.format.manifest import (
    ADJUSTMENT_KINDS,
    BLEND_MODES,
    CURRENT_VERSION,
    Manifest,
    ManifestError,
    load_manifest,
    save_manifest,
    validate_manifest,
)


def _make_minimal_manifest(width: int = 4, height: int = 3) -> Manifest:
    return Manifest(
        version=CURRENT_VERSION,
        color_space="sRGB",
        document_id="00000000-0000-0000-0000-000000000001",
        width=width,
        height=height,
        layers=[
            {
                "id": "AAAA1111-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
                "name": "Background",
                "isVisible": True,
                "isGroup": False,
                "opacity": 1.0,
                "blendMode": "Normal",
                "transform": {
                    "origin": [0.0, 0.0],
                    "size": [float(width), float(height)],
                    "rotation": 0.0,
                    "flipX": False,
                    "flipY": False,
                    "sampling": "High quality",
                },
                "imageFile": "AAAA1111-AAAA-AAAA-AAAA-AAAAAAAAAAAA.png",
            }
        ],
        active_layer_id="AAAA1111-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
    )


def test_minimal_roundtrip(tmp_path: Path) -> None:
    m = _make_minimal_manifest()
    out = tmp_path / "manifest.json"
    save_manifest(m, out)
    loaded = load_manifest(out)
    assert loaded.width == m.width
    assert loaded.height == m.height
    assert loaded.version == CURRENT_VERSION
    assert loaded.active_layer_id == m.active_layer_id
    assert loaded.layers[0]["name"] == "Background"
    assert loaded.layers[0]["transform"]["size"] == [4, 3]


def test_atomic_write_replaces_file(tmp_path: Path) -> None:
    m = _make_minimal_manifest()
    out = tmp_path / "manifest.json"
    save_manifest(m, out)
    first_bytes = out.read_bytes()

    m.layers[0]["name"] = "Renamed"
    save_manifest(m, out)
    second_bytes = out.read_bytes()
    assert second_bytes != first_bytes

    loaded = load_manifest(out)
    assert loaded.layers[0]["name"] == "Renamed"


def test_unknown_version_rejected(tmp_path: Path) -> None:
    bad = _make_minimal_manifest().to_dict()
    bad["version"] = 99
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(out)


def test_duplicate_layer_ids_rejected(tmp_path: Path) -> None:
    m = _make_minimal_manifest()
    m.layers.append(copy.deepcopy(m.layers[0]))
    with pytest.raises(ManifestError):
        save_manifest(m, tmp_path / "manifest.json")


def test_group_with_image_rejected() -> None:
    m = _make_minimal_manifest()
    m.layers.append({
        "id": "BBBB1111-BBBB-BBBB-BBBB-BBBBBBBBBBBB",
        "name": "Folder",
        "isVisible": True,
        "isGroup": True,
        "transform": {
            "origin": [0.0, 0.0],
            "size": [4.0, 3.0],
            "rotation": 0.0,
            "flipX": False,
            "flipY": False,
            "sampling": "High quality",
        },
        "imageFile": "should-not-be-here.png",
    })
    with pytest.raises(ManifestError):
        validate_manifest(m.to_dict())


def test_adjustment_layer_roundtrip() -> None:
    m = _make_minimal_manifest()
    m.layers.append({
        "id": "CCCC1111-CCCC-CCCC-CCCC-CCCCCCCCCCCC",
        "name": "Warm Grade",
        "isVisible": True,
        "isGroup": False,
        "opacity": 1.0,
        "blendMode": "Normal",
        "transform": {
            "origin": [0.0, 0.0],
            "size": [4.0, 3.0],
            "rotation": 0.0,
            "flipX": False,
            "flipY": False,
            "sampling": "High quality",
        },
        "adjustment": {"kind": "Curves"},
    })
    payload = m.to_dict()
    validate_manifest(payload)
    m.layers[0]["adjustment"] = {"kind": "Hue/Saturation", "hue": 30, "saturation": 10}
    with pytest.raises(ManifestError):
        validate_manifest(m.to_dict())  # first layer can't be an adjustment w/o imageFile


def test_canonical_blend_modes_present() -> None:
    # The spec calls for exactly these spellings.
    assert "Linear Dodge (Add)" in BLEND_MODES
    assert "Black & White" not in BLEND_MODES  # that's an adjustment kind
    assert len(BLEND_MODES) == 24
    # 12 original kinds + 7 new (Vignette, Bloom, Dither, Scanlines, Tonal Contrast, Lens Correction, Remove Background).
    assert len(ADJUSTMENT_KINDS) == 19
    for k in (
        "Vignette", "Bloom", "Dither", "Scanlines", "Tonal Contrast",
        "Lens Correction", "Remove Background",
    ):
        assert k in ADJUSTMENT_KINDS


def test_nesting_limit_enforced() -> None:
    m = _make_minimal_manifest()
    # Build a chain 65 deep.
    parent_id = None
    last_layer = None
    layers = []
    for i in range(65):
        lid = f"{i:08d}-0000-0000-0000-000000000000"
        layers.append({
            "id": lid,
            "name": f"L{i}",
            "isVisible": True,
            "isGroup": True,
            "parentID": parent_id,
            "transform": {
                "origin": [0.0, 0.0],
                "size": [4.0, 3.0],
                "rotation": 0.0,
                "flipX": False,
                "flipY": False,
                "sampling": "High quality",
            },
        })
        parent_id = lid
    layers.append(copy.deepcopy(m.layers[0]))
    layers[-1]["parentID"] = parent_id
    m.layers = layers
    with pytest.raises(ManifestError):
        validate_manifest(m.to_dict())
