"""Project-store tests: atomic write, image bookkeeping, round-trip."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compositor.document.editor_session import EditorSession
from compositor.document.layers import ImageLayer, Transform, new_id
from compositor.format.manifest import CURRENT_VERSION, Manifest
from compositor.io.image_importer import encode_png
from compositor.io.project_store import ProjectStore, ProjectStoreError


def _new_project(tmp_path: Path) -> tuple[Path, Manifest]:
    root = tmp_path / "Example.comp"
    manifest = Manifest(
        version=CURRENT_VERSION,
        color_space="sRGB",
        document_id="00000000-0000-0000-0000-000000000001",
        width=8,
        height=8,
        layers=[],
    )
    ProjectStore.create(root, manifest)
    return root, manifest


def test_create_and_open(tmp_path: Path) -> None:
    root, _ = _new_project(tmp_path)
    assert (root / "manifest.json").is_file()
    assert (root / "images").is_dir()
    store = ProjectStore.open(root)
    loaded = store.load_manifest()
    assert loaded.width == 8
    assert loaded.height == 8


def test_create_rejects_non_empty(tmp_path: Path) -> None:
    root = tmp_path / "Busy.comp"
    root.mkdir()
    (root / "stray.txt").write_text("hi")
    with pytest.raises(ProjectStoreError):
        ProjectStore.create(root, Manifest(
            version=CURRENT_VERSION, color_space="sRGB", document_id="DOC1",
            width=8, height=8, layers=[],
        ))


def test_image_roundtrip(tmp_path: Path) -> None:
    root, manifest = _new_project(tmp_path)
    store = ProjectStore.open(root)
    layer_id = new_id()
    arr = np.full((8, 8, 4), 200, dtype=np.uint8)
    arr[..., 3] = 255
    png_bytes = encode_png(arr)
    path = store.write_image(layer_id, png_bytes)
    assert path.is_file()
    assert path.name == f"{layer_id}.png"


def test_remove_orphan_images(tmp_path: Path) -> None:
    root, manifest = _new_project(tmp_path)
    store = ProjectStore.open(root)
    keep = new_id()
    drop = new_id()
    arr = np.full((4, 4, 4), 200, dtype=np.uint8)
    arr[..., 3] = 255
    png_bytes = encode_png(arr)
    store.write_image(keep, png_bytes)
    store.write_image(drop, png_bytes)
    deleted = store.remove_orphan_images([keep])
    assert len(deleted) == 1
    assert store.image_path(keep).exists()
    assert not store.image_path(drop).exists()


def test_open_missing_project(tmp_path: Path) -> None:
    with pytest.raises(ProjectStoreError):
        ProjectStore.open(tmp_path / "Nope.comp")


def test_full_roundtrip(tmp_path: Path) -> None:
    root, _ = _new_project(tmp_path)
    store = ProjectStore.open(root)
    session = EditorSession(width=8, height=8)
    layer = ImageLayer(
        id=new_id(),
        name="Background",
        transform=Transform(size=(8, 8)),
    )
    session.add_layer(layer)
    arr = np.full((8, 8, 4), 200, dtype=np.uint8)
    arr[..., 3] = 255
    store.write_image(layer.id, encode_png(arr))

    # Save manifest and reload.
    manifest = store.load_manifest()
    manifest.layers = session.to_manifest_layers()
    manifest.active_layer_id = layer.id
    store.save_manifest(manifest)

    reloaded = store.load_manifest()
    assert reloaded.layers[0]["name"] == "Background"
    assert store.image_path(layer.id).exists()
