"""ProjectStore — read/write a .comp project folder.

Implements the atomic write protocol described in
``docs/writing-comp-files.md`` so the running app sees either the old
manifest or the new one, never part of one.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ..format.manifest import (
    CURRENT_VERSION,
    Manifest,
    ManifestError,
    load_manifest,
    save_manifest,
    validate_manifest,
)


class ProjectStoreError(OSError):
    """Raised when a project cannot be read or written."""


@dataclass
class ProjectPaths:
    """Resolved paths for a .comp project."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def images(self) -> List[Path]:
        if not self.images_dir.is_dir():
            return []
        return sorted(self.images_dir.iterdir())

    @property
    def manifest_tmp(self) -> Path:
        # Atomic-write target. Compositor uses the same convention.
        return self.root / ".manifest.json.tmp"


class ProjectStore:
    """Read/write a .comp folder.

    Usage::

        store = ProjectStore.open("/path/to/Example.comp")
        manifest = store.load_manifest()
        ...
        store.save_manifest(manifest)
    """

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    # ---- factories ------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> "ProjectStore":
        root = Path(path)
        if not root.is_dir():
            raise ProjectStoreError(f"project root does not exist: {root}")
        return cls(ProjectPaths(root=root))

    @classmethod
    def create(cls, path: str | Path, manifest: Manifest) -> "ProjectStore":
        root = Path(path)
        if root.exists() and any(root.iterdir()):
            raise ProjectStoreError(f"project root is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(exist_ok=True)
        store = cls(ProjectPaths(root=root))
        store.save_manifest(manifest)
        return store

    # ---- manifest -------------------------------------------------------

    def load_manifest(self) -> Manifest:
        if not self.paths.manifest.is_file():
            raise ProjectStoreError(f"manifest.json missing at {self.paths.manifest}")
        return load_manifest(self.paths.manifest)

    def save_manifest(self, manifest: Manifest) -> None:
        """Atomic save.

        Writes to ``.manifest.json.tmp`` then renames over ``manifest.json``.
        On Windows ``os.replace`` is atomic on the same volume.
        """
        try:
            manifest.version = max(manifest.version, CURRENT_VERSION)
            # Validate before touching the disk.
            validate_manifest(manifest.to_dict())
        except ManifestError as exc:
            raise ProjectStoreError(str(exc)) from exc

        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.images_dir.mkdir(exist_ok=True)

        tmp_path = self.paths.manifest_tmp
        save_manifest(manifest, tmp_path)
        # Cross-volume ``os.replace`` falls back to copy+unlink. That's fine.
        os.replace(tmp_path, self.paths.manifest)

    # ---- image file helpers --------------------------------------------

    def write_image(self, layer_id: str, png_bytes: bytes, *, mask: bool = False) -> Path:
        """Write a layer PNG atomically (write to .tmp, then rename)."""
        if not layer_id:
            raise ProjectStoreError("layer id required")
        filename = f"{layer_id}.mask.png" if mask else f"{layer_id}.png"
        dest = self.paths.images_dir / filename
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(png_bytes)
        os.replace(tmp, dest)
        return dest

    def remove_image(self, layer_id: str, *, mask: bool = False) -> None:
        filename = f"{layer_id}.mask.png" if mask else f"{layer_id}.png"
        target = self.paths.images_dir / filename
        if target.exists():
            target.unlink()

    def image_path(self, layer_id: str, *, mask: bool = False) -> Path:
        filename = f"{layer_id}.mask.png" if mask else f"{layer_id}.png"
        return self.paths.images_dir / filename

    def remove_orphan_images(self, kept_ids: Iterable[str]) -> List[Path]:
        """Delete ``<id>.png`` and ``<id>.mask.png`` files no longer referenced.

        Returns the list of files actually deleted.
        """
        kept = set(kept_ids)
        deleted: List[Path] = []
        if not self.paths.images_dir.is_dir():
            return deleted
        for path in self.paths.images_dir.iterdir():
            if not path.is_file():
                continue
            name = path.name
            if not name.endswith(".png"):
                continue
            stem = name[:-4]
            if stem.endswith(".mask"):
                layer_id = stem[:-5]
            else:
                layer_id = stem
            if layer_id not in kept:
                path.unlink()
                deleted.append(path)
        return deleted
