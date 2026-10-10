"""Drag & drop + file import for the main window.

``DropImportMixin`` owns everything that brings foreign content into the
app: OS file drags (images / PSD / RAW / ``.comp`` folders), raw image
data drags (WeChat / browser without a file URL), and the ``import_*``
helpers the File menu reuses. Split from ``app.py`` to keep the window
class focused on UI wiring.

The mixin expects the host to provide ``open_project()``,
``_current_state``, ``_refresh_tool_panels()`` and the usual QMainWindow
surface.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFileDialog, QMessageBox

from .. import i18n
from ..i18n import tr
from ..document.editor_session import EditorSession
from ..document.layers import ImageLayer, Transform, new_id
from ..format.manifest import CURRENT_VERSION, Manifest
from ..io import image_importer
from ..io.image_importer import encode_png, encode_png_mask
from ..io.project_store import ProjectStore, ProjectStoreError


class DropImportMixin:
    """Mixin for :class:`~compositor.app.MainWindow` — see module docstring."""

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        md = event.mimeData()
        if md.hasUrls() or md.hasImage():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        md = event.mimeData()
        if md.hasUrls() or md.hasImage():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        md = event.mimeData()
        if md.hasUrls():
            for url in md.urls():
                local = url.toLocalFile()
                if not local:
                    continue
                p = Path(local)
                ext = p.suffix.lower()
                if p.is_dir() and (p / "manifest.json").is_file():
                    self.open_project(p)
                elif ext == ".json" and p.name == "manifest.json":
                    self.open_project(p.parent)
                elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".heic", ".heif"):
                    self.import_image(p)
                elif ext in (".psd", ".psb"):
                    self.import_psd(p)
                elif ext in (".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf"):
                    self.import_raw(p)
            event.acceptProposedAction()
            return
        # No file URLs — WeChat / browser drags often carry raw image data.
        if md.hasImage():
            img = QImage(md.imageData())
            if not img.isNull():
                self.import_qimage(img)
                event.acceptProposedAction()

    def import_qimage(self, img: "QImage") -> None:
        """Import a QImage (dragged from another app) as a new layer."""
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        h, w = img.height(), img.width()
        bpl = img.bytesPerLine()
        arr = np.frombuffer(img.constBits(), dtype=np.uint8)
        arr = arr.reshape(h, bpl // 4, 4)[:, :w, :].copy()
        if self._current_state is None:
            self._new_project_for_pixels(arr)
            return
        session = self._current_state.session
        lid = new_id()
        layer = ImageLayer(
            id=lid,
            name="Dropped Image",
            image_file=f"{lid}.png",
            transform=Transform(size=(w, h)),
        )
        session.add_layer(layer)
        if not hasattr(session, "_image_cache"):
            session._image_cache = {}  # type: ignore[attr-defined]
        session._image_cache[layer.id] = arr  # type: ignore[attr-defined]
        session.active_layer_id = layer.id
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()

    def _new_project_for_pixels(self, arr: np.ndarray) -> None:
        """No project open — create one sized to the dropped image."""
        h, w = arr.shape[:2]
        manifest = Manifest(
            version=CURRENT_VERSION,
            color_space="sRGB",
            document_id=new_id(),
            width=w,
            height=h,
            layers=[],
        )
        target = Path.home() / "Compositor Drops" / f"{new_id()}"
        try:
            proj = ProjectStore.create(target, manifest)
        except ProjectStoreError as exc:
            QMessageBox.critical(self, "Cannot create project", str(exc))
            return
        session = EditorSession(width=w, height=h)
        lid = new_id()
        layer = ImageLayer(
            id=lid,
            name="Dropped Image",
            image_file=f"{lid}.png",
            transform=Transform(size=(w, h)),
        )
        session.add_layer(layer)
        session._image_cache = {layer.id: arr}  # type: ignore[attr-defined]
        session.active_layer_id = layer.id
        manifest.layers = session.to_manifest_layers()
        proj.write_image(layer.id, encode_png(arr))
        proj.save_manifest(manifest)
        self.open_project(target)

    def import_psd(self, path: Path) -> None:
        """Open a PSD/PSB as a new .comp project."""
        from ..io.psd_importer import import_psd

        try:
            result = import_psd(path)
        except RuntimeError as exc:
            QMessageBox.critical(self, tr("PSD import failed"), str(exc))
            return
        self.open_imported_project(result, default_stem=path.stem)

    def open_imported_project(self, result: dict, *, default_stem: str) -> bool:
        """Write an ``import_psd`` result out as a new .comp project and open it.

        Shared by the PSD drop and by pulling a document out of Photoshop.
        Returns whether a project was actually created.
        """
        from ..format.manifest import Manifest
        from ..document.editor_session import EditorSession
        from .import_report import show_import_report

        target = QFileDialog.getExistingDirectory(
            self, tr("Choose folder for imported project")
        )
        if not target:
            return False

        target_path = Path(target) / (default_stem + ".comp")
        manifest = Manifest(
            version=CURRENT_VERSION,
            color_space="sRGB",
            document_id=new_id(),
            width=result["width"],
            height=result["height"],
            layers=result["layers"],
        )
        try:
            proj = ProjectStore.create(target_path, manifest)
        except ProjectStoreError as exc:
            QMessageBox.critical(self, tr("Cannot create project"), str(exc))
            return False

        session = EditorSession(width=manifest.width, height=manifest.height)
        session.load_manifest_layers(result["layers"])
        # ``pixels``/``masks`` hold the bitmaps the manifest only names.
        for layer in session.layers:
            if not isinstance(layer, ImageLayer) or not layer.image_file:
                continue
            pixels = result["pixels"].get(layer.id)
            if pixels is not None:
                proj.write_image(layer.id, encode_png(pixels))
            mask = result["masks"].get(layer.id)
            if mask is not None:
                proj.write_image(layer.id, encode_png_mask(mask), mask=True)
        proj.save_manifest(manifest)

        show_import_report(self, title=default_stem, result=result)
        self.open_project(target_path)
        return True

    def import_raw(self, path: Path) -> None:
        """Open a camera RAW file as a new layer."""
        from ..io.raw_importer import import_raw
        try:
            pixels = import_raw(path)
        except RuntimeError as exc:
            QMessageBox.critical(self, "RAW import failed", str(exc))
            return
        self.import_image(path)
        # Replace the just-imported layer with the demosaiced pixels.
        if self._current_state is not None:
            layers = self._current_state.session.layers
            if layers:
                top = layers[-1]
                cache = getattr(self._current_state.session, "_image_cache", {})
                cache[top.id] = pixels
                self._current_state.canvas.set_session(self._current_state.session)

    def import_image(self, path: Path) -> None:
        try:
            image = image_importer.load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, tr("Import failed"), str(exc))
            return
        if self._current_state is None:
            # No project open — dropping a file auto-creates one sized to it.
            self._new_project_for_pixels(image)
            return
        session = self._current_state.session
        h, w = image.shape[:2]
        layer = ImageLayer(
            id=new_id(),
            name=path.name,
            image_file=None,
            transform=Transform(size=(w, h)),
        )
        session.add_layer(layer)
        session._image_cache[layer.id] = image  # type: ignore[attr-defined]
        session.active_layer_id = layer.id
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()
