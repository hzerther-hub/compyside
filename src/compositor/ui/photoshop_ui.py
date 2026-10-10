"""Photoshop bridge UI — connect, push layers out, pull documents in.

The connection itself lives in :mod:`compositor.io.photoshop_bridge`; this
module is the Qt side of it: a connect sheet, a script console, and the
mixin the main window uses to wire up the Photoshop menu.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import i18n
from ..document.editor_session import EditorSession
from ..document.layers import AdjustmentLayer, GroupLayer, ImageLayer, Layer
from ..io.image_importer import encode_png
from ..io.photoshop_bridge import (
    DEFAULT_PORT,
    PhotoshopConnection,
    PhotoshopError,
    script_document_info,
    script_open_document,
    script_place_layers,
    script_save_active_document,
)
from ..rendering.layer_renderer import CompositorOptions, composite_document

_SETTINGS = QSettings("Compositor", "Compositor")


def tr(text: str) -> str:
    return i18n.tr(text)


# ---------------------------------------------------------------------------
# Flattening a document into transferable layer images
# ---------------------------------------------------------------------------


def flatten_layers_for_photoshop(
    session: EditorSession,
    *,
    include_invisible: bool = False,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Render each image layer (and group) to its own transparent canvas.

    Compositor's group tree has no Photoshop equivalent in a plain image
    transfer, so groups become flat layers named ``Group/Layer``. Returns
    ``(entries, skipped)`` where each entry carries the pixels, the crop
    offset, opacity and blend mode the bridge needs.
    """
    cache: Dict[str, np.ndarray] = getattr(session, "_image_cache", {})
    entries: List[Dict[str, object]] = []
    skipped: List[str] = []

    def walk(
        layers: Sequence[Layer], prefix: str, group_opacity: float, *, root: bool = False
    ) -> None:
        for layer in layers:
            # ``session.layers`` is a flat list, so the first pass sees every
            # layer in the document. Anything that belongs to a group is
            # reached through that group instead, or it would be sent twice.
            if root and layer.parent_id is not None:
                continue
            name = f"{prefix}{layer.name}"
            if isinstance(layer, AdjustmentLayer):
                skipped.append(name)
                continue
            if isinstance(layer, GroupLayer):
                walk(
                    layer_group_children(session, layer),
                    f"{name}/",
                    group_opacity * layer.opacity,
                )
                continue
            if not isinstance(layer, ImageLayer):
                continue
            if not layer.is_visible and not include_invisible:
                continue
            rendered = _render_single(
                session, layer, cache, group_opacity, force_visible=include_invisible
            )
            if rendered is None:
                skipped.append(name)
                continue
            rendered["name"] = name
            entries.append(rendered)

    walk(session.layers, "", 1.0, root=True)
    return entries, skipped


def layer_group_children(session: EditorSession, group: GroupLayer) -> Sequence[Layer]:
    """The layers nested directly inside ``group``."""
    return [
        layer
        for layer in session.layers
        if layer.parent_id == group.id
    ]


def _render_single(
    session: EditorSession,
    layer: ImageLayer,
    cache: Dict[str, np.ndarray],
    group_opacity: float,
    force_visible: bool = False,
) -> Optional[Dict[str, object]]:
    if layer.id not in cache:
        return None

    import copy

    flat = copy.copy(layer)
    flat.parent_id = None
    # Group opacity has no Photoshop counterpart once the group is flattened,
    # so it is baked into the pixels. The layer's own opacity is passed
    # through instead — baking that too would apply it twice.
    flat.opacity = max(0.0, min(1.0, group_opacity))
    flat.is_visible = True if force_visible else layer.is_visible
    # Photoshop applies the blend mode itself, against the layers below.
    flat.blend_mode = layer.blend_mode

    holder = SimpleNamespace(
        layers=[flat],
        width=session.width,
        height=session.height,
        _mask_cache=getattr(session, "_mask_cache", {}),
    )
    canvas = composite_document(
        holder,  # type: ignore[arg-type]
        CompositorOptions(
            canvas_width=session.width,
            canvas_height=session.height,
            image_loader=cache.get,
            mask_loader=getattr(session, "_mask_cache", {}).get,
        ),
    )

    pixels, offset = _crop_to_content(canvas)
    if pixels is None:
        return None
    return {
        "name": layer.name,
        "pixels": pixels,
        "offset": offset,
        "opacity": layer.opacity,
        "blend_mode": layer.blend_mode.value,
    }


def _crop_to_content(canvas: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    """Trim fully transparent margins so Photoshop gets a tight layer."""
    alpha = canvas[..., 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return None, (0, 0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    y0, y1 = int(ys[0]), int(ys[-1])
    x0, x1 = int(xs[0]), int(xs[-1])
    return canvas[y0 : y1 + 1, x0 : x1 + 1].copy(), (x0, y0)


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------


class PhotoshopConnectSheet(QDialog):
    """Host / port / password, plus the connection result."""

    def __init__(self, parent: Optional[QWidget] = None, connection: Optional[PhotoshopConnection] = None):
        super().__init__(parent)
        self.setWindowTitle(tr("Connect to Photoshop"))
        self.connection: Optional[PhotoshopConnection] = None

        form = QFormLayout()
        self.host_edit = QLineEdit(str(_SETTINGS.value("photoshop/host", "127.0.0.1")))
        form.addRow(tr("Host"), self.host_edit)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(int(_SETTINGS.value("photoshop/port", DEFAULT_PORT)))
        form.addRow(tr("Port"), self.port_spin)
        self.password_edit = QLineEdit(str(_SETTINGS.value("photoshop/password", "")))
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText(tr("Listener password, if any"))
        form.addRow(tr("Password"), self.password_edit)

        self.remember_check = QCheckBox(tr("Remember password"))
        self.remember_check.setChecked(bool(_SETTINGS.value("photoshop/remember", False, type=bool)))
        form.addRow("", self.remember_check)

        layout = QVBoxLayout(self)
        layout.addLayout(form)

        hint = QLabel(
            tr(
                "Photoshop must be running with Enable scripting turned on "
                "(Edit ▸ Preferences ▸ Plug-ins). The password must match the one "
                "Photoshop's scripting listener uses."
            )
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("Connect"))
        buttons.accepted.connect(self._on_connect)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if connection is not None and connection.is_connected:
            self.status_label.setText(tr("Already connected."))

    def _on_connect(self) -> None:
        host = self.host_edit.text().strip() or "127.0.0.1"
        port = int(self.port_spin.value())
        password = self.password_edit.text()
        connection = PhotoshopConnection(password=password, host=host, port=port)
        try:
            connection.connect()
        except PhotoshopError as exc:
            connection.close()
            self.status_label.setText(tr("Could not connect: {err}").format(err=str(exc)))
            return
        self.connection = connection
        self._persist(host, port, password)
        self.accept()

    def _persist(self, host: str, port: int, password: str) -> None:
        _SETTINGS.setValue("photoshop/host", host)
        _SETTINGS.setValue("photoshop/port", port)
        _SETTINGS.setValue("photoshop/remember", self.remember_check.isChecked())
        _SETTINGS.setValue(
            "photoshop/password", password if self.remember_check.isChecked() else ""
        )


class RunScriptSheet(QDialog):
    """A small ExtendScript console for the connected Photoshop."""

    def __init__(self, parent: Optional[QWidget] = None, connection: Optional[PhotoshopConnection] = None):
        super().__init__(parent)
        self.setWindowTitle(tr("Run ExtendScript"))
        self.connection = connection
        self.resize(640, 480)

        layout = QVBoxLayout(self)
        self.script_edit = QPlainTextEdit()
        self.script_edit.setPlainText(
            "#target photoshop\napp.version;\n"
        )
        layout.addWidget(self.script_edit, 1)

        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        layout.addWidget(QLabel(tr("Result")))
        layout.addWidget(self.output_edit, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("Run"))
        buttons.accepted.connect(self._on_run)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_run(self) -> None:
        if self.connection is None or not self.connection.is_connected:
            self.output_edit.setPlainText(tr("Not connected to Photoshop."))
            return
        try:
            result = self.connection.evaluate(self.script_edit.toPlainText())
        except PhotoshopError as exc:
            self.output_edit.setPlainText(tr("Error: {err}").format(err=str(exc)))
            return
        self.output_edit.setPlainText(result or tr("(no result)"))


# ---------------------------------------------------------------------------
# Main window mixin
# ---------------------------------------------------------------------------


class PhotoshopBridgeMixin:
    """Photoshop menu actions for :class:`~compositor.app.MainWindow`.

    Expects the host to provide ``_current_state``, ``_refresh_tool_panels``
    and the usual QMainWindow surface.
    """

    connection: Optional[PhotoshopConnection] = None

    # ---- connection -----------------------------------------------------

    def _photoshop_connect(self) -> None:
        sheet = PhotoshopConnectSheet(self, self.connection)
        if sheet.exec() and sheet.connection is not None:
            self.connection = sheet.connection
            self.statusBar().showMessage(
                tr("Connected to Photoshop"), 4000
            )
            self._refresh_menu_state()

    def _photoshop_disconnect(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.statusBar().showMessage(tr("Disconnected from Photoshop"), 4000)
        self._refresh_menu_state()

    def _require_connection(self) -> Optional[PhotoshopConnection]:
        if self.connection is not None and self.connection.is_connected:
            return self.connection
        QMessageBox.information(
            self,
            "Photoshop",
            tr("Connect to Photoshop first (File ▸ Connect to Photoshop…)."),
        )
        return None

    # ---- sending --------------------------------------------------------

    def _photoshop_send_merged(self) -> None:
        connection = self._require_connection()
        if connection is None:
            return
        state = getattr(self, "_current_state", None)
        if state is None:
            return
        session = state.session
        cache = getattr(session, "_image_cache", {})
        merged = composite_document(
            session,
            CompositorOptions(
                canvas_width=session.width,
                canvas_height=session.height,
                image_loader=cache.get,
                mask_loader=getattr(session, "_mask_cache", {}).get,
            ),
        )
        with tempfile.TemporaryDirectory(prefix="compositor-ps-") as tmp:
            path = Path(tmp) / f"{_safe_name(session)}.png"
            path.write_bytes(encode_png(merged))
            try:
                connection.evaluate(script_open_document(path))
            except PhotoshopError as exc:
                QMessageBox.critical(self, "Photoshop", str(exc))
                return
        self.statusBar().showMessage(tr("Sent the merged image to Photoshop"), 4000)

    def _photoshop_send_layers(self) -> None:
        connection = self._require_connection()
        if connection is None:
            return
        state = getattr(self, "_current_state", None)
        if state is None:
            return
        entries, skipped = flatten_layers_for_photoshop(state.session)
        if not entries:
            QMessageBox.information(
                self, tr("Send Layers to Photoshop"), tr("This document has no image layers to send.")
            )
            return

        with tempfile.TemporaryDirectory(prefix="compositor-ps-") as tmp:
            paths, names, opacities, modes, offsets = [], [], [], [], []
            for index, entry in enumerate(entries):
                path = Path(tmp) / f"layer-{index:04d}.png"
                path.write_bytes(encode_png(entry["pixels"]))
                paths.append(path)
                names.append(entry["name"])
                opacities.append(entry["opacity"])
                modes.append(entry["blend_mode"])
                offsets.append(entry["offset"])
            script = script_place_layers(
                paths, names, opacities=opacities, blend_modes=modes, offsets=offsets
            )
            try:
                connection.evaluate(script)
            except PhotoshopError as exc:
                QMessageBox.critical(self, "Photoshop", str(exc))
                return

        message = tr("Sent {count} layers to Photoshop").format(count=len(entries))
        if skipped:
            message += " " + tr("Skipped {count}: {names}").format(
                count=len(skipped), names=", ".join(skipped)
            )
        self.statusBar().showMessage(message, 8000)

    # ---- pulling --------------------------------------------------------

    def _photoshop_pull(self) -> None:
        connection = self._require_connection()
        if connection is None:
            return
        from ..io.psd_importer import import_psd

        with tempfile.TemporaryDirectory(prefix="compositor-ps-") as tmp:
            target = Path(tmp) / "from-photoshop.psd"
            try:
                connection.evaluate(script_save_active_document(target))
            except PhotoshopError as exc:
                QMessageBox.critical(self, "Photoshop", str(exc))
                return
            if not target.is_file():
                QMessageBox.warning(
                    self,
                    "Photoshop",
                    tr("Photoshop did not write the document — save it manually first."),
                )
                return
            try:
                result = import_psd(target)
            except RuntimeError as exc:
                QMessageBox.critical(self, tr("PSD import failed"), str(exc))
                return
        self.open_imported_project(result, default_stem="from-photoshop")

    # ---- info -----------------------------------------------------------

    def _photoshop_run_script(self) -> None:
        sheet = RunScriptSheet(self, self.connection)
        sheet.exec()

    def _photoshop_info(self) -> None:
        connection = self._require_connection()
        if connection is None:
            return
        try:
            raw = connection.evaluate(script_document_info())
        except PhotoshopError as exc:
            QMessageBox.critical(self, "Photoshop", str(exc))
            return
        try:
            info = json.loads(raw)
        except ValueError:
            QMessageBox.information(self, tr("Photoshop info"), raw)
            return

        lines = [tr("Photoshop {version} — {count} open document(s)").format(
            version=info.get("photoshop", "?"), count=info.get("documents", 0)
        )]
        if info.get("documents"):
            lines.append(
                tr("Active: {name} — {w}×{h}, {count} layers").format(
                    name=info.get("name", ""),
                    w=info.get("width", 0),
                    h=info.get("height", 0),
                    count=len(info.get("layers", [])),
                )
            )
            for layer in info.get("layers", [])[:20]:
                lines.append(f"  · {layer['name']} ({layer['kind']})")
        QMessageBox.information(self, tr("Photoshop info"), "\n".join(lines))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_name(session: EditorSession) -> str:
    for layer in reversed(session.layers):
        if isinstance(layer, ImageLayer) and layer.name:
            stem = Path(layer.name).stem
            return "".join(ch for ch in stem if ch.isalnum() or ch in "-_ ") or "document"
    return "document"
