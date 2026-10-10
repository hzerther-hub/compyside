"""Layers panel — full layer-list with visibility, blend mode, opacity, rename, etc."""
from __future__ import annotations

import json
from typing import Callable, Optional

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QAction, QDrag, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import i18n
from ..document.editor_session import EditorSession
from ..document.layers import (
    AdjustmentKind,
    AdjustmentLayer,
    AnyLayer,
    BlendMode,
    GroupLayer,
    ImageLayer,
)
from ..format.manifest import BLEND_MODES

LAYER_MIME = "application/x-compositor-layer"


def tr(text: str) -> str:
    return i18n.tr(text)


class LayersPanel(QWidget):
    """Layers panel with visibility, blend mode, opacity, rename, reorder."""

    selection_changed = Signal(str)
    add_layer_requested = Signal(str)  # "image" | "adjustment" | "group"
    delete_layer_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session: Optional[EditorSession] = None
        self._building = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.list = QListWidget(self)
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.itemSelectionChanged.connect(self._on_selection_changed)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._show_context_menu)
        # Cross-project drag support.
        self.list.setDragEnabled(True)
        self.list.setDragDropMode(QListWidget.DragDropMode.DragDrop)
        self.list.setAcceptDrops(True)
        self.list.startDrag = self._start_drag  # type: ignore[method-assign]
        self.list.dragEnterEvent = self._drag_enter_event  # type: ignore[method-assign]
        self.list.dragMoveEvent = self._drag_enter_event  # type: ignore[method-assign]
        self.list.dropEvent = self._drop_event  # type: ignore[method-assign]
        layout.addWidget(self.list, 1)

        # Inspector.
        inspector = QWidget(self)
        il = QVBoxLayout(inspector)
        il.setContentsMargins(0, 0, 0, 0)

# Blend mode.
        blend_row = QHBoxLayout()
        self.blend_label = QLabel(tr("Blend"))
        blend_row.addWidget(self.blend_label)
        self.blend_combo = QComboBox()
        for mode in BLEND_MODES:
            self.blend_combo.addItem(mode)
        self.blend_combo.currentTextChanged.connect(self._on_blend_changed)
        blend_row.addWidget(self.blend_combo, 1)
        il.addLayout(blend_row)

        # Opacity.
        op_row = QHBoxLayout()
        self.opacity_label = QLabel(tr("Opacity"))
        op_row.addWidget(self.opacity_label)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.valueChanged.connect(self._on_opacity_changed)
        op_row.addWidget(self.opacity_spin, 1)
        il.addLayout(op_row)

        # Visibility.
        vis_row = QHBoxLayout()
        self.visible_btn = QPushButton(tr("Visible"))
        self.visible_btn.setCheckable(True)
        self.visible_btn.toggled.connect(self._on_visibility_toggled)
        vis_row.addWidget(self.visible_btn)
        il.addLayout(vis_row)

        # Rename.
        rename_row = QHBoxLayout()
        self.name_label = QLabel(tr("Name"))
        rename_row.addWidget(self.name_label)
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._on_rename)
        rename_row.addWidget(self.name_edit, 1)
        il.addLayout(rename_row)

        layout.addWidget(inspector)

        # Buttons. Kept as attributes so a language switch can relabel them.
        btn_row = QHBoxLayout()
        self._add_buttons = []
        for key, kind in (
            ("+ Image", "image"),
            ("+ Group", "group"),
            ("+ Adjust", "adjustment"),
            ("Delete", "delete"),
        ):
            b = QPushButton(tr(key))
            self._add_buttons.append((b, key))
            if kind == "delete":
                b.clicked.connect(self._on_delete)
            else:
                b.clicked.connect(lambda _=False, k=kind: self.add_layer_requested.emit(k))
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        # Refresh labels when the language changes.
        i18n.on_change(self._refresh_labels)

    # ---- public API ------------------------------------------------------

    def set_session(self, session: Optional[EditorSession]) -> None:
        self.session = session
        self._refresh()

    # ---- populate --------------------------------------------------------

    def _refresh(self) -> None:
        self._building = True
        self.list.clear()
        if self.session is None:
            self._building = False
            return
        # Top of the list = top of the stack (reverse of the bottom-to-top array).
        for layer in reversed(self.session.layers):
            item = QListWidgetItem(self._label_for(layer))
            item.setData(Qt.ItemDataRole.UserRole, layer.id)
            self.list.addItem(item)
            if layer.id == self.session.active_layer_id:
                self.list.setCurrentItem(item)
        self._refresh_inspector()
        self._building = False

    def _refresh_labels(self, _code: str = "") -> None:
        """Re-render labels after a language switch.

        The static captions are set once in ``__init__``, so switching
        language has to put the new text back — repopulating the list alone
        would leave them in the previous language.
        """
        self.blend_label.setText(tr("Blend"))
        self.opacity_label.setText(tr("Opacity"))
        self.name_label.setText(tr("Name"))
        self.visible_btn.setText(tr("Visible"))
        for button, key in self._add_buttons:
            button.setText(tr(key))
        self._refresh()

    def _refresh_inspector(self) -> None:
        self._building = True
        layer = self._active_layer()
        if layer is None:
            self.blend_combo.setEnabled(False)
            self.opacity_spin.setEnabled(False)
            self.visible_btn.setEnabled(False)
            self.name_edit.setEnabled(False)
            return
        self.blend_combo.setEnabled(True)
        self.opacity_spin.setEnabled(True)
        self.visible_btn.setEnabled(True)
        self.name_edit.setEnabled(True)
        if isinstance(layer, GroupLayer):
            self.blend_combo.setEnabled(False)  # Groups are pass-through.
        idx = self.blend_combo.findText(layer.blend_mode.value)
        if idx >= 0:
            self.blend_combo.setCurrentIndex(idx)
        self.opacity_spin.setValue(layer.opacity)
        self.visible_btn.setChecked(layer.is_visible)
        self.name_edit.setText(layer.name)
        self._building = False

    def _active_layer(self) -> Optional[AnyLayer]:
        if self.session is None:
            return None
        return self.session.layer_by_id(self.session.active_layer_id)

    def _label_for(self, layer: AnyLayer) -> str:
        """Build the list-item label.

        Adjustment kinds show translated + English (e.g. "色阶 (Levels)" or
        just "Levels" when the UI is in English). Image and group layers use
        their user-supplied name.
        """
        if isinstance(layer, AdjustmentLayer):
            return f"⚙ {i18n.display_name(layer.kind.value)}"
        if isinstance(layer, GroupLayer):
            return f"📁 {layer.name}"
        opacity = int(round(layer.opacity * 100))
        return f"🖼 {layer.name}  ·  {opacity}%"

    # ---- events ----------------------------------------------------------

    def _on_selection_changed(self) -> None:
        if self._building or self.session is None:
            return
        item = self.list.currentItem()
        if item is None:
            return
        layer_id = item.data(Qt.ItemDataRole.UserRole)
        self.session.active_layer_id = layer_id
        self._refresh_inspector()
        self.selection_changed.emit(layer_id)

    def _on_blend_changed(self, mode: str) -> None:
        if self._building:
            return
        layer = self._active_layer()
        if layer is None or isinstance(layer, GroupLayer):
            return
        self.session.set_property(layer.id, "blend_mode", BlendMode(mode))

    def _on_opacity_changed(self, value: float) -> None:
        if self._building:
            return
        layer = self._active_layer()
        if layer is None:
            return
        self.session.set_property(layer.id, "opacity", value)

    def _on_visibility_toggled(self, checked: bool) -> None:
        if self._building:
            return
        layer = self._active_layer()
        if layer is None:
            return
        self.session.set_property(layer.id, "is_visible", checked)

    def _on_rename(self) -> None:
        if self._building:
            return
        layer = self._active_layer()
        if layer is None:
            return
        new_name = self.name_edit.text().strip() or layer.name
        self.session.set_property(layer.id, "name", new_name)
        item = self.list.currentItem()
        if item is not None:
            item.setText(self._label_for(layer))

    def _on_delete(self) -> None:
        layer = self._active_layer()
        if layer is None:
            return
        self.delete_layer_requested.emit(layer.id)

    def _show_context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None or self.session is None:
            return
        layer_id = item.data(Qt.ItemDataRole.UserRole)
        layer = self.session.layer_by_id(layer_id)
        if layer is None:
            return
        menu = QMenu(self)
        duplicate = menu.addAction(tr("Duplicate"))
        rename = menu.addAction(tr("Rename"))
        merge_down = menu.addAction(tr("Merge Down"))
        edit_text = None
        if isinstance(layer, ImageLayer) and hasattr(layer, "_type_settings"):
            edit_text = menu.addAction(tr("Edit Text…"))
        delete = menu.addAction(tr("Delete"))
        chosen = menu.exec(self.list.mapToGlobal(pos))
        if chosen is duplicate:
            self._duplicate(layer)
        elif chosen is rename:
            self.list.editItem(item)
        elif chosen is merge_down:
            self._merge_down(layer)
        elif chosen is edit_text:
            from ..tools.type_tool import open_text_editor
            cache = getattr(self.session, "_image_cache", {})
            open_text_editor(self.window(), layer, cache)
            self._refresh()
        elif chosen is delete:
            self.delete_layer_requested.emit(layer_id)

    def _duplicate(self, layer: AnyLayer) -> None:
        if self.session is None:
            return
        from copy import deepcopy
        new_layer = deepcopy(layer)
        new_layer.id = new_id_safe()
        idx = self.session.layer_index(layer.id)
        self.session.add_layer(new_layer, at_index=idx + 1)

    def _merge_down(self, layer: AnyLayer) -> None:
        if self.session is None:
            return
        idx = self.session.layer_index(layer.id)
        if idx == 0:
            return
        from copy import deepcopy
        below = deepcopy(self.session.layers[idx - 1])
        new_layer = deepcopy(layer)
        new_layer.name = below.name
        below.id = new_id_safe()
        self.session.add_layer(below, at_index=idx)
        self.session.add_layer(new_layer, at_index=idx + 1)
        self.session.remove_layer(layer.id)

    # ---- cross-project drag ---------------------------------------------

    def _drag_enter_event(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasFormat(LAYER_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drop_event(self, event) -> None:  # type: ignore[no-untyped-def]
        if not event.mimeData().hasFormat(LAYER_MIME):
            event.ignore()
            return
        if self.session is None:
            event.ignore()
            return
        try:
            payload = json.loads(bytes(event.mimeData().data(LAYER_MIME)).decode("utf-8"))
        except Exception:
            event.ignore()
            return
        from copy import deepcopy
        from ..document.layers import layer_from_dict, new_id
        raw = payload.get("layer", {})
        new_layer = layer_from_dict(deepcopy(raw))
        new_layer.id = new_id()
        if isinstance(new_layer, ImageLayer):
            new_layer.image_file = f"{new_layer.id}.png"
        # Append to the end of the stack (top).
        self.session.add_layer(new_layer, at_index=len(self.session.layers))
        if isinstance(new_layer, ImageLayer) and "png" in payload:
            from ..io.image_importer import decode_png
            png_bytes = bytes.fromhex(payload["png"])
            cache = getattr(self.session, "_image_cache", {})
            cache[new_layer.id] = decode_png(png_bytes)
        self.session.active_layer_id = new_layer.id
        self.selection_changed.emit(new_layer.id)
        event.acceptProposedAction()

    def _start_drag(self, supported_actions) -> None:  # type: ignore[no-untyped-def]
        if self.session is None:
            return
        layer = self._active_layer()
        if layer is None:
            return
        # Encode the layer's manifest dict + its PNG bytes (if image) into the
        # custom MIME payload as JSON.
        payload: dict = {"layer": layer.to_dict()}
        if isinstance(layer, ImageLayer):
            cache = getattr(self.session, "_image_cache", {})
            if layer.id in cache:
                from ..io.image_importer import encode_png
                payload["png"] = encode_png(cache[layer.id]).hex()
        mime = QMimeData()
        mime.setData(LAYER_MIME, json.dumps(payload).encode("utf-8"))
        mime.setText(layer.name)
        drag = QDrag(self.list)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


def new_id_safe() -> str:
    from ..document.layers import new_id
    return new_id()
