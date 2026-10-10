"""Transform Inspector — exact X / Y / W / H / rotation for the active layer."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import i18n
from ..document.editor_session import EditorSession
from ..document.layers import (
    AdjustmentLayer,
    AnyLayer,
    GroupLayer,
    ImageLayer,
    Transform,
)


def tr(text: str) -> str:
    return i18n.tr(text)


class TransformInspector(QWidget):
    """Live-edit the active layer's position / size / rotation / flip."""

    value_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session: Optional[EditorSession] = None
        self._building = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.title_label = QLabel(tr("Transform"))
        layout.addWidget(self.title_label)

        form = QFormLayout()

        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-30000, 30000)
        self.x_spin.setSingleStep(1.0)
        self.x_spin.valueChanged.connect(self._on_x_changed)
        self.x_label = QLabel(tr("X"))
        form.addRow(self.x_label, self.x_spin)

        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-30000, 30000)
        self.y_spin.setSingleStep(1.0)
        self.y_spin.valueChanged.connect(self._on_y_changed)
        self.y_label = QLabel(tr("Y"))
        form.addRow(self.y_label, self.y_spin)

        self.w_spin = QDoubleSpinBox()
        self.w_spin.setRange(1, 30000)
        self.w_spin.setSingleStep(1.0)
        self.w_spin.valueChanged.connect(self._on_w_changed)
        self.w_label = QLabel(tr("Width"))
        form.addRow(self.w_label, self.w_spin)

        self.h_spin = QDoubleSpinBox()
        self.h_spin.setRange(1, 30000)
        self.h_spin.setSingleStep(1.0)
        self.h_spin.valueChanged.connect(self._on_h_changed)
        self.h_label = QLabel(tr("Height"))
        form.addRow(self.h_label, self.h_spin)

        self.rot_spin = QDoubleSpinBox()
        self.rot_spin.setRange(-360, 360)
        self.rot_spin.setSingleStep(1.0)
        self.rot_spin.setSuffix("°")
        self.rot_spin.valueChanged.connect(self._on_rotation_changed)
        self.rot_label = QLabel(tr("Rotation"))
        form.addRow(self.rot_label, self.rot_spin)

        layout.addLayout(form)

        # Flip buttons.
        flip_row = QHBoxLayout()
        self.flip_h_btn = QPushButton(tr("Flip Horizontal"))
        self.flip_h_btn.clicked.connect(self._on_flip_h)
        flip_row.addWidget(self.flip_h_btn)
        self.flip_v_btn = QPushButton(tr("Flip Vertical"))
        self.flip_v_btn.clicked.connect(self._on_flip_v)
        flip_row.addWidget(self.flip_v_btn)
        layout.addLayout(flip_row)

        # Refresh on language change.
        i18n.on_change(self._refresh_labels)

    # ---- public ---------------------------------------------------------

    def set_session(self, session: Optional[EditorSession]) -> None:
        self.session = session
        self._refresh()

    # ---- internals ------------------------------------------------------

    def _active_layer(self) -> Optional[AnyLayer]:
        if self.session is None:
            return None
        return self.session.layer_by_id(self.session.active_layer_id)

    def _refresh(self, *_args) -> None:
        self._building = True
        layer = self._active_layer()
        has = layer is not None and not isinstance(layer, (GroupLayer, AdjustmentLayer))
        for w in (self.x_spin, self.y_spin, self.w_spin, self.h_spin,
                  self.rot_spin, self.flip_h_btn, self.flip_v_btn):
            w.setEnabled(has)
        if not has or layer is None:
            self._building = False
            return
        t = layer.transform
        self.x_spin.setValue(t.origin[0])
        self.y_spin.setValue(t.origin[1])
        self.w_spin.setValue(t.size[0])
        self.h_spin.setValue(t.size[1])
        self.rot_spin.setValue(t.rotation)
        self._building = False

    def _refresh_labels(self, _code: str = "") -> None:
        """Put the static captions back in the new language.

        ``QFormLayout`` keeps the label widgets, so they are relabelled
        here rather than rebuilding the form.
        """
        self.title_label.setText(tr("Transform"))
        self.x_label.setText(tr("X"))
        self.y_label.setText(tr("Y"))
        self.w_label.setText(tr("Width"))
        self.h_label.setText(tr("Height"))
        self.rot_label.setText(tr("Rotation"))
        self.flip_h_btn.setText(tr("Flip Horizontal"))
        self.flip_v_btn.setText(tr("Flip Vertical"))
        self._refresh()

    def _set_transform_attr(self, attr: str, value) -> None:
        if self._building:
            return
        layer = self._active_layer()
        if layer is None or isinstance(layer, (GroupLayer, AdjustmentLayer)):
            return
        t = layer.transform
        if attr == "origin":
            new_t = Transform(origin=value, size=t.size, rotation=t.rotation,
                              flip_x=t.flip_x, flip_y=t.flip_y, sampling=t.sampling)
        elif attr == "size":
            new_t = Transform(origin=t.origin, size=value, rotation=t.rotation,
                              flip_x=t.flip_x, flip_y=t.flip_y, sampling=t.sampling)
        elif attr == "rotation":
            new_t = Transform(origin=t.origin, size=t.size, rotation=value,
                              flip_x=t.flip_x, flip_y=t.flip_y, sampling=t.sampling)
        else:
            return
        self.session.set_property(layer.id, "transform", new_t)
        self.value_changed.emit()

    def _on_x_changed(self, v: float):
        layer = self._active_layer()
        if layer is None:
            return
        ox, oy = layer.transform.origin
        self._set_transform_attr("origin", (v, oy))

    def _on_y_changed(self, v: float):
        layer = self._active_layer()
        if layer is None:
            return
        ox, oy = layer.transform.origin
        self._set_transform_attr("origin", (ox, v))

    def _on_w_changed(self, v: float):
        layer = self._active_layer()
        if layer is None:
            return
        w, h = layer.transform.size
        self._set_transform_attr("size", (v, h))

    def _on_h_changed(self, v: float):
        layer = self._active_layer()
        if layer is None:
            return
        w, h = layer.transform.size
        self._set_transform_attr("size", (w, v))

    def _on_rotation_changed(self, v: float):
        self._set_transform_attr("rotation", v)

    def _on_flip_h(self):
        layer = self._active_layer()
        if layer is None:
            return
        self.session.set_property(layer.id, "transform", Transform(
            origin=layer.transform.origin, size=layer.transform.size,
            rotation=layer.transform.rotation,
            flip_x=not layer.transform.flip_x, flip_y=layer.transform.flip_y,
            sampling=layer.transform.sampling,
        ))

    def _on_flip_v(self):
        layer = self._active_layer()
        if layer is None:
            return
        self.session.set_property(layer.id, "transform", Transform(
            origin=layer.transform.origin, size=layer.transform.size,
            rotation=layer.transform.rotation,
            flip_x=layer.transform.flip_x, flip_y=not layer.transform.flip_y,
            sampling=layer.transform.sampling,
        ))
