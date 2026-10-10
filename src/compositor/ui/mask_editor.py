"""Mask editing UI — paint on / invert / feather / apply the active layer's mask."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import i18n
from ..document.editor_session import EditorSession


def tr(text: str) -> str:
    return i18n.tr(text)


class MaskEditor(QWidget):
    """A small panel that lets the user invert / clear / feather / re-enable the mask."""

    action_requested = Signal(str)  # "invert" | "clear" | "feather" | "apply" | "disable"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session: Optional[EditorSession] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._title = QLabel(tr("Mask"))
        layout.addWidget(self._title)

        self.enabled_checkbox = QCheckBox(tr("Mask enabled"))
        self.enabled_checkbox.toggled.connect(lambda _: self.action_requested.emit("toggle_enabled"))
        layout.addWidget(self.enabled_checkbox)

        row = QHBoxLayout()
        self.invert_btn = QPushButton(tr("Invert"))
        self.invert_btn.clicked.connect(lambda: self.action_requested.emit("invert"))
        row.addWidget(self.invert_btn)
        self.feather_btn = QPushButton(tr("Feather…"))
        self.feather_btn.clicked.connect(lambda: self.action_requested.emit("feather"))
        row.addWidget(self.feather_btn)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        self.clear_btn = QPushButton(tr("Clear"))
        self.clear_btn.clicked.connect(lambda: self.action_requested.emit("clear"))
        row2.addWidget(self.clear_btn)
        self.apply_btn = QPushButton(tr("Apply"))
        self.apply_btn.clicked.connect(lambda: self.action_requested.emit("apply"))
        row2.addWidget(self.apply_btn)
        layout.addLayout(row2)

        i18n.on_change(self._refresh_labels)

    def set_session(self, session: Optional[EditorSession]) -> None:
        self.session = session
        self._refresh()

    def _refresh(self, *_args) -> None:
        if self.session is None:
            return
        active = self.session.layer_by_id(self.session.active_layer_id)
        if active is None or not hasattr(active, "mask_enabled"):
            self.setEnabled(False)
            return
        self.setEnabled(True)
        self.enabled_checkbox.setChecked(active.mask_enabled)

    def _refresh_labels(self, _code: str = "") -> None:
        self._title.setText(tr("Mask"))
        self.enabled_checkbox.setText(tr("Mask enabled"))
        self.invert_btn.setText(tr("Invert"))
        self.feather_btn.setText(tr("Feather…"))
        self.clear_btn.setText(tr("Clear"))
        self.apply_btn.setText(tr("Apply"))
