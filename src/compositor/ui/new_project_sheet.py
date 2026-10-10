"""New-project sheet — canvas size in one dialog.

``_new_project`` used to ask for width, then ask for height, in two modal
round trips, so cancelling the second threw the first away. Width and
height belong on screen together, with a link so a square stays square.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from .. import i18n


def tr(text: str) -> str:
    return i18n.tr(text)


# The manifest caps width and height at 30000.
MAX_DIMENSION = 30000
DEFAULT_SIZE = 1024


class NewProjectSheet(QDialog):
    """Pick the canvas size."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("New Project"))

        form = QFormLayout()

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, MAX_DIMENSION)
        self.width_spin.setValue(int(width))
        form.addRow(tr("Width"), self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, MAX_DIMENSION)
        self.height_spin.setValue(int(height))
        form.addRow(tr("Height"), self.height_spin)

        self.link_check = QCheckBox(tr("Lock aspect ratio"))
        self.link_check.setChecked(False)
        form.addRow("", self.link_check)

        # Set while the link is writing into the other spin box, so the
        # resulting signal doesn't bounce straight back.
        self._linking = False

        layout = QHBoxLayout(self)
        layout.addLayout(form)

        self.preset_label = QLabel(self._preset_text())
        layout.addWidget(self.preset_label)
        layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Whichever field is edited drives the other while the link is on.
        self.width_spin.valueChanged.connect(self._on_width_changed)
        self.height_spin.valueChanged.connect(self._on_height_changed)
        self.link_check.toggled.connect(self._on_link_toggled)

        i18n.on_change(self._refresh_labels)
        self._refresh_labels()

    # ---- values ---------------------------------------------------------

    def size(self) -> Tuple[int, int]:
        return int(self.width_spin.value()), int(self.height_spin.value())

    # ---- link ------------------------------------------------------------

    def _on_width_changed(self, value: int) -> None:
        if self.link_check.isChecked() and not self._linking:
            self._linking = True
            self.height_spin.setValue(value)
            self._linking = False
        self._refresh_preset()

    def _on_height_changed(self, value: int) -> None:
        if self.link_check.isChecked() and not self._linking:
            self._linking = True
            self.width_spin.setValue(value)
            self._linking = False
        self._refresh_preset()

    def _on_link_toggled(self, checked: bool) -> None:
        if checked:
            # Start the square from whichever side the user left alone.
            self.width_spin.setValue(self.height_spin.value())
        self._refresh_preset()

    # ---- labels ----------------------------------------------------------

    def _preset_text(self) -> str:
        width, height = self.size()
        return tr("Canvas {w} × {h}").format(w=width, h=height)

    def _refresh_preset(self) -> None:
        self.preset_label.setText(self._preset_text())

    def _refresh_labels(self, _code: str = "") -> None:
        self.setWindowTitle(tr("New Project"))
        self.link_check.setText(tr("Lock aspect ratio"))
        self._refresh_preset()

