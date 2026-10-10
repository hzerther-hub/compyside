"""AI backend settings dialog.

Persists the user's CPU/GPU/auto choice in ``QSettings`` so the next launch
remembers. The dialog also surfaces a status table showing every backend,
whether it's installed, and which device it'll use under the current mode.

On ``OK`` we call :func:`compositor.rendering.ai_backends.set_execution_mode`,
which invalidates the per-device model caches so the next call loads under
the new rule.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import i18n
from ..rendering import ai_backends


SETTINGS_ORG = "comPySide"
SETTINGS_APP = "Compositor"
SETTINGS_KEY_MODE = "ai/execution_mode"


def _load_mode() -> str:
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    val = settings.value(SETTINGS_KEY_MODE, "auto")
    if isinstance(val, bytes):
        val = val.decode("utf-8", "ignore")
    return str(val).lower()


def _save_mode(mode: str) -> None:
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    settings.setValue(SETTINGS_KEY_MODE, mode)


def restore_mode() -> None:
    """Apply the persisted mode at startup so the first call uses the right device."""
    ai_backends.set_execution_mode(_load_mode())


class AISettingsDialog(QDialog):
    """Configure AI execution mode and inspect backend status."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(i18n.tr("AI Backend Settings"))
        self.resize(620, 420)

        layout = QVBoxLayout(self)

        # Mode chooser.
        form = QFormLayout()
        self.mode_combo = QComboBox()
        for label, value in (
            ("Auto — pick GPU when available, otherwise CPU", "auto"),
            ("Force CPU", "cpu"),
            ("Force GPU", "gpu"),
        ):
            self.mode_combo.addItem(label, value)
        current = _load_mode()
        for i in range(self.mode_combo.count()):
            if self.mode_combo.itemData(i) == current:
                self.mode_combo.setCurrentIndex(i)
                break
        form.addRow("Execution mode", self.mode_combo)
        layout.addLayout(form)

        # Status table.
        status_label = QLabel("Backend status")
        f = QFont()
        f.setBold(True)
        status_label.setFont(f)
        layout.addWidget(status_label)

        self.status_table = QTableWidget(0, 4, self)
        self.status_table.setHorizontalHeaderLabels(["Backend", "Status", "Device", "Notes"])
        self.status_table.horizontalHeader().setStretchLastSection(True)
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.status_table, 1)

        # Hint label.
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #666;")
        layout.addWidget(self.hint_label)

        # Buttons: Refresh + OK + Cancel.
        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_status)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

        self._refresh_status()

    def _refresh_status(self) -> None:
        diag = ai_backends.diagnose()
        # Reflect the combo's currently-displayed mode in the diagnostic.
        current = self.mode_combo.currentData()
        if current and current != diag.global_mode:
            ai_backends.set_execution_mode(current)
            diag = ai_backends.diagnose()

        self.status_table.setRowCount(len(diag.backends))
        for row, status in enumerate(diag.backends):
            name_item = QTableWidgetItem(status.name)
            status_item = QTableWidgetItem(
                "✓ " + status.reason if status.available else "✗ " + status.reason
            )
            device_item = QTableWidgetItem(status.device)
            notes_item = QTableWidgetItem(status.notes)

            if not status.available:
                for col in range(4):
                    item = self.status_table.item(row, col)
                    if item is None:
                        continue
                    item.setForeground(Qt.GlobalColor.gray)
            elif status.notes:
                notes_item.setForeground(Qt.GlobalColor.darkYellow)

            self.status_table.setItem(row, 0, name_item)
            self.status_table.setItem(row, 1, status_item)
            self.status_table.setItem(row, 2, device_item)
            self.status_table.setItem(row, 3, notes_item)
        self.status_table.resizeColumnsToContents()

        if diag.gpu_available:
            gpu_hint = "GPU acceleration is available on this machine."
        else:
            gpu_hint = (
                "No CUDA GPU detected — every backend will run on CPU. "
                "Install onnxruntime-gpu / a CUDA-enabled torch wheel to unlock GPU."
            )
        self.hint_label.setText(
            f"Global mode: {diag.global_mode}.  {gpu_hint}"
        )

    def _on_accept(self) -> None:
        mode = self.mode_combo.currentData()
        if mode:
            _save_mode(mode)
            ai_backends.set_execution_mode(mode)
        self.accept()
