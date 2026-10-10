"""Keyboard shortcut registry — JSON-driven, remappable from a dialog."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


DEFAULT_SHORTCUTS = {
    "File > Open": "Ctrl+O",
    "File > Save": "Ctrl+S",
    "File > Quit": "Ctrl+Q",
    "Edit > Undo": "Ctrl+Z",
    "Edit > Redo": "Ctrl+Shift+Z",
    "Edit > Copy Merged": "Ctrl+Shift+C",
    "Image > Canvas Size…": "Ctrl+Alt+C",
    "Image > Image Size…": "Ctrl+Alt+I",
    "Layer > New Layer": "Ctrl+Shift+N",
    "Layer > Duplicate Layer": "Ctrl+J",
    "Layer > Delete Layer": "Ctrl+Shift+Delete",
    "Layer > Merge Down": "Ctrl+E",
    "Layer > Flip Horizontal": "Ctrl+Shift+H",
    "Select > All": "Ctrl+A",
    "Select > Deselect": "Ctrl+D",
    "Select > Inverse": "Ctrl+Shift+I",
    "Filter > Last Filter": "Ctrl+Alt+F",
    "View > Zoom In": "Ctrl+=",
    "View > Zoom Out": "Ctrl+-",
    "View > Fit on Screen": "Ctrl+0",
    "View > Toggle Fullscreen": "F",
    "View > Toggle Rulers": "Ctrl+R",
    "View > Toggle Grid": "Ctrl+'",
    "Tools > Move": "V",
    "Tools > Brush": "B",
    "Tools > Eraser": "E",
    "Tools > Gradient": "G",
    "Tools > Crop": "C",
    "Tools > Eyedropper": "I",
    "Tools > Rectangle Marquee": "M",
    "Tools > Ellipse Marquee": "Shift+M",
    "Tools > Lasso": "L",
    "Tools > Magic Wand": "W",
    "Tools > Type": "T",
    "Tools > Shape": "U",
    "Tools > Clone Stamp": "S",
    "Tools > Blur": "Ctrl+B",
    "Tools > Heal": "J",
    "Help > Commands": "Ctrl+F",
}


class KeyboardShortcuts:
    """JSON-backed shortcut map."""

    def __init__(self, path: Optional[Path] = None):
        self._shortcuts: Dict[str, str] = dict(DEFAULT_SHORTCUTS)
        self._path = path
        if path is not None and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._shortcuts.update(data)
            except Exception:
                pass

    def get(self, action: str) -> Optional[str]:
        return self._shortcuts.get(action)

    def set(self, action: str, sequence: str) -> None:
        self._shortcuts[action] = sequence

    def save(self) -> None:
        if self._path is None:
            return
        self._path.write_text(json.dumps(self._shortcuts, indent=2), encoding="utf-8")

    @property
    def all(self) -> Dict[str, str]:
        return dict(self._shortcuts)


class KeyboardShortcutsDialog(QDialog):
    """Lets the user re-bind every shortcut in ``DEFAULT_SHORTCUTS``."""

    def __init__(self, parent: Optional[QWidget] = None, shortcuts: Optional[KeyboardShortcuts] = None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(480, 480)
        self._shortcuts = shortcuts or KeyboardShortcuts()

        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        inner = QWidget(scroll)
        inner_layout = QVBoxLayout(inner)

        self.table = QTableWidget(0, 2, inner)
        self.table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        inner_layout.addWidget(self.table)

        for action, sequence in self._shortcuts.all.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(action))
            edit = QKeySequenceEdit(QKeySequence(sequence))
            edit.setObjectName(action)
            self.table.setCellWidget(row, 1, edit)

        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        for row in range(self.table.rowCount()):
            action_item = self.table.item(row, 0)
            edit = self.table.cellWidget(row, 1)
            if action_item is None or edit is None:
                continue
            action = action_item.text()
            sequence = edit.keySequence().toString()
            self._shortcuts.set(action, sequence)
        self._shortcuts.save()
        self.accept()


def install_shortcut(action, sequence: str, target: Callable[[], None], parent: Optional[QWidget] = None):
    """Helper to bind a QAction with a parsed key sequence."""
    from PySide6.QtGui import QAction
    act = QAction(parent)
    act.setShortcut(QKeySequence(sequence))
    act.triggered.connect(target)
    if action is not None:
        action.addAction(act)
    return act
