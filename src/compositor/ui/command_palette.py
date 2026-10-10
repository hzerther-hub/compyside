"""Command palette — fuzzy-search every menu action by name (⌘F)."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class CommandPaletteDialog(QDialog):
    """Press Return to run the highlighted command."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        commands: Optional[List[Tuple[str, Callable[[], None]]]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Commands")
        self.resize(480, 360)
        self._commands = commands or []
        layout = QVBoxLayout(self)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Type a command…")
        self.search.textChanged.connect(self._refresh)
        self.search.returnPressed.connect(self._run_selected)
        layout.addWidget(self.search)

        self.list = QListWidget(self)
        self.list.itemActivated.connect(lambda _: self._run_selected())
        self.list.itemSelectionChanged.connect(lambda: None)
        layout.addWidget(self.list, 1)

        self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        query = self.search.text().lower().strip()
        scored = []
        for name, fn in self._commands:
            score = _fuzzy_score(query, name.lower())
            if query == "" or score > 0:
                scored.append((score, name, fn))
        scored.sort(key=lambda x: -x[0])
        for score, name, fn in scored[:20]:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, fn)
            self.list.addItem(item)
        if self.list.count() > 0:
            self.list.setCurrentRow(0)

    def _run_selected(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        fn = item.data(Qt.ItemDataRole.UserRole)
        if fn is not None:
            self.accept()
            fn()


def _fuzzy_score(query: str, target: str) -> int:
    if not query:
        return 1
    if query in target:
        return 100 - target.index(query)
    score = 0
    qi = 0
    for ch in target:
        if qi < len(query) and ch == query[qi]:
            score += 5
            qi += 1
    return score if qi == len(query) else 0
