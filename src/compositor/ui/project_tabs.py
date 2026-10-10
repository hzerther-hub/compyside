"""Multi-project tab widget."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QPushButton,
    QTabWidget,
    QToolButton,
    QWidget,
)

from ..document.editor_session import EditorSession


class ProjectTabs(QTabWidget):
    """Holds one widget per open project. Switching tabs swaps the canvas."""

    project_changed = Signal(int)  # index

    def __init__(self, parent: Optional[QWidget] = None, on_close: Optional[Callable[[int], None]] = None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabCloseRequested.connect(self._on_close)
        self.currentChanged.connect(self._on_change)
        self._on_close_callback = on_close

    def add_project(self, label: str, widget: QWidget) -> int:
        idx = self.addTab(widget, label)
        self.setCurrentIndex(idx)
        return idx

    def remove_project(self, index: int) -> None:
        self.removeTab(index)

    def widget_for(self, index: int) -> Optional[QWidget]:
        return self.widget(index)

    def _on_close(self, index: int) -> None:
        if self._on_close_callback is not None:
            self._on_close_callback(index)
        else:
            self.removeTab(index)

    def _on_change(self, index: int) -> None:
        self.project_changed.emit(index)
