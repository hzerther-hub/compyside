"""Brush controls panel."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .. import i18n
from ..rendering import brush as brush_engine


def tr(text: str) -> str:
    return i18n.tr(text)


class BrushControls(QWidget):
    """Adjusts the active brush tool's settings."""

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._tool: Optional[brush_engine.BrushSettings] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._title = QLabel(tr("Brush"))
        layout.addWidget(self._title)

        # Mode.
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([tr("Paint"), tr("Erase")])
        self.mode_combo.currentTextChanged.connect(self._emit)
        layout.addWidget(self.mode_combo)

        # Size.
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(1, 500)
        self.size_spin.setValue(24)
        self.size_spin.valueChanged.connect(self._emit)
        layout.addLayout(_row("Size", self.size_spin))

        # Hardness.
        self.hardness_spin = QDoubleSpinBox()
        self.hardness_spin.setRange(0.0, 1.0)
        self.hardness_spin.setSingleStep(0.05)
        self.hardness_spin.setValue(0.8)
        self.hardness_spin.valueChanged.connect(self._emit)
        layout.addLayout(_row("Hardness", self.hardness_spin))

        # Opacity.
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(1.0)
        self.opacity_spin.valueChanged.connect(self._emit)
        layout.addLayout(_row("Opacity", self.opacity_spin))

        # Flow.
        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.0, 1.0)
        self.flow_spin.setSingleStep(0.05)
        self.flow_spin.setValue(1.0)
        self.flow_spin.valueChanged.connect(self._emit)
        layout.addLayout(_row("Flow", self.flow_spin))

        # Spacing.
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.01, 2.0)
        self.spacing_spin.setSingleStep(0.05)
        self.spacing_spin.setValue(0.1)
        self.spacing_spin.valueChanged.connect(self._emit)
        layout.addLayout(_row("Spacing", self.spacing_spin))

        # Smoothing.
        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0.0, 1.0)
        self.smoothing_spin.setSingleStep(0.05)
        self.smoothing_spin.setValue(0.5)
        self.smoothing_spin.valueChanged.connect(self._emit)
        layout.addLayout(_row("Smoothing", self.smoothing_spin))

        # Refresh labels when the language changes.
        i18n.on_change(self._refresh_labels)

    def _refresh_labels(self, _code: str) -> None:
        self._title.setText(tr("Brush"))
        self.mode_combo.clear()
        self.mode_combo.addItems([tr("Paint"), tr("Erase")])

    def bind(self, tool) -> None:
        if tool is None:
            return
        self._tool = tool.settings
        self.size_spin.setValue(tool.settings.size)
        self.hardness_spin.setValue(tool.settings.hardness)
        self.opacity_spin.setValue(tool.settings.opacity)
        self.flow_spin.setValue(tool.settings.flow)
        self.spacing_spin.setValue(tool.settings.spacing)
        self.smoothing_spin.setValue(tool.settings.smoothing)
        self.mode_combo.setCurrentText("Erase" if tool.settings.mode == "erase" else "Paint")

    def _emit(self) -> None:
        if self._tool is None:
            return
        self._tool.size = int(self.size_spin.value())
        self._tool.hardness = self.hardness_spin.value()
        self._tool.opacity = self.opacity_spin.value()
        self._tool.flow = self.flow_spin.value()
        self._tool.spacing = self.spacing_spin.value()
        self._tool.smoothing = self.smoothing_spin.value()
        self._tool.mode = "erase" if self.mode_combo.currentText() == "Erase" else "paint"
        self.changed.emit()


def _row(label: str, widget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(QLabel(tr(label)))
    row.addWidget(widget, 1)
    return row
