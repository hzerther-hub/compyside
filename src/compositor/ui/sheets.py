"""Adjustment sheets — Levels, Curves, Hue/Saturation, Exposure, Color Balance, Invert, B&W."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class _AdjustmentDialog(QDialog):
    """Base class — accepts a settings dict and lets the user mutate it."""

    accepted_with_settings = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, settings: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self._settings: Dict[str, Any] = dict(settings or {})
        self.setLayout(QVBoxLayout())
        body = QWidget(self)
        self.layout().addWidget(body)
        self._build(body)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)

    def _build(self, body: QWidget) -> None:
        raise NotImplementedError

    def _on_accept(self) -> None:
        self.accepted_with_settings.emit(self._settings)
        self.accept()


class LevelsSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        for k in ("inputBlack", "inputWhite", "gamma", "outputBlack", "outputWhite"):
            spin = QDoubleSpinBox()
            spin.setRange(0, 255)
            spin.setValue(self._settings.get(k, 0 if "Black" in k or k == "gamma" else 255))
            if k == "gamma":
                spin.setRange(0.1, 5.0)
            spin.valueChanged.connect(lambda v, key=k: self._settings.update({key: v}))
            form.addRow(k, spin)


class CurvesSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        layout = QVBoxLayout(body)
        layout.addWidget(QLabel("Curves editor — pick control points (x: 0-255, y: 0-255)"))
        # A simplified model: one big QLineEdit accepting JSON-ish points.
        from PySide6.QtWidgets import QPlainTextEdit
        self.text = QPlainTextEdit()
        self.text.setPlainText(str(self._settings.get("points", [{"x": 0, "y": 0}, {"x": 255, "y": 255}])))
        layout.addWidget(self.text)
        self.text.textChanged.connect(lambda: self._settings.update({"points": self.text.toPlainText()}))


class HueSaturationSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        for k, lo, hi in (
            ("hue", -180, 180),
            ("saturation", -100, 100),
            ("lightness", -100, 100),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(self._settings.get(k, 0))
            spin.valueChanged.connect(lambda v, key=k: self._settings.update({key: v}))
            form.addRow(k, spin)
        self.colorize = QComboBox()
        self.colorize.addItems(["Off", "On"])
        self.colorize.setCurrentText("On" if self._settings.get("colorize") else "Off")
        self.colorize.currentTextChanged.connect(lambda v: self._settings.update({"colorize": v == "On"}))
        form.addRow("Colorize", self.colorize)


class ExposureSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        spin = QDoubleSpinBox()
        spin.setRange(-5, 5)
        spin.setSingleStep(0.1)
        spin.setValue(self._settings.get("exposure", 0))
        spin.valueChanged.connect(lambda v: self._settings.update({"exposure": v}))
        form.addRow("Stops", spin)


class ColorBalanceSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        for label, key in (
            ("Shadow R/C", "shadowCyanRed"),
            ("Shadow M/Y", "shadowMagentaGreen"),
            ("Shadow Y/B", "shadowYellowBlue"),
            ("Mid R/C", "midCyanRed"),
            ("Mid M/Y", "midMagentaGreen"),
            ("Mid Y/B", "midYellowBlue"),
            ("Highlight R/C", "highlightCyanRed"),
            ("Highlight M/Y", "highlightMagentaGreen"),
            ("Highlight Y/B", "highlightYellowBlue"),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(-100, 100)
            spin.setValue(self._settings.get(key, 0))
            spin.valueChanged.connect(lambda v, k=key: self._settings.update({k: v}))
            form.addRow(label, spin)


class GradientMapSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        layout = QVBoxLayout(body)
        self.shadows = _color_picker(self._settings.get("shadows", [0, 0, 0]))
        self.highlights = _color_picker(self._settings.get("highlights", [255, 255, 255]))
        layout.addLayout(_labeled_row("Shadows", self.shadows))
        layout.addLayout(_labeled_row("Highlights", self.highlights))
        self.reversed_combo = QComboBox()
        self.reversed_combo.addItems(["No", "Yes"])
        self.reversed_combo.setCurrentText("Yes" if self._settings.get("reversed") else "No")
        self.reversed_combo.currentTextChanged.connect(lambda v: self._settings.update({"reversed": v == "Yes"}))
        layout.addLayout(_labeled_row("Reversed", self.reversed_combo))

    def accept(self) -> None:  # type: ignore[override]
        self._settings["shadows"] = _read_color(self.shadows)
        self._settings["highlights"] = _read_color(self.highlights)
        super().accept()


class GrainSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setValue(self._settings.get("amount", 0.1))
        spin.valueChanged.connect(lambda v: self._settings.update({"amount": v}))
        form.addRow("Amount", spin)

        size = QDoubleSpinBox()
        size.setRange(0.5, 8.0)
        size.setSingleStep(0.5)
        size.setValue(self._settings.get("size", 1.0))
        size.valueChanged.connect(lambda v: self._settings.update({"size": v}))
        form.addRow("Size", size)


class GaussianBlurSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 200.0)
        spin.setSingleStep(1.0)
        spin.setValue(self._settings.get("radius", 5.0))
        spin.valueChanged.connect(lambda v: self._settings.update({"radius": v}))
        form.addRow("Radius", spin)


class MotionBlurSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        d = QDoubleSpinBox()
        d.setRange(1.0, 500.0)
        d.setValue(self._settings.get("distance", 10.0))
        d.valueChanged.connect(lambda v: self._settings.update({"distance": v}))
        form.addRow("Distance", d)
        a = QDoubleSpinBox()
        a.setRange(-360.0, 360.0)
        a.setValue(self._settings.get("angle", 0.0))
        a.valueChanged.connect(lambda v: self._settings.update({"angle": v}))
        form.addRow("Angle", a)


class AddNoiseSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 1.0)
        amount.setValue(self._settings.get("amount", 0.1))
        amount.valueChanged.connect(lambda v: self._settings.update({"amount": v}))
        form.addRow("Amount", amount)
        dist = QComboBox()
        dist.addItems(["uniform", "gaussian"])
        dist.setCurrentText(self._settings.get("distribution", "uniform"))
        dist.currentTextChanged.connect(lambda v: self._settings.update({"distribution": v}))
        form.addRow("Distribution", dist)


class InvertSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        layout = QVBoxLayout(body)
        layout.addWidget(QLabel("Invert RGB. Alpha is preserved."))


class VignetteSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        amount = QDoubleSpinBox()
        amount.setRange(-1.0, 1.0)
        amount.setSingleStep(0.05)
        amount.setValue(self._settings.get("amount", 0.5))
        amount.valueChanged.connect(lambda v: self._settings.update({"amount": v}))
        form.addRow("Amount", amount)

        midpoint = QDoubleSpinBox()
        midpoint.setRange(0.0, 1.0)
        midpoint.setValue(self._settings.get("midpoint", 0.5))
        midpoint.valueChanged.connect(lambda v: self._settings.update({"midpoint": v}))
        form.addRow("Midpoint", midpoint)

        feather = QDoubleSpinBox()
        feather.setRange(0.05, 1.0)
        feather.setValue(self._settings.get("feather", 0.5))
        feather.valueChanged.connect(lambda v: self._settings.update({"feather": v}))
        form.addRow("Feather", feather)

        self.color_btn = _color_picker(self._settings.get("color", (0, 0, 0)))
        form.addRow("Colour", self.color_btn)

    def accept(self) -> None:  # type: ignore[override]
        self._settings["color"] = _read_color(self.color_btn)
        super().accept()


class BloomSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        threshold = QDoubleSpinBox()
        threshold.setRange(0.0, 1.0)
        threshold.setSingleStep(0.05)
        threshold.setValue(self._settings.get("threshold", 0.7))
        threshold.valueChanged.connect(lambda v: self._settings.update({"threshold": v}))
        form.addRow("Threshold", threshold)

        intensity = QDoubleSpinBox()
        intensity.setRange(0.0, 4.0)
        intensity.setValue(self._settings.get("intensity", 1.0))
        intensity.valueChanged.connect(lambda v: self._settings.update({"intensity": v}))
        form.addRow("Intensity", intensity)

        radius = QDoubleSpinBox()
        radius.setRange(0.5, 100.0)
        radius.setValue(self._settings.get("radius", 15.0))
        radius.valueChanged.connect(lambda v: self._settings.update({"radius": v}))
        form.addRow("Radius", radius)


class DitherSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        levels = QDoubleSpinBox()
        levels.setRange(2, 32)
        levels.setDecimals(0)
        levels.setSingleStep(1)
        levels.setValue(float(self._settings.get("levels", 2)))
        levels.valueChanged.connect(lambda v: self._settings.update({"levels": int(v)}))
        form.addRow("Levels", levels)

        strength = QDoubleSpinBox()
        strength.setRange(0.0, 1.0)
        strength.setSingleStep(0.05)
        strength.setValue(self._settings.get("strength", 1.0))
        strength.valueChanged.connect(lambda v: self._settings.update({"strength": v}))
        form.addRow("Strength", strength)


class ScanlinesSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        spacing = QDoubleSpinBox()
        spacing.setRange(1, 16)
        spacing.setDecimals(0)
        spacing.setSingleStep(1)
        spacing.setValue(float(self._settings.get("spacing", 2)))
        spacing.valueChanged.connect(lambda v: self._settings.update({"spacing": int(v)}))
        form.addRow("Spacing", spacing)

        intensity = QDoubleSpinBox()
        intensity.setRange(0.0, 1.0)
        intensity.setSingleStep(0.05)
        intensity.setValue(self._settings.get("intensity", 0.5))
        intensity.valueChanged.connect(lambda v: self._settings.update({"intensity": v}))
        form.addRow("Intensity", intensity)


class TonalContrastSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 2.0)
        amount.setSingleStep(0.05)
        amount.setValue(self._settings.get("amount", 0.5))
        amount.valueChanged.connect(lambda v: self._settings.update({"amount": v}))
        form.addRow("Amount", amount)

        radius = QDoubleSpinBox()
        radius.setRange(0.5, 30.0)
        radius.setValue(self._settings.get("radius", 3.0))
        radius.valueChanged.connect(lambda v: self._settings.update({"radius": v}))
        form.addRow("Radius", radius)


class LensCorrectionSheet(_AdjustmentDialog):
    def _build(self, body: QWidget) -> None:
        form = QFormLayout(body)
        strength = QDoubleSpinBox()
        strength.setRange(-1.0, 1.0)
        strength.setSingleStep(0.05)
        strength.setValue(self._settings.get("strength", 0.0))
        strength.valueChanged.connect(lambda v: self._settings.update({"strength": v}))
        form.addRow("Strength", strength)

        cx = QDoubleSpinBox()
        cx.setRange(0.0, 1.0)
        cx.setSingleStep(0.05)
        cx.setValue(self._settings.get("cx", 0.5))
        cx.valueChanged.connect(lambda v: self._settings.update({"cx": v}))
        form.addRow("Centre X", cx)

        cy = QDoubleSpinBox()
        cy.setRange(0.0, 1.0)
        cy.setSingleStep(0.05)
        cy.setValue(self._settings.get("cy", 0.5))
        cy.valueChanged.connect(lambda v: self._settings.update({"cy": v}))
        form.addRow("Centre Y", cy)


def _color_picker(initial) -> QWidget:
    from PySide6.QtWidgets import QPushButton
    btn = QPushButton()
    btn.setText("#%02x%02x%02x" % tuple(initial[:3]))
    btn._rgb = list(initial[:3])  # type: ignore[attr-defined]

    def _open() -> None:
        from PySide6.QtWidgets import QColorDialog
        c = QColorDialog.getColor()
        if c.isValid():
            btn._rgb = [c.red(), c.green(), c.blue()]  # type: ignore[attr-defined]
            btn.setText(c.name())

    btn.clicked.connect(_open)
    return btn


def _read_color(btn) -> list:
    return list(getattr(btn, "_rgb", [0, 0, 0]))


def _labeled_row(label: str, widget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    row.addWidget(widget, 1)
    return row
