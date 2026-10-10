"""Camera Raw panel — develop controls for rawpy imports."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# Each panel: (key, lo, hi, default, label)
_LIGHT_KNOBS = [
    ("exposure", -5.0, 5.0, 0.0, "Exposure"),
    ("contrast", 0.0, 2.0, 1.0, "Contrast"),
    ("highlights", -1.0, 1.0, 0.0, "Highlights"),
    ("shadows", -1.0, 1.0, 0.0, "Shadows"),
    ("whites", -1.0, 1.0, 0.0, "Whites"),
    ("blacks", -1.0, 1.0, 0.0, "Blacks"),
]

_COLOR_KNOBS = [
    ("temperature", -100.0, 100.0, 0.0, "Temperature"),
    ("tint", -100.0, 100.0, 0.0, "Tint"),
    ("saturation", 0.0, 2.0, 1.0, "Saturation"),
    ("vibrance", -1.0, 1.0, 0.0, "Vibrance"),
]

_CURVES_KNOBS = [
    ("clarity", -1.0, 1.0, 0.0, "Clarity"),
    ("dehaze", -1.0, 1.0, 0.0, "Dehaze"),
]

_MIXER_KNOBS = [
    # Each colour channel hue/sat/lum — surfaces a 3-row × 8-col mini form.
]

_GRADING_KNOBS = [
    ("shadow_hue", 0.0, 360.0, 220.0, "Shadow hue"),
    ("shadow_sat", 0.0, 1.0, 0.5, "Shadow sat"),
    ("highlight_hue", 0.0, 360.0, 50.0, "Highlight hue"),
    ("highlight_sat", 0.0, 1.0, 0.5, "Highlight sat"),
    ("balance", -1.0, 1.0, 0.0, "Balance"),
]

_DETAIL_KNOBS = [
    ("sharpening_amount", 0.0, 5.0, 0.0, "Sharpening"),
    ("sharpening_radius", 0.5, 3.0, 1.0, "Radius"),
    ("noise_reduction", 0.0, 5.0, 0.0, "Noise reduction"),
]

_OPTICS_KNOBS = [
    ("lens_strength", -1.0, 1.0, 0.0, "Distortion"),
    ("lens_cx", 0.0, 1.0, 0.5, "Centre X"),
    ("lens_cy", 0.0, 1.0, 0.5, "Centre Y"),
]

_GEOMETRY_KNOBS = [
    ("rotate", -180.0, 180.0, 0.0, "Rotate"),
    ("aspect", 0.5, 2.0, 1.0, "Aspect"),
    ("scale", 0.05, 4.0, 1.0, "Scale"),
]


class _Panel(QWidget):
    """A single Camera Raw tab — a QFormLayout populated from a knob list."""

    changed = Signal()

    def __init__(self, knobs: list, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QFormLayout(self)
        self.spins: dict = {}
        for key, lo, hi, default, label in knobs:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            if hi - lo <= 4:
                spin.setSingleStep(0.05)
            spin.setValue(default)
            spin.valueChanged.connect(self.changed.emit)
            self.spins[key] = spin
            layout.addRow(label, spin)

    def values(self) -> dict:
        return {key: spin.value() for key, spin in self.spins.items()}


class _MixerPanel(QWidget):
    """HSL colour mixer — 8 hues × {hue shift, sat, lum}."""

    changed = Signal()

    HUES = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Adjust hue/saturation/luminance per colour."))
        form = QFormLayout()
        self.spins: dict = {}
        for h in self.HUES:
            for op, default in (("hue", 0.0), ("sat", 0.0), ("lum", 0.0)):
                key = f"mix_{h.lower()}_{op}"
                spin = QDoubleSpinBox()
                spin.setRange(-100.0, 100.0)
                spin.setSingleStep(5.0)
                spin.setValue(default)
                spin.valueChanged.connect(self.changed.emit)
                self.spins[key] = spin
                form.addRow(f"{h} {op}", spin)
        layout.addLayout(form)

    def values(self) -> dict:
        return {key: spin.value() for key, spin in self.spins.items()}


class _CurvesPanel(QWidget):
    """Tonal curves — 3 LUT sliders (R/G/B) plus a contrast curve."""

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Tonal curves — adjust shadows/midtones/highlights."))
        form = QFormLayout()
        self.spins: dict = {}
        for key, default, label in (
            ("curve_darks", 0.0, "Darks"),
            ("curve_lights", 0.0, "Lights"),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(5.0)
            spin.setValue(default)
            spin.valueChanged.connect(self.changed.emit)
            self.spins[key] = spin
            form.addRow(label, spin)
        layout.addLayout(form)
        for k, v in _CURVES_KNOBS:
            spin = QDoubleSpinBox()
            spin.setRange(v[1], v[2])
            spin.setSingleStep(0.05)
            spin.setValue(v[3])
            spin.valueChanged.connect(self.changed.emit)
            self.spins[k] = spin
            form.addRow(v[4], spin)
        layout.addStretch(1)

    def values(self) -> dict:
        return {key: spin.value() for key, spin in self.spins.items()}


class CameraRawDialog(QDialog):
    """Apply Camera Raw develop controls to a raw image and produce an RGBA uint8 array.

    Organised as 7 Photoshop-Camera-Raw-style panels: Light, Color, Curves,
    Mixer, Grading, Detail, Optics, Geometry.
    """

    developed = Signal(np.ndarray)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Camera Raw Filter")
        self.resize(820, 560)
        self._raw: Optional[np.ndarray] = None
        self._preview: Optional[np.ndarray] = None

        layout = QHBoxLayout(self)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)

        open_btn = QPushButton("Open RAW…")
        open_btn.clicked.connect(self._open_raw)
        left_layout.addWidget(open_btn)

        self.tabs = QTabWidget(left_panel)
        self._panels = []
        for label, factory in (
            ("Light", lambda: _Panel(_LIGHT_KNOBS)),
            ("Color", lambda: _Panel(_COLOR_KNOBS)),
            ("Curves", lambda: _CurvesPanel()),
            ("Mixer", lambda: _MixerPanel()),
            ("Grading", lambda: _Panel(_GRADING_KNOBS)),
            ("Detail", lambda: _Panel(_DETAIL_KNOBS)),
            ("Optics", lambda: _Panel(_OPTICS_KNOBS)),
            ("Geometry", lambda: _Panel(_GEOMETRY_KNOBS)),
        ):
            panel = factory()
            panel.changed.connect(self._render_preview)
            self.tabs.addTab(panel, label)
            self._panels.append(panel)
        left_layout.addWidget(self.tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        left_layout.addWidget(buttons)

        layout.addWidget(left_panel, 1)

        from .common import PreviewLabel
        self.preview = PreviewLabel(self)
        layout.addWidget(self.preview, 2)

    def _open_raw(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Camera RAW",
            "",
            "Camera RAW (*.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef *.srw)",
        )
        if not path:
            return
        try:
            from ..io.raw_importer import import_raw
            self._raw = import_raw(path)
        except RuntimeError as exc:
            self.preview.setText(str(exc))
            return
        self._render_preview()

    def _collect_kwargs(self) -> dict:
        out: dict = {}
        for panel in self._panels:
            out.update(panel.values())
        return out

    def _render_preview(self) -> None:
        if self._raw is None:
            return
        from ..io.raw_importer import develop
        kwargs = self._collect_kwargs()
        self._preview = develop(self._raw, **kwargs)
        self.preview.set_array(self._preview)

    def _on_accept(self) -> None:
        if self._preview is not None:
            self.developed.emit(self._preview)
        self.accept()
