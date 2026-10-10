"""Export-as dialog (PNG / JPEG / PDF)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ExportAsSheet(QDialog):
    exported = Signal(str)  # path written

    def __init__(self, parent: Optional[QWidget] = None, width: int = 1024, height: int = 1024):
        super().__init__(parent)
        self.setWindowTitle("Export As")
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG", "PDF"])
        form.addRow("Format", self.format_combo)

        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(92)
        form.addRow("JPEG Quality", self.quality_spin)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 30000)
        self.width_spin.setValue(width)
        form.addRow("Width", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 30000)
        self.height_spin.setValue(height)
        form.addRow("Height", self.height_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.05, 8.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(1.0)
        form.addRow("Scale", self.scale_spin)

        layout.addLayout(form)

        path_row = QHBoxLayout()
        self.path_label = QLabel("(no file selected)")
        path_btn = QPushButton("Choose…")
        path_btn.clicked.connect(self._choose_path)
        path_row.addWidget(self.path_label, 1)
        path_row.addWidget(path_btn)
        layout.addLayout(path_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._path: Optional[Path] = None

    def _choose_path(self) -> None:
        fmt = self.format_combo.currentText().lower()
        path, _ = QFileDialog.getSaveFileName(self, "Export As", f"untitled.{fmt}", f"{fmt.upper()} (*.{fmt})")
        if path:
            self._path = Path(path)
            self.path_label.setText(str(self._path))

    def _on_accept(self) -> None:
        if self._path is None:
            return
        self.accept()
        self.exported.emit(str(self._path))


class JpegPreviewSheet(QDialog):
    """JPEG export with a live preview that re-encodes as the slider moves."""

    exported = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, image: Optional[np.ndarray] = None):
        super().__init__(parent)
        self.setWindowTitle("Export JPEG (live preview)")
        self.setMinimumSize(640, 480)
        self._image = image if image is not None else np.zeros((64, 64, 4), dtype=np.uint8)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(85)
        self.quality_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Quality", self.quality_spin)

        self.size_label = QLabel("—")
        form.addRow("File size", self.size_label)

        layout.addLayout(form)

        self.preview_label = QLabel()
        self.preview_label.setMinimumHeight(320)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background:#222;")
        layout.addWidget(self.preview_label, 1)

        # Debounced re-encode.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._refresh_preview)

        path_row = QHBoxLayout()
        self.path_label = QLabel("(no file selected)")
        path_btn = QPushButton("Choose…")
        path_btn.clicked.connect(self._choose_path)
        path_row.addWidget(self.path_label, 1)
        path_row.addWidget(path_btn)
        layout.addLayout(path_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._path: Optional[Path] = None
        self._refresh_preview()

    def _schedule_preview(self) -> None:
        # Restart the debounce timer so rapid slider movement doesn't re-encode.
        self._timer.start()

    def _refresh_preview(self) -> None:
        try:
            import io
            from PIL import Image
            pil = Image.fromarray(self._image[..., :3])
            buf = io.BytesIO()
            quality = int(self.quality_spin.value())
            pil.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            self.size_label.setText(f"{len(data) / 1024:.1f} KB @ q{quality}")
            qimg = QImage()
            qimg.loadFromData(data, "JPEG")
            pix = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(
                pix.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.size_label.setText(f"preview error: {exc}")

    def _choose_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export JPEG", "untitled.jpg", "JPEG (*.jpg *.jpeg)")
        if path:
            self._path = Path(path)
            self.path_label.setText(str(self._path))

    def _on_accept(self) -> None:
        if self._path is None:
            return
        try:
            from PIL import Image
            pil = Image.fromarray(self._image[..., :3])
            quality = int(self.quality_spin.value())
            pil.save(self._path, format="JPEG", quality=quality)
        except Exception as exc:  # noqa: BLE001
            self.size_label.setText(f"save failed: {exc}")
            return
        self.accept()
        self.exported.emit(str(self._path))
