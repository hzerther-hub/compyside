"""UI helpers shared between sheets."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


class PreviewLabel(QLabel):
    """A QLabel that scales to fit and shows a NumPy image."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setMinimumSize(160, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._array: np.ndarray | None = None

    def set_array(self, array: np.ndarray) -> None:
        self._array = array
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._array is None:
            return super().paintEvent(event)
        h, w = self._array.shape[:2]
        img = QImage(self._array.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        painter = QPainter(self)
        target = self.rect()
        pix = QPixmap.fromImage(img).scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap(target.topLeft() + (target.size() - pix.size()) / 2, pix)
        painter.end()


def preview_array_into(label: QLabel, array: np.ndarray) -> None:
    label._array = array  # type: ignore[attr-defined]
    label.update()
