"""Color picker — dialog with hue/sat grid + eyedropper button."""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class _ColorGrid(QWidget):
    color_changed = Signal(tuple)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self._hue = 0
        self._color = (255, 0, 0)
        self._image = self._render()

    def set_hue(self, hue: int) -> None:
        self._hue = hue % 360
        self._image = self._render()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)
        # Selector.
        h, s, _ = self._color
        x = int(s * self.width() / 255)
        y = int((255 - self._color[2]) * self.height() / 255)
        painter.setPen(QColor(255, 255, 255))
        painter.drawEllipse(x - 4, y - 4, 8, 8)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._sample(event.position().x(), event.position().y())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._sample(event.position().x(), event.position().y())

    def _sample(self, x: float, y: float) -> None:
        s = int(round(x * 255 / self.width()))
        v = int(round((1 - y / self.height()) * 255))
        s = max(0, min(255, s))
        v = max(0, min(255, v))
        self._color = (self._hue, s, v)
        self.color_changed.emit(self._color)
        self.update()

    def _render(self) -> QImage:
        img = QImage(self.width(), self.height(), QImage.Format.Format_RGB888)
        hue = self._hue / 360.0
        for y in range(self.height()):
            for x in range(self.width()):
                s = x / max(self.width() - 1, 1)
                v = 1 - y / max(self.height() - 1, 1)
                rgb = QColor.fromHsvF(hue, s, v)
                img.setPixelColor(x, y, rgb)
        return img


class ColorPickerDialog(QDialog):
    color_picked = Signal(tuple)

    def __init__(self, parent: Optional[QWidget] = None, initial: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__(parent)
        self.setWindowTitle("Color Picker")
        self.setLayout(QVBoxLayout())
        self._initial = initial

        self.grid = _ColorGrid(self)
        hue = QColor(*initial).hueF()
        if hue < 0:
            hue = 0
        self.grid.set_hue(int(hue * 360))
        self.grid._color = QColor(*initial).getHsv()[:3]
        self.grid.color_changed.connect(self._on_grid)
        self.layout().addWidget(self.grid)

        self.hue_slider = QSlider(Qt.Orientation.Horizontal)
        self.hue_slider.setRange(0, 359)
        self.hue_slider.setValue(int((hue or 0) * 360))
        self.hue_slider.valueChanged.connect(self.grid.set_hue)
        self.layout().addWidget(self.hue_slider)

        self.hex_edit = QLineEdit(self)
        self.hex_edit.setText("#%02x%02x%02x" % initial)
        self.layout().addWidget(self.hex_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)

    def _on_grid(self, hsv) -> None:
        h, s, v = hsv
        rgb = QColor.fromHsv(h, s, v)
        self.hex_edit.setText(rgb.name())

    def _on_accept(self) -> None:
        color = QColor(self.hex_edit.text())
        if color.isValid():
            self.color_picked.emit((color.red(), color.green(), color.blue()))
        self.accept()
