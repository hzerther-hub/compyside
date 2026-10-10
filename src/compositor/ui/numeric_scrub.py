"""NumericScrub — drag a label horizontally to change its value."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)


class NumericScrub(QWidget):
    """A label whose value changes when the user drags horizontally over it.

    Photoshop-style: hover the cursor over the label, click+drag to scrub.
    Right-click resets to default. The label is paired with a spin box so
    the value can also be typed.
    """

    value_changed = None  # Signal from QDoubleSpinBox

    DEFAULT_STEP = 1.0

    def __init__(
        self,
        label_text: str,
        minimum: float = 0.0,
        maximum: float = 100.0,
        step: float = DEFAULT_STEP,
        default: Optional[float] = None,
        suffix: str = "",
        parent: Optional[QWidget] = None,
        on_change: Optional[Callable[[float], None]] = None,
    ):
        super().__init__(parent)
        self._step = step
        self._default = default if default is not None else minimum
        self._drag_start_x: Optional[float] = None
        self._drag_start_value: Optional[float] = None
        self._on_change = on_change

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = QLabel(label_text, self)
        self.label.setMinimumWidth(80)
        self.label.setCursor(Qt.CursorShape.SizeHorCursor)
        layout.addWidget(self.label)

        self.spin = QDoubleSpinBox(self)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setValue(self._default)
        if on_change is not None:
            self.spin.valueChanged.connect(on_change)
        layout.addWidget(self.spin)

        self.label.mousePressEvent = self._label_mouse_press
        self.label.mouseMoveEvent = self._label_mouse_move
        self.label.mouseReleaseEvent = self._label_mouse_release
        self.label.wheelEvent = self._label_wheel

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        self.spin.setValue(value)

    def setRange(self, minimum: float, maximum: float) -> None:
        self.spin.setRange(minimum, maximum)

    def set_step(self, step: float) -> None:
        self._step = step
        self.spin.setSingleStep(step)

    # ---- mouse handling on the label -----------------------------------

    def _label_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.spin.setValue(self._default)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_x = event.position().x()
            self._drag_start_value = self.spin.value()
            event.accept()

    def _label_mouse_move(self, event: QMouseEvent) -> None:
        if self._drag_start_x is None or self._drag_start_value is None:
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        delta = event.position().x() - self._drag_start_x
        # 1 pixel of drag = one step. Holding Shift = 10x finer.
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            delta /= 10.0
        new_value = self._drag_start_value + delta * self._step
        new_value = max(self.spin.minimum(), min(self.spin.maximum(), new_value))
        self.spin.setValue(new_value)
        event.accept()

    def _label_mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_x = None
            self._drag_start_value = None

    def _label_wheel(self, event) -> None:
        delta = 1 if event.angleDelta().y() > 0 else -1
        self.spin.setValue(self.spin.value() + delta * self._step)
        event.accept()
