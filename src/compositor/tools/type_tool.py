"""Type tool — click to place a text layer with inline multi-line editing.

The full Photoshop-style rich-text overlay is enormous. This MVP:
  * places a flat text layer on click (rasterised),
  * exposes a multi-line editor dialog (`open_text_editor`) that re-rasterises
    on commit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

from ..document.layers import ImageLayer, Transform, new_id
from .base import Tool, ToolContext


@dataclass
class TypeSettings:
    text: str = "Type"
    font_family: str = "Arial"
    font_size: int = 48
    color: Tuple[int, int, int, int] = (0, 0, 0, 255)
    align: str = "left"  # left | centre | right
    line_spacing: float = 1.2


def rasterize_text(
    text: str,
    font_family: str = "Arial",
    font_size: int = 48,
    color: Tuple[int, int, int, int] = (0, 0, 0, 255),
    align: str = "left",
    line_spacing: float = 1.2,
    max_width: int = 0,
) -> np.ndarray:
    """Rasterise multi-line text to an RGBA uint8 NumPy buffer.

    ``align`` is one of ``"left"``, ``"centre"``, ``"right"``. ``max_width`` in
    pixels enables word-wrap (0 = no wrap).
    """
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(font_family + ".ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype(font_family, font_size)
        except OSError:
            font = ImageFont.load_default()
    # Split into lines, then word-wrap each.
    raw_lines = (text or "Type").splitlines() or [""]
    if max_width > 0:
        wrapped = []
        for line in raw_lines:
            words = line.split(" ")
            if not words:
                wrapped.append("")
                continue
            current = words[0]
            for word in words[1:]:
                candidate = current + " " + word
                bbox = font.getbbox(candidate)
                if bbox[2] - bbox[0] <= max_width:
                    current = candidate
                else:
                    wrapped.append(current)
                    current = word
            wrapped.append(current)
        raw_lines = wrapped
    # Measure line height.
    line_bbox = font.getbbox("Ay")
    line_height = max(1, int((line_bbox[3] - line_bbox[1]) * line_spacing))
    line_widths = [font.getbbox(line)[2] - font.getbbox(line)[0] for line in raw_lines]
    width = max(line_widths) + 8 if line_widths else 64
    height = line_height * max(1, len(raw_lines)) + 4
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, (line, lw) in enumerate(zip(raw_lines, line_widths)):
        if align == "centre":
            x = (width - lw) // 2
        elif align == "right":
            x = width - lw - 4
        else:
            x = 4
        y = i * line_height + 2
        draw.text((x, y), line, fill=color, font=font)
    return np.asarray(img, dtype=np.uint8)


class TypeTool(Tool):
    name = "type"
    cursor = Qt.CursorShape.IBeamCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = TypeSettings()

    def mouse_press(self, context: ToolContext, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cx, cy = self.to_canvas_coords(context, event.position().x(), event.position().y())
        pixels = rasterize_text(
            self.settings.text or "Type",
            font_family=self.settings.font_family,
            font_size=self.settings.font_size,
            color=self.settings.color,
            align=self.settings.align,
            line_spacing=self.settings.line_spacing,
        )
        h, w = pixels.shape[:2]
        layer = ImageLayer(
            id=new_id(),
            name=self.settings.text or "Type",
            transform=Transform(origin=(cx, cy), size=(w, h)),
            image_file=f"{new_id()}.png",
        )
        context.session.add_layer(layer)
        cache = getattr(context.session, "_image_cache", {})
        cache[layer.id] = pixels
        # Stash the type settings on the layer so future edits know the font.
        setattr(layer, "_type_settings", self.settings)
        context.request_repaint()


def open_text_editor(parent, layer: ImageLayer, cache: dict) -> None:
    """Open an inline multi-line text editor for an existing text layer.

    On commit, re-rasterise the layer's pixels from the new text.
    """
    from PySide6.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
    )
    settings = getattr(layer, "_type_settings", TypeSettings())

    dialog = QDialog(parent)
    dialog.setWindowTitle("Edit Text")
    dialog.resize(420, 320)
    layout = QVBoxLayout(dialog)

    form = QFormLayout()
    family_combo = QComboBox()
    for fam in ("Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana"):
        family_combo.addItem(fam)
    family_combo.setCurrentText(settings.font_family)
    form.addRow("Font", family_combo)

    size_spin = QSpinBox()
    size_spin.setRange(8, 512)
    size_spin.setValue(settings.font_size)
    form.addRow("Size", size_spin)

    align_combo = QComboBox()
    align_combo.addItems(["left", "centre", "right"])
    align_combo.setCurrentText(settings.align)
    form.addRow("Align", align_combo)

    line_spin = QSpinBox()
    line_spin.setRange(80, 300)
    line_spin.setValue(int(settings.line_spacing * 100))
    form.addRow("Line spacing %", line_spin)

    layout.addLayout(form)

    editor = QPlainTextEdit()
    editor.setPlainText(settings.text)
    layout.addWidget(editor, 1)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return

    new_text = editor.toPlainText()
    new_settings = TypeSettings(
        text=new_text,
        font_family=family_combo.currentText(),
        font_size=size_spin.value(),
        color=settings.color,
        align=align_combo.currentText(),
        line_spacing=line_spin.value() / 100.0,
    )
    pixels = rasterize_text(**{**new_settings.__dict__})
    cache[layer.id] = pixels
    # Resize the layer transform to match the new rasterised extent.
    h, w = pixels.shape[:2]
    layer.transform = Transform(
        origin=layer.transform.origin,
        size=(w, h),
        rotation=layer.transform.rotation,
        flip_x=layer.transform.flip_x,
        flip_y=layer.transform.flip_y,
        sampling=layer.transform.sampling,
        corners=layer.transform.corners,
    )
    setattr(layer, "_type_settings", new_settings)


# QFontMetrics is needed for width calc; import lazily so non-Qt tests don't break.
try:
    from PySide6.QtGui import QFontMetrics  # type: ignore
except ImportError:  # pragma: no cover
    class QFontMetrics:  # type: ignore[no-redef]
        def __init__(self, font):
            self.font = font
        def horizontalAdvance(self, text):
            return len(text) * (self.font.pointSize() or 12) // 2
