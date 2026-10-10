"""Import report sheet — what an imported file could and could not keep.

PSD/PSB import is lossy by nature (vector text, smart objects and adjustment
kinds Compositor does not model all have to go somewhere). Rather than let
the difference vanish, :class:`ImportReportSheet` shows the document's own
flattened preview next to the list of conversions, so the user can see what
arrived and what did not.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import i18n


def tr(text: str) -> str:
    return i18n.tr(text)


class ImportReportSheet(QDialog):
    """Preview + conversion list for one imported document."""

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        title: str,
        preview: Optional[np.ndarray],
        layer_count: int,
        width: int,
        height: int,
        report: Sequence[Dict[str, str]],
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 420)

        layout = QVBoxLayout(self)

        body = QHBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(220, 220)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background:#2b2b2b; border:1px solid #444;")
        body.addWidget(self.preview_label)
        layout.addLayout(body)

        self.summary_label = QLabel(
            tr("Imported {count} layers — {w}×{h}").format(count=layer_count, w=width, h=height)
        )
        layout.addWidget(self.summary_label)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        for entry in report:
            # Language-neutral: only the layer name and the note vary.
            item = QListWidgetItem(
                f"{entry.get('name', '')} — {entry.get('message', '')}"
            )
            item.setToolTip(str(entry.get("kind", "")))
            self.list_widget.addItem(item)
        if not report:
            self.list_widget.addItem(tr("Everything in this file was imported."))
        layout.addWidget(self.list_widget, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._set_preview(preview)
        i18n.on_change(self._refresh_labels)
        self._refresh_labels()

    def _set_preview(self, preview: Optional[np.ndarray]) -> None:
        if preview is None or preview.size == 0:
            self.preview_label.setText(tr("(no preview)"))
            return
        arr = np.ascontiguousarray(preview.astype(np.uint8))
        height, width = arr.shape[:2]
        img = QImage(arr.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
        self.preview_label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _refresh_labels(self, _code: str = "") -> None:
        self.setWindowTitle(tr("Import report"))


def show_import_report(
    parent: Optional[QWidget],
    *,
    title: str,
    result: Dict[str, Any],
) -> None:
    """Open the report sheet, non-modal — it is informational only."""
    layers: List[Dict[str, Any]] = result.get("layers", [])
    sheet = ImportReportSheet(
        parent,
        title=title,
        preview=result.get("flattened"),
        layer_count=len([entry for entry in layers if not entry.get("isGroup")]),
        width=int(result.get("width", 0)),
        height=int(result.get("height", 0)),
        report=result.get("report", []),
    )
    sheet.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    sheet.show()
