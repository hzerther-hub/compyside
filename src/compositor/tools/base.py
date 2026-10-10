"""Tool base class.

Tools receive Qt mouse / keyboard events from the canvas and turn them into
mutations on the EditorSession. Heavy work (painting, blurring) is delegated
to the rendering helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QMouseEvent


@dataclass
class ToolContext:
    """Carried with every tool call.

    The canvas fills in everything the tool needs without giving the tool
    direct access to Qt widgets — keeps tools testable.
    """

    session: object  # EditorSession
    canvas_width: int
    canvas_height: int
    zoom: float = 1.0
    pan: tuple = (0, 0)
    selection: Optional[object] = None
    active_layer_id: Optional[str] = None
    request_repaint: Callable[[], None] = lambda: None
    request_reload: Callable[[], None] = lambda: None


class Tool(QObject):
    name = "base"
    cursor = Qt.CursorShape.ArrowCursor

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

    # Lifecycle hooks (override as needed) ----------------------------------

    def activate(self, context: ToolContext) -> None:  # noqa: D401
        """Called when the tool becomes active."""
        pass

    def deactivate(self, context: ToolContext) -> None:
        pass

    # Mouse hooks ------------------------------------------------------------

    def mouse_press(self, context: ToolContext, event: QMouseEvent) -> None:
        pass

    def mouse_move(self, context: ToolContext, event: QMouseEvent) -> None:
        pass

    def mouse_release(self, context: ToolContext, event: QMouseEvent) -> None:
        pass

    # Keyboard hooks ---------------------------------------------------------

    def key_press(self, context: ToolContext, event) -> None:  # type: ignore[no-untyped-def]
        pass

    # Helpers ----------------------------------------------------------------

    def to_canvas_coords(self, context: ToolContext, x: float, y: float) -> tuple:
        """Convert widget-space pixel coords to document-space pixel coords."""
        return ((x - context.pan[0]) / context.zoom, (y - context.pan[1]) / context.zoom)
