"""Tools — every interactive tool the canvas can be in.

Each tool owns mouse handling for its phase (down, drag, up), keyboard
modifiers, and an optional cursor. The canvas widget dispatches events
to whichever tool is active.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QObject, Qt

from .base import Tool, ToolContext
from .brush_tool import BrushTool, EraserTool
from .selection_tools import (
    EllipseMarqueeTool,
    LassoTool,
    PolygonalLassoTool,
    RectangleMarqueeTool,
    WandTool,
)
from .transform_tool import MoveTool
from .gradient_tool import GradientTool
from .shape_tool import ShapeTool
from .eyedropper_tool import EyedropperTool
from .crop_tool import CropTool
from .type_tool import TypeTool
from .clone_tool import CloneTool
from .blur_tool import BlurTool, HealTool
from .object_select_tool import ObjectSelectTool


def default_registry() -> "ToolRegistry":
    r = ToolRegistry()
    for tool in (
        MoveTool(),
        BrushTool(),
        EraserTool(),
        RectangleMarqueeTool(),
        EllipseMarqueeTool(),
        LassoTool(),
        PolygonalLassoTool(),
        WandTool(),
        CropTool(),
        GradientTool(),
        ShapeTool(),
        TypeTool(),
        CloneTool(),
        BlurTool(),
        HealTool(),
        ObjectSelectTool(),
        EyedropperTool(),
    ):
        r.register(tool)
    return r


class ToolRegistry(QObject):
    """Holds every tool keyed by name; the canvas asks for the active one."""

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools.keys())


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "default_registry",
    "BrushTool",
    "EraserTool",
    "RectangleMarqueeTool",
    "EllipseMarqueeTool",
    "LassoTool",
    "PolygonalLassoTool",
    "WandTool",
    "MoveTool",
    "CropTool",
    "GradientTool",
    "ShapeTool",
    "TypeTool",
    "CloneTool",
    "BlurTool",
    "HealTool",
    "EyedropperTool",
]
