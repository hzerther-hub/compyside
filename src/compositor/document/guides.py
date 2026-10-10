"""Document guides — drag-from-ruler reference lines.

A guide is a horizontal or vertical line placed in the document's coordinate
space. Used for alignment and for snapping layers / tools to a fixed position.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal, Tuple


def new_guide_id() -> str:
    """Generate a unique guide id."""
    return str(uuid.uuid4()).upper()


@dataclass(frozen=True)
class Guide:
    """A single guide line.

    ``orientation`` is ``"h"`` (horizontal — drawn across the document at a
    fixed ``y``) or ``"v"`` (vertical — fixed ``x``).
    ``position`` is in document pixels.
    """

    id: str
    orientation: Literal["h", "v"]
    position: float
    is_visible: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "orientation": self.orientation,
            "position": self.position,
            "isVisible": self.is_visible,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Guide":
        return cls(
            id=str(data["id"]),
            orientation=data["orientation"],
            position=float(data["position"]),
            is_visible=bool(data.get("isVisible", True)),
        )
