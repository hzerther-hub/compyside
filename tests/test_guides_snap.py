"""Tests for guides + snap algorithm + Transform Inspector."""
from __future__ import annotations

import numpy as np
import pytest

from compositor.document.editor_session import EditorSession
from compositor.document.guides import Guide, new_guide_id
from compositor.document.layers import (
    AdjustmentLayer,
    AdjustmentKind,
    ImageLayer,
    Transform,
    new_id,
)
from compositor.rendering.layer_renderer import snap_position


# ---------------------------------------------------------------------------
# Guides
# ---------------------------------------------------------------------------


def _session_with_layer(width=100, height=100):
    s = EditorSession(width=width, height=height)
    s.add_layer(ImageLayer(id=new_id(), name="L", transform=Transform(size=(10, 10))))
    return s


def test_add_guide_appends():
    s = _session_with_layer()
    assert s.guides == []
    g = Guide(id=new_guide_id(), orientation="h", position=42.0)
    s.add_guide(g)
    assert len(s.guides) == 1
    assert s.guides[0].position == 42.0


def test_remove_guide_undoable():
    s = _session_with_layer()
    g = Guide(id=new_guide_id(), orientation="v", position=10.0)
    s.add_guide(g)
    s.remove_guide(g.id)
    assert s.guides == []
    s.undo()
    assert len(s.guides) == 1
    assert s.guides[0].id == g.id


def test_move_guide_undoable():
    s = _session_with_layer()
    g = Guide(id=new_guide_id(), orientation="h", position=10.0)
    s.add_guide(g)
    s.move_guide(g.id, 50.0)
    assert s.guide_by_id(g.id).position == 50.0
    s.undo()
    assert s.guide_by_id(g.id).position == 10.0


def test_guide_to_from_dict_roundtrip():
    g = Guide(id="ABCD", orientation="v", position=123.5)
    d = g.to_dict()
    g2 = Guide.from_dict(d)
    assert g2.id == "ABCD"
    assert g2.orientation == "v"
    assert g2.position == 123.5


def test_manifest_roundtrip_preserves_guides(tmp_path):
    from compositor.format.manifest import CURRENT_VERSION, Manifest
    from compositor.io.project_store import ProjectStore

    m = Manifest(
        version=CURRENT_VERSION,
        color_space="sRGB",
        document_id=new_id(),
        width=10,
        height=10,
        layers=[],
        guides=[
            {"id": "A", "orientation": "h", "position": 5.0},
            {"id": "B", "orientation": "v", "position": 8.0},
        ],
    )
    store = ProjectStore.create(tmp_path / "P.comp", m)
    m2 = store.load_manifest()
    assert len(m2.guides) == 2
    assert m2.guides[0]["position"] == 5.0


# ---------------------------------------------------------------------------
# Snap algorithm
# ---------------------------------------------------------------------------


def test_snap_to_canvas_edge():
    nx, ny = snap_position(
        3, 0,
        canvas_width=100, canvas_height=100,
        other_layers=[], guides=[],
        snap_to="canvas",
    )
    assert nx == 0
    assert ny == 0


def test_snap_to_canvas_center():
    nx, ny = snap_position(
        49, 48,
        canvas_width=100, canvas_height=100,
        other_layers=[], guides=[],
        snap_to="canvas",
    )
    assert nx == 50  # half = 50
    assert ny == 50


def test_no_snap_when_outside_threshold():
    nx, ny = snap_position(
        20, 20,
        canvas_width=100, canvas_height=100,
        other_layers=[], guides=[],
        snap_to="canvas",
        threshold=2.0,
    )
    # 20 is not within 2 of any canvas anchor (0, 50, 100).
    assert nx == 20
    assert ny == 20


def test_snap_to_other_layer():
    other = ImageLayer(id=new_id(), name="X", transform=Transform(origin=(50, 50), size=(40, 40)))
    nx, ny = snap_position(
        51, 51,
        canvas_width=200, canvas_height=200,
        other_layers=[other], guides=[],
        snap_to="layers",
    )
    # The other layer has anchors at x = 50, 70, 90 and y = 50, 70, 90.
    # 51 is closest to 50; 51 is closest to 50.
    assert nx == 50
    assert ny == 50


def test_snap_to_guide():
    guide = Guide(id="G1", orientation="v", position=200.0)
    nx, _ = snap_position(
        198, 50,
        canvas_width=300, canvas_height=300,
        other_layers=[], guides=[guide],
        snap_to="guides",
    )
    assert nx == 200


def test_x_and_y_snap_independently():
    # snap x to canvas, y to a guide.
    other = ImageLayer(id=new_id(), name="X", transform=Transform(origin=(0, 0), size=(50, 50)))
    guide = Guide(id="G", orientation="h", position=100.0)
    nx, ny = snap_position(
        1, 99,
        canvas_width=400, canvas_height=400,
        other_layers=[other], guides=[guide],
        snap_to="all",
    )
    assert nx == 0   # snapped to canvas edge
    assert ny == 100  # snapped to horizontal guide
