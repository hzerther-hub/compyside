"""Photoshop transfer tests — flattening a Compositor document for Photoshop.

Photoshop gets flat image layers, so the interesting part is the walk that
decides what to send, in what order, under what name, and where on the
canvas.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from compositor.document.editor_session import EditorSession  # noqa: E402
from compositor.document.layers import (  # noqa: E402
    AdjustmentKind,
    AdjustmentLayer,
    BlendMode,
    GroupLayer,
    ImageLayer,
    Transform,
    new_id,
)
from compositor.ui.photoshop_ui import flatten_layers_for_photoshop  # noqa: E402


def _image_layer(session, name, origin, color, *, size=(20, 20), visible=True):
    layer_id = new_id()
    layer = ImageLayer(
        id=layer_id,
        name=name,
        image_file=f"{layer_id}.png",
        transform=Transform(origin=origin, size=size),
    )
    layer.is_visible = visible
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[..., :3] = color
    pixels[..., 3] = 255
    session._image_cache[layer_id] = pixels
    return layer


@pytest.fixture()
def session():
    s = EditorSession(width=100, height=80)
    s._image_cache = {}
    s._mask_cache = {}
    return s


def test_order_is_bottom_to_top(session):
    session.layers = [
        _image_layer(session, "Bottom", (10, 5), (255, 0, 0)),
        _image_layer(session, "Top", (40, 30), (0, 255, 0)),
    ]
    entries, skipped = flatten_layers_for_photoshop(session)
    assert [e["name"] for e in entries] == ["Bottom", "Top"]
    assert skipped == []


def test_layers_keep_their_document_position(session):
    session.layers = [
        _image_layer(session, "Bottom", (10, 5), (255, 0, 0)),
        _image_layer(session, "Top", (40, 30), (0, 255, 0)),
    ]
    entries, _ = flatten_layers_for_photoshop(session)
    assert entries[0]["offset"] == (10, 5)
    assert entries[1]["offset"] == (40, 30)


def test_pixels_are_cropped_to_content(session):
    session.layers = [_image_layer(session, "Bottom", (10, 5), (255, 0, 0))]
    entries, _ = flatten_layers_for_photoshop(session)
    # A 20x20 layer at (10,5) crops to its own extent.
    assert entries[0]["pixels"].shape == (20, 20, 4)
    assert tuple(entries[0]["pixels"][0, 0]) == (255, 0, 0, 255)


def test_group_children_are_sent_once_with_the_group_prefix(session):
    group = GroupLayer(id=new_id(), name="Grp")
    inside = _image_layer(session, "Inside", (0, 60), (0, 0, 255))
    inside.parent_id = group.id
    bottom = _image_layer(session, "Bottom", (0, 0), (255, 0, 0))
    # ``session.layers`` is flat, as it is everywhere else in the app.
    session.layers = [bottom, group, inside]

    entries, _ = flatten_layers_for_photoshop(session)
    names = [e["name"] for e in entries]
    assert names == ["Bottom", "Grp/Inside"], names


def test_nested_groups_stack_their_prefixes(session):
    outer = GroupLayer(id=new_id(), name="Outer")
    inner = GroupLayer(id=new_id(), name="Inner")
    inner.parent_id = outer.id
    leaf = _image_layer(session, "Leaf", (0, 0), (1, 2, 3))
    leaf.parent_id = inner.id
    session.layers = [outer, inner, leaf]

    entries, _ = flatten_layers_for_photoshop(session)
    assert [e["name"] for e in entries] == ["Outer/Inner/Leaf"]


def test_group_opacity_is_baked_into_the_pixels(session):
    group = GroupLayer(id=new_id(), name="Grp", opacity=0.5)
    inside = _image_layer(session, "Inside", (0, 0), (10, 10, 10))
    inside.parent_id = group.id
    session.layers = [group, inside]

    entries, _ = flatten_layers_for_photoshop(session)
    # The group is gone by the time Photoshop sees this layer, so its opacity
    # has to live in the pixels...
    assert entries[0]["pixels"][0, 0, 3] == pytest.approx(128, abs=1)
    # ...while the layer's own opacity is passed through for Photoshop to
    # apply, not baked in as well.
    assert entries[0]["opacity"] == pytest.approx(1.0)


def test_layer_opacity_is_passed_through_not_baked(session):
    layer = _image_layer(session, "Faded", (0, 0), (10, 10, 10))
    layer.opacity = 0.5
    session.layers = [layer]

    entries, _ = flatten_layers_for_photoshop(session)
    assert entries[0]["opacity"] == pytest.approx(0.5)
    # Full alpha in the pixels; Photoshop multiplies by its own opacity.
    assert entries[0]["pixels"][0, 0, 3] == 255


def test_adjustment_layers_are_skipped_by_name(session):
    adjustment = AdjustmentLayer(id=new_id(), name="Grade", kind=AdjustmentKind.INVERT)
    session.layers = [
        _image_layer(session, "Bottom", (0, 0), (1, 1, 1)),
        adjustment,
    ]
    entries, skipped = flatten_layers_for_photoshop(session)
    assert [e["name"] for e in entries] == ["Bottom"]
    assert skipped == ["Grade"]


def test_hidden_layers_are_left_out_unless_asked(session):
    session.layers = [
        _image_layer(session, "Shown", (0, 0), (1, 1, 1)),
        _image_layer(session, "Hidden", (30, 30), (2, 2, 2), visible=False),
    ]
    entries, _ = flatten_layers_for_photoshop(session)
    assert [e["name"] for e in entries] == ["Shown"]

    entries, _ = flatten_layers_for_photoshop(session, include_invisible=True)
    assert [e["name"] for e in entries] == ["Shown", "Hidden"]


def test_layer_without_cached_pixels_is_reported_not_sent(session):
    orphan = ImageLayer(
        id=new_id(), name="Orphan", image_file="missing.png",
        transform=Transform(origin=(0, 0), size=(10, 10)),
    )
    session.layers = [orphan]
    entries, skipped = flatten_layers_for_photoshop(session)
    assert entries == []
    assert skipped == ["Orphan"]


def test_fully_transparent_layer_is_reported_not_sent(session):
    layer = _image_layer(session, "Empty", (0, 0), (5, 5, 5))
    layer.transform = Transform(origin=(0, 0), size=(20, 20))
    session._image_cache[layer.id] = np.zeros((20, 20, 4), dtype=np.uint8)
    session.layers = [layer]
    entries, skipped = flatten_layers_for_photoshop(session)
    assert entries == []
    assert skipped == ["Empty"]


def test_blend_mode_is_carried_through(session):
    layer = _image_layer(session, "Mul", (0, 0), (9, 9, 9))
    layer.blend_mode = BlendMode.MULTIPLY
    session.layers = [layer]
    entries, _ = flatten_layers_for_photoshop(session)
    assert entries[0]["blend_mode"] == "Multiply"
