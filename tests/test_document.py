"""Document model tests — undo/redo and layer access."""
from __future__ import annotations

import copy

import pytest

from compositor.document.editor_session import EditorSession
from compositor.document.layers import (
    AdjustmentKind,
    AdjustmentLayer,
    BlendMode,
    GroupLayer,
    ImageLayer,
    Sampling,
    Transform,
    new_id,
)


def _session() -> EditorSession:
    return EditorSession(width=10, height=10)


def test_add_layer_records_undo() -> None:
    s = _session()
    layer = ImageLayer(id=new_id(), name="L1", transform=Transform(size=(10, 10)))
    s.add_layer(layer)
    assert s.can_undo()
    assert s.layers[0].name == "L1"

    s.undo()
    assert s.layers == []
    assert s.can_redo()


def test_redo_replays_add_layer() -> None:
    s = _session()
    layer = ImageLayer(id=new_id(), name="L1", transform=Transform(size=(10, 10)))
    s.add_layer(layer)
    s.undo()
    s.redo()
    assert len(s.layers) == 1
    assert s.layers[0].name == "L1"


def test_set_property_undo() -> None:
    s = _session()
    layer = ImageLayer(id=new_id(), name="L1", transform=Transform(size=(10, 10)))
    s.add_layer(layer)
    s.set_property(layer.id, "opacity", 0.5)
    assert s.layers[0].opacity == 0.5
    s.undo()
    assert s.layers[0].opacity == 1.0
    s.redo()
    assert s.layers[0].opacity == 0.5


def test_reorder_layer_undo() -> None:
    s = _session()
    a = ImageLayer(id=new_id(), name="A", transform=Transform(size=(10, 10)))
    b = ImageLayer(id=new_id(), name="B", transform=Transform(size=(10, 10)))
    c = ImageLayer(id=new_id(), name="C", transform=Transform(size=(10, 10)))
    s.add_layer(a)
    s.add_layer(b)
    s.add_layer(c)
    assert [l.name for l in s.layers] == ["A", "B", "C"]

    s.reorder_layer(a.id, new_index=2)
    assert [l.name for l in s.layers] == ["B", "C", "A"]
    s.undo()
    assert [l.name for l in s.layers] == ["A", "B", "C"]


def test_remove_layer_undo_restores() -> None:
    s = _session()
    a = ImageLayer(id=new_id(), name="A", transform=Transform(size=(10, 10)))
    s.add_layer(a)
    s.remove_layer(a.id)
    assert s.layers == []
    s.undo()
    assert len(s.layers) == 1
    assert s.layers[0].name == "A"


def test_adjustment_layer_roundtrip() -> None:
    s = _session()
    adj = AdjustmentLayer(
        id=new_id(),
        name="Warm",
        kind=AdjustmentKind.HUE_SATURATION,
        settings={"hue": 30, "saturation": -10},
        transform=Transform(size=(10, 10)),
    )
    s.add_layer(adj)
    payload = s.to_manifest_layers()
    assert payload[0]["adjustment"]["kind"] == "Hue/Saturation"
    assert payload[0]["adjustment"]["hue"] == 30

    s2 = _session()
    s2.load_manifest_layers(copy.deepcopy(payload))
    assert isinstance(s2.layers[0], AdjustmentLayer)
    assert s2.layers[0].kind == AdjustmentKind.HUE_SATURATION


def test_resize_canvas_undo() -> None:
    s = _session()
    s.resize_canvas(20, 30)
    assert s.width == 20
    assert s.height == 30
    s.undo()
    assert s.width == 10
    assert s.height == 10
    s.redo()
    assert s.width == 20
    assert s.height == 30


def test_scale_image_undo() -> None:
    s = _session()
    s.scale_image(64, 64)
    assert s.width == 64
    assert s.height == 64
    s.undo()
    assert s.width == 10
    assert s.height == 10


def test_transform_corners_roundtrip() -> None:
    """Free-distort corners survive manifest round-trip."""
    s = _session()
    layer = ImageLayer(
        id=new_id(),
        name="L",
        transform=Transform(
            size=(10, 10),
            corners=((0, 0), (15, 0), (15, 15), (0, 15)),
        ),
    )
    s.add_layer(layer)
    payload = s.to_manifest_layers()
    assert "corners" in payload[0]["transform"]
    s2 = _session()
    s2.load_manifest_layers(copy.deepcopy(payload))
    rt_layer = s2.layers[0]
    assert rt_layer.transform.corners is not None
    assert tuple(rt_layer.transform.corners[0]) == (0, 0)
    assert tuple(rt_layer.transform.corners[2]) == (15, 15)


def test_group_mask_roundtrip() -> None:
    """A group with a mask survives manifest round-trip (v6+ feature)."""
    s = _session()
    grp = GroupLayer(
        id=new_id(),
        name="Group",
        mask_file=f"{new_id()}.png",
        mask_enabled=True,
    )
    s.add_layer(grp)
    payload = s.to_manifest_layers()
    assert payload[0]["maskFile"]
    assert payload[0]["maskEnabled"] is True
    s2 = _session()
    s2.load_manifest_layers(copy.deepcopy(payload))
    rt = s2.layers[0]
    assert isinstance(rt, GroupLayer)
    assert rt.mask_file is not None
    assert rt.mask_enabled is True


# ---------------------------------------------------------------------------
# Merge Down / Trim
# ---------------------------------------------------------------------------


def test_merge_down_replaces_pair_and_undo_restores() -> None:
    s = _session()
    bottom = ImageLayer(id=new_id(), name="Bottom", transform=Transform(size=(10, 10)))
    top = ImageLayer(id=new_id(), name="Top", transform=Transform(size=(10, 10)))
    s.add_layer(bottom)
    s.add_layer(top)
    s.active_layer_id = top.id

    cache = {bottom.id: "px-bottom", top.id: "px-top"}
    merged = ImageLayer(id=bottom.id, name="Bottom", transform=Transform(size=(10, 10)))
    s.merge_down(top.id, merged, new_pixels="px-merged", pixel_cache=cache)

    assert [l.name for l in s.layers] == ["Bottom"]
    assert s.active_layer_id == bottom.id
    assert cache[bottom.id] == "px-merged"

    s.undo()
    assert [l.name for l in s.layers] == ["Bottom", "Top"]
    assert s.active_layer_id == top.id
    assert cache[bottom.id] == "px-bottom"
    assert cache[top.id] == "px-top"

    s.redo()
    assert [l.name for l in s.layers] == ["Bottom"]
    assert cache[bottom.id] == "px-merged"


def test_merge_down_skips_foreign_parent_layers() -> None:
    s = _session()
    outside = ImageLayer(id=new_id(), name="Outside", transform=Transform(size=(10, 10)))
    group = GroupLayer(id=new_id(), name="G")
    inner_a = ImageLayer(
        id=new_id(), name="InnerA", parent_id=group.id, transform=Transform(size=(10, 10))
    )
    inner_b = ImageLayer(
        id=new_id(), name="InnerB", parent_id=group.id, transform=Transform(size=(10, 10))
    )
    for layer in (outside, group, inner_a, inner_b):
        s.add_layer(layer)
    s.active_layer_id = inner_b.id

    merged = ImageLayer(
        id=inner_a.id, name="InnerA", parent_id=group.id, transform=Transform(size=(10, 10))
    )
    s.merge_down(inner_b.id, merged)

    # The root-level layer below the group must be untouched.
    assert [l.name for l in s.layers] == ["Outside", "G", "InnerA"]
    assert s.layers[2].id == inner_a.id
    s.undo()
    assert [l.name for l in s.layers] == ["Outside", "G", "InnerA", "InnerB"]


def test_trim_document_shifts_layers_and_undo_restores() -> None:
    s = EditorSession(width=100, height=80)
    layer = ImageLayer(
        id=new_id(), name="L", transform=Transform(origin=(30.0, 20.0), size=(40.0, 30.0))
    )
    s.add_layer(layer)

    s.trim_document(30, 20, 40, 30)
    assert (s.width, s.height) == (40, 30)
    assert layer.transform.origin == (0.0, 0.0)
    assert layer.transform.size == (40.0, 30.0)  # size untouched

    s.undo()
    assert (s.width, s.height) == (100, 80)
    assert layer.transform.origin == (30.0, 20.0)

    s.redo()
    assert (s.width, s.height) == (40, 30)
    assert layer.transform.origin == (0.0, 0.0)
