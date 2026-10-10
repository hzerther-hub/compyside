"""Layer renderer tests."""
from __future__ import annotations

import numpy as np

from compositor.document.editor_session import EditorSession
from compositor.document.layers import AdjustmentKind, AdjustmentLayer, GroupLayer, ImageLayer, Transform, new_id
from compositor.rendering.adjustments import apply_adjustment
from compositor.rendering.layer_renderer import CompositorOptions, composite_document


def _session_with_red_square(width: int = 32, height: int = 32) -> tuple:
    session = EditorSession(width=width, height=height)
    layer = ImageLayer(id=new_id(), name="Red", transform=Transform(size=(width, height)))
    session.add_layer(layer)
    red = np.zeros((height, width, 4), dtype=np.uint8)
    red[..., 0] = 255
    red[..., 3] = 255
    cache = {layer.id: red}
    return session, layer.id, cache


def test_image_layer_red() -> None:
    session, layer_id, cache = _session_with_red_square()
    out = composite_document(session, CompositorOptions(
        canvas_width=session.width, canvas_height=session.height, image_loader=cache.get
    ))
    assert (out[..., 0] == 255).all()
    assert (out[..., 1] == 0).all()
    assert (out[..., 2] == 0).all()


def test_group_passes_through() -> None:
    session = EditorSession(width=10, height=10)
    session.add_layer(ImageLayer(id=new_id(), name="A", transform=Transform(size=(10, 10))))
    session.add_layer(GroupLayer(id=new_id(), name="Group"))
    session.add_layer(ImageLayer(id=new_id(), name="B", transform=Transform(size=(10, 10))))
    cache = {}
    for layer in session.layers:
        if isinstance(layer, ImageLayer):
            img = np.zeros((10, 10, 4), dtype=np.uint8)
            img[..., 0] = 255
            img[..., 3] = 255
            cache[layer.id] = img
    out = composite_document(session, CompositorOptions(
        canvas_width=10, canvas_height=10, image_loader=cache.get
    ))
    # Top layer should be visible.
    assert (out[..., 0] == 255).all()


def test_adjustment_layer_levels_identity() -> None:
    session, _, cache = _session_with_red_square()
    identity = {"black": 0, "gamma": 1.0, "white": 255, "outputBlack": 0, "outputWhite": 255}
    layer = AdjustmentLayer(
        id=new_id(),
        name="Levels",
        kind=AdjustmentKind.LEVELS,
        settings={"levels": {"ranges": [identity] * 4}},
        transform=Transform(size=(session.width, session.height)),
    )
    session.add_layer(layer)
    out = composite_document(session, CompositorOptions(
        canvas_width=session.width, canvas_height=session.height, image_loader=cache.get
    ))
    # Identity levels should leave the red square unchanged.
    assert (out[..., 0] == 255).all()


def test_apply_adjustment_invert() -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[..., 0] = 200
    arr[..., 3] = 255
    out = apply_adjustment(arr, AdjustmentKind.INVERT, {})
    np.testing.assert_array_equal(out[..., 0], 55)
    np.testing.assert_array_equal(out[..., 3], 255)


def test_apply_adjustment_hue_saturation() -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[..., 0] = 255
    arr[..., 3] = 255
    out = apply_adjustment(arr, AdjustmentKind.HUE_SATURATION, {"hue": 120.0})
    # After a 120° hue shift in HSV, red becomes green-ish.
    assert out[..., 1].mean() > out[..., 0].mean()


def test_apply_adjustment_vignette_darkens() -> None:
    arr = np.full((16, 16, 4), 200, dtype=np.uint8)
    arr[..., 3] = 255
    out = apply_adjustment(arr, AdjustmentKind.VIGNETTE, {"amount": 1.0, "midpoint": 0.0, "feather": 1.0})
    assert out[0, 0, 0] < 200  # corner darker than centre
    # Centre is at (7.5, 7.5); pixel (8,8) is essentially the centre.
    assert out[8, 8, 0] >= 195  # within a few units of the source


def test_apply_adjustment_bloom_runs() -> None:
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[:, 4:, :3] = 255
    arr[..., 3] = 255
    out = apply_adjustment(arr, AdjustmentKind.BLOOM, {"threshold": 0.5, "intensity": 1.0, "radius": 1.0})
    assert out.shape == arr.shape
    assert out[..., 3].max() == 255


def test_apply_adjustment_dither_runs() -> None:
    arr = np.linspace(0, 255, 256, dtype=np.uint8).reshape(1, 256, 1).repeat(4, axis=-1)
    out = apply_adjustment(arr, AdjustmentKind.DITHER, {"levels": 4, "strength": 1.0})
    assert out.shape == arr.shape


def test_apply_adjustment_scanlines_runs() -> None:
    arr = np.full((4, 4, 4), 100, dtype=np.uint8)
    arr[..., 3] = 255
    out = apply_adjustment(arr, AdjustmentKind.SCANLINES, {"spacing": 2, "intensity": 1.0})
    assert out.shape == arr.shape
    assert out[1, 0, 0] == 0  # dark row


def test_apply_adjustment_tonal_contrast_runs() -> None:
    arr = np.full((8, 8, 4), 128, dtype=np.uint8)
    arr[..., 3] = 255
    out = apply_adjustment(arr, AdjustmentKind.TONAL_CONTRAST, {"amount": 0.5, "radius": 1.0})
    np.testing.assert_array_equal(out[..., 3], arr[..., 3])  # alpha untouched


def test_apply_adjustment_lens_correction_runs() -> None:
    arr = np.random.randint(0, 256, size=(16, 16, 4), dtype=np.uint8)
    out = apply_adjustment(arr, AdjustmentKind.LENS_CORRECTION, {"strength": 0.5, "cx": 0.5, "cy": 0.5})
    assert out.shape == arr.shape


def test_apply_adjustment_remove_background_runs() -> None:
    arr = np.full((16, 16, 4), 200, dtype=np.uint8)
    arr[4:12, 4:12, :3] = 50
    arr[..., 3] = 255
    out = apply_adjustment(arr, AdjustmentKind.REMOVE_BACKGROUND, {"margin": 0.2})
    assert out.shape == arr.shape


# ---------------------------------------------------------------------------
# Merge Down rendering
# ---------------------------------------------------------------------------


def test_merge_layers_pixels_bakes_pair_and_crops() -> None:
    from compositor.rendering.layer_renderer import merge_layers_pixels

    session = EditorSession(width=40, height=20)
    bottom = ImageLayer(
        id=new_id(), name="Bottom", transform=Transform(origin=(10.0, 5.0), size=(10.0, 10.0))
    )
    top = ImageLayer(
        id=new_id(), name="Top", transform=Transform(origin=(25.0, 5.0), size=(10.0, 10.0))
    )
    session.add_layer(bottom)
    session.add_layer(top)

    def solid(rgb):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[..., :3] = rgb
        img[..., 3] = 255
        return img

    cache = {bottom.id: solid((255, 0, 0)), top.id: solid((0, 0, 255))}
    pixels, origin, size = merge_layers_pixels(session, cache, bottom=bottom, top=top)

    # Bottom spans x=10..19, top x=25..34 -> union 25 px wide; y=5..14 -> 10 px tall.
    assert size == (25.0, 10.0)
    assert origin == (10.0, 5.0)
    assert pixels.shape == (10, 25, 4)
    # Left half: red from the bottom layer; right half: blue from the top.
    assert (pixels[:, :10, 0] == 255).all()
    assert (pixels[:, 15:, 2] == 255).all()
    # The gap between the two layers stays transparent.
    assert (pixels[:, 10:15, 3] == 0).all()


def test_merge_layers_pixels_fully_transparent_keeps_canvas() -> None:
    from compositor.rendering.layer_renderer import merge_layers_pixels

    session = EditorSession(width=8, height=6)
    bottom = ImageLayer(id=new_id(), name="Bottom", transform=Transform(size=(8.0, 6.0)))
    top = ImageLayer(id=new_id(), name="Top", transform=Transform(size=(8.0, 6.0)))
    session.add_layer(bottom)
    session.add_layer(top)
    cache = {bottom.id: np.zeros((6, 8, 4), dtype=np.uint8),
             top.id: np.zeros((6, 8, 4), dtype=np.uint8)}
    pixels, origin, size = merge_layers_pixels(session, cache, bottom=bottom, top=top)
    assert origin == (0.0, 0.0)
    assert size == (8.0, 6.0)
    assert pixels.shape == (6, 8, 4)
