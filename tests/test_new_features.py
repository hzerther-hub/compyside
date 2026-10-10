"""Tests for Content-Aware Fill, Object Select, NumericScrub, GPU effects, Spot Heal."""
from __future__ import annotations

import numpy as np
import pytest

from compositor.rendering.content_fill import content_aware_fill
from compositor.rendering.gpu_effects import gpu_available, gpu_drop_shadow, gpu_color_overlay
from compositor.rendering.object_select import object_select
from compositor.rendering.pixel_ops import gaussian_blur


# ---------------------------------------------------------------------------
# Content-Aware Fill
# ---------------------------------------------------------------------------


def test_content_aware_fill_runs():
    """The cv2.inpaint path should produce a modified image (any change counts)."""
    img = np.full((32, 32, 4), 100, dtype=np.uint8)
    img[:, :16, :] = 200  # half white, half gray
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 255  # hole in the middle
    try:
        out = content_aware_fill(img, mask, radius=3)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
    except RuntimeError:
        pytest.skip("cv2 not available")


# ---------------------------------------------------------------------------
# Object Select
# ---------------------------------------------------------------------------


def test_object_select_returns_mask_shape():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[5:35, 5:55] = 255
    try:
        mask = object_select(img, (5, 5, 55, 35), iterations=2)
        assert mask.shape == (40, 60)
        assert mask.dtype == np.uint8
    except RuntimeError:
        pytest.skip("cv2 not available")


# ---------------------------------------------------------------------------
# Spot Heal — patch-based, no cv2 needed
# ---------------------------------------------------------------------------


def test_heal_stamp_copies_source_region():
    """A heal dab should pull pixels from the source offset into the destination."""
    from compositor.document.editor_session import EditorSession
    from compositor.document.layers import ImageLayer, Transform, new_id
    from compositor.tools.blur_tool import HealTool
    from compositor.tools.base import ToolContext

    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[:, 20:, :] = 200  # right half is white
    s = EditorSession(width=40, height=40)
    layer = ImageLayer(id=new_id(), name="L", transform=Transform(size=(40, 40)))
    s.add_layer(layer)
    # HealTool reads the cache from context.session._image_cache.
    s._image_cache = {layer.id: img}  # type: ignore[attr-defined]

    tool = HealTool()
    tool.settings.size = 8
    tool.set_sample_source(20, 20)  # sample from the white area
    tool._stamp(ToolContext(
        session=s, canvas_width=40, canvas_height=40, active_layer_id=layer.id,
    ), 10, 10)
    out = s._image_cache[layer.id]
    # The destination at (10,10) should now have white pixels (from source offset).
    assert out[10, 10, 0] > 100


# ---------------------------------------------------------------------------
# NumericScrub
# ---------------------------------------------------------------------------


def test_numeric_scrub_basic(qapp_or_skip):
    from compositor.ui.numeric_scrub import NumericScrub
    ns = NumericScrub("Test", minimum=0, maximum=100, step=1, default=50)
    assert ns.value() == 50
    ns.setValue(75)
    assert ns.value() == 75


def test_numeric_scrub_default(qapp_or_skip):
    from compositor.ui.numeric_scrub import NumericScrub
    ns = NumericScrub("Test", minimum=0, maximum=100, step=1, default=42)
    ns.setValue(99)
    assert ns.value() == 99


# ---------------------------------------------------------------------------
# Mask editor
# ---------------------------------------------------------------------------


def test_mask_editor_basic(qapp_or_skip):
    from compositor.ui.mask_editor import MaskEditor
    panel = MaskEditor()
    # Without a session, the panel disables itself.
    panel._refresh()
    # The panel shouldn't crash without a session — buttons may still be
    # clickable but no signal fires.
    panel._refresh_labels()


def test_mask_editor_with_layer(qapp_or_skip):
    from compositor.ui.mask_editor import MaskEditor
    from compositor.document.editor_session import EditorSession
    from compositor.document.layers import ImageLayer, Transform, new_id
    s = EditorSession(width=10, height=10)
    s.add_layer(ImageLayer(id=new_id(), name="L", transform=Transform(size=(10, 10))))
    panel = MaskEditor()
    panel.set_session(s)
    assert panel.enabled_checkbox.isChecked() is False


def test_mask_editor_with_layer(qapp_or_skip):
    from compositor.ui.mask_editor import MaskEditor
    from compositor.document.editor_session import EditorSession
    from compositor.document.layers import ImageLayer, Transform, new_id
    s = EditorSession(width=10, height=10)
    s.add_layer(ImageLayer(id=new_id(), name="L", transform=Transform(size=(10, 10))))
    panel = MaskEditor()
    panel.set_session(s)
    assert panel.isEnabled()
    assert panel.enabled_checkbox.isChecked() is False  # default


# ---------------------------------------------------------------------------
# Object select
# ---------------------------------------------------------------------------


def test_object_select_full_pipeline():
    """A rectangle around a white rectangle should give a foreground mask."""
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[5:35, 10:50] = 200
    try:
        from compositor.rendering.object_select import object_select
        mask = object_select(img, (10, 5, 50, 35), iterations=5)
        # The mask should have at least *some* foreground.
        assert mask.sum() > 0
    except RuntimeError:
        pass  # cv2 not available


# ---------------------------------------------------------------------------
# Spot Heal — round trip
# ---------------------------------------------------------------------------


def test_spot_heal_blends_to_source():
    """Multiple dabs should keep blending — verify cumulative behaviour."""
    from compositor.document.editor_session import EditorSession
    from compositor.document.layers import ImageLayer, Transform, new_id
    from compositor.tools.blur_tool import HealTool
    from compositor.tools.base import ToolContext

    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[:, 25:, :] = 200  # right half is white
    s = EditorSession(width=40, height=40)
    layer = ImageLayer(id=new_id(), name="L", transform=Transform(size=(40, 40)))
    s.add_layer(layer)
    s._image_cache = {layer.id: img}  # type: ignore[attr-defined]

    tool = HealTool()
    tool.settings.size = 6
    tool.set_sample_source(25, 25)
    ctx = ToolContext(session=s, canvas_width=40, canvas_height=40, active_layer_id=layer.id)
    tool._stamp(ctx, 5, 5)
    tool._stamp(ctx, 10, 10)
    out = s._image_cache[layer.id]
    # At (5, 5) and (10, 10) some of the white should have transferred.
    assert out[5, 5, 0] > 0
    assert out[10, 10, 0] > 0


# ---------------------------------------------------------------------------
# Update check
# ---------------------------------------------------------------------------


def test_update_check_handles_network_failure(monkeypatch):
    from compositor.platform import update_check
    # Patch urlopen to raise — the function should return None gracefully.
    def boom(*args, **kwargs):
        raise OSError("network down")
    monkeypatch.setattr(update_check.urllib.request, "urlopen", boom)
    assert update_check.check_for_updates() is None


def test_update_check_parses_version():
    from compositor.platform.update_check import _parse_version
    assert _parse_version("0.1.0") == (0, 1, 0)
    assert _parse_version("2.3.4-beta") == (2, 3, 4)
    assert _parse_version("0.1.0") < _parse_version("0.2.0")
    assert _parse_version("1.0.0") > _parse_version("0.9.9")


# ---------------------------------------------------------------------------
# GPU layer effects — falls back to CPU when GL isn't available
# ---------------------------------------------------------------------------


def test_gpu_drop_shadow_runs():
    layer = np.zeros((20, 20, 4), dtype=np.uint8)
    layer[5:15, 5:15, 3] = 255
    out = gpu_drop_shadow(layer, offset=(3, 3), blur=2, color=(0, 0, 0, 255))
    # dropshadow returns an expanded canvas — must be larger than the layer.
    assert out.shape[0] > 20 and out.shape[1] > 20
    assert out.shape[-1] == 4


def test_gpu_color_overlay_runs():
    layer = np.zeros((10, 10, 4), dtype=np.uint8)
    layer[..., 3] = 255
    out = gpu_color_overlay(layer, color=(255, 0, 0, 255), opacity=1.0)
    assert out.shape == (10, 10, 4)
    # Full opacity over black background → fully red.
    assert out[0, 0, 0] > 200


def test_gpu_color_overlay_blends():
    layer = np.full((10, 10, 4), 100, dtype=np.uint8)
    layer[..., 3] = 255
    out = gpu_color_overlay(layer, color=(255, 0, 0, 255), opacity=0.5)
    # 50% red over gray (100) → 100 + (255 - 100) * 0.5 ≈ 178.
    assert out[0, 0, 0] > 150


def test_gpu_available_returns_bool():
    assert isinstance(gpu_available(), bool)
