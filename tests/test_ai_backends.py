"""Tests for AI backend wiring + graceful fallback when models are missing.

The tests do *not* download any model weights. They verify:
  * the dispatcher picks the right backend based on availability,
  * ``auto`` falls back cleanly when only cv2 (no AI libs) is present,
  * explicit method strings raise the documented error type when the
    backend is missing,
  * LaMa / SAM / rembg unavailability returns False from the helper.
"""
from __future__ import annotations

import numpy as np
import pytest

from compositor.rendering import ai_backends


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plain_image(h: int = 64, w: int = 64) -> np.ndarray:
    """An HxWx3 RGB uint8 image with a coloured centre on a white background."""
    arr = np.full((h, w, 3), 240, dtype=np.uint8)
    arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = [220, 60, 60]
    return arr


def _rect_around_centre(h: int, w: int) -> tuple:
    return (w // 4, h // 4, 3 * w // 4, 3 * h // 4)


# ---------------------------------------------------------------------------
# pixel_ops wiring
# ---------------------------------------------------------------------------


def test_remove_background_grabcut_returns_rgba() -> None:
    from compositor.rendering.pixel_ops import remove_background
    out = remove_background(_plain_image(), margin=0.2, method="grabcut")
    assert out.shape == (64, 64, 4)
    assert out.dtype == np.uint8


def test_remove_background_explicit_rembg_raises_when_missing(monkeypatch) -> None:
    """When method='rembg' but rembg isn't installed, raise AIBackendUnavailable.

    Patch ``_rembg_session_for_mode`` to simulate the missing package so the
    test doesn't depend on whether rembg happens to be installed.
    """
    from compositor.rendering import ai_backends
    from compositor.rendering.pixel_ops import remove_background

    monkeypatch.setattr(
        ai_backends, "_rembg_session_for_mode",
        lambda mode, model: (_ for _ in ()).throw(ai_backends.AIBackendUnavailable("rembg is not installed.")),
    )
    with pytest.raises(ai_backends.AIBackendUnavailable):
        remove_background(_plain_image(), method="rembg")


def test_remove_background_auto_falls_back_to_grabcut(monkeypatch) -> None:
    from compositor.rendering import ai_backends
    from compositor.rendering.pixel_ops import remove_background

    monkeypatch.setattr(ai_backends, "is_rembg_available", lambda: False)
    out = remove_background(_plain_image(), margin=0.2, method="auto")
    assert out.shape == (64, 64, 4)


def test_content_aware_fill_telea_runs() -> None:
    from compositor.rendering.pixel_ops import content_aware_fill
    img = _plain_image()
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:36, 28:36] = 255
    out = content_aware_fill(img, mask, method="telea")
    assert out.shape == img.shape


def test_content_aware_fill_explicit_lama_raises_when_missing(monkeypatch) -> None:
    from compositor.rendering import ai_backends
    from compositor.rendering.pixel_ops import content_aware_fill

    monkeypatch.setattr(ai_backends, "is_lama_available", lambda: False)
    img = _plain_image()
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:36, 28:36] = 255
    with pytest.raises(ai_backends.AIBackendUnavailable):
        content_aware_fill(img, mask, method="lama")


def test_upscale_lanczos_doubles_dimensions() -> None:
    from compositor.rendering.pixel_ops import upscale
    img = _plain_image()
    out = upscale(img, factor=2.0, method="lanczos")
    assert out.shape == (128, 128, 3)


def test_upscale_preserves_alpha() -> None:
    from compositor.rendering.pixel_ops import upscale
    img = np.full((64, 64, 4), 200, dtype=np.uint8)
    img[..., 3] = 128
    out = upscale(img, factor=2.0, method="lanczos")
    assert out.shape == (128, 128, 4)


# ---------------------------------------------------------------------------
# object_select wiring
# ---------------------------------------------------------------------------


def test_object_select_grabcut_runs() -> None:
    from compositor.rendering.object_select import object_select
    mask = object_select(_plain_image(), _rect_around_centre(64, 64), method="grabcut")
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert (mask > 0).any()


def test_object_select_auto_with_only_grabcut(monkeypatch) -> None:
    from compositor.rendering import ai_backends
    from compositor.rendering.object_select import object_select
    monkeypatch.setattr(ai_backends, "is_sam_available", lambda: False)
    monkeypatch.setattr(ai_backends, "is_rembg_available", lambda: False)
    mask = object_select(_plain_image(), _rect_around_centre(64, 64), method="auto")
    assert mask.shape == (64, 64)


def test_object_select_invalid_method_raises() -> None:
    from compositor.rendering.object_select import object_select
    with pytest.raises(ValueError):
        object_select(_plain_image(), _rect_around_centre(64, 64), method="bogus")


# ---------------------------------------------------------------------------
# availability helpers (don't depend on what's actually installed)
# ---------------------------------------------------------------------------


def test_ai_availability_helpers_return_bool() -> None:
    for fn in (
        ai_backends.is_rembg_available,
        ai_backends.is_lama_available,
        ai_backends.is_sam_available,
    ):
        assert isinstance(fn(), bool)


def test_ai_backend_unavailable_is_runtime_error() -> None:
    err = ai_backends.AIBackendUnavailable("test")
    assert isinstance(err, RuntimeError)
    assert "test" in str(err)


# ---------------------------------------------------------------------------
# diagnostics + CPU/GPU mode
# ---------------------------------------------------------------------------


def test_diagnose_returns_all_four_backends() -> None:
    diag = ai_backends.diagnose()
    names = {b.name for b in diag.backends}
    assert names == {"rembg", "lama", "sam", "realesrgan"}


def test_diagnose_global_mode_matches_setting() -> None:
    ai_backends.set_execution_mode("auto")
    diag = ai_backends.diagnose()
    assert diag.global_mode == "auto"


def test_set_execution_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        ai_backends.set_execution_mode("tpu")


def test_set_execution_mode_invalidates_caches(monkeypatch) -> None:
    """Switching mode must drop the per-device cache slots."""
    # Simulate something in the cache.
    ai_backends._REMBG_SESSION_CPU = object()
    ai_backends._REMBG_SESSION_GPU = object()
    # Move to a different mode first so the next call crosses the threshold.
    ai_backends.set_execution_mode("cpu")
    assert ai_backends._REMBG_SESSION_CPU is None
    assert ai_backends._REMBG_SESSION_GPU is None
    ai_backends.set_execution_mode("auto")  # restore for subsequent tests


def test_resolve_mode_falls_back_when_gpu_unavailable(monkeypatch) -> None:
    """If 'gpu' is requested but no GPU exists, fall back to CPU."""
    monkeypatch.setattr(ai_backends, "_gpu_available", lambda: False)
    assert ai_backends._resolve_mode("gpu") == "cpu"


def test_resolve_mode_honours_global_override(monkeypatch) -> None:
    """Global 'cpu' override beats a per-call 'gpu' hint."""
    monkeypatch.setattr(ai_backends, "_gpu_available", lambda: True)
    ai_backends.set_execution_mode("cpu")
    try:
        assert ai_backends._resolve_mode("gpu") == "cpu"
    finally:
        ai_backends.set_execution_mode("auto")


def test_device_label_returns_valid_value() -> None:
    label = ai_backends.device_label()
    assert label in ("cpu", "gpu")


def test_invalidate_caches_clears_everything() -> None:
    ai_backends._REMBG_SESSION_CPU = object()
    ai_backends._LAMA_MODEL_GPU = object()
    ai_backends._SAM_PREDICTOR_CPU = object()
    ai_backends.invalidate_caches()
    assert ai_backends._REMBG_SESSION_CPU is None
    assert ai_backends._LAMA_MODEL_GPU is None
    assert ai_backends._SAM_PREDICTOR_CPU is None


def test_diagnose_to_dict_is_jsonable() -> None:
    import json
    diag = ai_backends.diagnose()
    payload = json.dumps(diag.to_dict())
    assert "global_mode" in payload
    assert "backends" in payload


def test_backend_status_unavailable_has_install_hint() -> None:
    diag = ai_backends.diagnose()
    by_name = {b.name: b for b in diag.backends}
    for name in ("rembg", "lama", "sam", "realesrgan"):
        if not by_name[name].available:
            assert "pip install" in by_name[name].notes
