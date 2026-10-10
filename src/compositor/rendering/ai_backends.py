"""AI backend registry — pluggable heavy models for inpainting, background
removal, and segmentation.

Each backend is loaded lazily and cached at module level. A missing
dependency or a missing model file raises :class:`AIBackendUnavailable`,
which the calling code treats as "fall back to a classical algorithm".

Available backends:

* ``rembg`` — background removal (U²-Net family). Install via
  ``pip install rembg``. Model is downloaded on first use to
  ``~/.u2net/`` (~176 MB for the default U²-Net).
* ``lama``  — LaMa inpainting (Resolution-robust Large Mask Inpainting).
  Install via ``pip install simple-lama-inpainting`` (downloads ~200 MB on
  first call).
* ``sam``   — Meta Segment Anything (heavy: ~350 MB). Install via
  ``pip install segment-anything`` and drop a ``sam_vit_b_01ec64.pth``
  checkpoint under ``~/.cache/compositor/sam/``.
* ``realesrgan`` — Real-ESRGAN upscaler. CPU by default; pass
  ``gpuid=0`` to use Vulkan GPU when available.

CPU/GPU mode:
    Each backend respects ``_execution_mode`` (one of ``"auto"``,
    ``"cpu"``, ``"gpu"``). ``"auto"`` (the default) picks GPU when
    available, otherwise falls back to CPU. Override at runtime with
    :func:`set_execution_mode`, or via the env var
    ``COMPOSITOR_AI_DEVICE=cpu|gpu|auto``.

Each entry point mirrors the classical algorithm's signature so the
calling code is identical regardless of which path runs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


class AIBackendUnavailable(RuntimeError):
    """Raised when an AI backend is requested but cannot be loaded."""


_REMBG_SESSION: Optional[object] = None
_LAMA_MODEL: Optional[object] = None
_SAM_PREDICTOR: Optional[object] = None

# Per-mode (cpu/gpu) cache slots so toggling mode doesn't unload a model.
_REMBG_SESSION_CPU: Optional[object] = None
_REMBG_SESSION_GPU: Optional[object] = None
_LAMA_MODEL_CPU: Optional[object] = None
_LAMA_MODEL_GPU: Optional[object] = None
_SAM_PREDICTOR_CPU: Optional[object] = None
_SAM_PREDICTOR_GPU: Optional[object] = None

# Module-level execution mode.
_execution_mode: Literal["auto", "cpu", "gpu"] = "auto"


def set_execution_mode(mode: str) -> None:
    """Set the global AI execution mode.

    ``mode`` is one of ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    Switching mode invalidates the per-device caches so the next call
    loads under the new rule.
    """
    global _execution_mode
    mode = (mode or "auto").lower()
    if mode not in ("auto", "cpu", "gpu"):
        raise ValueError(f"unknown execution mode: {mode!r}")
    if mode != _execution_mode:
        _execution_mode = mode
        invalidate_caches()


def get_execution_mode() -> str:
    return _execution_mode


def invalidate_caches() -> None:
    """Drop every cached model so the next call re-loads under the current mode."""
    global _REMBG_SESSION, _REMBG_SESSION_CPU, _REMBG_SESSION_GPU
    global _LAMA_MODEL, _LAMA_MODEL_CPU, _LAMA_MODEL_GPU
    global _SAM_PREDICTOR, _SAM_PREDICTOR_CPU, _SAM_PREDICTOR_GPU
    _REMBG_SESSION = None
    _REMBG_SESSION_CPU = None
    _REMBG_SESSION_GPU = None
    _LAMA_MODEL = None
    _LAMA_MODEL_CPU = None
    _LAMA_MODEL_GPU = None
    _SAM_PREDICTOR = None
    _SAM_PREDICTOR_CPU = None
    _SAM_PREDICTOR_GPU = None


def _resolve_mode(device_hint: str) -> str:
    """Return the concrete device to use given a backend's hint.

    ``device_hint`` is one of ``"auto"``, ``"cpu"``, ``"gpu"``. The
    effective mode honours the global override then the hint, finally
    falling back to CPU when the requested device isn't usable.
    """
    mode = _execution_mode
    if mode == "auto":
        mode = device_hint
    if mode == "gpu" and not _gpu_available():
        return "cpu"
    return mode


def _gpu_available() -> bool:
    """Best-effort detection of any usable accelerator."""
    # CUDA via torch.
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    # DirectML / ROCm — same torch.cuda path usually.
    # onnxruntime with GPU provider installed.
    try:
        import onnxruntime as ort  # type: ignore
        providers = ort.get_available_providers()
        if any(p.startswith("CUDA") or p.startswith("Dml") for p in providers):
            return True
    except ImportError:
        pass
    return False


def device_label() -> str:
    """Human-readable description of the device the next call will use.

    Always returns ``"cpu"`` or ``"gpu"``; ``"auto"`` is internal-only.
    """
    mode = _resolve_mode("auto")
    return mode if mode in ("cpu", "gpu") else "cpu"


# ---------------------------------------------------------------------------
# Diagnostics — report the live state of every backend so the UI / logs /
# first-run wizard can decide whether to install more packages.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendStatus:
    """Live status of one AI backend."""

    name: str
    available: bool
    reason: str
    device: str  # 'cpu' / 'gpu' / 'n/a' — what the next call would actually use
    notes: str = ""


@dataclass(frozen=True)
class AIDiagnostics:
    """Aggregate result of `diagnose()`."""

    global_mode: str
    gpu_available: bool
    backends: list

    def to_dict(self) -> dict:
        return {
            "global_mode": self.global_mode,
            "gpu_available": self.gpu_available,
            "backends": [b.__dict__ for b in self.backends],
        }


def diagnose() -> AIDiagnostics:
    """Inspect every backend and report its live status.

    Use this from the AI Settings dialog or the first-run wizard. The
    result tells the user exactly which packages to install and which
    device each backend will land on under the current mode.
    """
    backends: list = []

    # rembg
    if is_rembg_available():
        mode = _resolve_mode("auto")
        note = ""
        try:
            import onnxruntime as ort  # type: ignore
            providers = ort.get_available_providers()
            if mode == "gpu" and not any(p.startswith("CUDA") for p in providers):
                note = "GPU requested but onnxruntime CUDA provider missing"
        except ImportError:
            if mode == "gpu":
                note = "onnxruntime not installed; CPU fallback"
        backends.append(BackendStatus(
            name="rembg",
            available=True,
            reason="installed",
            device=mode,
            notes=note,
        ))
    else:
        backends.append(BackendStatus(
            name="rembg",
            available=False,
            reason="not installed",
            device="n/a",
            notes="pip install rembg",
        ))

    # LaMa
    if is_lama_available():
        mode = _resolve_mode("auto")
        note = ""
        try:
            import torch  # type: ignore
            if mode == "gpu" and not torch.cuda.is_available():
                note = "GPU requested but torch CUDA not available"
        except ImportError:
            if mode == "gpu":
                note = "torch not installed; CPU fallback"
        backends.append(BackendStatus(
            name="lama",
            available=True,
            reason="installed",
            device=mode,
            notes=note,
        ))
    else:
        backends.append(BackendStatus(
            name="lama",
            available=False,
            reason="not installed",
            device="n/a",
            notes="pip install simple-lama-inpainting",
        ))

    # SAM
    if is_sam_available():
        mode = _resolve_mode("auto")
        note = ""
        from pathlib import Path
        ckpt = Path.home() / ".cache/compositor/sam/sam_vit_b_01ec64.pth"
        if not ckpt.is_file():
            note = f"checkpoint missing at {ckpt}"
        backends.append(BackendStatus(
            name="sam",
            available=True,
            reason="installed",
            device=mode,
            notes=note,
        ))
    else:
        backends.append(BackendStatus(
            name="sam",
            available=False,
            reason="not installed",
            device="n/a",
            notes="pip install segment-anything torch",
        ))

    # Real-ESRGAN
    if is_realesrgan_available():
        mode = _resolve_mode("auto")
        note = "Vulkan" if mode == "gpu" else "CPU"
        backends.append(BackendStatus(
            name="realesrgan",
            available=True,
            reason="installed",
            device=mode,
            notes=note,
        ))
    else:
        backends.append(BackendStatus(
            name="realesrgan",
            available=False,
            reason="not installed",
            device="n/a",
            notes="pip install realesrgan-ncnn-py",
        ))

    return AIDiagnostics(
        global_mode=_execution_mode,
        gpu_available=_gpu_available(),
        backends=backends,
    )


# ---------------------------------------------------------------------------
# rembg — background removal
# ---------------------------------------------------------------------------


def is_rembg_available() -> bool:
    try:
        import rembg  # noqa: F401
        return True
    except ImportError:
        return False


def _rembg_session_for_mode(mode: str, model_name: str):
    """Return a (cached) rembg session, picking the slot for ``mode``."""
    global _REMBG_SESSION_CPU, _REMBG_SESSION_GPU
    slot = _REMBG_SESSION_GPU if mode == "gpu" else _REMBG_SESSION_CPU
    if slot is not None:
        return slot
    try:
        from rembg import new_session  # type: ignore
    except ImportError as exc:
        raise AIBackendUnavailable("rembg is not installed. Run `pip install rembg`.") from exc
    # When GPU is requested, try to register the CUDA provider with the
    # underlying onnxruntime session. The Python rembg API doesn't expose
    # providers directly, so we patch via the session's internal attribute.
    providers: Optional[list] = None
    if mode == "gpu":
        try:
            import onnxruntime as ort  # type: ignore
            avail = ort.get_available_providers()
            if any(p.startswith("CUDA") for p in avail):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except ImportError:
            providers = None
    sess = new_session(model_name, providers=providers) if providers else new_session(model_name)
    if mode == "gpu":
        _REMBG_SESSION_GPU = sess
    else:
        _REMBG_SESSION_CPU = sess
    return sess


def rembg_remove_background(
    image: "np.ndarray", model: str = "u2net", *, device: str = "auto"
) -> np.ndarray:
    """Remove the background from an RGB/RGBA uint8 image using U²-Net.

    The returned array has an alpha channel where opaque = foreground.
    If the input is already RGBA the alpha is replaced; otherwise one is
    added.

    ``device`` is ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    """
    mode = _resolve_mode(device)
    session = _rembg_session_for_mode(mode, model)
    try:
        from rembg import remove  # type: ignore
    except ImportError as exc:
        raise AIBackendUnavailable("rembg is not installed.") from exc
    if image.ndim == 3 and image.shape[-1] == 4:
        rgb = image[..., :3]
    else:
        rgb = image
    out = remove(rgb, session=session)
    return np.asarray(out, dtype=np.uint8)


# ---------------------------------------------------------------------------
# LaMa — inpainting
# ---------------------------------------------------------------------------


def is_lama_available() -> bool:
    try:
        import simple_lama_inpainting  # noqa: F401
        return True
    except ImportError:
        return False


def _lama_model_for_mode(mode: str):
    global _LAMA_MODEL_CPU, _LAMA_MODEL_GPU
    slot = _LAMA_MODEL_GPU if mode == "gpu" else _LAMA_MODEL_CPU
    if slot is not None:
        return slot
    try:
        from simple_lama_inpainting import SimpleLama  # type: ignore
    except ImportError as exc:
        raise AIBackendUnavailable(
            "simple-lama-inpainting is not installed. Run `pip install simple-lama-inpainting`."
        ) from exc
    model = SimpleLama()
    if mode == "gpu":
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                # Move the underlying torch model to GPU.
                inner = getattr(model, "model", None) or getattr(model, "_model", None)
                if inner is not None and hasattr(inner, "to"):
                    inner.to("cuda")
        except ImportError:
            pass
        _LAMA_MODEL_GPU = model
    else:
        _LAMA_MODEL_CPU = model
    return model


def lama_inpaint(image: "np.ndarray", mask: np.ndarray, *, device: str = "auto") -> np.ndarray:
    """Inpaint the masked pixels using LaMa.

    ``device`` is ``"auto"``, ``"cpu"``, or ``"gpu"``.
    """
    from PIL import Image
    mode = _resolve_mode(device)
    if image.shape[-1] == 4:
        rgb = image[..., :3]
    else:
        rgb = image
    model = _lama_model_for_mode(mode)
    pil_image = Image.fromarray(rgb)
    pil_mask = Image.fromarray(mask).convert("L")
    result = model(pil_image, pil_mask)  # PIL.Image, RGB
    out_rgb = np.asarray(result, dtype=np.uint8)
    if image.shape[-1] == 4:
        return np.concatenate([out_rgb, image[..., 3:4]], axis=-1)
    return out_rgb


# ---------------------------------------------------------------------------
# SAM — Segment Anything (heavy)
# ---------------------------------------------------------------------------


def is_sam_available() -> bool:
    try:
        import segment_anything  # noqa: F401
        return True
    except ImportError:
        return False


def _sam_predictor_for_mode(mode: str, checkpoint: Optional[str]):
    global _SAM_PREDICTOR_CPU, _SAM_PREDICTOR_GPU
    slot = _SAM_PREDICTOR_GPU if mode == "gpu" else _SAM_PREDICTOR_CPU
    if slot is not None:
        return slot
    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise AIBackendUnavailable(
            "segment-anything is not installed. Run `pip install segment-anything torch`."
        ) from exc
    import os
    from pathlib import Path

    ckpt = (
        checkpoint
        or os.environ.get("COMPOSITOR_SAM_CHECKPOINT")
        or str(Path.home() / ".cache/compositor/sam/sam_vit_b_01ec64.pth")
    )
    if not Path(ckpt).is_file():
        raise AIBackendUnavailable(
            f"SAM checkpoint not found at {ckpt}. "
            "Download sam_vit_b_01ec64.pth from "
            "https://github.com/facebookresearch/segment-anything#model-checkpoints "
            f"and place it at {ckpt}."
        )
    model_type = "vit_b"
    device = "cuda" if mode == "gpu" and torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    if mode == "gpu":
        _SAM_PREDICTOR_GPU = predictor
    else:
        _SAM_PREDICTOR_CPU = predictor
    return predictor


def sam_segment(
    image: "np.ndarray",
    point_xy: tuple,
    *,
    checkpoint: Optional[str] = None,
    device: str = "auto",
) -> np.ndarray:
    """Return a binary mask (HxW uint8, 0/255) covering the object at
    ``point_xy``. ``device`` is ``"auto"`` / ``"cpu"`` / ``"gpu"``.
    """
    mode = _resolve_mode(device)
    predictor = _sam_predictor_for_mode(mode, checkpoint)
    rgb = image[..., :3] if image.shape[-1] == 4 else image
    predictor.set_image(rgb)
    point_coords = np.array([point_xy], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    return (masks[0].astype(np.uint8)) * 255


# ---------------------------------------------------------------------------
# Real-ESRGAN — upscaler (CPU/Vulkan GPU)
# ---------------------------------------------------------------------------


def is_realesrgan_available() -> bool:
    try:
        import realesrgan_ncnn_py  # noqa: F401
        return True
    except ImportError:
        return False


def realesrgan_upscale(
    image: "np.ndarray", factor: float = 2.0, *, device: str = "auto"
) -> np.ndarray:
    """Upscale via Real-ESRGAN. ``device`` is ``"auto"`` / ``"cpu"`` / ``"gpu"``.

    CPU mode sets ``gpuid=-1``; GPU mode uses Vulkan (``gpuid=0``) when
    available, otherwise falls back to CPU.
    """
    try:
        import realesrgan_ncnn_py  # type: ignore
    except ImportError as exc:
        raise AIBackendUnavailable(
            "realesrgan-ncnn-py is not installed. Run `pip install realesrgan-ncnn-py`."
        ) from exc
    from PIL import Image
    import numpy as _np

    mode = _resolve_mode(device)
    gpuid = 0 if mode == "gpu" else -1
    upsampler = realesrgan_ncnn_py.Realesrgan(gpuid=gpuid)
    pil = Image.fromarray(image[..., :3] if image.shape[-1] == 4 else image)
    out = upsampler.process(pil)
    out_arr = _np.asarray(out, dtype=_np.uint8)
    if image.shape[-1] == 4:
        h, w = image.shape[:2]
        nw, nh = int(round(w * factor)), int(round(h * factor))
        alpha_pil = Image.fromarray(image[..., 3], mode="L").resize(
            (nw, nh), Image.Resampling.LANCZOS
        )
        return _np.concatenate([out_arr, _np.array(alpha_pil)[..., None]], axis=-1)
    return out_arr


# ---------------------------------------------------------------------------
# Module init — honour env var.
# ---------------------------------------------------------------------------

_env_mode = os.environ.get("COMPOSITOR_AI_DEVICE", "").lower().strip()
if _env_mode in ("cpu", "gpu"):
    _execution_mode = _env_mode
