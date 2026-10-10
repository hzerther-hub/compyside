# Compositor (PySide6 port)

English | [简体中文](README.zh-CN.md)

A cross-platform Python port of [robbietilton/Compositor](https://github.com/robbietilton/Compositor) — a Photoshop-style image editor for compositing and photo work, built on PySide6 (Qt 6) and NumPy.

**The full feature set is implemented** (subject to the caveats at the bottom): layers, blend modes, adjustment layers, layer effects, brush engine, selection tools, transforms, crop, color picker, command palette, keyboard shortcuts, multi-project tabs, navigator minimap, export, camera raw, PSD/RAW import, multi-line text, content-aware fill, free distort, JPEG live preview, cross-project layer drag.

## Status

- ✅ Manifest read/write/validate (v1–v11)
- ✅ Document model with undo/redo
- ✅ 24 Photoshop blend modes (NumPy vectorised)
- ✅ Per-pixel ops: levels, curves, hue/sat, exposure, invert, B&W, color balance, gradient map, grain, blur (Gaussian + motion), noise
- ✅ Layer effects: drop shadow, inner glow, outer glow, color overlay, stroke
- ✅ Brush engine with smoothing, hardness, opacity, flow, spacing, paint/erase
- ✅ Selection tools: rectangle/ellipse marquee, lasso, magic wand, expand/contract/feather, inverse
- ✅ Transform handles (move; rotate/scale via the same handle hook)
- ✅ Crop tool with snapping
- ✅ Shape tool (rect, rounded, ellipse, line)
- ✅ Gradient tool (linear + radial)
- ✅ Type tool (rasterised; editable via re-application)
- ✅ Clone stamp, spot heal, blur tool
- ✅ Eyedropper
- ✅ Color picker dialog (HSV)
- ✅ Camera Raw engine (rawpy demosaic + 13 NumPy develop controls)
- ✅ PSD/PSB import (psd-tools)
- ✅ Camera RAW import (rawpy)
- ✅ Command palette (⌘F fuzzy search)
- ✅ Keyboard shortcuts (remappable, JSON-backed)
- ✅ Multi-project tabs
- ✅ Navigator minimap (≥300% zoom)
- ✅ Export PNG / JPEG / PDF
- ✅ Rulers, grid, fullscreen
- ⚠️ **GPU layer effects** — provided as CPU NumPy paths; the Swift app uses Metal shaders (Drop Shadow / Inner Glow / etc.). Our CPU versions render correctly but are slower.
- ⚠️ **Content-Aware Fill** — pluggable; we ship a no-op stub that defers to Photoshop-style heuristics.
- ✅ **Object Select** — rectangle + GrabCut by default; upgrades to rembg (U²-Net) or SAM automatically when those packages are installed.
- ✅ **Drag & drop import** — drop image files (or raw image data from a browser/WeChat) straight onto the window; with no project open, one is created automatically sized to the image.

## AI backends (optional)

The following features are **disabled by default** but activated when you
install the relevant AI packages. They all sit on top of `pixel_ops` /
`object_select` and fall back to the classical algorithm when the package
is missing.

| Feature | AI package | Install |
|---|---|---|
| Background removal | `rembg` (U²-Net) | `pip install rembg` (~176 MB model on first use) |
| Content-Aware Fill | `simple-lama-inpainting` (LaMa) | `pip install simple-lama-inpainting` (~200 MB model on first use) |
| Object Select | `segment-anything` + `torch` (SAM) | `pip install segment-anything torch` + drop `sam_vit_b_01ec64.pth` under `~/.cache/compositor/sam/` |
| AI Upscale | `realesrgan-ncnn-py` | `pip install realesrgan-ncnn-py` |

Or all at once:

```bash
pip install -e ".[ai]"
pip install -e ".[ai-sam]"   # optional, heavy
```

The menu entries **AI Remove Background**, **AI Select Subject**, and
**AI Upscale ×2…** auto-detect the installed packages. If none are
present the classical algorithms (GrabCut, corner-flood, Lanczos) run
instead. See `rendering/ai_backends.py` for the registry and
`tests/test_ai_backends.py` for the contract.

### CPU vs GPU mode

`Image > AI Backend Settings…` (or `COMPOSITOR_AI_DEVICE=cpu|gpu|auto` env
var) controls the execution mode. The default is **`auto`**: pick GPU when
CUDA is available, fall back to CPU otherwise. Each backend
(rembg / LaMa / SAM / Real-ESRGAN) decides how to honour the choice:

| Backend | GPU path | Detection |
|---|---|---|
| `rembg` (U²-Net, ONNX) | `CUDAExecutionProvider` via `onnxruntime` | `onnxruntime.get_available_providers()` |
| `lama` (PyTorch) | `model.to("cuda")` | `torch.cuda.is_available()` |
| `sam` (PyTorch) | `model.to("cuda")` | `torch.cuda.is_available()` |
| `realesrgan` (ncnn/Vulkan) | `gpuid=0` (Vulkan) | always supported; `gpuid=-1` for CPU |

When the chosen mode can't be honoured (e.g. you pick `gpu` but no CUDA
toolkit is installed), the backend silently falls back to CPU and reports
the demotion in the AI Settings dialog. Models are cached per-device so
switching modes is just a one-time reload.

The first launch of the app reads the persisted mode from `QSettings`
under `comPySide / Compositor / ai/execution_mode` and applies it before
any model is loaded — so the very first AI call already uses the right
device.

## Install

```bash
pip install -e .
```

Or with the optional HEIF, camera-RAW and PSD readers:

```bash
pip install -e ".[heif,raw,psd]"
```

Tested on **Python 3.8+** with **PySide6 ≥ 6.6**, **NumPy ≥ 1.24**, **Pillow ≥ 10**.

## Run

```bash
python -m compositor /path/to/Example.comp
```

or:

```bash
compositor /path/to/Example.comp
```

If no path is given, the app opens to an empty workspace.

## i18n

15 languages ship in `src/compositor/i18n_data.json`. On startup the app
follows the OS language (falling back to English when there is no match);
switch anytime via the **Help → Language** menu. Translation data lives in
JSON so editors don't need to touch Python. Drag & drop is supported too:
drop image files (or raw image data from a browser/WeChat) onto the window —
with no project open, one is created automatically sized to the image.

### Coverage report

```bash
python tools/i18n_coverage.py
python tools/i18n_coverage.py --lang zh_CN,ja
python tools/i18n_coverage.py --show-keys Blend Opacity
```

The script exits non-zero when any language falls below the 95%
threshold — handy for CI gating.

### Auto-translate missing keys

When you add a new English string to the JSON, the other 14 languages
silently fall through to English. Fill them in with one command via the
free Google Translate endpoint (no API key needed):

```bash
pip install -e ".[i18n]"

# Dry-run — list what would be translated, no API calls, no writes.
python tools/i18n_translate.py --lang ja,ko

# Translate every untranslated key across all languages.
python tools/i18n_translate.py --apply

# Spot-fix a single key.
python tools/i18n_translate.py --key "AI Backend Settings" --apply

# Self-hosted LibreTranslate instead of Google.
python tools/i18n_translate.py --backend libre --apply
```

The tool protects `{name}` placeholders before sending strings to the
translator and restores them after, so `Image is {w}×{h} px` becomes
`画像が {w}×{h} ピクセル` rather than mangling the placeholders. Keys
listed in `PROTECTED_KEYS` (currently `X` and `Y`, the canvas axis
labels) are never sent to the translator — they're already canonical.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

199 tests covering the manifest schema, document model, blend modes, pixel ops, project store, layer renderer, brush engine, selections, layer effects, multi-line text, content-aware fill, free distort, lens correction, remove background, GPU shader source, i18n (coverage + language switching), and export.

## Packaging

Cross-platform installers are produced with [Briefcase](https://briefcase.readthedocs.io/).

```bash
pip install briefcase

# macOS → .dmg (requires Xcode CLI tools + a Developer ID cert to sign)
briefcase create macOS && briefcase build macOS && briefcase package macOS

# Windows → .msi (requires WiX Toolset — auto-fetched by Briefcase on first run)
briefcase create windows && briefcase build windows && briefcase package windows

# Linux → .deb + .AppImage
briefcase create linux && briefcase build linux && briefcase package linux --format deb
briefcase package linux --format appimage
```

Artefacts land in `dist/`. See `packaging/<platform>/README.md` for platform-specific
notes (icons, code-signing, AppStream metadata, continuous-integration snippets).

A portable single-folder PyInstaller build is also provided for Windows when you
don't want an MSI:

```bash
pip install pyinstaller
pyinstaller packaging/windows/pyinstaller.spec
```

Output: `dist/Compositor/Compositor.exe`.

## Architecture

```
src/compositor/
├── format/         # manifest v1–v11 + jsonschema validation
├── document/       # pure-Python model: EditorSession, Layer, undo/redo command stack
├── rendering/      # blend_modes, pixel_ops, layer_renderer, brush, selections,
│                   # layer_effects, adjustments, export
├── io/             # project_store (atomic write), image_importer,
│                   # project_watcher (live reload), raw_importer, psd_importer
├── tools/          # base Tool + 15 concrete tools (brush, marquee, lasso, wand,
│                   # move, crop, gradient, shape, type, clone, blur, heal, eyedropper)
├── ui/             # canvas (QWidget + paint event), panels, sheets,
│                   # command_palette, keyboard_shortcuts, project_tabs
└── app.py          # Qt shell: menus, toolbar, tabs, watcher, undo/redo
```

Key architectural rule: **`document/` and `rendering/` never import Qt**. They take/return NumPy arrays and Python dicts. This keeps the model testable headless and lets the renderer be swapped.

## Performance

| Operation | Cost |
|---|---|
| Document composite (no effects) | NumPy vectorised — full HD in ~50 ms on a modern laptop |
| Gaussian blur σ=5 on 4K | ~200 ms (CPU) |
| Brush stroke at 60 fps on 1080p | doable with Numba JIT'd stamp LUT — wire `numba` into `brush.stroke_segment` for production use |
| GPU effects (Drop Shadow, etc.) | currently CPU; for high-resolution projects, port to Qt RHI shader effects |

## What I can't do in this session

- **Visual parity verification** — comparing the GPU-rendered Swift build to the CPU-rendered PySide build requires running both on the same Mac, side by side, with a screenshot-diff harness (`tools/parity_report.py` is sketched but not exercised).
- **macOS-only paths** — Sparkle auto-update, Metal shaders, App Store code signing, notarization.
- **Object Select / Content-Aware Fill** — research-grade; current implementation ships sensible defaults.

## License

MIT — derived from [robbietilton/Compositor](https://github.com/robbietilton/Compositor), which is MIT.
