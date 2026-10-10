# Compositor → PySide6: Survey and Design Doc

**Goal.** Re-implement [robbietilton/Compositor](https://github.com/robbietilton/Compositor) as a cross-platform image editor in Python + PySide6 (Qt 6), preserving the `.comp` project format and the user-visible workflow, and gaining Windows + Linux support.

**Out of scope for this doc.** Pixel-perfect parity with the Swift implementation, notarized releases, App Store distribution, iPad/iOS.

---

## 1. What the existing app actually is

Surveyed from `E:\compositor\compositor-temp\` (commit at clone time, 2026-10-10).

| Area | Files | LoC (approx) | Language |
|---|---|---|---|
| UI controls / sheets | `Compositor/UI/` (49 files) | ~12k | Swift (SwiftUI + AppKit) |
| Document model, edits | `Compositor/Document/` (57 files) | ~22k | Swift |
| Rendering | `Compositor/Rendering/` (~50 files) | ~15k Swift + 2.8k C | Swift + C |
| File I/O, PSD | `Compositor/IO/` (15 files) + `IO/PSD/` (6 files) | ~9k | Swift |
| App entry, appkit glue | `Compositor/CompositorApp.swift`, `ContentView.swift` | ~1k | Swift |

**C bridging surface** (`Compositor-Bridging-Header.h`): nine pixel-processing modules totaling **2,762 lines of C**:

| Module | Lines | Purpose |
|---|---|---|
| `AdjustPixels` | 1185 | Levels / Curves / Hue / Saturation / Exposure / Color Balance kernels |
| `DitherPixels` | 546 | Bayer / scanline / tonal contrast / vignette dither |
| `HealPixels` | 259 | Spot-heal compositing |
| `WandPixels` | 205 | Magic wand / color-range selection |
| `ContentFill` | 93 | Content-aware fill patch |
| `LevelsPixels` | 60 | Levels math primitives |
| `NoisePixels` | 49 | Grain / noise generation |
| `LensPixels` | 37 | Lens correction |
| `BrushPixels` | 50 | Brush stamp lookup |

**GPU layer (Metal):** `MetalLayerEffects.swift`, `MetalWarp.swift`, `MetalBrushCoverage.swift`, `GPUCanvas.swift` — Drop shadow, inner glow, color overlay, blur, warps, brush coverage. All Metal Shading Language.

**Update mechanism:** [Sparkle](https://sparkle-project.org/) (`import Sparkle` in `CompositorApp.swift`). On macOS this can stay Sparkle via `pysparkle`; on Win/Linux we fall back to a built-in "check GitHub releases" updater.

---

## 2. Tech-stack mapping

### 2.1 Tier-by-tier replacement

| Concern | Compositor (today) | PySide port | Notes |
|---|---|---|---|
| UI toolkit | SwiftUI + AppKit | **PySide6** (QtWidgets + Qt Quick / QML where it helps) | Qt 6 has full coverage of every widget used: panels, sheets, color pickers, sliders, scrubbable numerics, command palette, tabs, scroll views, rulers, overlays. |
| Window/document model | SwiftUI `@SceneStorage` / AppKit `NSDocument` | `QMainWindow` + custom `Document` class; QFileSystemWatcher for live `.comp` reloads | |
| GPU / canvas | Metal via `MTKView` and Metal shaders | **Qt RHI** (the Qt Quick scene graph) for the canvas surface; **ModernGL** or **PyOpenGL** for direct control when needed | Qt 6's RHI runs OpenGL / Vulkan / Metal / Direct3D backends transparently — cross-platform for free. |
| Pixel kernels | Hand-written C | **NumPy** for vectorised ops + **Numba** JIT or **Cython** for the inner loops that NumPy can't express cleanly | NumPy covers ~70% of the C surface with 1-line vectorised replacements. |
| Color management | Core Image / ICC core | **littlecms** via `python-colormath` or `PyLittleCMS` (`pip install littlecms`) | |
| PSD/PSB parsing | Custom Swift (`IO/PSD/`) | **psd-tools** (`pip install psd-tools`) for read; write path can reuse it or stay read-only initially | psd-tools covers the layer structure; some vector text falls back to rasterised, which Compositor already does |
| Camera RAW | CoreImage / proprietary | **rawpy** (LibRaw wrapper, `pip install rawpy`) | |
| PNG / JPEG / TIFF / HEIC / SVG | ImageIO / Core Graphics | **Pillow** + **pillow-heif** + **cairosvg** | |
| Live file watch | FSEvents / DispatchSource | **QFileSystemWatcher** | |
| Keyboard shortcuts | AppKit key events + JSON map | **QShortcut** + JSON map (already in `KeyboardShortcuts.swift`) | |
| Auto-update | Sparkle | **Sparkle-CLI / pysparkle** on macOS, **custom GitHub-releases check** on Win/Linux | |
| Packaging | `scripts/release.sh` → `.dmg` | **PyInstaller** or **Briefcase** (BeeWare) — Briefcase handles Win `.msi`, macOS `.app/.dmg`, Linux `.AppImage/.deb` natively | |
| Color picker | `NSColorWell` | **`QColorDialog`** + custom eyedropper overlay | |
| Numeric scrub widget | `NumericScrub.swift` (drag-label-to-scrub) | Custom `QWidget` reimplementation (small, well-defined gesture) | |

### 2.2 What stays the same (cross-cutting)

These don't change because they're language-agnostic:

- **`.comp` file format** — same `manifest.json` + `images/<uuid>.png`. v11 stays the current version. Tests written against the Swift app's fixtures must still pass.
- **PNG-as-layer** convention — 8-bit RGBA per layer, 8-bit grayscale masks.
- **Atomic write protocol** for live editing: write new PNGs first, then rename a `.manifest.json.tmp` over `manifest.json`.
- **Blend-mode names** (24 enum values, exact spelling required).
- **All keyboard shortcuts** as a JSON map.

---

## 3. Subsystem-by-subsystem rewrite cost

Legend: **L** = little effort (drop-in library), **M** = medium (rewrite but well-defined spec), **H** = heavy (research / weeks of work), **X** = explore (no off-the-shelf solution, custom work).

### 3.1 UI layer (`UI/`)

| Subsystem | Cost | PySide equivalent |
|---|---|---|
| `LayersPanel`, `BlendModePicker`, `NativeLayerList` | **L** | `QTreeView` with custom delegate for blend-mode badge |
| `CommandPalette` / `CommandPaletteView` (⌘F search) | **L** | `QCompleter` + fuzzy filter; same JSON manifest as Swift version |
| `CanvasRulers`, `GridSettingsSheet` | **L** | Custom paint on `QWidget`; rulers are 2 cm of code |
| `FloatingPanel` | **L** | `QDockWidget` (it's literally what it's for) |
| All `*Controls.swift` and `*Sheet.swift` (filter / adjustment panels) | **L–M** | `QWidget` subclasses with `QSlider`, `QDoubleSpinBox`, `QPushButton`, custom curve editor |
| `NavigatorMinimap` (overlay minimap at 300%+) | **M** | Custom `QWidget` painted on top of canvas viewport |
| `ColorPickerSheet` + eyedropper | **M** | `QColorDialog` + global screen-grab via `QScreen.grabWindow()` / `mss` |
| `KeyboardShortcuts` (remappable map) | **L** | `QShortcut` per binding; same JSON config |
| `NumericScrub` (drag-label-to-scrub) | **M** | Custom `QWidget` with mouse-down + delta math |
| `InlineTextEditor` | **H** | Switch to Qt's `QTextEdit` over a transformed overlay; embedding rich text on a transformed canvas is fiddly |
| `RawDevelopSheet`, `CameraRaw*Controls` (the whole Camera Raw panel) | **M** | Lift-and-shift UI; the work is the *engine* (see 3.4) |

### 3.2 Document model (`Document/`)

| Subsystem | Cost | Notes |
|---|---|---|
| `EditorSession` (top-level document state, undo/redo) | **M** | Replace with a pure-Python `EditorSession` class; undo/redo via a command-stack pattern (`cmdpattern` or hand-rolled ~150 LoC) |
| `LayerTransform`, `LayerGroups`, `LayerMask`, `LayerEffects`, `LayerMerge`, `LayerFlip`, `LayerAppearance` | **M** | Pure data + math. Mostly translation from Swift structs to `@dataclass`es, with `__eq__`, `to_dict`, `from_dict`. |
| Selection tools: `ColorRangeSelection`, `FloatingSelection`, marquee, lasso | **M** | Mask-of-bool `numpy.ndarray` matching canvas size; ops are `np.where`, `np.any`, `np.logical_and/or/xor` |
| `Adjustments` (Hue/Sat, Levels, Curves, Exposure, …) | **M** | Apply to `numpy` arrays; see 3.4 for kernel strategy |
| `Distort`, `CanvasSize`, `ImageTrim`, `Crop`, `CanvasRotation` | **M** | `cv2.warpAffine` / `cv2.remap` |
| `CloneStamp`, `SpotHealing`, `BlurTool`, `Gradient`, `Shape`, `Type` | **M–H** | Brush engine is its own work (see 3.4). Healing is `X`-grade research. |

### 3.3 Rendering & I/O

| Subsystem | Cost | Notes |
|---|---|---|
| C pixel kernels (`Rendering/*.c`) | **M** (large LoC, but mechanical) | Translate to NumPy. The 1185-line `AdjustPixels.c` is the bulk — most of it is per-channel ops over a `uint8` array, which NumPy expresses in one line each. Use Numba `@njit` for the loops that don't vectorise cleanly (e.g. brush stamp lookup). |
| Metal shader layer effects (`MetalLayerEffects.swift` etc.) | **H** | Port MSL → GLSL (use **shaderc** to cross-compile, or hand-translate). Drive them through Qt's RHI shader-effect pipeline. This is the most visually consequential work. |
| `LayerRenderer` / `TiledLayerRenderer` / `GPUCanvas` | **H** | Replace with a tiled `QQuickItem` (Qt Quick scene graph) that renders each visible tile. Tiling logic is well-defined. |
| `DownsampleCache` (sharp downsampled mipmaps) | **L** | `cv2.pyrDown` / Pillow `Image.resize` with `LANCZOS` |
| `ProjectStore` / `ProjectWatcher` (atomic write + live reload) | **L** | The protocol is language-agnostic; use `os.replace` for atomic rename and `QFileSystemWatcher` for live reload |
| `ImageImporter` (PNG/JPEG/HEIC/TIFF/SVG) | **L** | Pillow + pillow-heif + cairosvg |
| `ImageExporter` (PNG/JPEG/PDF) | **L** | Pillow + `reportlab` (or `cairo` for PDF) |
| `RawImporter` (camera RAW) | **L** | `rawpy.postprocess()` |
| `IO/PSD/` (read PSD/PSB) | **M** | `psd-tools` covers layers/masks/groups; missing pieces (vector text → raster) match the Swift behavior |
| `RecentProjects` | **L** | `QSettings` |
| `CompositorApplicationDelegate` | **L** | `QApplication` subclass |

### 3.4 The hard bits — non-trivial algorithms

These are the items where "swap the library" is the wrong mental model. Each is a self-contained research project.

| Subsystem | Cost | Strategy |
|---|---|---|
| **Camera Raw engine** (demosaic, exposure, curves, color grading, detail/optics/geometry panels) | **H** | Lift the **UI** cheaply with `rawpy`. The **panel logic** (light/color/curves/mixer/grading/detail/optics/geometry on top of the demosaiced buffer) is its own ~2000 LoC subsystem. Use `rawpy` for demosaic, then a NumPy pipeline for the rest. Cross-check against the Swift version's output on a fixed set of RAW fixtures. |
| **Content-Aware Fill** | **X** | The Swift implementation is non-trivial and not fully described in the README. Two options: (a) read `ContentFill.c` carefully and port the algorithm, (b) integrate **OpenCV's `cv2.inpaint`** as a pragmatic v1 and ship a "basic" mode behind the same menu item. |
| **Wand / Object select** | **H** | Color-range (wand) → `numpy` flood-fill with tolerance, fairly easy. **Object select** is segmentation territory; v1 can ship wand-only, defer object-select. |
| **Spot heal / Clone stamp** | **H** | Clone stamp is straightforward offset+sample. Spot heal is patch-based texture synthesis; port `HealPixels.c` (~250 lines) and verify visually against the Swift version on a battery of test images. |
| **Brush engine** (size, hardness, opacity, smoothing, paint vs erase, stamp lookup) | **M** | The C version is only 50 lines of stamp LUT — the real work is the smoothing algorithm and pressure curve. Port `BrushPixels.c`, use Numba for the stamp LUT, smooth via `numpy` + a small state filter. |
| **GPU layer effects** (drop shadow, inner glow, outer glow, stroke, color overlay) | **H** | Translate MSL → GLSL. Each effect is ~50–150 lines of GLSL. Drive via Qt RHI's `QSGShaderEffect`. Test by screenshot diff against the Swift build on a fixed scene. |
| **Metal warp / distort** (`MetalWarp.swift`) | **H** | Free distort is `cv2.remap` + a small UI. Other warps (liquify-style) are deferred to v2. |
| **Live project watch + atomic reload** | **L** | `QFileSystemWatcher` + `os.replace`. The protocol is well-specified in `docs/writing-comp-files.md`. |
| **Undo/redo across all the above** | **M** | Command-stack pattern; each operation produces a `(do, undo)` pair of pure-data diffs. Photoshop's undo granularity is fine — no need for operation-level coalescing for v1. |

---

## 4. Architecture (target)

```
comPySide/
├── pyproject.toml                 # Poetry/pip build; Briefcase project metadata
├── README.md
├── LICENSE                        # MIT (Compositor is MIT)
├── src/
│   └── compositor/
│       ├── __main__.py            # python -m compositor
│       ├── app.py                 # QApplication setup, theme, dark mode
│       ├── ui/                    # all Qt widgets / sheets / panels
│       │   ├── layers_panel.py
│       │   ├── command_palette.py
│       │   ├── sheets/            # filter / adjustment sheets
│       │   ├── controls/          # sliders, scrubbers, color picker
│       │   └── canvas/            # QQuickView-based canvas surface
│       ├── document/              # pure-Python model (no Qt imports here)
│       │   ├── editor_session.py
│       │   ├── layers.py          # @dataclass Layer, Group, AdjustmentLayer
│       │   ├── transforms.py
│       │   ├── selections.py
│       │   ├── adjustments/       # levels.py, curves.py, hue_sat.py, ...
│       │   ├── tools/             # brush.py, clone.py, heal.py, ...
│       │   └── history.py         # undo/redo command stack
│       ├── rendering/
│       │   ├── pixel_ops.py       # NumPy kernels (translates AdjustPixels.c et al)
│       │   ├── numba_kernels.py   # JIT'd hot loops (brush LUT, heal patch)
│       │   ├── blend_modes.py     # vectorised blend modes (24 modes)
│       │   ├── canvas_item.py     # QQuickItem that paints the tiled canvas
│       │   ├── shaders/           # GLSL translations of MetalLayerEffects etc.
│       │   └── tiles.py           # tile cache, mipmaps
│       ├── io/
│       │   ├── project_store.py   # atomic .comp read/write
│       │   ├── project_watcher.py # QFileSystemWatcher wrapper
│       │   ├── image_importer.py  # PNG/JPEG/HEIC/TIFF/SVG
│       │   ├── image_exporter.py  # PNG/JPEG/PDF
│       │   ├── raw_importer.py    # rawpy
│       │   ├── psd_reader.py      # psd-tools wrapper
│       │   └── recent_projects.py # QSettings
│       ├── format/
│       │   └── manifest.py        # v1–v11 schema, validation (jsonschema)
│       └── platform/
│           ├── macos.py           # Sparkle update
│           ├── windows.py         # GitHub-releases update + toast
│           └── linux.py           # AppImage update
├── tests/
│   ├── fixtures/                  # .comp files copied from Swift app for parity
│   ├── test_format_roundtrip.py   # open Swift-saved .comp, re-save, diff
│   ├── test_blend_modes.py        # vectorised blend vs Photoshop reference
│   ├── test_pixel_ops.py          # NumPy kernel vs C reference output
│   └── test_io.py
├── tools/
│   ├── screenshot_diff.py         # render same .comp on both apps, diff PNG
│   └── parity_report.py
└── packaging/
    ├── briefcase/                 # Briefcase project files
    └── pyinstaller/               # fallback single-binary build
```

**Key architectural rule:** the `document/` package has **no Qt imports**. It takes `numpy.ndarray`s in and out. This keeps the model unit-testable without a display server and makes the rendering layer swappable (the Swift app's renderer can be re-targeted later if needed).

---

## 5. Phased delivery plan

Realistic for one developer working on it part-time. Each phase ends with something runnable.

### Phase 0 — Skeleton (1–2 weeks)
- Briefcase project, `QMainWindow`, empty canvas, dark theme.
- CI on GitHub Actions: Windows, macOS, Linux runners build and run a `pytest` smoke test.
- Result: a window that opens on every platform.

### Phase 1 — MVP viewer (3–4 weeks)
- Open `.comp` folder (read manifest, load PNG layers).
- Render stacked layers on a `QQuickItem` canvas with pan / zoom.
- Atomic write protocol implementation; live reload via `QFileSystemWatcher`.
- Round-trip test: open Swift-saved `.comp`, save, diff against original.
- Result: a read-only `.comp` viewer that works on Win/macOS/Linux. Useful immediately for AI agents writing `.comp` files on any OS.

### Phase 2 — Layer ops + history (4–6 weeks)
- Layers panel (visibility, opacity, blend mode, rename, reorder, nest, duplicate, delete).
- Transform handles on canvas (move/scale/rotate/flip, with snapping).
- Layer masks (paint, fill, invert, blur, feather).
- Clipping masks and group masks.
- Adjustment layers: Levels, Curves, Hue/Sat, Exposure, Invert, B&W, Color Balance (the "easy" set).
- Undo/redo.
- Result: a usable editor for non-destructive compositing, matching Compositor v1.0-era feature parity.

### Phase 3 — Tools (6–8 weeks)
- Brush (paint / erase, size/hardness/opacity/smoothing).
- Clone stamp, spot heal, blur tool.
- Shape tool, gradient tool, type tool.
- Selections: marquee, lasso, wand.
- Floating selection, modify selection (expand/contract/feather).
- Crop tool, canvas/image size, trim.
- Result: full painting + compositing workflow.

### Phase 4 — GPU layer effects + filters (4–6 weeks)
- Port `MetalLayerEffects.swift` → GLSL; drive via Qt RHI `QSGShaderEffect`.
- Stroke, drop shadow, inner/outer glow, color overlay.
- Gaussian blur, motion blur, add noise, vignette, bloom, dither, scanlines.
- Result: visual parity with the Swift app's effects.

### Phase 5 — File I/O breadth (3–4 weeks)
- Import: JPEG, PNG, HEIC, TIFF, SVG, camera RAW (rawpy develop step).
- Import: PSD/PSB (via psd-tools, with conversion report sheet).
- Export: PNG, JPEG, PDF (single page at print size).
- Drag-and-drop import, drag-and-drop between projects.
- Result: Compositor can act as a Photoshop import target on every OS.

### Phase 6 — Camera Raw + Content-Aware (6–8 weeks)
- Camera Raw filter panel + engine (light, color, curves, mixer, grading, detail, optics, geometry).
- Content-Aware Fill (port `ContentFill.c`, or integrate `cv2.inpaint` as a basic mode first).
- Magic Wand → Object Select (segmentation).
- Result: advanced selection / raw development parity.

### Phase 7 — Polish + distribution (4 weeks)
- Command palette (⌘F), keyboard remapping, numeric scrub.
- Fullscreen, navigator minimap, rulers, guides, grid, snap-to.
- Multi-project tabs, recent projects, automatic updates (Sparkle on macOS, GitHub releases elsewhere).
- Briefcase-packaged installers for Win (`.msi`), macOS (`.dmg`), Linux (`.AppImage`, `.deb`).
- Result: shippable 1.0.

**Total realistic estimate:** ~30–45 weeks for one developer. The "killer features" (Camera Raw, Content-Aware, GPU effects) dominate the second half.

---

## 6. Risks and open questions

1. **Performance.** NumPy + Numba will get us close, but brush stroke latency at 4K with a heavy layer stack will need profiling. Tile the canvas (the Swift version already does — `TiledLayerRenderer.swift`) and only re-render dirty tiles.
2. **Metal → GLSL translation.** Test against a screenshot-diff battery from day one of Phase 4. The Swift shader code is the source of truth; treat the GLSL ports as a regression risk.
3. **`.comp` parity.** A `tools/parity_report.py` that opens the same `.comp` in both apps and reports pixel-level diff is the single most important test harness. Start it in Phase 1.
4. **PSD import fidelity.** `psd-tools` is good but not perfect. Spot-check Compositor's existing PSD fixtures early.
5. **Sparkle on non-macOS.** If `pysparkle` doesn't have Win/Linux bindings, write a thin updater that hits GitHub Releases API. ~200 LoC.
6. **HEIC support on Linux.** `pillow-heif` covers it, but Linux distros may need libheif installed. Document it in the README.
7. **The Swift app will keep moving.** Every port milestone should be cross-checked against the current main. Pin a Compositor commit as the parity baseline; revisit quarterly.

---

## 7. What I can usefully build here

Since this session is on Windows 11, I can:

- Author all the Python source files and write tests against synthetic fixtures (no display server needed for the `document/` and `io/` packages — they're Qt-free by design).
- Validate NumPy kernels against the Swift app's C reference output if you provide the test images.
- Run `pytest` to confirm the round-trip tests pass.
- Stand up the Briefcase project and confirm CI builds on all three platforms.

What I **can't** do here:

- Verify visual parity (no GPU/shader runtime).
- Run the macOS-only Camera Raw / Sparkle code paths.
- Run the full Qt UI smoke test without an X server / Wayland / Windows GUI session.

**Suggested first deliverable:** Phase 1 (MVP viewer). It proves the architecture and is testable on every platform including this Windows box. From there we sequence the phases above.

---

## Sources

- [robbietilton/Compositor](https://github.com/robbietilton/Compositor) — surveyed at commit on 2026-10-10
- [Compositor README](https://github.com/robbietilton/Compositor/blob/main/README.md)
- [Compositor AGENTS.md](https://github.com/robbietilton/Compositor/blob/main/AGENTS.md)
- [Compositor project format (v1–11)](https://github.com/robbietilton/Compositor/blob/main/docs/project-format.md)
- [Writing .comp files (live-edit protocol)](https://github.com/robbietilton/Compositor/blob/main/docs/writing-comp-files.md)
- [psd-tools](https://psd-tools.readthedocs.io/) — PSD/PSB reader for Python
- [rawpy](https://pypi.org/project/rawpy/) — LibRaw wrapper
- [littlecms Python bindings](https://pypi.org/project/littlecms/)
- [Briefcase](https://briefcase.readthedocs.io/) — BeeWare cross-platform packaging
- [Qt RHI and shader effects](https://doc.qt.io/qt-6/qsgshadereffect.html)
