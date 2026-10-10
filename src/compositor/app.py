"""Qt application shell — the full Compositor PySide window.

Wires every panel, sheet, and tool together. Loads and saves .comp
projects, dispatches keyboard shortcuts, hosts multi-project tabs,
and runs the live-reload watcher.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QImage, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from . import i18n
from .document.editor_session import EditorSession
from .document.layers import (
    AdjustmentKind,
    AdjustmentLayer,
    AnyLayer,
    BlendMode,
    GroupLayer,
    ImageLayer,
    Transform,
    new_id,
)
from .format.manifest import (
    CURRENT_VERSION,
    Manifest,
    ManifestError,
    load_manifest,
    save_manifest,
)
from .io import image_importer
from .io.image_importer import encode_png
from .io.project_store import ProjectStore, ProjectStoreError
from .io.project_watcher import ProjectWatcher
from .tools import default_registry
from .tools.base import ToolContext
from .ui.brush_controls import BrushControls
from .ui.canvas import CanvasWidget
from .ui.drop_import import DropImportMixin
from .ui.photoshop_ui import PhotoshopBridgeMixin
from .ui.command_palette import CommandPaletteDialog
from .ui.export_sheet import ExportAsSheet, JpegPreviewSheet
from .ui.keyboard_shortcuts import KeyboardShortcuts, KeyboardShortcutsDialog
from .ui.layers_panel import LayersPanel
from .ui.project_tabs import ProjectTabs
from .ui.transform_inspector import TransformInspector
from .ui.sheets import (
    AddNoiseSheet,
    BloomSheet,
    ColorBalanceSheet,
    CurvesSheet,
    DitherSheet,
    ExposureSheet,
    GaussianBlurSheet,
    GradientMapSheet,
    GrainSheet,
    HueSaturationSheet,
    InvertSheet,
    LevelsSheet,
    LensCorrectionSheet,
    MotionBlurSheet,
    ScanlinesSheet,
    TonalContrastSheet,
    VignetteSheet,
)
from .ui.color_picker import ColorPickerDialog


def tr(text: str) -> str:
    """Shorthand for ``i18n.tr`` — keeps menu code readable."""
    return i18n.tr(text)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compositor (PySide6)")
    parser.add_argument("project", nargs="?", help="Path to a .comp project folder or manifest.json")
    args = parser.parse_args(argv)

    # Apply persisted AI execution mode (auto / cpu / gpu) before any model
    # is loaded.
    try:
        from .ui.ai_settings import restore_mode
        restore_mode()
    except Exception:
        pass

    app = QApplication.instance() or QApplication(sys.argv)
    # Follow the OS language on startup; falls back to English when the
    # system locale has no bundled translation.
    i18n.set_language(i18n.system_language())
    window = MainWindow()
    window.show()
    if args.project:
        window.open_project(Path(args.project))
    return app.exec()


# Menus (and File entries) that need an open project to make sense; they are
# disabled until one is opened.
_DOC_MENUS = {"&Edit", "&Image", "&Layer", "&Select", "F&ilter", "&View", "&Tools"}
_DOC_FILE_LABELS = {
    "&Save",
    "&Close Project",
    "&Export PNG…",
    "Export &JPEG (live preview)…",
    "Export &As…",
}

# The tool list on the left edge. ``text-align: left`` is the whole point:
# Qt centres button text, and every translation has a different label
# width, which turns the column into a staircase. Left as the native
# colours otherwise, so the panel follows whatever theme the OS applies.
_TOOL_PANEL_QSS = """
QPushButton {
    text-align: left;
    padding: 1px 6px;
}
"""


class MainWindow(DropImportMixin, PhotoshopBridgeMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("app_title"))
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        self._tools = default_registry()
        self._shortcuts = KeyboardShortcuts()

        # Multi-project tabs.
        self.tabs = ProjectTabs(self, on_close=self._close_tab)
        self.tabs.project_changed.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        # Layers dock.
        self._layers_panel = LayersPanel(self)
        self._layers_panel.add_layer_requested.connect(self._add_layer_kind)
        self._layers_panel.delete_layer_requested.connect(self._delete_layer_id)
        self._layers_panel.selection_changed.connect(self._on_layers_selection_changed)
        # Kept as attributes so a language switch can retitle the dock.
        self._layers_dock = QDockWidget(tr("Layers"), self)
        self._layers_dock.setWidget(self._layers_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._layers_dock)

        # Transform Inspector dock.
        self._transform_inspector = TransformInspector(self)
        self._transform_dock = QDockWidget(tr("Transform"), self)
        self._transform_dock.setWidget(self._transform_inspector)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._transform_dock)

        # Per-project state. Keyed by tab index.
        self._states: Dict[int, "ProjectState"] = {}
        self._current_state: Optional[ProjectState] = None

        # Menu + toolbar.
        self._build_menu()
        self._build_toolbar()
        self._refresh_menu_state()
        self.setStatusBar(QStatusBar(self))

        # Layer effects + camera raw dialogs lazy-loaded.
        self._camera_raw_dialog = None

    # ---- menus / toolbar -------------------------------------------------

    def _build_menu(self) -> None:
        bar = self.menuBar()
        bar.clear()
        # ``bar.clear()`` removes the actions but leaves the QMenu objects
        # parented to the bar (a Qt quirk). Delete them explicitly so a
        # rebuild doesn't leak stale menus that later title lookups — and
        # the user — could confuse for the live ones.
        for stale in getattr(self, "_top_menus", []):
            stale.deleteLater()
        self._top_menus: List = []
        self._menus: Dict[str, List[Tuple[str, str, Callable[[], None]]]] = {}
        self._menu_actions: Dict[str, List[QAction]] = {}
        self._language_actions: Dict[str, QAction] = {}
        self._doc_menu_actions: List[QAction] = []
        self._doc_file_actions: List[QAction] = []

        def add(menu_label: str, items: List[Tuple[str, str, Callable[[], None]]]) -> QMenu:
            menu = bar.addMenu(tr(menu_label))
            self._top_menus.append(menu)
            self._menus[menu_label] = items
            self._menu_actions[menu_label] = []
            if menu_label in _DOC_MENUS:
                self._doc_menu_actions.append(menu.menuAction())
            for label, sequence, fn in items:
                # An empty label is a separator — keep the tuple shape so
                # menus can interleave dividers without a second list format.
                if not label:
                    menu.addSeparator()
                    continue
                act = QAction(tr(label), self)
                if sequence:
                    parsed = QKeySequence(sequence)
                    if not parsed.isEmpty():
                        act.setShortcut(parsed)
                act.triggered.connect(fn)
                menu.addAction(act)
                self._menu_actions[menu_label].append(act)
                if menu_label == "&File" and label in _DOC_FILE_LABELS:
                    self._doc_file_actions.append(act)
            return menu

        add("&File", [
            ("&New Project…", "Ctrl+N", self._new_project),
            ("&Open Project…", self._shortcuts.get("File > Open"), self._open_project_dialog),
            ("&Save", self._shortcuts.get("File > Save"), self._save_project),
            ("&Close Project", "", self._close_current_tab),
            ("&Export PNG…", "Ctrl+Shift+E", lambda: self._export("PNG")),
            ("Export &JPEG (live preview)…", "Ctrl+Shift+Alt+S", self._export_jpeg_live),
            ("Export &As…", "Ctrl+Shift+Alt+W", lambda: self._export_dialog()),
            ("&Quit", "Ctrl+Q", self.close),
        ])
        add("&Edit", [
            ("&Undo", self._shortcuts.get("Edit > Undo"), self._undo),
            ("&Redo", self._shortcuts.get("Edit > Redo"), self._redo),
            ("Copy &Merged", self._shortcuts.get("Edit > Copy Merged"), self._copy_merged),
            ("Content-&Aware Fill", "Shift+Backspace", self._content_aware_fill_active),
            ("Select &Subject", "Ctrl+Alt+A", self._select_subject),
            ("Keyboard &Shortcuts…", "", self._edit_shortcuts),
        ])
        add("&Image", [
            ("&Canvas Size…", self._shortcuts.get("Image > Canvas Size…"), self._canvas_size),
            ("&Image Size…", self._shortcuts.get("Image > Image Size…"), self._image_size),
            ("&Trim", "", self._trim_document),
            ("Camera Raw &Filter…", "", self._open_camera_raw),
            ("AI &Remove Background", "", self._ai_remove_background_active),
            ("AI &Select Subject", "", self._ai_select_subject),
            ("AI &Upscale ×2…", "", self._ai_upscale_active),
            ("AI Backend &Settings…", "", self._open_ai_settings),
        ])
        add("&Layer", [
            ("&New Layer", self._shortcuts.get("Layer > New Layer"), lambda: self._add_layer_kind("image")),
            ("New &Group", "", lambda: self._add_layer_kind("group")),
            ("New &Adjustment Layer", "", self._new_adjustment_layer),
            ("&Duplicate Layer", self._shortcuts.get("Layer > Duplicate Layer"), self._duplicate_active),
            ("&Delete Layer", self._shortcuts.get("Layer > Delete Layer"), self._delete_active),
            ("&Merge Down", self._shortcuts.get("Layer > Merge Down"), self._merge_down_active),
            ("Remove &Background", "", self._remove_background_active),
            ("Flip &Horizontal", self._shortcuts.get("Layer > Flip Horizontal"), lambda: self._flip("x")),
            ("Flip &Vertical", "", lambda: self._flip("y")),
        ])
        add("&Select", [
            ("Select &All", self._shortcuts.get("Select > All"), self._select_all),
            ("&Deselect", self._shortcuts.get("Select > Deselect"), self._deselect),
            ("&Inverse", self._shortcuts.get("Select > Inverse"), self._inverse_selection),
            ("&Expand…", "", lambda: self._select_modify("expand")),
            ("&Contract…", "", lambda: self._select_modify("contract")),
            ("&Feather…", "", lambda: self._select_modify("feather")),
        ])
        add("F&ilter", [
            ("&Last Filter", self._shortcuts.get("Filter > Last Filter"), self._apply_last_filter),
            ("&Gaussian Blur…", "", lambda: self._apply_adjustment(AdjustmentKind.GAUSSIAN_BLUR)),
            ("&Motion Blur…", "", lambda: self._apply_adjustment(AdjustmentKind.MOTION_BLUR)),
            ("Add &Noise…", "", lambda: self._apply_adjustment(AdjustmentKind.ADD_NOISE)),
            ("&Invert", "", lambda: self._apply_adjustment(AdjustmentKind.INVERT)),
            ("&Vignette…", "", lambda: self._apply_adjustment(AdjustmentKind.VIGNETTE)),
            ("&Bloom…", "", lambda: self._apply_adjustment(AdjustmentKind.BLOOM)),
            ("&Dither…", "", lambda: self._apply_adjustment(AdjustmentKind.DITHER)),
            ("S&canlines…", "", lambda: self._apply_adjustment(AdjustmentKind.SCANLINES)),
            ("Tonal &Contrast…", "", lambda: self._apply_adjustment(AdjustmentKind.TONAL_CONTRAST)),
            ("&Lens Correction…", "", lambda: self._apply_adjustment(AdjustmentKind.LENS_CORRECTION)),
        ])
        add("&View", [
            ("Zoom &In", self._shortcuts.get("View > Zoom In"), self._zoom_in),
            ("Zoom &Out", self._shortcuts.get("View > Zoom Out"), self._zoom_out),
            ("Fit on &Screen", self._shortcuts.get("View > Fit on Screen"), self._fit_window),
            ("Toggle &Rulers", self._shortcuts.get("View > Toggle Rulers"), self._toggle_rulers),
            ("Toggle &Grid", self._shortcuts.get("View > Toggle Grid"), self._toggle_grid),
            ("Toggle &Fullscreen", self._shortcuts.get("View > Toggle Fullscreen"), self._toggle_fullscreen),
        ])
        add("&Tools", self._tool_menu_items())
        add("&Photoshop", [
            ("&Connect to Photoshop…", "", self._photoshop_connect),
            ("&Disconnect", "", self._photoshop_disconnect),
            ("", "", lambda: None),  # separator
            ("Send &Layers to Photoshop", "", self._photoshop_send_layers),
            ("Send &Merged Image to Photoshop", "", self._photoshop_send_merged),
            ("&Pull Active Document from Photoshop", "", self._photoshop_pull),
            ("", "", lambda: None),  # separator
            ("Run &ExtendScript…", "", self._photoshop_run_script),
            ("Photoshop &Info", "", self._photoshop_info),
        ])
        help_menu = add("&Help", [
            ("&Commands", self._shortcuts.get("Help > Commands"), self._open_command_palette),
            ("&About", "", self._about),
        ])

        # Language submenu goes straight into the Help menu we just built —
        # no title lookups. (The old findChildren-based lookup matched a
        # stale Help menu after a language switch, which made the submenu
        # vanish when switching back to English.)
        lang_menu = QMenu(tr("&Language"), self)
        self._language_group = QActionGroup(self)
        self._language_group.setExclusive(True)
        for code, native_label in i18n.available_languages():
            act = QAction(native_label, self, checkable=True)
            act.setChecked(code == i18n.current_language())
            act.triggered.connect(lambda _=False, c=code: self._switch_language(c))
            lang_menu.addAction(act)
            self._language_group.addAction(act)
            self._language_actions[code] = act
        help_menu.addMenu(lang_menu)
        self._top_menus.append(lang_menu)

        # Rebuild the command palette aggregator with translated names.
        self._command_index = {}
        for menu, items in self._menus.items():
            for label, _, fn in items:
                if not label:
                    continue  # separator
                stripped = label.replace("&", "")
                self._command_index[f"{tr(menu).replace('&', '')} > {tr(label).replace('&', '')}"] = fn
                self._command_index[f"{stripped}"] = fn  # also register English form
        for name in self._tools.names():
            self._command_index[f"{tr('Tools')} > {tr(name.title())}"] = lambda n=name: self._activate_tool(n)

        # Listen for future language changes.
        i18n.on_change(self._on_language_changed)

    def _tool_menu_items(self) -> List[Tuple[str, str, Callable[[], None]]]:
        out = []
        shortcuts = {
            "move": "V",
            "brush": "B",
            "eraser": "E",
            "marquee-rect": "M",
            "marquee-ellipse": "Shift+M",
            "lasso": "L",
            "polygon-lasso": "Shift+L",
            "wand": "W",
            "crop": "C",
            "gradient": "G",
            "type": "T",
            "shape": "U",
            "clone-stamp": "S",
            "blur": "Ctrl+B",
            "heal": "J",
            "eyedropper": "I",
        }
        for name in self._tools.names():
            label = tr(name.replace("-", " ").title())
            seq = shortcuts.get(name, "")
            out.append((label, seq, self._make_tool_activator(name)))
        return out

    def _make_tool_activator(self, name: str) -> Callable[[], None]:
        def _activate() -> None:
            self._activate_tool(name)
        return _activate

    def _switch_language(self, code: str) -> None:
        if not i18n.set_language(code):
            return
        # The listener registered in _build_menu will rebuild the menu bar.
        self.setWindowTitle(tr("app_title"))

    def _on_language_changed(self, code: str) -> None:
        # Defer the rebuild to the next event-loop tick. If we rebuilt
        # synchronously inside a QAction.triggered handler, ``bar.clear()``
        # would delete the very action whose signal we're dispatching —
        # Qt hangs waiting for that dispatch to finish.
        QTimer.singleShot(0, self._rebuild_ui)

    def _rebuild_ui(self) -> None:
        self._build_menu()
        self._build_toolbar()
        self._refresh_menu_state()
        # The docks are built once, so their titles need the new language too.
        self._layers_dock.setWindowTitle(tr("Layers"))
        self._transform_dock.setWindowTitle(tr("Transform"))
        self._refresh_tool_panels()
        self._refresh_canvas_placeholder()

    def _refresh_menu_state(self) -> None:
        """Gray out everything that needs an open project when none is."""
        has_doc = self._current_state is not None
        for act in getattr(self, "_doc_menu_actions", []):
            act.setEnabled(has_doc)
        for act in getattr(self, "_doc_file_actions", []):
            act.setEnabled(has_doc)
        tools_dock = getattr(self, "_tools_dock", None)
        if tools_dock is not None:
            tools_dock.setEnabled(has_doc)

    def _on_tab_changed(self, index: int) -> None:
        self._current_state = self._states.get(index)
        self._refresh_menu_state()
        if self._current_state is not None:
            self._activate_tool(self._current_state.tool_name or "move")
        self._refresh_tool_panels()

    def _refresh_canvas_placeholder(self) -> None:
        if self._current_state is None:
            return
        self._current_state.canvas.set_session(self._current_state.session)

    def _build_toolbar(self) -> None:
        """Tool buttons down the left edge, each label flush left.

        Every translation has a different label width, so centre-aligned
        buttons read as a ragged staircase. QToolBar centres its button text
        and ignores ``text-align``; QToolButton widgets do too. A QPushButton
        with ``text-align: left`` actually honours it, so the tool list is a
        column of those, with ``autoExclusive`` standing in for the old
        QActionGroup.
        """
        previous = getattr(self, "_tools_dock", None)
        if previous is not None:
            self.removeDockWidget(previous)
            previous.deleteLater()
        for tb in self.findChildren(QToolBar):
            self.removeToolBar(tb)

        panel = QWidget(self)
        panel.setStyleSheet(_TOOL_PANEL_QSS)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self._tool_buttons: Dict[str, QPushButton] = {}
        active = self._current_state.tool_name if self._current_state else None
        for name in self._tools.names():
            # display_name() returns "Translated (English)" for non-English
            # languages so users can still recognise tools by their
            # English name. For English it returns just the name.
            button = QPushButton(i18n.display_name(f"tool.{name}"), panel)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if name == active:
                button.setChecked(True)
            button.clicked.connect(lambda _=False, n=name: self._activate_tool(n))
            self._tool_buttons[name] = button
            layout.addWidget(button)

        dock = QDockWidget(tr("Tools"), self)
        # A dock would otherwise grow a title bar; the old toolbar had none.
        dock.setTitleBarWidget(QWidget())
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self._tools_dock = dock
        self._tools_panel = panel

    # ---- tool dispatch ---------------------------------------------------

    def _activate_tool(self, name: str) -> None:
        if self._current_state is None:
            return
        tool = self._tools.get(name)
        self._current_state.canvas.set_active_tool(name, tool)
        self._current_state.tool_name = name
        button = getattr(self, "_tool_buttons", {}).get(name)
        if button is not None:
            button.setChecked(True)
        self._refresh_tool_panels()

    # ---- project open / save --------------------------------------------

    def _new_project(self) -> None:
        w, ok = QInputDialog.getInt(self, "New Project", "Width", 1024, 1, 30000)
        if not ok:
            return
        h, ok = QInputDialog.getInt(self, "New Project", "Height", 1024, 1, 30000)
        if not ok:
            return
        manifest = Manifest(
            version=CURRENT_VERSION,
            color_space="sRGB",
            document_id=new_id(),
            width=w,
            height=h,
            layers=[],
        )
        path = QFileDialog.getExistingDirectory(self, "New Project Folder")
        if not path:
            return
        try:
            store = ProjectStore.create(Path(path), manifest)
        except ProjectStoreError as exc:
            QMessageBox.critical(self, "Cannot create project", str(exc))
            return
        self._open_project_store(store)

    def _open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open .comp project",
            "",
            "Compositor project (manifest.json);;All files (*)",
        )
        if not path:
            return
        p = Path(path)
        if p.name == "manifest.json":
            p = p.parent
        self.open_project(p)

    def open_project(self, path: Path) -> None:
        try:
            store = ProjectStore.open(path)
            self._open_project_store(store)
        except ProjectStoreError as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))

    def _open_project_store(self, store: ProjectStore) -> None:
        manifest = store.load_manifest()
        session = EditorSession(width=manifest.width, height=manifest.height)
        session.color_space = manifest.color_space
        session.document_id = manifest.document_id
        session.resolution = manifest.resolution
        session.load_manifest_layers(manifest.layers)
        session.active_layer_id = manifest.active_layer_id

        # Load layer PNGs into the in-memory cache.
        image_cache: Dict[str, np.ndarray] = {}
        mask_cache: Dict[str, np.ndarray] = {}
        for layer in session.layers:
            if isinstance(layer, ImageLayer) and layer.image_file:
                img_path = store.paths.images_dir / layer.image_file
                if img_path.is_file():
                    image_cache[layer.id] = image_importer.load_image(img_path)
            if layer.mask_file:
                mask_path = store.paths.images_dir / layer.mask_file
                if mask_path.is_file():
                    mask_cache[layer.id] = image_importer.decode_png_mask(mask_path.read_bytes())
        session._image_cache = image_cache  # type: ignore[attr-defined]
        session._mask_cache = mask_cache  # type: ignore[attr-defined]

        state = ProjectState(canvas=CanvasWidget(self.tabs), session=session, store=store, watcher=ProjectWatcher(store.paths.root, parent=self))
        state.canvas.set_session(session)
        state.canvas.document_changed.connect(self._save_project)
        state.watcher.changed.connect(lambda s=state: self._reload_state(s))

        idx = self.tabs.add_project(store.paths.root.name or str(store.paths.root), state.canvas)
        self._states[idx] = state
        self._current_state = state
        state.canvas.set_session(session)
        state.canvas.sync_caches_from_session()
        # Load guides from the manifest.
        session.load_manifest_guides(manifest.guides)
        self._activate_tool(state.tool_name or "move")
        self._refresh_tool_panels()
        self._refresh_menu_state()
        state.watcher.watch()
        self.statusBar().showMessage(tr("Opened {path}").format(path=str(store.paths.root)), 5000)

    def _reload_state(self, state: "ProjectState") -> None:
        try:
            manifest = state.store.load_manifest()
        except ManifestError as exc:
            self.statusBar().showMessage(tr("Reload failed: {err}").format(err=str(exc)), 5000)
            return
        state.session.load_manifest_layers(manifest.layers)
        state.session.active_layer_id = manifest.active_layer_id
        for layer in state.session.layers:
            if isinstance(layer, ImageLayer) and layer.image_file:
                img_path = state.store.paths.images_dir / layer.image_file
                if img_path.is_file() and layer.id not in state.session._image_cache:
                    state.session._image_cache[layer.id] = image_importer.load_image(img_path)
        state.canvas.set_session(state.session)
        state.canvas.sync_caches_from_session()
        self._refresh_tool_panels()
        self.statusBar().showMessage(tr("Project reloaded"), 3000)

    def _save_project(self) -> None:
        if self._current_state is None:
            return
        state = self._current_state
        manifest = Manifest(
            version=CURRENT_VERSION,
            color_space=state.session.color_space,
            document_id=state.session.document_id,
            width=state.session.width,
            height=state.session.height,
            layers=state.session.to_manifest_layers(),
            guides=state.session.to_manifest_guides(),
            resolution=state.session.resolution,
            active_layer_id=state.session.active_layer_id,
        )
        # Flush any in-memory layer images back to disk as PNGs.
        cache = getattr(state.session, "_image_cache", {})
        for layer in state.session.layers:
            if isinstance(layer, ImageLayer) and layer.id in cache:
                png_bytes = encode_png(cache[layer.id])
                state.store.write_image(layer.id, png_bytes)
                # Patch the layer's imageFile in the manifest if missing.
                for entry in manifest.layers:
                    if entry["id"] == layer.id and not entry.get("imageFile"):
                        entry["imageFile"] = f"{layer.id}.png"
        try:
            state.store.save_manifest(manifest)
            self.statusBar().showMessage(tr("Saved"), 3000)
        except ProjectStoreError as exc:
            QMessageBox.critical(self, tr("Save failed"), str(exc))

    # ---- tabs ------------------------------------------------------------

    def _close_current_tab(self) -> None:
        idx = self.tabs.currentIndex()
        if idx >= 0:
            self._close_tab(idx)

    def _close_tab(self, index: int) -> None:
        state = self._states.pop(index, None)
        if state is not None and state.watcher is not None:
            state.watcher.stop()
        self.tabs.remove_project(index)
        # Tab indices shift after a removal — re-key the remaining states
        # by matching each tab page to its canvas widget.
        remap: Dict[int, "ProjectState"] = {}
        for i in range(self.tabs.count()):
            for st in self._states.values():
                if self.tabs.widget(i) is st.canvas:
                    remap[i] = st
                    break
        self._states = remap
        self._current_state = self._states.get(self.tabs.currentIndex())
        self._refresh_menu_state()
        self._refresh_tool_panels()

    # ---- layer ops -------------------------------------------------------

    def _add_layer_kind(self, kind: str) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        if kind == "group":
            layer = GroupLayer(id=new_id(), name="Group")
        elif kind == "image":
            layer = ImageLayer(
                id=new_id(),
                name="Layer",
                image_file=None,
                transform=Transform(size=(session.width, session.height)),
            )
            empty = np.zeros((session.height, session.width, 4), dtype=np.uint8)
            self._current_state.session._image_cache[layer.id] = empty  # type: ignore[attr-defined]
        elif kind == "adjustment":
            self._new_adjustment_layer()
            return
        else:
            return
        session.add_layer(layer)
        session.active_layer_id = layer.id
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()

    def _new_adjustment_layer(self) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        items = [k.value for k in AdjustmentKind]
        kind, ok = QInputDialog.getItem(self, "New Adjustment Layer", "Kind", items, 0, False)
        if not ok:
            return
        layer = AdjustmentLayer(
            id=new_id(),
            name=f"{kind}",
            kind=AdjustmentKind(kind),
            transform=Transform(size=(session.width, session.height)),
        )
        session.add_layer(layer)
        session.active_layer_id = layer.id
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()

    def _duplicate_active(self) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        layer = session.layer_by_id(session.active_layer_id)
        if layer is None:
            return
        from copy import deepcopy
        new_layer = deepcopy(layer)
        new_layer.id = new_id()
        if isinstance(new_layer, ImageLayer) and new_layer.image_file:
            new_layer.image_file = f"{new_layer.id}.png"
        idx = session.layer_index(layer.id)
        session.add_layer(new_layer, at_index=idx + 1)
        session.active_layer_id = new_layer.id
        cache = getattr(session, "_image_cache", {})
        if layer.id in cache:
            cache[new_layer.id] = cache[layer.id].copy()
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()

    def _delete_active(self) -> None:
        if self._current_state is None:
            return
        self._delete_layer_id(self._current_state.session.active_layer_id)

    def _delete_layer_id(self, layer_id: Optional[str]) -> None:
        if self._current_state is None or layer_id is None:
            return
        session = self._current_state.session
        if session.layer_by_id(layer_id) is None:
            return
        session.remove_layer(layer_id)
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()

    def _on_layers_selection_changed(self, layer_id: str) -> None:
        if self._current_state is None:
            return
        if self._current_state.session.layer_by_id(layer_id) is None:
            return
        self._current_state.session.active_layer_id = layer_id
        self._refresh_tool_panels()

    def _merge_down_active(self) -> None:
        # Compose the active layer over the one below, replacing the result.
        if self._current_state is None:
            return
        state = self._current_state
        session = state.session
        top = session.layer_by_id(session.active_layer_id)
        if top is None or not isinstance(top, ImageLayer):
            self._show_message(tr("Select an image layer first."))
            return
        # Walk from just below the active layer downwards — starting the
        # scan at the stack top would hit ``top`` first and bail out.
        below = None
        for i in range(session.layer_index(top.id) - 1, -1, -1):
            candidate = session.layers[i]
            if candidate.parent_id == top.parent_id:
                below = candidate
                break
        if below is None or not isinstance(below, ImageLayer):
            self._show_message(tr("No image layer below to merge into."))
            return
        cache = getattr(session, "_image_cache", {})
        if top.id not in cache or below.id not in cache:
            self._show_message(tr("Active layer has no pixels to process."))
            return

        from .rendering.layer_renderer import merge_layers_pixels
        pixels, origin, size = merge_layers_pixels(session, cache, bottom=below, top=top)

        merged = ImageLayer(
            id=below.id,
            name=below.name,
            parent_id=below.parent_id,
            transform=Transform(origin=origin, size=size),
            image_file=f"{below.id}.png",
        )
        session.merge_down(top.id, merged, new_pixels=pixels, pixel_cache=cache)
        state.canvas.set_session(session)
        self._refresh_tool_panels()
        self.statusBar().showMessage(tr("Merge Down"), 3000)

    def _flip(self, axis: str) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        layer = session.layer_by_id(session.active_layer_id)
        if layer is None:
            return
        if axis == "x":
            session.set_property(layer.id, "transform", Transform(
                origin=layer.transform.origin,
                size=layer.transform.size,
                rotation=layer.transform.rotation,
                flip_x=not layer.transform.flip_x,
                flip_y=layer.transform.flip_y,
                sampling=layer.transform.sampling,
            ))
        else:
            session.set_property(layer.id, "transform", Transform(
                origin=layer.transform.origin,
                size=layer.transform.size,
                rotation=layer.transform.rotation,
                flip_x=layer.transform.flip_x,
                flip_y=not layer.transform.flip_y,
                sampling=layer.transform.sampling,
            ))

    def _remove_background_active(self) -> None:
        """Bake a foreground-extracted alpha into the active image layer."""
        if self._current_state is None:
            return
        session = self._current_state.session
        layer = session.layer_by_id(session.active_layer_id)
        if layer is None or not isinstance(layer, ImageLayer):
            self._show_message(tr("Select an image layer first."))
            return
        cache = getattr(session, "_image_cache", {})
        if layer.id not in cache:
            self._show_message(tr("Active layer has no pixels to process."))
            return
        from .rendering.pixel_ops import remove_background
        new_pixels = remove_background(cache[layer.id])
        cache[layer.id] = new_pixels
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()
        self.statusBar().showMessage(tr("Removed background on active layer"), 3000)

    def _ai_remove_background_active(self) -> None:
        """AI Remove Background — prefer rembg (U²-Net) over cv2 GrabCut."""
        if self._current_state is None:
            return
        session = self._current_state.session
        layer = session.layer_by_id(session.active_layer_id)
        if layer is None or not isinstance(layer, ImageLayer):
            self._show_message(tr("Select an image layer first."))
            return
        cache = getattr(session, "_image_cache", {})
        if layer.id not in cache:
            self._show_message(tr("Active layer has no pixels to process."))
            return
        from .rendering import ai_backends
        if not ai_backends.is_rembg_available():
            QMessageBox.warning(
                self,
                tr("rembg not installed"),
                tr("Install rembg for AI-powered background removal. Run: pip install rembg"),
            )
            return
        from .rendering.pixel_ops import remove_background
        new_pixels = remove_background(cache[layer.id], method="rembg")
        cache[layer.id] = new_pixels
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()
        self.statusBar().showMessage(tr("AI background removal applied (U²-Net)"), 3000)

    def _ai_select_subject(self) -> None:
        """AI Select Subject — fill the current selection mask with the model's foreground."""
        if self._current_state is None:
            return
        session = self._current_state.session
        composite = self._current_state.canvas._composite()
        from .rendering import ai_backends
        if ai_backends.is_rembg_available():
            rgba = ai_backends.rembg_remove_background(composite[..., :3])
            mask = rgba[..., 3]
            session._selection_mask = mask  # type: ignore[attr-defined]
            self._current_state.canvas.set_session(session)
            self._refresh_tool_panels()
            self.statusBar().showMessage(tr("AI Select Subject applied (rembg)"), 3000)
            return
        # Fall back to the existing heuristic.
        self._select_subject()

    def _ai_upscale_active(self) -> None:
        """AI Upscale ×2 — promote the active image layer to 2× resolution."""
        if self._current_state is None:
            return
        session = self._current_state.session
        layer = session.layer_by_id(session.active_layer_id)
        if layer is None or not isinstance(layer, ImageLayer):
            self._show_message(tr("Select an image layer first."))
            return
        cache = getattr(session, "_image_cache", {})
        if layer.id not in cache:
            self._show_message(tr("Active layer has no pixels to process."))
            return
        from PIL import Image
        from .rendering.pixel_ops import upscale
        new_pixels = upscale(cache[layer.id], factor=2.0)
        cache[layer.id] = new_pixels
        # Update the layer transform size to match.
        h, w = new_pixels.shape[:2]
        layer.transform = Transform(
            origin=layer.transform.origin,
            size=(w, h),
            rotation=layer.transform.rotation,
            flip_x=layer.transform.flip_x,
            flip_y=layer.transform.flip_y,
            sampling=layer.transform.sampling,
            corners=layer.transform.corners,
        )
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()
        self.statusBar().showMessage(tr("Upscaled active layer to {w}×{h}").format(w=w, h=h), 3000)

    def _open_ai_settings(self) -> None:
        from .ui.ai_settings import AISettingsDialog
        dialog = AISettingsDialog(self)
        dialog.exec()

    def _content_aware_fill_active(self) -> None:
        """Fill the current selection on the active layer with surrounding pixels."""
        if self._current_state is None:
            return
        session = self._current_state.session
        layer = session.layer_by_id(session.active_layer_id)
        if layer is None or not isinstance(layer, ImageLayer):
            self._show_message(tr("Select an image layer first."))
            return
        mask = getattr(session, "_selection_mask", None)
        if mask is None or not (mask > 0).any():
            self._show_message("Make a selection first (then run Content-Aware Fill).")
            return
        cache = getattr(session, "_image_cache", {})
        if layer.id not in cache:
            self._show_message(tr("Active layer has no pixels to process."))
            return
        from .rendering.pixel_ops import content_aware_fill
        new_pixels = content_aware_fill(cache[layer.id], mask)
        cache[layer.id] = new_pixels
        # Clear the selection after filling.
        session._selection_mask = None  # type: ignore[attr-defined]
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()
        self.statusBar().showMessage(tr("Content-Aware Fill applied"), 3000)

    def _select_subject(self) -> None:
        """Build a selection mask covering the most-likely subject area."""
        if self._current_state is None:
            return
        session = self._current_state.session
        composite = self._current_state.canvas._composite()
        if composite.shape[-1] == 4:
            alpha = composite[..., 3]
        else:
            alpha = (composite[..., :3].max(axis=-1) > 0).astype(np.uint8) * 255
        ys, xs = np.where(alpha > 0)
        if len(xs) == 0:
            self._show_message("No content to select.")
            return
        # Heuristic: subject = non-transparent pixels; tighten the bbox by
        # ~10% on each side so we don't grab edge artefacts.
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        x0 = min(max(x0 + margin_x, 0), session.width - 1)
        y0 = min(max(y0 + margin_y, 0), session.height - 1)
        x1 = max(min(x1 - margin_x, session.width), x0 + 1)
        y1 = max(min(y1 - margin_y, session.height), y0 + 1)
        mask = np.zeros((session.height, session.width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        session._selection_mask = mask  # type: ignore[attr-defined]
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()
        self.statusBar().showMessage(tr("Subject selected"), 3000)

    # ---- selection ops ---------------------------------------------------

    def _select_all(self) -> None:
        if self._current_state is None:
            return
        mask = np.full((self._current_state.session.height, self._current_state.session.width), 255, dtype=np.uint8)
        self._current_state.session._selection_mask = mask  # type: ignore[attr-defined]

    def _deselect(self) -> None:
        if self._current_state is None:
            return
        self._current_state.session._selection_mask = None  # type: ignore[attr-defined]

    def _inverse_selection(self) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        existing = getattr(session, "_selection_mask", None)
        if existing is None:
            mask = np.zeros((session.height, session.width), dtype=np.uint8)
        else:
            mask = 255 - existing
        session._selection_mask = mask  # type: ignore[attr-defined]

    def _select_modify(self, op: str) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        mask = getattr(session, "_selection_mask", None)
        if mask is None:
            return
        from .rendering import selections as sel
        if op == "expand":
            v, ok = QInputDialog.getInt(self, "Expand Selection", "Pixels", 5, 0, 200)
            if ok:
                mask = sel.expand(mask, v)
        elif op == "contract":
            v, ok = QInputDialog.getInt(self, "Contract Selection", "Pixels", 5, 0, 200)
            if ok:
                mask = sel.contract(mask, v)
        elif op == "feather":
            v, ok = QInputDialog.getDouble(self, "Feather Selection", "Radius", 2.0, 0.0, 100.0, 1)
            if ok:
                mask = sel.feather(mask, v)
        session._selection_mask = mask  # type: ignore[attr-defined]

    # ---- view ops --------------------------------------------------------

    def _zoom_in(self) -> None:
        if self._current_state is None:
            return
        self._current_state.canvas.zoom_in()

    def _zoom_out(self) -> None:
        if self._current_state is None:
            return
        self._current_state.canvas.zoom_out()

    def _fit_window(self) -> None:
        if self._current_state is None:
            return
        self._current_state.canvas.fit_to_window()

    def _toggle_rulers(self) -> None:
        if self._current_state is None:
            return
        self._current_state.canvas.toggle_rulers()

    def _toggle_grid(self) -> None:
        if self._current_state is None:
            return
        self._current_state.canvas.toggle_grid()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ---- undo/redo -------------------------------------------------------

    def _undo(self) -> None:
        if self._current_state is None:
            return
        if self._current_state.session.undo():
            self._current_state.canvas.set_session(self._current_state.session)
            self._refresh_tool_panels()

    def _redo(self) -> None:
        if self._current_state is None:
            return
        if self._current_state.session.redo():
            self._current_state.canvas.set_session(self._current_state.session)
            self._refresh_tool_panels()

    # ---- image ops -------------------------------------------------------

    def _trim_document(self) -> None:
        """Crop away transparent borders around the document content."""
        if self._current_state is None:
            return
        state = self._current_state
        session = state.session
        composite = state.canvas._composite()
        alpha = composite[..., 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        if not rows.any() or not cols.any():
            self._show_message(tr("Trim: image is empty"))
            return
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        new_w, new_h = int(x1 - x0 + 1), int(y1 - y0 + 1)
        old_w, old_h = session.width, session.height
        if (x0, y0, new_w, new_h) == (0, 0, old_w, old_h):
            return  # already tight — nothing to trim
        session.trim_document(int(x0), int(y0), new_w, new_h)
        state.canvas.set_session(session)
        self._refresh_tool_panels()
        self.statusBar().showMessage(
            tr("Trimmed {a}×{b} → {c}×{d}").format(a=old_w, b=old_h, c=new_w, d=new_h), 4000
        )

    # ---- filter / adjustment application ---------------------------------

    def _apply_adjustment(self, kind: AdjustmentKind) -> None:
        if self._current_state is None:
            return
        mapping = {
            AdjustmentKind.LEVELS: LevelsSheet,
            AdjustmentKind.CURVES: CurvesSheet,
            AdjustmentKind.HUE_SATURATION: HueSaturationSheet,
            AdjustmentKind.EXPOSURE: ExposureSheet,
            AdjustmentKind.COLOR_BALANCE: ColorBalanceSheet,
            AdjustmentKind.GRADIENT_MAP: GradientMapSheet,
            AdjustmentKind.GRAIN: GrainSheet,
            AdjustmentKind.GAUSSIAN_BLUR: GaussianBlurSheet,
            AdjustmentKind.MOTION_BLUR: MotionBlurSheet,
            AdjustmentKind.ADD_NOISE: AddNoiseSheet,
            AdjustmentKind.INVERT: InvertSheet,
            AdjustmentKind.VIGNETTE: VignetteSheet,
            AdjustmentKind.BLOOM: BloomSheet,
            AdjustmentKind.DITHER: DitherSheet,
            AdjustmentKind.SCANLINES: ScanlinesSheet,
            AdjustmentKind.TONAL_CONTRAST: TonalContrastSheet,
            AdjustmentKind.LENS_CORRECTION: LensCorrectionSheet,
        }
        cls = mapping.get(kind)
        if cls is None:
            return
        dialog = cls(self)
        dialog.accepted_with_settings.connect(
            lambda settings, k=kind: self._commit_adjustment_layer(k, settings)
        )
        dialog.exec()

    def _commit_adjustment_layer(self, kind: AdjustmentKind, settings: dict) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        layer = AdjustmentLayer(
            id=new_id(),
            name=kind.value,
            kind=kind,
            settings=settings,
            transform=Transform(size=(session.width, session.height)),
        )
        session.add_layer(layer)
        session.active_layer_id = layer.id
        # Cache for "Last Filter" (⌃⌘F).
        self._last_filter = (kind, dict(settings))
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()

    def _apply_last_filter(self) -> None:
        if self._current_state is None:
            return
        last = getattr(self, "_last_filter", None)
        if last is None:
            self._show_message("No filter has been run yet.")
            return
        kind, settings = last
        self._commit_adjustment_layer(kind, settings)

    # ---- copy / clipboard ------------------------------------------------

    def _copy_merged(self) -> None:
        if self._current_state is None:
            return
        composite = self._current_state.canvas._composite()
        from .io.image_importer import encode_png
        from PySide6.QtCore import QMimeData
        from PySide6.QtGui import QGuiApplication, QImage
        # Composite is HxWx4 uint8 RGBA. Convert to QImage via Format_RGBA8888.
        h, w = composite.shape[:2]
        qimg = QImage(composite.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        png_bytes = encode_png(composite)
        mime = QMimeData()
        mime.setImageData(qimg)
        mime.setData("PNG", png_bytes)
        QGuiApplication.clipboard().setMimeData(mime)
        self.statusBar().showMessage(tr("Copied merged to clipboard"), 3000)

    # ---- image ops -------------------------------------------------------

    def _canvas_size(self) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        w, ok = QInputDialog.getInt(self, "Canvas Size", "Width", session.width, 1, 30000)
        if not ok:
            return
        h, ok = QInputDialog.getInt(self, "Canvas Size", "Height", session.height, 1, 30000)
        if not ok:
            return
        old_w, old_h = session.width, session.height
        session.resize_canvas(w, h)
        # Clamp layer transform origins so layers stay anchored to top-left
        # of the new canvas when it shrinks.
        from PySide6.QtCore import QRectF
        for layer in session.layers:
            t = layer.transform
            new_origin = (max(0.0, min(t.origin[0], w)), max(0.0, min(t.origin[1], h)))
            if new_origin != t.origin:
                session.set_property(layer.id, "transform", Transform(
                    origin=new_origin,
                    size=t.size,
                    rotation=t.rotation,
                    flip_x=t.flip_x,
                    flip_y=t.flip_y,
                    sampling=t.sampling,
                ))
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()
        self.statusBar().showMessage(
            tr("Canvas resized {a}×{b} → {c}×{d}").format(a=old_w, b=old_h, c=w, d=h), 5000
        )

    def _image_size(self) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        w, ok = QInputDialog.getInt(self, "Image Size", "Width", session.width, 1, 30000)
        if not ok:
            return
        h, ok = QInputDialog.getInt(self, "Image Size", "Height", session.height, 1, 30000)
        if not ok:
            return
        old_w, old_h = session.width, session.height
        session.scale_image(w, h)
        # Resample every ImageLayer's pixels to the new canvas size.
        cache = getattr(session, "_image_cache", {})
        from PIL import Image
        for layer in session.layers:
            if not isinstance(layer, ImageLayer):
                continue
            if layer.id in cache:
                arr = cache[layer.id]
                pil = Image.fromarray(arr)
                # Resample only if the size actually changed.
                if (pil.width, pil.height) != (w, h):
                    pil = pil.resize((w, h), Image.Resampling.LANCZOS)
                    cache[layer.id] = np.array(pil)
        # Same for masks.
        mask_cache = getattr(session, "_mask_cache", {})
        for layer_id, mask in list(mask_cache.items()):
            if mask.ndim == 2 and mask.shape != (h, w):
                pil = Image.fromarray(mask, mode="L")
                pil = pil.resize((w, h), Image.Resampling.LANCZOS)
                mask_cache[layer_id] = np.array(pil)
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()
        self.statusBar().showMessage(
            tr("Image resized to {w}×{h}").format(w=w, h=h), 5000
        )

    # ---- export ----------------------------------------------------------

    def _export(self, fmt: str) -> None:
        if self._current_state is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt}", f"untitled.{fmt.lower()}", f"{fmt} (*.{fmt.lower()})")
        if not path:
            return
        self._do_export(Path(path), fmt)

    def _export_dialog(self) -> None:
        if self._current_state is None:
            return
        sheet = ExportAsSheet(self, self._current_state.session.width, self._current_state.session.height)
        sheet.exported.connect(lambda p: self._do_export(Path(p), sheet.format_combo.currentText()))
        sheet.exec()

    def _export_jpeg_live(self) -> None:
        if self._current_state is None:
            return
        composite = self._current_state.canvas._composite()
        sheet = JpegPreviewSheet(self, composite)
        sheet.exported.connect(lambda p: self.statusBar().showMessage(tr("Saved {p}").format(p=p), 5000))
        sheet.exec()

    def _do_export(self, path: Path, fmt: str) -> None:
        if self._current_state is None:
            return
        composite = self._current_state.canvas._composite()
        from .rendering import export as exp
        if fmt.upper() == "PNG":
            exp.export_png(composite, path)
        elif fmt.upper() == "JPEG":
            exp.export_jpeg(composite, path)
        elif fmt.upper() == "PDF":
            exp.export_pdf(composite, path)
        self.statusBar().showMessage(f"Exported {path}", 5000)

    # ---- panels / refresh ------------------------------------------------

    def _refresh_tool_panels(self) -> None:
        # Re-bind the brush controls to whichever brush/eraser tool is active.
        # With no project open, panels are detached (set_session(None)) so
        # they don't keep showing the closed project's layers.
        has_state = self._current_state is not None
        for dock in self.findChildren(QDockWidget):
            widget = dock.widget()
            if hasattr(widget, "set_session"):
                widget.set_session(self._current_state.session if has_state else None)
            if has_state and hasattr(widget, "bind"):
                active = self._current_state.canvas._active_tool
                if active is not None and hasattr(active, "settings"):
                    widget.bind(active)

    def _show_message(self, msg: str) -> None:
        QMessageBox.information(self, "Compositor", msg)

    def _edit_shortcuts(self) -> None:
        dialog = KeyboardShortcutsDialog(self, self._shortcuts)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("Shortcuts updated", 3000)

    def _open_command_palette(self) -> None:
        items = list(self._command_index.items())
        palette = CommandPaletteDialog(self, [(name, fn) for name, fn in items])
        palette.exec()

    def _about(self) -> None:
        QMessageBox.about(
            self,
            tr("About Compositor (PySide)"),
            tr(
                "<h3>Compositor (PySide6)</h3>"
                "<p>A cross-platform Python port of <a href='https://github.com/robbietilton/Compositor'>Compositor</a>.</p>"
                "<p>MIT licensed. Built with PySide6, NumPy, OpenCV, Pillow.</p>"
            ),
        )

    def _open_camera_raw(self) -> None:
        if self._current_state is None:
            return
        from .ui.camera_raw import CameraRawDialog
        dialog = CameraRawDialog(self)
        dialog.developed.connect(self._add_camera_raw_layer)
        dialog.exec()

    def _add_camera_raw_layer(self, image: np.ndarray) -> None:
        if self._current_state is None:
            return
        session = self._current_state.session
        h, w = image.shape[:2]
        layer = ImageLayer(
            id=new_id(),
            name="Camera Raw",
            image_file=None,
            transform=Transform(size=(w, h)),
        )
        session.add_layer(layer)
        session._image_cache[layer.id] = image  # type: ignore[attr-defined]
        session.active_layer_id = layer.id
        self._current_state.canvas.set_session(session)
        self._refresh_tool_panels()


class ProjectState:
    """Per-tab state — the canvas, the document, the store, the watcher."""

    def __init__(self, canvas: CanvasWidget, session: EditorSession, store: ProjectStore, watcher: ProjectWatcher):
        self.canvas = canvas
        self.session = session
        self.store = store
        self.watcher = watcher
        self.tool_name: str = "move"


if __name__ == "__main__":
    sys.exit(main())
