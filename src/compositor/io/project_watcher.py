"""ProjectWatcher — coalesce filesystem events on a .comp folder.

The Compositor spec says several writes in quick succession should arrive as
a single update. ``ProjectWatcher`` wraps ``QFileSystemWatcher`` and
coalesces events into a single Qt signal with a small debounce delay.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QFileSystemWatcher, QObject, QTimer, Signal


class ProjectWatcher(QObject):
    """Watch a .comp project folder for changes.

    Emits :pyattr:`changed` after a short quiet period (default 300 ms — the
    Compositor spec calls for ~1/3 second).
    """

    changed = Signal()

    def __init__(
        self,
        project_root: str | Path,
        parent: Optional[QObject] = None,
        debounce_ms: int = 300,
    ):
        super().__init__(parent)
        self._root = Path(project_root)
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_event)
        self._watcher.directoryChanged.connect(self._on_event)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self._fire)
        self._watched_paths: set[str] = set()

    # ---- public API -----------------------------------------------------

    def watch(self) -> None:
        """(Re)register the manifest + images directory with the watcher."""
        # QFileSystemWatcher silently drops duplicate paths.
        self._unwatch_all()
        manifest = self._root / "manifest.json"
        images = self._root / "images"
        for p in (manifest, images):
            if p.exists():
                self._watcher.addPath(str(p))
                self._watched_paths.add(str(p))

    def stop(self) -> None:
        self._unwatch_all()

    # ---- internals ------------------------------------------------------

    def _unwatch_all(self) -> None:
        if self._watched_paths:
            self._watcher.removePaths(list(self._watched_paths))
            self._watched_paths.clear()

    def _on_event(self, _path: str) -> None:
        # Some editors unlink and recreate; re-register in case the file
        # came back with a different inode.
        self._timer.start()
        self.watch()

    def _fire(self) -> None:
        self.changed.emit()
