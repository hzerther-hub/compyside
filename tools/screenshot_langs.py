"""Capture screenshots of the Compositor app in multiple languages."""
import os
import shutil
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from compositor import i18n
from compositor.app import MainWindow
from compositor.document.layers import ImageLayer, Transform, new_id
from compositor.format.manifest import CURRENT_VERSION, Manifest
from compositor.io.image_importer import encode_png
from compositor.io.project_store import ProjectStore


def build_demo(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    W, H = 256, 192
    m = Manifest(version=CURRENT_VERSION, color_space="sRGB",
                 document_id=new_id(), width=W, height=H, layers=[])
    store = ProjectStore.create(root, m)

    bg_id = new_id()
    bg = np.zeros((H, W, 4), dtype=np.uint8)
    for y in range(H):
        t = y / H
        bg[y, :, 0] = int(40 + 60 * t)
        bg[y, :, 1] = int(80 + 100 * t)
        bg[y, :, 2] = int(160 + 80 * t)
        bg[y, :, 3] = 255
    store.write_image(bg_id, encode_png(bg))

    fg_id = new_id()
    fg = np.zeros((H, W, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    circle = ((xx - 96) ** 2 + (yy - 96) ** 2) <= 50 ** 2
    fg[circle, 0] = 230
    fg[circle, 3] = 255
    store.write_image(fg_id, encode_png(fg))

    m.layers = [
        {"id": bg_id, "name": "Sky", "isVisible": True, "isGroup": False, "opacity": 1.0,
         "blendMode": "Normal",
         "transform": {"origin": [0, 0], "size": [W, H], "rotation": 0, "flipX": False,
                       "flipY": False, "sampling": "High quality"},
         "imageFile": f"{bg_id}.png"},
        {"id": fg_id, "name": "Shapes", "isVisible": True, "isGroup": False, "opacity": 1.0,
         "blendMode": "Normal",
         "transform": {"origin": [0, 0], "size": [W, H], "rotation": 0, "flipX": False,
                       "flipY": False, "sampling": "High quality"},
         "imageFile": f"{fg_id}.png"},
    ]
    m.active_layer_id = fg_id
    store.save_manifest(m)


def main() -> int:
    demo = Path("E:/comPySide/_demo")
    build_demo(demo)

    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.show()
    w.open_project(demo)

    langs = ["en", "zh_CN", "ja", "ko", "fr", "de"]
    idx = [0]

    def shoot() -> None:
        if idx[0] >= len(langs):
            app.quit()
            return
        code = langs[idx[0]]
        i18n.set_language(code)
        QTimer.singleShot(200, _capture)

    def _capture() -> None:
        out = f"E:/comPySide/screenshot_{code}.png"
        w.grab().save(out, "PNG")
        print(f"{code}: {out}")
        idx[0] += 1
        QTimer.singleShot(200, shoot)

    QTimer.singleShot(400, shoot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
