"""Take a Qt screenshot of the loaded demo project.

Run: QT_QPA_PLATFORM=offscreen python tools/screenshot_demo.py
"""
import os
import shutil
import sys
from pathlib import Path

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
    manifest = Manifest(version=CURRENT_VERSION, color_space="sRGB",
                        document_id=new_id(), width=W, height=H, layers=[])
    store = ProjectStore.create(root, manifest)

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
    square = ((xx >= 130) & (xx <= 200) & (yy >= 50) & (yy <= 120))
    stripe = ((xx - yy) > 0) & ((xx - yy) < 30) & (yy > 100)
    fg[circle, 0] = 230; fg[circle, 3] = 255
    fg[square, 1] = 200; fg[square, 3] = 255
    fg[stripe, 0] = 255; fg[stripe, 1] = 220; fg[stripe, 2] = 0; fg[stripe, 3] = 255
    store.write_image(fg_id, encode_png(fg))

    m = Manifest(version=CURRENT_VERSION, color_space="sRGB",
                 document_id=manifest.document_id, width=W, height=H, layers=[
        {"id": bg_id, "name": "Sky", "isVisible": True, "isGroup": False,
         "opacity": 1.0, "blendMode": "Normal",
         "transform": {"origin": [0, 0], "size": [W, H], "rotation": 0,
                       "flipX": False, "flipY": False, "sampling": "High quality"},
         "imageFile": f"{bg_id}.png"},
        {"id": fg_id, "name": "Shapes", "isVisible": True, "isGroup": False,
         "opacity": 1.0, "blendMode": "Normal",
         "transform": {"origin": [0, 0], "size": [W, H], "rotation": 0,
                       "flipX": False, "flipY": False, "sampling": "High quality"},
         "imageFile": f"{fg_id}.png"},
    ])
    m.active_layer_id = fg_id
    store.save_manifest(m)


def main() -> int:
    demo = Path("E:/comPySide/_demo")
    build_demo(demo)

    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.show()
    w.open_project(demo)

    def grab_en() -> None:
        w.grab().save("E:/comPySide/screenshot_en.png", "PNG")
        i18n.set_language("zh_CN")
        # Schedule the second shot after the menu rebuild settles.
        QTimer.singleShot(300, grab_zh)

    def grab_zh() -> None:
        w.grab().save("E:/comPySide/screenshot_zh.png", "PNG")
        state = w._current_state
        print(f"Window {w.width()}x{w.height()}, "
              f"layers={len(state.session.layers) if state else 0}, "
              f"tool={state.tool_name if state else 'n/a'}, "
              f"language={i18n.CURRENT_LANGUAGE}")
        app.quit()

    QTimer.singleShot(500, grab_en)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
