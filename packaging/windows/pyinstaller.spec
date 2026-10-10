# PyInstaller spec for Compositor — single-folder build that mirrors the
# Briefcase MSI but skips WiX. Use when you only need an unpacked distributable
# (e.g. for portable USB stick or internal QA).
#
#   pip install pyinstaller
#   pyinstaller packaging/windows/pyinstaller.spec
#
# Output: dist/Compositor/Compositor.exe + a sibling Python runtime.

# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

# Pull the app's data path so PyInstaller can find the package.
sys.path.insert(0, os.path.abspath("src"))

a = Analysis(
    ["src/compositor/__main__.py"],
    pathex=[os.path.abspath("src")],
    binaries=[],
    datas=[],
    hiddenimports=[
        "compositor",
        "compositor.document",
        "compositor.format",
        "compositor.io",
        "compositor.rendering",
        "compositor.tools",
        "compositor.ui",
        "compositor.platform",
        "PySide6.QtSvg",
        "PySide6.QtXml",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "pandas",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Compositor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    manifest="packaging/windows/app.manifest",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Compositor",
)
