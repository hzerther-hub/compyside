# Linux packaging notes

## 1. Briefcase → `.deb` + `.AppImage`

```bash
briefcase create linux
briefcase build linux
briefcase package linux --format deb
briefcase package linux --format appimage
```

Outputs:
* `dist/Compositor-0.1.0-amd64.deb`
* `dist/Compositor-0.1.0-x86_64.AppImage`

## 2. Manual `.deb`

If you don't want Briefcase (e.g. building inside a stripped-down CI), the
recipe is roughly:

```bash
mkdir -p build/compositor/DEBIAN
mkdir -p build/compositor/usr/{bin,lib,share/{applications,icons/hicolor/256x256/apps,metainfo}}

cp packaging/linux/compositor.desktop \
   build/compositor/usr/share/applications/

cp packaging/linux/org.compositor.Compositor.metainfo.xml \
   build/compositor/usr/share/metainfo/

cp packaging/linux/icon.png \
   build/compositor/usr/share/icons/hicolor/256x256/apps/compositor.png

# Install the Python package into the staging area
PREFIX=/usr pip install --root=build/compositor --prefix=/usr .

cat > build/compositor/DEBIAN/control <<EOF
Package: compositor
Version: 0.1.0
Section: graphics
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.10), python3-pyqt6 | python3-pyside6, libgl1, libegl1
Maintainer: comPySide contributors <devnull@example.com>
Description: Cross-platform PySide6 port of Compositor (image editor)
EOF

dpkg-deb --build build/compositor dist/compositor_0.1.0_amd64.deb
```

## Runtime dependencies

The Briefcase `system_requires` in `pyproject.toml` lists the apt packages
needed at build time on Debian / Ubuntu 22.04+:

* `libgl1-mesa-glx` — OpenGL
* `libegl1` — EGL (Qt RHI fallback)
* `libxkbcommon-x11-0` — keyboard handling
* `libdbus-1-3` — desktop integration
* `libxcb-cursor0` — Qt cursor theme

Optional:
* `libheif-dev` — for `pillow-heif` (HEIC import)
* `libraw-dev` — for `rawpy` (camera RAW import)

## Icon

`icon.png` (256×256) should sit next to the desktop file. Drop it at
`packaging/linux/icon.png` and reference it from `compositor.desktop`.

## Continuous integration

```yaml
runs-on: ubuntu-22.04
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with: { python-version: "3.11" }
  - run: sudo apt-get install -y libgl1-mesa-glx libegl1 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0
  - run: pip install briefcase
  - run: briefcase create linux
  - run: briefcase build linux
  - run: briefcase package linux --format deb
  - run: briefcase package linux --format appimage
  - uses: actions/upload-artifact@v4
    with: { name: compositor-linux, path: dist/*.{deb,AppImage} }
```
