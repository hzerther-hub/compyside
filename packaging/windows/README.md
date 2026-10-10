# Windows packaging notes

Two build paths are supported:

## 1. Briefcase → signed `.msi`

```cmd
briefcase create windows
briefcase build windows
briefcase package windows
```

The `.msi` lands in `dist\Compositor-0.1.0.msi`. To sign with a real
certificate:

```cmd
briefcase package windows ^
    --identity "CN=Your Company, O=Your Company, C=US" ^
    --entitlements packaging\windows\app.manifest
```

## 2. PyInstaller single-folder (portable build)

When you need a portable `.exe` (no MSI install), use the spec file. It
references `packaging/windows/app.manifest` for PerMonitorV2 DPI awareness.

```cmd
pip install pyinstaller
pyinstaller packaging\windows\pyinstaller.spec
```

Output: `dist\Compositor\Compositor.exe`. Zip the `Compositor/` folder and
distribute.

## What you still need to add

* `icon.ico` (256x256 with embedded 16/32/48/64/128 sizes). Drop it next to
  the spec / `pyproject.toml`. Briefcase falls back to its default icon if
  missing.
* Code-signing `.pfx` (for `signtool sign /f … /p … /t …`).
* Optional: MSIX packaging project if you want Microsoft Store distribution.

## Continuous integration

A reasonable GitHub Actions matrix on `windows-latest`:

```yaml
- uses: actions/setup-python@v5
  with: { python-version: "3.11" }
- run: pip install briefcase
- run: briefcase create windows
- run: briefcase build windows
- uses: actions/upload-artifact@v4
  with: { name: compositor-windows, path: dist/*.msi }
```
