# CI Pipeline

GitHub Actions workflows under `.github/workflows/`:

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push to `main`, PR, manual | `pytest` on Win/macOS/Linux + parity-report dry run; uploads parity artefacts; then builds `.deb`, `.AppImage`, `.msi`, `.dmg` and uploads them as build artefacts |
| `release.yml` | push a `v*` tag | Same packaging, but attaches artefacts to a GitHub release |

## Badge

Add this to the top of `README.md`:

```markdown
[![CI](https://github.com/yourname/comPySide/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/comPySide/actions/workflows/ci.yml)
```

## Required secrets (only for the release workflow)

| Secret | Purpose |
|---|---|
| `MACOS_SIGNING_IDENTITY` | `Developer ID Application: Your Name` — used by `codesign --sign` |
| `MACOS_NOTARYTOOL_PROFILE` | `xcrun notarytool store-credentials <profile>` profile name |
| `WINDOWS_CERT_PFX_BASE64` | Base64-encoded `.pfx` for `signtool sign` |
| `WINDOWS_CERT_PASSWORD` | PFX password |

Without these, the release workflow falls back to unsigned builds (still
functional; just not Gatekeeper-cleared).

## Dependabot

`.github/dependabot.yml` opens weekly PRs for both `pip` and `github-actions`
updates, grouped so runtime and optional deps stay in their own PRs.

## Manual smoke test

```bash
# Install briefcase and try a local build for your OS:
pip install briefcase
briefcase create  # uses the current OS
briefcase build
briefcase package
briefcase run    # launch the built app
```

## CI caching

The `setup-python@v5` step uses `cache: pip` to cache wheels between runs.
For Briefcase builds the wheel cache is also reused via
`~/.cache/briefcase`.
