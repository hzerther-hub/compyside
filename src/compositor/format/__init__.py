"""Manifest format: read, validate, write .comp project files.

Supports versions 1–11 (additive schema — see docs/project-format.md).
"""
from .manifest import (
    CURRENT_VERSION,
    Manifest,
    ManifestError,
    load_manifest,
    save_manifest,
    validate_manifest,
)

__all__ = [
    "CURRENT_VERSION",
    "Manifest",
    "ManifestError",
    "load_manifest",
    "save_manifest",
    "validate_manifest",
]
