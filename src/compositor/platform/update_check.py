"""Auto-update check via GitHub Releases API.

Fetches the latest release tag from the source repository and compares
against the running version. Honours ``check_for_updates`` config flag
(set via environment variable ``COMPOSITOR_UPDATE_CHECK`` — defaults to
``"1"``).
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional


CURRENT_VERSION = "0.1.0"
RELEASES_URL = (
    "https://api.github.com/repos/robbietilton/Compositor/"
    "releases/latest"
)
PORT_URL = "https://api.github.com/repos/robbietilton/Compositor-Port-PySide/releases/latest"
TIMEOUT_SEC = 4.0


@dataclass
class UpdateInfo:
    current: str
    latest: str
    url: str
    is_update_available: bool


def _parse_version(v: str) -> tuple:
    """Turn '0.1.0' into (0, 1, 0) for comparison."""
    parts = []
    for chunk in v.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits or 0))
    return tuple(parts[:3])


def check_for_updates() -> Optional[UpdateInfo]:
    """Return ``UpdateInfo`` if a newer release exists, else ``None``.

    Respects the ``COMPOSITOR_UPDATE_CHECK`` env var:
      - "0" disables the check entirely
      - "1" (default) enables
    """
    if os.environ.get("COMPOSITOR_UPDATE_CHECK", "1") == "0":
        return None
    try:
        req = urllib.request.Request(
            PORT_URL,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "Compositor-PySide/" + CURRENT_VERSION},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    latest_tag = data.get("tag_name", "").lstrip("v")
    if not latest_tag:
        return None
    html_url = data.get("html_url", "")
    is_new = _parse_version(latest_tag) > _parse_version(CURRENT_VERSION)
    return UpdateInfo(
        current=CURRENT_VERSION,
        latest=latest_tag,
        url=html_url,
        is_update_available=is_new,
    )
