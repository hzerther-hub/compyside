"""Shared pytest fixtures."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    """Provide a QApplication instance."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def qapp_or_skip(qapp):
    """Same as qapp — kept for clarity."""
    yield qapp
