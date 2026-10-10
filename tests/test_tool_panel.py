"""Tool panel tests.

The left-edge tool list used to be a QToolBar, which centres its button
text. Every translation has a different label width, so the column rendered
as a staircase. It is now a column of QPushButtons carrying a
``text-align: left`` stylesheet — these tests pin that, plus the check
state that the old QActionGroup provided.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QPushButton, QSizePolicy  # noqa: E402

from compositor import i18n  # noqa: E402
from compositor.app import _TOOL_PANEL_QSS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp):
    from compositor.app import MainWindow

    previous = i18n.current_language()
    win = MainWindow()
    yield win
    i18n.set_language(previous)
    win.close()
    win.deleteLater()


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_stylesheet_left_aligns():
    assert "text-align: left" in _TOOL_PANEL_QSS


def test_stylesheet_is_applied_to_the_panel(window):
    assert "text-align: left" in window._tools_panel.styleSheet()


def test_tool_buttons_fill_the_panel_width(window):
    """A shrink-to-fit button would centre its own text, which is the bug."""
    for button in window._tool_buttons.values():
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding


def test_panel_is_built_from_push_buttons_not_a_toolbar(window):
    assert window._tools_panel.findChildren(QPushButton)
    # No QToolBar is added to the window any more.
    from PySide6.QtWidgets import QToolBar

    assert not [tb for tb in window.findChildren(QToolBar) if tb.isVisible()]


# ---------------------------------------------------------------------------
# Check state (used to come from the QActionGroup)
# ---------------------------------------------------------------------------


def test_buttons_are_checkable_and_mutually_exclusive(window):
    buttons = window._tool_buttons
    assert buttons
    assert all(b.isCheckable() for b in buttons.values())
    assert all(b.autoExclusive() for b in buttons.values())

    window._tools_dock.setEnabled(True)
    buttons["brush"].click()
    assert buttons["brush"].isChecked()

    buttons["crop"].click()
    assert buttons["crop"].isChecked()
    assert not buttons["brush"].isChecked()


def test_active_tool_is_reflected_on_startup(window):
    from compositor.document.editor_session import EditorSession
    from compositor.document.layers import ImageLayer, Transform, new_id

    session = EditorSession(width=64, height=64)
    layer = ImageLayer(
        id=new_id(),
        name="L",
        image_file=f"{new_id()}.png",
        transform=Transform(size=(10, 10)),
    )
    session.add_layer(layer)
    state = window._states.setdefault(0, None) or _FakeState(session)
    window._states[0] = state
    state.tool_name = "brush"
    window._current_state = state
    window._build_toolbar()

    assert window._tool_buttons["brush"].isChecked()
    assert not window._tool_buttons["crop"].isChecked()


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------


def test_labels_follow_the_language(window):
    i18n.set_language("zh_CN")
    window._build_toolbar()
    assert window._tool_buttons["blur"].text() == "模糊 (Blur)"

    i18n.set_language("en")
    window._build_toolbar()
    assert window._tool_buttons["blur"].text() == "Blur"


class _FakeState:
    """Just enough of ProjectState for the toolbar to read ``tool_name``."""

    def __init__(self, session):
        self.session = session
        self.tool_name = None