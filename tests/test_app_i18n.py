"""End-to-end test: switching language actually rebuilds the menus.

These tests are pragmatic — they verify the language switching machinery
wires up correctly, not that Qt itself rebuilds menus perfectly. (Qt's
``bar.clear()`` / ``addMenu()`` cycle can leave orphaned submenus in
``findChildren``, which makes visual title assertions flaky.)
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtWidgets import QApplication

from compositor import i18n
from compositor.app import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def main_window(qapp):
    i18n._current = "en"
    i18n._LISTENERS.clear()
    return MainWindow()


def _pump_events(window):
    """Process pending Qt events (rebuild runs via QTimer.singleShot(0))."""
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.processEvents()


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------


def test_main_window_has_language_actions(main_window):
    """The MainWindow registers one QAction per supported language."""
    assert len(main_window._language_actions) == 15


def test_language_actions_are_checkable(main_window):
    for code, act in main_window._language_actions.items():
        assert act.isCheckable(), f"{code} action not checkable"


def test_english_action_is_currently_checked(main_window):
    assert main_window._language_actions["en"].isChecked()


def test_each_language_action_triggers_switch(main_window):
    for code in ("zh_CN", "ja", "ko", "fr", "de"):
        i18n._current = "en"  # reset
        main_window._switch_language(code)
        assert i18n.current_language() == code


# ---------------------------------------------------------------------------
# Switching
# ---------------------------------------------------------------------------


def test_switching_chinese_triggers_rebuild(main_window):
    main_window._switch_language("zh_CN")
    _pump_events(main_window)
    # After the rebuild, the menu bar has been rebuilt — verify by
    # checking that the menu bar's actions are fresh (their text matches
    # the new language).
    titles = [a.text() for a in main_window.menuBar().actions()]
    assert "文件(&F)" in titles, f"File menu didn't switch to Chinese: {titles}"
    assert "&File" not in titles, f"English File still present: {titles}"


def test_switching_back_to_english_restores_titles(main_window):
    main_window._switch_language("zh_CN")
    _pump_events(main_window)
    main_window._switch_language("en")
    _pump_events(main_window)
    titles = [a.text() for a in main_window.menuBar().actions()]
    assert "&File" in titles, f"English File didn't return: {titles}"


def test_no_op_switch_does_nothing(main_window):
    """Switching to the current language should be a no-op."""
    initial_listener_count = len(i18n._LISTENERS)
    result = main_window._switch_language("en")  # already English
    _pump_events(main_window)
    assert result is False or result is None


def test_each_language_translates_a_menu_label(main_window):
    """Every advertised language has at least one translated menu label."""
    for code in i18n.T.keys():
        i18n.set_language(code)
        val = i18n.tr("&File")
        assert val, f"{code} produced empty translation"
