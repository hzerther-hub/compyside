"""New-project sheet and dock-visibility tests."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMenu  # noqa: E402

from compositor import i18n  # noqa: E402
from compositor.ui.new_project_sheet import (  # noqa: E402
    DEFAULT_SIZE,
    MAX_DIMENSION,
    NewProjectSheet,
)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def restore_language():
    previous = i18n.current_language()
    yield
    i18n.set_language(previous)


# ---------------------------------------------------------------------------
# New-project sheet
# ---------------------------------------------------------------------------


def test_defaults_match_the_old_dialog(qapp, restore_language):
    sheet = NewProjectSheet()
    assert sheet.size() == (DEFAULT_SIZE, DEFAULT_SIZE)
    assert DEFAULT_SIZE == 1024


def test_both_dimensions_are_editable_at_once(qapp, restore_language):
    """The old flow asked for width, then height, in two modal dialogs."""
    sheet = NewProjectSheet()
    assert sheet.width_spin.isEnabled()
    assert sheet.height_spin.isEnabled()
    sheet.width_spin.setValue(1280)
    sheet.height_spin.setValue(720)
    assert sheet.size() == (1280, 720)


def test_dimension_range_matches_the_manifest(qapp, restore_language):
    sheet = NewProjectSheet()
    assert sheet.width_spin.maximum() == MAX_DIMENSION
    assert sheet.height_spin.maximum() == MAX_DIMENSION
    assert sheet.width_spin.minimum() == 1


def test_width_and_height_are_independent_when_unlinked(qapp, restore_language):
    sheet = NewProjectSheet()
    sheet.width_spin.setValue(1920)
    assert sheet.size() == (1920, DEFAULT_SIZE)
    sheet.height_spin.setValue(600)
    assert sheet.size() == (1920, 600)


def test_link_keeps_them_square(qapp, restore_language):
    sheet = NewProjectSheet()
    sheet.link_check.setChecked(True)
    sheet.width_spin.setValue(800)
    assert sheet.size() == (800, 800)


def test_linking_does_not_recurse(qapp, restore_language):
    sheet = NewProjectSheet()
    sheet.link_check.setChecked(True)
    sheet.height_spin.setValue(500)
    assert sheet.size() == (500, 500)
    assert sheet._linking is False


def test_unlinking_lets_them_diverge(qapp, restore_language):
    sheet = NewProjectSheet()
    sheet.link_check.setChecked(True)
    sheet.height_spin.setValue(500)
    assert sheet.size() == (500, 500)

    sheet.link_check.setChecked(False)
    sheet.height_spin.setValue(600)
    assert sheet.size() == (500, 600)


def test_preset_label_shows_the_current_size(qapp, restore_language):
    i18n.set_language("zh_CN")
    sheet = NewProjectSheet()
    sheet.width_spin.setValue(1920)
    sheet.height_spin.setValue(1080)
    assert "1920" in sheet.preset_label.text()
    assert "1080" in sheet.preset_label.text()


def test_labels_follow_the_language(qapp, restore_language):
    i18n.set_language("zh_CN")
    sheet = NewProjectSheet()
    assert sheet.windowTitle() == "新建项目"
    assert sheet.link_check.text() == "锁定宽高比"

    i18n.set_language("en")
    assert sheet.windowTitle() == "New Project"
    assert sheet.link_check.text() == "Lock aspect ratio"


# ---------------------------------------------------------------------------
# View ▸ Panels
# ---------------------------------------------------------------------------


@pytest.fixture()
def window(qapp):
    from PySide6.QtCore import QCoreApplication, QEvent

    from compositor.app import MainWindow

    previous = i18n.current_language()
    win = MainWindow()
    win.show()
    yield win
    # Restore the language while the window is still alive: its
    # _refresh_labels listener is never unregistered, so calling it after
    # the C++ object is gone is not safe.
    i18n.set_language(previous)
    win.close()
    win.deleteLater()
    # Each rebuild queues deleteLater on the dock and menus it replaces.
    # Drain those before the next test, and once more at the end, or the
    # pile-up keeps the Qt event loop alive and pytest never exits.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _panels_menu(window):
    """The View ▸ Panels submenu, found by position.

    Deliberately does not switch language: set_language defers a rebuild
    with singleShot(0), and pumping the queue from inside the rebuild
    re-enters _build_menu.
    """
    view = [a.menu() for a in window.menuBar().actions()
            if a.menu() and a.menu().title() == i18n.tr("&View")][0]
    submenus = [m.menu() for m in view.actions() if m.menu()]
    assert submenus, "View has no submenu"
    return submenus[-1]


def test_panels_menu_exists(window):
    assert _panels_menu(window) is not None


def test_panels_start_checked(window):
    """isVisible() is False before the window is shown, which unticked them."""
    actions = _panels_menu(window).actions()
    assert [a.isChecked() for a in actions] == [True] * len(actions)


def test_menu_reopens_a_closed_dock(window, qapp):
    actions = _panels_menu(window).actions()
    window._layers_dock.close()
    qapp.processEvents()
    assert not actions[0].isChecked()

    actions[0].setChecked(True)
    qapp.processEvents()
    assert window._layers_dock.isVisible()


def test_closing_the_dock_unticks_the_menu(window, qapp):
    actions = _panels_menu(window).actions()
    window._transform_dock.close()
    qapp.processEvents()
    assert not actions[1].isChecked()


def test_every_panel_can_be_restored(window, qapp):
    actions = _panels_menu(window).actions()
    for action in actions:
        action.setChecked(False)
    qapp.processEvents()
    assert not any(window._panel_visibility.values())

    for action in actions:
        action.setChecked(True)
    qapp.processEvents()
    assert all(window._panel_visibility.values())
    assert window._layers_dock.isVisible()
    assert window._transform_dock.isVisible()


def test_panel_mnemonics_are_distinct(window):
    labels = [a.text() for a in _panels_menu(window).actions()]
    marks = [label[label.index("&") + 1] for label in labels if "&" in label]
    assert len(marks) == len(set(marks)), labels


def test_hidden_panel_stays_hidden_across_a_language_switch(window, qapp):
    actions = _panels_menu(window).actions()
    actions[2].setChecked(False)
    qapp.processEvents()
    assert not window._tools_dock.isVisible()

    # Drive the real path: switch language, then run the deferred rebuild the
    # window queued. Doing it in that order avoids re-entering _build_menu.
    i18n.set_language("en")
    window._rebuild_ui()
    qapp.processEvents()
    assert not window._tools_dock.isVisible()


def test_synchronous_rebuild_does_not_crash(window):
    """A submenu dies with its parent, so the stale-menu loop must not
    call deleteLater on it — the deferred rebuild in the GUI hides this."""
    window._layers_dock.close()
    for _ in range(3):
        window._rebuild_ui()