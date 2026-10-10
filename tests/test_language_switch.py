"""Language-switch regression tests.

Switching the UI language rebuilds the menu bar, but the dock panels are
built once. Their static captions used to be set in ``__init__`` only, so
switching to English left the whole right-hand side in the previous
language. These tests pin the panels and the dock titles to the current
language.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QPushButton  # noqa: E402

from compositor import i18n  # noqa: E402
from compositor.ui.layers_panel import LayersPanel  # noqa: E402
from compositor.ui.transform_inspector import TransformInspector  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def restore_language():
    previous = i18n.current_language()
    yield
    i18n.set_language(previous)


def _texts(widget):
    out = []
    for child in widget.findChildren(QLabel) + widget.findChildren(QPushButton):
        text = child.text().strip()
        if text:
            out.append(text)
    return out


def _panel_is_english(panel) -> bool:
    return all(not any("一" <= ch <= "鿿" for ch in text) for text in _texts(panel))


# ---------------------------------------------------------------------------
# LayersPanel
# ---------------------------------------------------------------------------


def test_layers_panel_labels_follow_the_language(qapp, restore_language):
    panel = LayersPanel()
    i18n.set_language("zh_CN")
    zh = _texts(panel)
    assert "混合" in zh and "不透明度" in zh and "+ 图像" in zh

    i18n.set_language("en")
    en = _texts(panel)
    assert "Blend" in en and "Opacity" in en and "+ Image" in en
    assert _panel_is_english(panel)


def test_layers_panel_buttons_all_follow(qapp, restore_language):
    panel = LayersPanel()
    i18n.set_language("en")
    labels = [button.text() for button, _key in panel._add_buttons]
    assert labels == ["+ Image", "+ Group", "+ Adjust", "Delete"]


# ---------------------------------------------------------------------------
# TransformInspector
# ---------------------------------------------------------------------------


def test_transform_inspector_labels_follow_the_language(qapp, restore_language):
    inspector = TransformInspector()
    i18n.set_language("zh_CN")
    zh = _texts(inspector)
    assert "宽度" in zh and "旋转" in zh and "水平翻转" in zh

    i18n.set_language("en")
    en = _texts(inspector)
    assert "Width" in en and "Rotation" in en and "Flip Horizontal" in en
    assert _panel_is_english(inspector)


def test_form_row_labels_follow_the_language(qapp, restore_language):
    """QFormLayout holds its labels, so they must be relabelled in place."""
    inspector = TransformInspector()
    i18n.set_language("zh_CN")
    assert inspector.w_label.text() == "宽度"
    i18n.set_language("en")
    assert inspector.w_label.text() == "Width"
    assert inspector.h_label.text() == "Height"


# ---------------------------------------------------------------------------
# MainWindow dock titles
# ---------------------------------------------------------------------------


def test_main_window_dock_titles_follow_the_language(qapp, restore_language, tmp_path):
    from compositor.app import MainWindow

    window = MainWindow()
    try:
        i18n.set_language("zh_CN")
        window._rebuild_ui()
        assert window._layers_dock.windowTitle() == "图层"
        assert window._transform_dock.windowTitle() == "变换"

        i18n.set_language("en")
        window._rebuild_ui()
        assert window._layers_dock.windowTitle() == "Layers"
        assert window._transform_dock.windowTitle() == "Transform"
    finally:
        window.close()
        window.deleteLater()