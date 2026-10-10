"""Tests for tools/i18n_translate.py — placeholder protection + missing detection.

The actual Google Translate call is exercised manually in dev; in CI we
keep it offline so the tests don't hit the network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import i18n_translate  # noqa: E402


# ---------------------------------------------------------------------------
# Placeholder protection
# ---------------------------------------------------------------------------


def test_protect_simple_placeholder() -> None:
    masked, phs = i18n_translate.protect_placeholders("Image is {w}×{h} px")
    assert phs == ["{w}", "{h}"]
    assert "{w}" not in masked and "{h}" not in masked


def test_restore_simple_placeholder() -> None:
    text = "Image is {w}×{h} px"
    masked, phs = i18n_translate.protect_placeholders(text)
    # Simulate translation that preserves the sentinels.
    out = i18n_translate.restore_placeholders(masked, phs)
    assert out == text


def test_protect_preserves_no_placeholders() -> None:
    masked, phs = i18n_translate.protect_placeholders("No placeholders here")
    assert phs == []
    assert masked == "No placeholders here"


def test_protect_with_format_spec() -> None:
    masked, phs = i18n_translate.protect_placeholders("Count: {n:03d}")
    assert phs == ["{n:03d}"]
    assert "{n:03d}" not in masked


def test_protect_with_underscore_in_name() -> None:
    masked, phs = i18n_translate.protect_placeholders("{path_with_underscore}")
    assert phs == ["{path_with_underscore}"]


def test_protect_handles_duplicate_names() -> None:
    """Two {w} placeholders → two separate sentinels so each survives."""
    masked, phs = i18n_translate.protect_placeholders("{w} × {w}")
    assert phs == ["{w}", "{w}"]
    out = i18n_translate.restore_placeholders(masked, phs)
    assert out == "{w} × {w}"


def test_restore_handles_extra_text() -> None:
    """If the translator wraps the sentinel in surrounding text, restore still finds it."""
    text = "Image is {w}×{h} px"
    masked, phs = i18n_translate.protect_placeholders(text)
    # Simulated translator output where the sentinel is preserved verbatim
    # but surrounded by translated text.
    fake = "图像是" + masked  # translator kept the masked tail unchanged
    out = i18n_translate.restore_placeholders(fake, phs)
    # Both placeholders are restored verbatim.
    assert "{w}" in out
    assert "{h}" in out
    # The masked sentinel no longer appears in the output.
    assert "␀PH␀" not in out


# ---------------------------------------------------------------------------
# Missing detection
# ---------------------------------------------------------------------------


def test_identify_missing_finds_english_fallbacks(tmp_path: Path) -> None:
    fake = {
        "en": {"a": "Hello", "b": "World"},
        "fr": {"a": "Bonjour", "b": "World"},   # b missing
        "de": {"a": "", "b": "Welt"},            # a empty
        "ja": {},                                  # whole table empty
    }
    items = i18n_translate.identify_missing(fake, ["fr", "de", "ja"])
    keys = {(code, k) for code, k, _ in items}
    assert ("fr", "b") in keys
    assert ("de", "a") in keys
    assert ("ja", "a") in keys
    assert ("ja", "b") in keys


def test_identify_missing_protects_short_keys() -> None:
    fake = {
        "en": {"X": "X", "Y": "Y"},
        "fr": {"X": "X", "Y": "Y"},  # identical to EN but should not be translated
    }
    items = i18n_translate.identify_missing(fake, ["fr"])
    # X and Y are in PROTECTED_KEYS so the tool refuses to send them to the translator.
    assert all(k != "X" and k != "Y" for _, k, _ in items)


def test_identify_missing_skips_en() -> None:
    fake = {"en": {"a": "Hello"}}
    items = i18n_translate.identify_missing(fake, ["en"])
    assert items == []


def test_is_missing_logic() -> None:
    assert i18n_translate._is_missing("", "Hello") is True
    assert i18n_translate._is_missing("Hello", "Hello") is True  # identical = fallback
    assert i18n_translate._is_missing("Bonjour", "Hello") is False
    assert i18n_translate._is_missing("Hello!", "Hello") is False  # different


# ---------------------------------------------------------------------------
# Offline backend integration
# ---------------------------------------------------------------------------


def test_offline_backend_passes_text_through() -> None:
    """The 'offline' backend must not call the network; it just returns the input."""
    from i18n_translate import Translator
    t = Translator("offline")
    assert t.translate("Hello", source="en", target="zh-CN") == "Hello"
    assert t.translate("", source="en", target="zh-CN") == ""


def test_unknown_backend_raises() -> None:
    from i18n_translate import Translator
    t = Translator("bogus")
    with pytest.raises(ValueError):
        t.translate("Hello", source="en", target="fr")
