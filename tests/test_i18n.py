"""Tests for the i18n subsystem."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from compositor import i18n


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_tr_returns_english_when_unknown_language(monkeypatch) -> None:
    i18n.set_language("en")
    assert i18n.tr("Tools") == "Tools"
    assert i18n.tr("AI Backend Settings") == "AI Backend Settings"


def test_tr_falls_back_to_english_when_key_missing() -> None:
    i18n.set_language("en")
    # Even if we ask for a fictional language, missing keys fall through to EN.
    i18n.set_language("de")
    assert i18n.tr("Definitely not in any dictionary") == "Definitely not in any dictionary"


def test_tr_format_args() -> None:
    i18n.set_language("en")
    out = i18n.tr("Upscaled active layer to {w}×{h}", w=512, h=512)
    assert "512" in out


def test_set_language_returns_false_for_unknown() -> None:
    assert i18n.set_language("xx_YY") is False


def test_set_language_no_op_when_same() -> None:
    i18n.set_language("en")
    assert i18n.set_language("en") is False


def test_listener_fires_on_change() -> None:
    calls: list[str] = []

    def listener(code: str) -> None:
        calls.append(code)

    i18n.on_change(listener)
    i18n.set_language("en")  # no-op since current is en
    assert calls == []
    i18n.set_language("zh_CN")
    assert calls == ["zh_CN"]
    i18n.set_language("en")
    assert calls == ["zh_CN", "en"]


def test_listener_exception_does_not_break_dispatch() -> None:
    def bad(_: str) -> None:
        raise RuntimeError("boom")

    i18n.on_change(bad)
    # Must not raise; the failure is swallowed.
    i18n.set_language("zh_CN")
    i18n.set_language("en")


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------


def test_coverage_returns_between_0_and_1() -> None:
    cov = i18n.coverage("zh_CN")
    assert 0.0 <= cov <= 1.0


def test_untranslated_keys_returns_list() -> None:
    missing = i18n.untranslated_keys("ja")
    assert isinstance(missing, list)
    # X/Y are protected keys that stay identical to EN in every
    # language, so even fully-translated locales report them.
    assert "X" in missing and "Y" in missing


def test_arabic_fully_translated() -> None:
    # Arabic used to be a stub; it now covers every EN key.
    assert i18n.coverage("ar") == 1.0


def test_zh_cn_covers_almost_everything() -> None:
    """zh_CN is the priority language — assert it covers ≥ 95% of EN."""
    cov = i18n.coverage("zh_CN")
    assert cov >= 0.95


def test_reload_data_rereads_json(tmp_path: Path, monkeypatch) -> None:
    """Edit the JSON file in place and verify reload_data picks it up."""
    data = json.loads(Path(i18n._DATA_PATH).read_text(encoding="utf-8"))
    data["en"]["__test_marker__"] = "before"
    Path(i18n._DATA_PATH).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    i18n.reload_data()
    assert i18n.tr("__test_marker__") == "before"
    # Restore by removing the marker.
    data = json.loads(Path(i18n._DATA_PATH).read_text(encoding="utf-8"))
    data["en"].pop("__test_marker__", None)
    Path(i18n._DATA_PATH).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    i18n.reload_data()


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------


def test_langs_contain_canonical_codes() -> None:
    codes = {code for code, _ in i18n.LANGS}
    for expected in ("en", "zh_CN", "ja", "ko", "fr", "de", "it", "pt", "es"):
        assert expected in codes


def test_available_languages_matches_registry() -> None:
    assert len(i18n.available_languages()) == len(i18n.LANGS)


# ---------------------------------------------------------------------------
# Every tr() key used in the codebase must exist in EN
# ---------------------------------------------------------------------------


def test_all_tr_keys_are_in_en() -> None:
    """No code path may ask for a key that the dictionary doesn't define."""
    import re
    used: set[str] = set()
    for p in Path("src/compositor").rglob("*.py"):
        txt = p.read_text(encoding="utf-8")
        for m in re.finditer(r'tr\(\"([^\"]+)\"', txt):
            used.add(m.group(1))
    missing = used - set(i18n.EN.keys())
    assert not missing, f"tr() keys without EN entry: {sorted(missing)}"


# ---------------------------------------------------------------------------
# System language detection
# ---------------------------------------------------------------------------


def _with_locale(monkeypatch, value):
    import locale
    for env in ("LC_ALL", "LANG"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(locale, "getdefaultlocale", lambda: value)


def test_system_language_exact_and_prefix(monkeypatch) -> None:
    _with_locale(monkeypatch, ("ja_JP", "cp932"))
    assert i18n.system_language() == "ja"
    _with_locale(monkeypatch, ("fr_CA", "utf8"))
    assert i18n.system_language() == "fr"
    _with_locale(monkeypatch, ("it_IT@euro", "cp1252"))
    assert i18n.system_language() == "it"


def test_system_language_chinese_variants_map_to_zh_cn(monkeypatch) -> None:
    _with_locale(monkeypatch, ("zh_TW", "cp950"))
    assert i18n.system_language() == "zh_CN"
    _with_locale(monkeypatch, ("zh-HK", "utf8"))
    assert i18n.system_language() == "zh_CN"


def test_system_language_falls_back_to_english(monkeypatch) -> None:
    _with_locale(monkeypatch, (None, None))
    assert i18n.system_language() == "en"
    _with_locale(monkeypatch, ("es_Unknown", "cp1252"))
    assert i18n.system_language() == "es"  # language part still matches
    _with_locale(monkeypatch, ("xx_YY", "utf8"))
    assert i18n.system_language() == "en"  # nothing close → English
