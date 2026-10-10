# coding=utf-8
"""Internationalisation — 15 languages.

Translation data lives in ``i18n_data.json`` so editors don't need to
touch Python. See ``tools/i18n_coverage.py`` to check per-language
coverage and find untranslated keys.

Public API (unchanged from the inline-data version):

* ``tr(key)`` / ``tr(key, *args, **kwargs)`` — look up the current
  language; missing keys fall back to English, then to ``key`` itself.
* ``set_language(code)`` — switch language; notify listeners on change.
* ``current_language()`` — the active language code.
* ``available_languages()`` — ``[(code, native_label), …]`` for the menu.
* ``code_label(code)`` — native label for a code.
* ``display_name(key)`` — ``"<translation> (<English>)"`` helper.
* ``on_change(fn)`` / ``untranslated_keys(code)`` / ``coverage(code)``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List


# ---------------------------------------------------------------------------
# Language registry — code, native name
# ---------------------------------------------------------------------------

LANGS: List[tuple] = [
    ("en", "English"),
    ("zh_CN", "简体中文"),
    ("ja", "日本語"),
    ("ko", "한국어"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("it", "Italiano"),
    ("pt", "Português"),
    ("es", "Español"),
    ("vi", "Tiếng Việt"),
    ("tr", "Türkçe"),
    ("ar", "العربية"),
    ("hi", "हिन्दी"),
    ("id", "Bahasa Indonesia"),
    ("th", "ภาษาไทย"),
]
LANG_NAMES = dict(LANGS)


# ---------------------------------------------------------------------------
# Translation data — loaded from a sibling JSON file
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).parent / "i18n_data.json"


def _load_data() -> Dict[str, Dict[str, str]]:
    with open(_DATA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


_DATA: Dict[str, Dict[str, str]] = _load_data()
T: Dict[str, Dict[str, str]] = _DATA
EN: Dict[str, str] = _DATA["en"]


def reload_data() -> None:
    """Re-read the JSON. Useful for tooling that mutates translations on
    disk and wants to see the effect without restarting the app."""
    global _DATA, T, EN
    _DATA = _load_data()
    T = _DATA
    EN = _DATA["en"]


# ---------------------------------------------------------------------------
# Translation lookup
# ---------------------------------------------------------------------------

_current = "en"

# Language-change listeners. Entries are either a plain callable or a
# ``weakref.WeakMethod`` wrapping a widget's bound ``_refresh_labels``, so a
# closed panel or dialog drops out of the list instead of being called back
# after its Qt object is gone.
_LISTENERS: List[Callable[[str], None]] = []


def tr(key: str, *args, **kwargs) -> str:
    """Look up ``key`` in the current language with smart fallback.

    Priority:
        1. Current language table.
        2. English table (default).
        3. ``key`` itself.

    Placeholders like ``{w}`` / ``{path}`` are filled by ``args`` (positional)
    or ``kwargs`` (named).
    """
    table = T.get(_current, {})
    val = table.get(key)
    if not val:
        val = T["en"].get(key, key)
    if args or kwargs:
        try:
            return val.format(*args, **kwargs)
        except Exception:
            return val
    return val


def set_language(code: str) -> bool:
    """Switch language; notify listeners on change."""
    global _current
    if code not in T:
        return False
    if code == _current:
        return False
    _current = code
    import weakref

    for entry in list(_LISTENERS):
        if isinstance(entry, weakref.WeakMethod):
            fn = entry()
            if fn is None:
                # The widget went away without unregistering.
                _LISTENERS.remove(entry)
                continue
        else:
            fn = entry
        try:
            fn(code)
        except Exception:
            # Losing one refresh is better than losing the whole switch.
            pass
    return True


def current_language() -> str:
    return _current


def available_languages() -> List[tuple]:
    return [(code, native) for code, native in LANGS]


def code_label(code: str) -> str:
    return LANG_NAMES.get(code, code)


def system_language() -> str:
    """Best supported match for the OS UI language; ``"en"`` when none.

    Windows reports the regional locale via ``locale.getdefaultlocale``;
    Unix convention uses ``$LC_ALL`` / ``$LANG``. Exact codes match first
    (``zh_CN``), then the language part (``fr_CA`` → ``fr``); any Chinese
    variant maps to ``zh_CN`` (the only bundled variant).
    """
    import os
    import sys

    code = ""
    if sys.platform == "win32":
        try:
            import locale
            code = locale.getdefaultlocale()[0] or ""
        except Exception:
            code = ""
    else:
        code = os.environ.get("LC_ALL") or os.environ.get("LANG") or ""
        if not code:
            try:
                import locale
                code = locale.getdefaultlocale()[0] or ""
            except Exception:
                code = ""
    code = code.replace("-", "_").split(".")[0].strip()
    if not code:
        return "en"
    if code in LANG_NAMES:
        return code
    lang = code.split("_")[0].lower()
    if lang == "zh":
        return "zh_CN"
    for supported in LANG_NAMES:
        if supported.split("_")[0].lower() == lang:
            return supported
    return "en"


def display_name(key: str) -> str:
    """Return ``"<translation> (<English>)"`` so users always see both."""
    english = T["en"].get(key, key)
    if _current == "en":
        return english
    translated = T.get(_current, {}).get(key, english)
    if not translated or translated == english:
        return english
    return f"{translated} ({english})"


def on_change(fn: Callable[[str], None]) -> None:
    """Register a callback fired when the language switches.

    A widget's bound method is held weakly: panels and dialogs register
    themselves in ``__init__`` and never unregister, so a strong reference
    would keep the listener alive after the widget is closed.
    """
    import weakref

    try:
        _LISTENERS.append(weakref.WeakMethod(fn))  # type: ignore[arg-type]
    except TypeError:
        _LISTENERS.append(fn)


def untranslated_keys(code: str) -> List[str]:
    """Keys in EN whose translation differs from EN (or is missing) in ``code``.

    A key is reported as untranslated when its value in ``T[code]`` is
    either missing or identical to the English string — i.e. the user
    would see English for that key.
    """
    if code not in T:
        return list(EN.keys())
    en = EN
    table = T[code]
    return [k for k in en if not table.get(k) or table[k] == en[k]]


def coverage(code: str) -> float:
    """Fraction of EN keys explicitly translated in ``code`` (0..1)."""
    if code not in T or not EN:
        return 0.0
    en_keys = set(EN.keys())
    table = T[code]
    translated = sum(1 for k in en_keys if table.get(k) and table[k] != EN[k])
    return translated / max(1, len(en_keys))


# Backwards-compatible alias.
CURRENT_LANGUAGE = property(lambda self: _current)  # type: ignore[assignment]
"""Compatibility shim — kept for the old ``i18n.CURRENT_LANGUAGE`` access."""
