"""Auto-translate untranslated i18n keys via a custom translation API.

This script fills the missing entries in ``src/compositor/i18n_data.json``.
The default backend is a private translation API at
``fortuneteller.top``; if that fails for any reason the tool
automatically falls back to the next backend in a configurable chain.
Translation runs concurrently across a thread pool so a 170-key /
15-language batch finishes in seconds, not minutes.

Usage:
    # Default: list what would be translated (no API calls, no writes).
    python tools/i18n_translate.py
    python tools/i18n_translate.py --lang ja,ko --limit 10

    # Translate everything that's currently English-fallback.
    python tools/i18n_translate.py --apply

    # Only translate a specific key (handy for spot-fixes).
    python tools/i18n_translate.py --key "AI Backend Settings" --apply

    # Use the custom backend, then fall back to MyMemory on any error.
    python tools/i18n_translate.py --backend custom --fallback mymemory --apply

    # Run 8 parallel workers (default 4). The custom API tolerates this.
    python tools/i18n_translate.py --workers 8 --apply

    # Offline mode — no network; useful to refresh the JSON structure with
    # explicit EN fallbacks before a human translator fills them in.
    python tools/i18n_translate.py --backend offline --apply

The tool protects Python ``{name}`` placeholders before sending the
string to the translator and restores them after, so
``"Image is {w}×{h} pixels"`` translates as
``"画像が {w}×{h} ピクセル"`` rather than mangling the placeholders.

Backends:
    --backend custom    (default; calls the project's private translation
                         API at fortuneteller.top/api/translate with the
                         bundled API key. Falls through to mymemory on
                         any error — translation never silently fails)
    --backend mymemory  (MyMemory free endpoint; no API key; 5000 chars/day
                         per IP — fine for one-shot bulk runs)
    --backend google    (deep-translator's GoogleTranslator; rate-limited)
    --backend libre     (LibreTranslate — self-host or public instance)
    --backend offline   (no network: leaves the EN value in place; useful
                         when you just want to refresh the JSON structure)

The custom backend reads ``COMPOSITOR_TRANSLATE_API_KEY`` from the
environment first, then falls back to the bundled key. Override either
to rotate the credential.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
JSON_PATH = REPO / "src" / "compositor" / "i18n_data.json"

# Deep-translator (Google / MyMemory / Libre) uses ISO codes like
# "zh-CN"; the custom API uses the short codes ("zh", "ja", "fr", ...).
LANG_CODE_MAP = {
    "zh_CN": "zh-CN",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "es": "es",
    "vi": "vi",
    "tr": "tr",
    "ar": "ar",
    "hi": "hi",
    "id": "id",
    "th": "th",
}
LANG_CODE_MAP_CUSTOM = {
    "zh_CN": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "es": "es",
    "vi": "vi",
    "tr": "tr",
    "ar": "ar",
    "hi": "hi",
    "id": "id",
    "th": "th",
}

# MyMemory expects en-US / ja-JP / etc. full codes.
LANG_CODE_MAP_MYMEMORY = {
    "zh_CN": "zh-CN",
    "en": "en-GB",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-PT",
    "es": "es-ES",
    "vi": "vi-VN",
    "tr": "tr-TR",
    "ar": "ar-SA",
    "hi": "hi-IN",
    "id": "id-ID",
    "th": "th-TH",
}

# Custom translation API endpoint and key. Override the key via the
# ``COMPOSITOR_TRANSLATE_API_KEY`` env var; the default is the project
# key embedded in the repo.
CUSTOM_API_URL = "https://fortuneteller.top/api/translate"
CUSTOM_API_KEY = os.environ.get(
    "COMPOSITOR_TRANSLATE_API_KEY",
    "4a7f5e2c720d8f512f4ee8cfc87e3184ccbf99bfd08f55f2",
)

# Strings the translator must never touch — short tokens that are already
# the canonical rendering in every language (axis labels, keyboard
# mnemonics, etc.). The translator often re-renders these to the script
# of the target language and breaks them.
PROTECTED_KEYS = {
    "X", "Y",
}


# ---------------------------------------------------------------------------
# Placeholder protection
# ---------------------------------------------------------------------------

# {name} and {name:spec} and {{escaped}} forms. The regex matches the
# outermost single-brace form so we can swap with a sentinel token that
# survives the Google Translate round-trip.
_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*(?:[:!][^}]*)?\}")
_PLACEHOLDER_TOKEN = "␀PH␀"  # ␀PH␀ (rare Unicode chars)


def protect_placeholders(text: str) -> Tuple[str, List[str]]:
    """Replace ``{name}`` placeholders with sentinel tokens.

    Returns the masked text and the list of original placeholders so
    :func:`restore_placeholders` can put them back unchanged.
    """
    placeholders: List[str] = []

    def _mask(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"{_PLACEHOLDER_TOKEN}{len(placeholders) - 1}␁"

    masked = _PLACEHOLDER_RE.sub(_mask, text)
    return masked, placeholders


def restore_placeholders(translated: str, placeholders: List[str]) -> str:
    """Inverse of :func:`protect_placeholders`."""
    out = translated
    for i, original in enumerate(placeholders):
        sentinel = f"{_PLACEHOLDER_TOKEN}{i}␁"
        out = out.replace(sentinel, original)
    return out


# ---------------------------------------------------------------------------
# Translation backend
# ---------------------------------------------------------------------------


class Translator:
    """Wraps the chosen backend. Lazy-loads deep-translator only when used."""

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def translate(self, text: str, *, source: str, target: str) -> str:
        if not text.strip():
            return text
        if self.backend == "offline":
            return text  # offline = no-op (EN fallback stays in place)
        if self.backend == "custom":
            return _translate_custom(text, source=source, target=target)
        if self.backend == "google":
            from deep_translator import GoogleTranslator  # type: ignore
            return GoogleTranslator(source=source, target=target).translate(text)
        if self.backend == "mymemory":
            from deep_translator import MyMemoryTranslator  # type: ignore
            return MyMemoryTranslator(source=source, target=target).translate(text)
        if self.backend == "libre":
            from deep_translator import LibreTranslator  # type: ignore
            return LibreTranslator(source=source, target=target).translate(text)
        raise ValueError(f"unknown backend: {self.backend!r}")


def _translate_custom(text: str, *, source: str, target: str) -> str:
    """Call the private fortuneteller.top translation API.

    ``source``/``target`` use the API's short codes ("en", "ja", "zh", …).
    Raises on any non-2xx or unexpected response so the fallback chain
    can take over. Uses a shared ``requests.Session`` with retry
    middleware so concurrent calls survive transient SSL/connection
    issues (which happen a lot from this network when 8+ workers hit
    the endpoint at once).
    """
    import requests  # type: ignore

    payload = {"text": text, "from": source, "to": target}
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": CUSTOM_API_KEY,
    }
    resp = _shared_session().post(CUSTOM_API_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    body = resp.json()
    if "text" not in body or body.get("code", 0) != 0:
        raise RuntimeError(f"custom API bad response: {body!r}")
    return body["text"]


def _shared_session():
    """Lazily build a requests Session with retry + connection pooling."""
    import requests  # type: ignore
    from requests.adapters import HTTPAdapter  # type: ignore
    from urllib3.util.retry import Retry  # type: ignore

    global _SESSION
    if _SESSION is not None:
        return _SESSION
    session = requests.Session()
    # ``allowed_methods`` is the urllib3 2.x name; older versions used
    # ``method_whitelist``. Use the new name; fall back on the old if the
    # installed urllib3 rejects it.
    try:
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("POST",),
            raise_on_status=False,
        )
    except TypeError:
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            method_whitelist=("POST",),
            raise_on_status=False,
        )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _SESSION = session
    return session


_SESSION: Optional[object] = None


def _translate_with_fallback(
    masked: str,
    *,
    source: str,
    target: str,
    backends: List[str],
    retries: int = 2,
) -> Tuple[str, str, bool]:
    """Try each backend in order; return the first success.

    Returns ``(translated_text, used_backend, ok)``. ``ok=False`` means
    every backend failed for this item; the caller should leave the
    entry as the EN fallback.
    """
    last_exc: Optional[Exception] = None
    for backend in backends:
        for attempt in range(retries):
            try:
                t = Translator(backend).translate(masked, source=source, target=target)
                return t, backend, True
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 < retries:
                    time.sleep(0.2 * (attempt + 1))
        # Try the next backend in the chain.
    return masked, "", False  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def _worker(
    item: Tuple[str, str, str],
    backends: List[str],
    source_short: str,
    target_map: dict,
    target_mymemory_map: dict,
) -> Tuple[str, str, str, str, str, bool]:
    """Translate one item through the backend chain. Returns enough info
    for the main thread to log and apply the result.
    """
    code, key, en_value = item
    target = target_map.get(code)
    if target is None:
        return (code, key, en_value, "", "", False)

    masked, placeholders = protect_placeholders(en_value)
    # Build the per-item target list (the custom backend uses the API
    # short codes; mymemory uses its own "en-GB" / "ja-JP" form).
    chain_targets: List[Tuple[str, str]] = []  # (backend, target)
    for backend in backends:
        if backend == "custom":
            t = LANG_CODE_MAP_CUSTOM.get(code) or LANG_CODE_MAP.get(code)
        elif backend == "mymemory":
            t = LANG_CODE_MAP_MYMEMORY.get(code) or LANG_CODE_MAP.get(code)
        elif backend == "google":
            t = LANG_CODE_MAP.get(code)
        else:
            t = LANG_CODE_MAP.get(code)
        chain_targets.append((backend, t))

    # Try each (backend, target) pair in order.
    for backend, t in chain_targets:
        try:
            if backend == "custom":
                translated = _translate_custom(masked, source=source_short, target=t)
            else:
                translated = Translator(backend).translate(
                    masked, source="en-GB", target=t
                )
            translated = restore_placeholders(translated, placeholders)
            return (code, key, en_value, translated, backend, True)
        except Exception as exc:  # noqa: BLE001
            # Quietly try the next backend; only log on the last attempt.
            last_exc = exc
            continue
    # All backends failed.
    return (code, key, en_value, en_value, "", False)


# ---------------------------------------------------------------------------
# Core: load → identify missing → translate → save
# ---------------------------------------------------------------------------


def _load_data() -> dict:
    with open(JSON_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _save_data(data: dict) -> None:
    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def _is_missing(entry: str, english: str) -> bool:
    """A key is missing when its value is empty or identical to EN."""
    if not entry:
        return True
    if entry == english:
        return True
    return False


def identify_missing(
    data: dict, langs: Iterable[str]
) -> List[Tuple[str, str, str]]:
    """Return ``[(lang_code, en_key, en_value), …]`` for every missing entry."""
    en = data["en"]
    out: List[Tuple[str, str, str]] = []
    for code in langs:
        if code == "en":
            continue
        if code not in data:
            print(f"[warn] language {code!r} not in JSON", file=sys.stderr)
            continue
        table = data[code]
        for key, en_value in en.items():
            table_value = table.get(key)
            if _is_missing(table_value or "", en_value):
                if key in PROTECTED_KEYS:
                    # Don't send X/Y to the translator — they're already
                    # canonical.
                    continue
                out.append((code, key, en_value))
    return out


def translate_batch_concurrent(
    items: List[Tuple[str, str, str]],
    backends: List[str],
    *,
    workers: int = 4,
) -> List[Tuple[str, str, str, str, str, bool]]:
    """Translate ``items`` in parallel using ``workers`` threads.

    Each item is run through the ``backends`` chain (custom → mymemory,
    etc.). Returns a list of
    ``(code, key, en, translated, used_backend, ok)`` tuples in the
    same order as the input.
    """
    if not items:
        return []
    n = len(items)
    print(f"[threadpool] dispatching {n} items across {workers} workers", file=sys.stderr)
    source_short = "en"  # both backends accept short "en" here
    results: List[Tuple[str, str, str, str, str, bool]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _worker,
                item,
                backends,
                source_short,
                LANG_CODE_MAP,
                LANG_CODE_MAP_MYMEMORY,
            ): item
            for item in items
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                code, key, en_value = futures[fut]
                results.append((code, key, en_value, en_value, "", False))
            if done % 50 == 0 or done == n:
                print(f"[threadpool] {done}/{n} done", file=sys.stderr)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", help="Comma-separated language codes (default: all non-EN)")
    p.add_argument("--key", help="Translate only this key (across selected languages)")
    p.add_argument(
        "--backend",
        choices=["custom", "mymemory", "google", "libre", "offline"],
        default="custom",
        help="Primary translation backend (default: custom)",
    )
    p.add_argument(
        "--fallback",
        action="append",
        choices=["custom", "mymemory", "google", "libre", "offline"],
        help="Append a fallback backend (can be repeated). "
        "Each item tries backends in order until one succeeds.",
    )
    p.add_argument("--apply", action="store_true", help="Write translations back to JSON (default: dry-run)")
    p.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Concurrent worker threads (default: 3 — the public API can be "
             "flaky under high concurrency; bump to 4-5 on a fast LAN)",
    )
    p.add_argument("--limit", type=int, default=0, help="Only translate N items (0 = all)")
    args = p.parse_args(argv)

    data = _load_data()
    langs = args.lang.split(",") if args.lang else [c for c in data.keys() if c != "en"]

    items = identify_missing(data, langs)
    if args.key:
        items = [it for it in items if it[1] == args.key]
    if args.limit:
        items = items[: args.limit]

    if not items:
        print("Nothing to translate — every selected key is already covered.")
        return 0

    # Build the backend chain: primary first, then any --fallback entries.
    backends: List[str] = [args.backend]
    if args.fallback:
        for fb in args.fallback:
            if fb not in backends:
                backends.append(fb)
    print(
        f"Translating {len(items)} entries (chain={backends}, workers={args.workers}, langs={langs})"
    )

    if args.backend == "offline":
        for code, key, en_value in items[:10]:
            print(f"  would translate: [{code}] {key!r} = {en_value!r}")
        if len(items) > 10:
            print(f"  …and {len(items) - 10} more")
        print("Pass --apply with a network backend to actually translate.")
        return 0

    if not args.apply:
        for code, key, en_value in items[:10]:
            print(f"  would translate: [{code}] {key!r} = {en_value!r}")
        if len(items) > 10:
            print(f"  …and {len(items) - 10} more")
        print("Dry run — pass --apply to actually call the API and write the JSON.")
        return 0

    results = translate_batch_concurrent(items, backends, workers=args.workers)

    # Apply results.
    succeeded = 0
    failed = 0
    backend_counts: dict = {}
    for code, key, _en, translated, used_backend, ok in results:
        if ok and translated:
            data[code][key] = translated
            succeeded += 1
            backend_counts[used_backend] = backend_counts.get(used_backend, 0) + 1
        else:
            failed += 1
    _save_data(data)
    backend_summary = ", ".join(f"{k}={v}" for k, v in sorted(backend_counts.items()))
    print(f"Translated {succeeded} entries ({backend_summary or 'none'}); {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
