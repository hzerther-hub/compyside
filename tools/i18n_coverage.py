"""i18n coverage report — list untranslated keys per language.

Usage:
    python tools/i18n_coverage.py
    python tools/i18n_coverage.py --lang zh_CN,ja,ko
    python tools/i18n_coverage.py --show-keys Blend Opacity

Exit code 0 if every language covers ≥ 95% of EN keys, else 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from compositor import i18n  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", help="Comma-separated language codes (default: all)")
    p.add_argument("--show-keys", nargs="*", help="Print full translation table for these keys")
    p.add_argument("--threshold", type=float, default=95.0, help="Pass % threshold")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args(argv)

    langs = args.lang.split(",") if args.lang else list(i18n.T.keys())
    if "en" not in langs:
        langs = ["en", *langs]

    if args.show_keys:
        for key in args.show_keys:
            row = [f"{key}"]
            for code in langs:
                table = i18n.T.get(code, {})
                val = table.get(key) or i18n.EN.get(key, "(missing)")
                row.append(val)
            print(" | ".join(row))
        return 0

    print(f"{'language':<10} {'translated':>11} {'total':>7} {'%':>6}")
    overall_fail = False
    for code in langs:
        table = i18n.T.get(code, {})
        en = i18n.EN
        translated = sum(1 for k in en if table.get(k) and table[k] != en[k])
        total = len(en)
        pct = 100 * translated / max(1, total)
        status = ""
        if pct < args.threshold and code != "en":
            overall_fail = True
            status = "  <-- below threshold"
        line = f"{code:<10} {translated:>11} {total:>7} {pct:>5.1f}%"
        print(line + status)
    return 1 if overall_fail else 0


if __name__ == "__main__":
    sys.exit(main())
