"""Check the numbers in the documents against results/*.json.

A tag goes right after a number and is invisible in rendered Markdown:

    8.5<!--=step5.scenarios.film.total:.1f-->

The key is a path in results/ (see reentry/results.py) and the format is as
in str.format (.1f, .0f, +.2f); without a format, .1f. The checks.* keys are
counted from the sources: checks.total is the number of test_* checks in all
verify_step*.py, checks.verify_stepN the number in one file.

    python check_docs.py              README.md and docs/REPORT.md
    python check_docs.py --fix        rewrite the numbers from the results
    python check_docs.py FILE ...     specific files

Exit code 1 if there are mismatches or unknown keys.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from reentry.results import lookup

ROOT = Path(__file__).resolve().parent
MARK = re.compile(r"(?P<num>[+\-−]?\d+(?:\.\d+)?)"
                  r"<!--=(?P<key>[\w.]+)(?::(?P<fmt>[^>]*?))?-->")


def default_docs() -> list[Path]:
    return [ROOT / "README.md", ROOT / "docs" / "REPORT.md"]


def _count_checks() -> dict:
    counts = {}
    for path in sorted(ROOT.glob("verify_step*.py")):
        tree = ast.parse(path.read_text())
        counts[path.stem] = sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                                for n in tree.body)
    counts["total"] = sum(counts.values())
    return counts


def resolve(key: str, cache: dict):
    if key.startswith("checks."):
        if "checks" not in cache:
            cache["checks"] = _count_checks()
        return cache["checks"][key.split(".", 1)[1]]
    return lookup(key, cache)


def _normalize(num: str) -> str:
    s = num.replace("−", "-")
    return s[1:] if s.startswith("+") and not s.startswith("+-") else s


def check_file(path: Path, fix: bool = False, cache: dict | None = None) -> list[str]:
    """Returns a list of problems; with fix=True rewrites the numbers."""
    cache = {} if cache is None else cache
    text = path.read_text()
    problems = []

    def repl(m: re.Match) -> str:
        key, fmt, num = m["key"], m["fmt"] or ".1f", m["num"]
        line = text.count("\n", 0, m.start()) + 1
        try:
            value = resolve(key, cache)
        except (KeyError, IndexError) as exc:
            problems.append(f"{path.name}:{line}: unknown key {key} ({exc})")
            return m.group(0)
        if value is None:
            problems.append(f"{path.name}:{line}: {key} = None")
            return m.group(0)
        want = format(value, fmt)
        want_cmp = _normalize(want)
        if want_cmp in ("-0", "-0.0", "-0.00"):
            want_cmp = want_cmp[1:]
        got_cmp = _normalize(num)
        if got_cmp in ("-0", "-0.0", "-0.00"):
            got_cmp = got_cmp[1:]
        if got_cmp == want_cmp:
            return m.group(0)
        problems.append(f"{path.name}:{line}: {key}: text has {num}, results have {want}")
        if "−" in num:
            want = want.replace("-", "−")
        return f"{want}<!--={key}{':' + m['fmt'] if m['fmt'] else ''}-->"

    new = MARK.sub(repl, text)
    if fix and new != text:
        path.write_text(new)
    return problems


def count_marks(path: Path) -> int:
    return len(MARK.findall(path.read_text()))


def main(argv: list[str]) -> int:
    fix = "--fix" in argv
    files = [Path(a) for a in argv if not a.startswith("--")] or default_docs()
    cache: dict = {}
    total = 0
    all_problems = []
    for path in files:
        n = count_marks(path)
        total += n
        problems = check_file(path, fix=fix, cache=cache)
        all_problems += problems
        status = "fixed" if fix and problems else ("OK" if not problems else "MISMATCHES")
        print(f"{path}: {n} tags, {len(problems)} mismatches: {status}")
    for p in all_problems:
        print("  " + p)
    if fix:
        unknown = [p for p in all_problems if "unknown key" in p or "= None" in p]
        return 1 if unknown else 0
    return 1 if all_problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
