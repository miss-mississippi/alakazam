"""Сверка чисел в документах с results/*.json.

Метка ставится сразу после числа и в отрендеренном Markdown не видна:

    8.5<!--=step5.scenarios.film.total:.1f-->

Ключ — путь в results/ (см. reentry/results.py), формат — как в str.format
(.1f, .0f, +.2f); без формата — .1f. Ключи checks.* считаются по исходникам:
checks.total — число проверок test_* во всех verify_step*.py,
checks.verify_stepN — в одном файле.

    python check_docs.py              README.md и ../REPORT_full.md (если есть)
    python check_docs.py --fix        переписать числа из результатов
    python check_docs.py FILE ...     конкретные файлы

Код выхода 1, если есть расхождения или неизвестные ключи.
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
    docs = [ROOT / "README.md"]
    report = ROOT.parent / "REPORT_full.md"
    if report.exists():
        docs.append(report)
    return docs


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
    """Возвращает список проблем; при fix=True переписывает числа."""
    cache = {} if cache is None else cache
    text = path.read_text()
    problems = []

    def repl(m: re.Match) -> str:
        key, fmt, num = m["key"], m["fmt"] or ".1f", m["num"]
        line = text.count("\n", 0, m.start()) + 1
        try:
            value = resolve(key, cache)
        except (KeyError, IndexError) as exc:
            problems.append(f"{path.name}:{line}: неизвестный ключ {key} ({exc})")
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
        problems.append(f"{path.name}:{line}: {key}: в тексте {num}, в результатах {want}")
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
        status = "исправлено" if fix and problems else ("OK" if not problems else "РАСХОЖДЕНИЯ")
        print(f"{path}: меток {n}, расхождений {len(problems)} — {status}")
    for p in all_problems:
        print("  " + p)
    if fix:
        unknown = [p for p in all_problems if "неизвестный ключ" in p or "= None" in p]
        return 1 if unknown else 0
    return 1 if all_problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
