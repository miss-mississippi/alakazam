"""Документы и результаты: числа в тексте совпадают с results/*.json,
а сами results/*.json совпадают с тем, что сейчас считает код.

Если падает test_docs_match_results — текст разошёлся с результатами:
    python check_docs.py          показать расхождения
    python check_docs.py --fix    переписать числа из результатов
Если падает test_results_fresh — результаты устарели: перезапустить run_step*.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import check_docs
from reentry.results import RESULTS_DIR, lookup

EXPECTED = ["step1", "step1b", "step2", "step3", "step4", "step5", "step5b",
            "verify_step1", "verify_step2", "verify_step3", "verify_step4",
            "verify_step5"]


def test_results_exist():
    missing = [n for n in EXPECTED if not (RESULTS_DIR / f"{n}.json").exists()]
    assert not missing, f"нет results/{missing}: запустить соответствующие скрипты"


@pytest.mark.parametrize("doc", check_docs.default_docs(), ids=lambda p: p.name)
def test_docs_match_results(doc):
    assert check_docs.count_marks(doc) > 0, f"в {doc.name} нет ни одной метки"
    problems = check_docs.check_file(doc)
    assert not problems, "\n".join(problems)


def test_results_consistent_across_scripts():
    """Одна и та же величина, посчитанная разными скриптами, совпадает."""
    pairs = [("step4.fragments.film.total.vap_al", "step5.scenarios.film.total"),
             ("step4.fragments.bare.total.vap_al", "step5.scenarios.bare.total"),
             ("step5b.transfer.p0.median", "step5.scenarios.film.median"),
             ("step5b.transfer.p0.mass", "step5.scenarios.film.total"),
             ("step1.base.h_peak_km", "verify_step1.ae.num.h_km"),
             ("step5.sensitivity.base.total", "step5.scenarios.film.total")]
    for a, b in pairs:
        assert lookup(a) == pytest.approx(lookup(b), rel=1e-9), f"{a} != {b}"


def test_results_fresh():
    """Ключевые числа пересчитываются и сверяются с сохранёнными."""
    from reentry import EntryState, ExponentialAtmosphere, Vehicle, integrate
    import run_step5 as S

    tr = integrate(Vehicle(mass=175.0, area=1.0, Cd=1.0), EntryState(),
                   ExponentialAtmosphere(), h_stop=30e3, earth_rotation=False)
    assert tr.peak_decel()[1] / 1e3 == pytest.approx(lookup("step1.base.h_peak_km"),
                                                     rel=1e-9)

    frags = S.build_fragments()
    for key, mat in (("film", S.SCENARIOS["плёнка активна"]),
                     ("bare", S.SCENARIOS["голый расплав"])):
        s = S.summarize(S.run_model(frags, mat))
        assert s["total"] == pytest.approx(lookup(f"step5.scenarios.{key}.total"),
                                           rel=1e-6), f"step5.scenarios.{key} устарел"
        assert s["median"] == pytest.approx(lookup(f"step5.scenarios.{key}.median"),
                                            abs=1e-6)


def test_check_counts_match_last_runs():
    """Число проверок в исходниках = числу, прошедших в последнем прогоне."""
    counts = check_docs._count_checks()
    for n in range(1, 6):
        name = f"verify_step{n}"
        assert lookup(f"{name}.total") == counts[name], f"{name}: перезапустить"
        assert lookup(f"{name}.passed") == counts[name], f"{name}: есть провалы"
    assert np.isfinite(counts["total"])
