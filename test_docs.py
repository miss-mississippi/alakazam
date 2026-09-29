"""Documents and results: the numbers in the text match results/*.json, and
results/*.json itself matches what the code computes now.

If test_docs_match_results fails, the text has drifted from the results:
    python check_docs.py          show the mismatches
    python check_docs.py --fix    rewrite the numbers from the results
If test_results_fresh fails, the results are stale: rerun run_step*.py.
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
    assert not missing, f"missing results/{missing}: run the corresponding scripts"


@pytest.mark.parametrize("doc", check_docs.default_docs(), ids=lambda p: p.name)
def test_docs_match_results(doc):
    assert check_docs.count_marks(doc) > 0, f"{doc.name} has no tags"
    problems = check_docs.check_file(doc)
    assert not problems, "\n".join(problems)


def test_results_consistent_across_scripts():
    """The same quantity computed by different scripts agrees."""
    pairs = [("step4.fragments.film.total.vap_al", "step5.scenarios.film.total"),
             ("step4.fragments.bare.total.vap_al", "step5.scenarios.bare.total"),
             ("step5b.transfer.p0.median", "step5.scenarios.film.median"),
             ("step5b.transfer.p0.mass", "step5.scenarios.film.total"),
             ("step1.base.h_peak_km", "verify_step1.ae.num.h_km"),
             ("step5.sensitivity.base.total", "step5.scenarios.film.total")]
    for a, b in pairs:
        assert lookup(a) == pytest.approx(lookup(b), rel=1e-9), f"{a} != {b}"


def test_results_fresh():
    """Key numbers are recomputed and compared with the saved ones.

    Tolerance 0.1%: the test catches stale results (any model change moves
    them more than that), not round-off differences between platforms; the
    adaptive LSODA picks slightly different steps on Linux and macOS.
    """
    from reentry import EntryState, ExponentialAtmosphere, Vehicle, integrate
    import run_step5 as S

    tr = integrate(Vehicle(mass=175.0, area=1.0, Cd=1.0), EntryState(),
                   ExponentialAtmosphere(), h_stop=30e3, earth_rotation=False)
    assert tr.peak_decel()[1] / 1e3 == pytest.approx(lookup("step1.base.h_peak_km"),
                                                     rel=1e-4)

    frags = S.build_fragments()
    for key, mat in (("film", S.SCENARIOS["active oxide film"]),
                     ("bare", S.SCENARIOS["bare melt"])):
        s = S.summarize(S.run_model(frags, mat))
        assert s["total"] == pytest.approx(lookup(f"step5.scenarios.{key}.total"),
                                           rel=1e-3), f"step5.scenarios.{key} is stale"
        assert s["median"] == pytest.approx(lookup(f"step5.scenarios.{key}.median"),
                                            abs=0.05)


def test_check_counts_match_last_runs():
    """The number of checks in the sources = the number passed in the last run."""
    counts = check_docs._count_checks()
    for n in range(1, 6):
        name = f"verify_step{n}"
        assert lookup(f"{name}.total") == counts[name], f"{name}: rerun it"
        assert lookup(f"{name}.passed") == counts[name], f"{name}: has failures"
    assert np.isfinite(counts["total"])
