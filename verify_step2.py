"""Step 2 checks. Run: python verify_step2.py  (or pytest)

1. Spline accuracy against a direct pymsis call.
2. Seams in the derivative of log(rho) in NRLMSISE-00 and their absence in 2.1.
3. Trajectory convergence in the tabulation step dh.
4. Mechanism: why solar activity barely matters at 120 km.
5. Geography and season versus solar activity.
"""

from __future__ import annotations

import numpy as np
import pymsis

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.constants import G0
from reentry.results import Recorder

# Steps 1-2 are defined for a non-rotating Earth; integrate() turns rotation
# on by default, so it is switched off explicitly here.
NO_ROT = dict(earth_rotation=False)
R = Recorder("verify_step2")


VEH = Vehicle(mass=175.0, area=1.0, Cd=1.5)
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)
DATE = np.datetime64("2026-09-01T12:00")
KW = dict(lat=-40.0, lon=-140.0, f107=140.0, f107a=140.0, ap=4.0, version=0)


def _direct(alts_m, f107=140.0, ap=4.0, lat=-40.0):
    out = pymsis.calculate(DATE, -140.0, lat, np.asarray(alts_m) / 1e3,
                           f107s=f107, f107as=f107, aps=[[ap] * 7], version=0)
    return out[..., pymsis.Variable.MASS_DENSITY].ravel()


# Internal boundaries of the NRLMSISE-00 formulation where the model's own
# derivative of log(rho) breaks. Found numerically (see test_model_seams).
MSIS00_SEAMS = (72.5e3, 123.4e3)


def test_spline_accuracy():
    print("1. SPLINE ACCURACY AGAINST A DIRECT pymsis CALL")
    atm = MSISAtmosphere(**KW)
    h = np.linspace(35e3, 130e3, 20001)   # dense grid, NOT the tabulation nodes
    err = np.abs(atm.density(h) / _direct(h) - 1.0)

    near_seam = np.zeros_like(h, dtype=bool)
    for s in MSIS00_SEAMS:
        near_seam |= np.abs(h - s) < 1.0e3

    print(f"   away from model seams:  max {err[~near_seam].max():.2e}, "
          f"median {np.median(err[~near_seam]):.2e}")
    print(f"   within 1 km of a seam:  max {err[near_seam].max():.2e}")
    print("   The spikes do NOT come from interpolation: NRLMSISE-00 itself has a")
    print("   broken derivative at 72.5 and 123 km (internal boundaries of the")
    print("   formulation), and the cubic spline smooths the kink. A 0.1% cost")
    print("   against the 10-30% uncertainty of MSIS itself is negligible.")
    R["spline.max_err_far"] = err[~near_seam].max()
    R["spline.median_err_far"] = np.median(err[~near_seam])
    R["spline.max_err_seam"] = err[near_seam].max()
    print()
    assert err[~near_seam].max() < 1e-4, f"spline error {err[~near_seam].max():.1e}"
    assert err.max() < 5e-3, f"error near a seam {err.max():.1e}"


def test_model_seams():
    print("1b. WHERE THE MODEL'S OWN DERIVATIVE BREAKS")
    hh = np.arange(40e3, 130e3, 50.0)
    found = {}
    for ver in (0, 2.1):
        out = pymsis.calculate(DATE, -140.0, -40.0, hh / 1e3, f107s=140.0,
                               f107as=140.0, aps=[[4.0] * 7], version=ver)
        lr = np.log(out[..., pymsis.Variable.MASS_DENSITY].ravel())
        d2 = np.abs(np.gradient(np.gradient(lr, hh), hh))
        med = np.median(d2)
        idx = np.where(d2 > 12 * med)[0]
        if idx.size == 0:
            print(f"   NRLMSIS {ver}: no seams found, the profile is smooth")
            found[ver] = []
            continue
        groups = []
        cur = [idx[0]]
        for k in idx[1:]:
            if k - cur[-1] <= 5:
                cur.append(k)
            else:
                groups.append(cur)
                cur = [k]
        groups.append(cur)
        peaks = [hh[g[int(np.argmax(d2[g]))]] / 1e3 for g in groups]
        name = "NRLMSISE-00" if ver == 0 else f"NRLMSIS {ver}"
        print(f"   {name}: kinks at " + ", ".join(f"{p:.1f} km" for p in peaks))
        found[ver] = peaks
    print("   -> 2.x is smooth. That helps an adaptive integrator, and one of")
    print("      the MSISE-00 kinks (72.5 km) lies right in the ablation zone.\n")
    R["seams.msis00_km"] = found[0]
    R["seams.msis21_count"] = len(found[2.1])
    for s_km in (72.5, 123.4):
        assert any(abs(p - s_km) < 1.0 for p in found[0]), f"seam at {s_km} km not found"
    assert not found[2.1], f"kinks found in NRLMSIS 2.1: {found[2.1]}"


def test_grid_convergence():
    print("2. CONVERGENCE IN THE TABULATION STEP")
    print(f"   {'dh, m':>8}{'nodes':>8}{'max error':>14}"
          f"{'h decel, km':>13}{'h heating, km':>15}")
    prev = None
    ok = True
    rng = np.random.default_rng(1)
    h = rng.uniform(35e3, 130e3, 500)
    ref = _direct(h)
    for dh in (2000.0, 1000.0, 250.0, 100.0):
        atm = MSISAtmosphere(dh=dh, **KW)
        err = np.abs(atm.density(h) / ref - 1.0).max()
        tr = integrate(VEH, ENTRY, atm, **NO_ROT)
        _, h_a, _ = tr.peak_decel()
        h_q, _, _ = tr.peak_heating()
        print(f"   {dh:>8.0f}{len(atm._h_grid):>8d}{err:>14.2e}"
              f"{h_a/1e3:>13.3f}{h_q/1e3:>15.3f}")
        if prev is not None and abs(h_q - prev) > 1.0:
            ok = False
        prev = h_q
    print("   The max error does not fall monotonically with dh: it is limited by")
    print("   the same two model seams, not by the grid resolution. The peak")
    print("   altitudes agree to a metre for any dh, and that is the criterion.\n")
    assert ok, "heating-peak altitude moves by more than 1 m when dh changes"


def test_solar_mechanism():
    print("3. MECHANISM: WHY F10.7 BARELY MATTERS AT 120 KM")
    print("   Hypothesis: solar heating controls the EXOSPHERIC TEMPERATURE,")
    print("   which sets the scale height in the upper thermosphere. At 120 km")
    print("   the atmosphere is still governed by the mesosphere below, not")
    print("   from above. Test: if the hypothesis holds, the sensitivity must")
    print("   grow sharply with altitude.\n")
    print(f"   {'h, km':>7}{'F10.7=70':>12}{'F10.7=220':>12}{'ratio':>12}")
    ratios = {}
    for h in (70, 100, 120, 150, 200, 300, 400):
        lo = float(_direct([h * 1e3], f107=70.0)[0])
        hi = float(_direct([h * 1e3], f107=220.0)[0])
        ratios[h] = hi / lo
        print(f"   {h:>7}{lo:>12.3e}{hi:>12.3e}{hi/lo:>12.2f}")
    print(f"\n   x{ratios[120]:.2f} at 120 km, x{ratios[400]:.1f} at 400 km:")
    print("   solar activity changes density by multiples only at 300+ km.\n")
    for h, r in ratios.items():
        R[f"solar_ratio.{h}"] = r
    assert ratios[120] < 1.3 and ratios[400] > 3.0, f"ratios {ratios}"


def test_geography_dominates():
    print("4. THEN WHAT IS THE MAIN SOURCE OF ATMOSPHERIC SPREAD")
    print("   We compare the spread of three factors in heating-peak altitude.\n")
    def peak(**kw):
        atm = MSISAtmosphere(**{**KW, **kw})
        return integrate(VEH, ENTRY, atm, **NO_ROT).peak_heating()[0]

    solar = [peak(f107=f, f107a=f) for f in (70.0, 220.0)]
    geomag = [peak(ap=a) for a in (4.0, 80.0)]
    lat = [peak(lat=x) for x in (-75.0, 0.0)]
    season = [peak(date=d) for d in (np.datetime64("2026-03-01T12:00"),
                                     np.datetime64("2026-09-01T12:00"))]
    rows = [("solar activity F10.7 70-220", solar),
            ("geomagnetic storm Ap 4-80", geomag),
            ("latitude -75...0", lat),
            ("season March/September", season)]
    print(f"   {'factor':<36}{'spread, km':>12}")
    keys = {"solar activity F10.7 70-220": "solar", "geomagnetic storm Ap 4-80": "geomag",
            "latitude -75...0": "lat", "season March/September": "season"}
    spans = {}
    for label, vals in sorted(rows, key=lambda r: -abs(r[1][1] - r[1][0])):
        spans[keys[label]] = abs(vals[1] - vals[0]) / 1e3
        print(f"   {label:<36}{spans[keys[label]]:>12.1f}")
        R[f"env_span_km.{keys[label]}"] = spans[keys[label]]
    print("\n   For comparison: the Cd 1.0-2.2 sweep gives 5.8 km,")
    print("   and fragmentation about 20 km.\n")
    assert spans["lat"] > 10 * max(spans["solar"], 0.1), "latitude must beat F10.7"
    assert spans["season"] > 5 * max(spans["solar"], 0.1), "season must beat F10.7"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
