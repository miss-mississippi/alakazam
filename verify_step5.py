"""Step 5 checks. Run: python verify_step5.py  (or pytest)

1. Planck -> Stefan-Boltzmann (an exact identity).
2. A constant eps is returned exactly.
3. Wien displacement.
4. Kirchhoff: reflectance -> eps -> back.
5. Mass conservation in the histogram.
6. The median does not depend on the bin width (regression).
7. The required spectral band.
8. Bare-metal eps from resistivity versus handbook data for polished Al.
"""

from __future__ import annotations

import numpy as np

from reentry.ablation import Aluminium, surface_thermal_model
from reentry.emissivity import (SIGMA_SB, WIEN_B, aluminium_resistivity,
                                bare_aluminium_emissivity, parker_abbott_emissivity,
                                planck_spectral_radiance, required_band,
                                total_emissivity, total_emissivity_from_reflectance)
import run_step5 as S
from reentry.results import Recorder

R = Recorder("verify_step5")


def test_stefan_boltzmann():
    print("1. PLANCK -> STEFAN-BOLTZMANN")
    print("   pi * int B(lam,T) dlam == sigma T^4, an identity with no freedom\n")
    lam = np.geomspace(1e-8, 1e-2, 200000)
    ok = True
    print(f"   {'T, K':>7}{'pi*int B':>15}{'sigma T^4':>15}{'rel. difference':>17}")
    for T in (1000.0, 1500.0, 2000.0, 2740.0):
        I = float(np.pi * np.trapezoid(planck_spectral_radiance(lam, T), lam))
        ref = SIGMA_SB * T ** 4
        d = I / ref - 1
        ok &= abs(d) < 1e-6
        R["stefan_boltzmann_max_diff"] = max(abs(d), R.data.get("stefan_boltzmann_max_diff", 0.0))
        print(f"   {T:>7.0f}{I:>15.6e}{ref:>15.6e}{d:>+17.2e}")
    print(f"\n   -> {'OK' if ok else 'FAIL'}: the integrator and constants agree.\n")
    assert ok, "pi*int(B) != sigma*T^4"


def test_flat_emissivity():
    print("2. A CONSTANT eps IS RETURNED EXACTLY")
    lam = np.geomspace(1e-7, 1e-3, 50000)
    ok = True
    for T in (1500.0, 2740.0):
        for c in (0.05, 0.35, 0.9):
            got = total_emissivity(lam, np.full_like(lam, c), T)
            ok &= abs(got - c) < 1e-9
            print(f"   T={T:.0f} eps={c:.2f} -> {got:.10f}")
    print(f"\n   -> {'OK' if ok else 'FAIL'}: the weighting is normalized.\n")
    assert ok, "a constant eps is not returned"


def test_wien():
    print("3. WIEN DISPLACEMENT")
    lam = np.geomspace(1e-7, 1e-4, 200000)
    ok = True
    print(f"   {'T, K':>7}{'numerical peak':>16}{'b/T':>12}{'difference':>12}")
    for T in (1500.0, 2000.0, 2740.0):
        pk = float(lam[int(np.argmax(planck_spectral_radiance(lam, T)))])
        ref = WIEN_B / T
        ok &= abs(pk / ref - 1) < 1e-3
        print(f"   {T:>7.0f}{pk*1e6:>13.4f} um{ref*1e6:>10.4f}{pk/ref-1:>+12.1e}")
    print(f"\n   -> {'OK' if ok else 'FAIL'}\n")
    assert ok, "the Planck peak does not match Wien's law"


def test_kirchhoff_roundtrip():
    print("4. KIRCHHOFF: REFLECTANCE -> eps -> BACK")
    print("   Exactly the path the instrument data will take.\n")
    lam = np.geomspace(0.3e-6, 20e-6, 4000)
    rng = np.random.default_rng(0)
    refl = np.clip(0.6 + 0.25 * np.sin(np.log(lam * 1e6))
                   + 0.02 * rng.normal(size=lam.size), 0.0, 1.0)
    e1 = total_emissivity_from_reflectance(lam, refl, 2000.0)
    e2 = total_emissivity(lam, 1.0 - refl, 2000.0)
    print(f"   from R:    {e1:.10f}")
    print(f"   from 1-R:  {e2:.10f}")
    ok = abs(e1 - e2) < 1e-12
    print(f"\n   -> {'OK' if ok else 'FAIL'}\n")
    assert ok, "the R -> eps path does not match the direct one"


def test_histogram_conservation():
    print("5. MASS CONSERVATION IN THE HISTOGRAM")
    print("   The sum over layers must equal the total evaporated Al mass")
    print("   computed directly over fragments with their Al fractions.\n")
    frags = S.build_fragments()
    ok = True
    print(f"   {'case':<32}{'histogram sum, kg':>18}{'direct sum, kg':>16}{'difference':>12}")
    cases = [("eps=0.05", Aluminium(emissivity=0.05), S.BASE_SPLIT),
             ("active oxide film", S.BASE, S.BASE_SPLIT),
             ("bare, Al in thin-walled parts", S.SCENARIOS["bare melt"],
              S.AL_SPLITS["all Al in thin-walled parts"])]
    for label, mat, split in cases:
        run = S.run_model(frags, mat, al_split=split)
        hist = S.histogram(run)
        direct = sum(surface_thermal_model(tr, veh, mat, wall_thickness=tw)["m_vap"] * f
                     for (_, veh, tr, tw), f in zip(frags, split))
        d = abs(hist.sum() / direct - 1) if direct > 0 else 0.0
        ok &= d < 1e-6
        print(f"   {label:<32}{hist.sum():>18.6f}{direct:>16.6f}{d:>12.1e}")
    print(f"\n   -> {'OK' if ok else 'FAIL'}: nothing is lost at the bin edges.\n")
    assert ok, "the histogram loses mass"


def test_bin_independence():
    print("6. INDEPENDENCE OF THE BIN WIDTH (regression)")
    print("   The median is computed from the raw series, so it does not depend")
    print("   on the bin by construction; the test catches a return to the")
    print("   histogram median, which moved by 1.5 km.\n")
    frags = S.build_fragments()
    run = S.run_model(frags, S.BASE)
    saved_bins, saved_c = S.BINS, S.CENTERS
    meds, tots = [], []
    print(f"   {'bin, km':>9}{'total, kg':>13}{'median, km':>14}")
    try:
        for w in (0.5, 1.0, 2.0, 5.0):
            S.BINS = np.arange(30.0, 102.0, w)
            S.CENTERS = 0.5 * (S.BINS[:-1] + S.BINS[1:])
            h = S.histogram(run)
            s = S.summarize(run)
            meds.append(s["median"]); tots.append(h.sum())
            print(f"   {w:>9.1f}{h.sum():>13.3f}{s['median']:>14.2f}")
    finally:
        S.BINS, S.CENTERS = saved_bins, saved_c
    ok = (max(meds) - min(meds) < 1e-9) and (max(tots) / min(tots) - 1 < 1e-6)
    print(f"\n   -> {'OK' if ok else 'FAIL'}\n")
    assert ok, "the median or the mass depends on the bin width"


def test_required_band():
    print("7. THE REQUIRED SPECTRAL BAND")
    lo_hot, _, _ = required_band(2200.0)
    _, hi_cold, _ = required_band(1500.0)
    R["band.lo_um"] = lo_hot * 1e6
    R["band.hi_um"] = hi_cold * 1e6
    print(f"   union over the 1500-2200 K working range: {lo_hot*1e6:.2f} - {hi_cold*1e6:.2f} um")
    print(f"   UV-Vis-NIR + mid-IR FTIR (2.5-15 um) cover it with margin")
    ok = lo_hot > 0.25e-6 and hi_cold < 15e-6
    print(f"\n   -> {'OK' if ok else 'FAIL'}: a standard pair of instruments is enough;")
    print("      a high-temperature emissometer is NOT needed.\n")
    assert ok, "the band is not covered by the standard pair of instruments"


def test_bare_metal_emissivity():
    print("8. BARE-METAL eps FROM RESISTIVITY (Parker-Abbott)")
    print("   The 'bare melt' scenario rests on an estimate, not a measurement.")
    print("   We check it where measurements exist: polished solid Al, handbook")
    print("   0.04-0.07 at 600-900 K. At 300-400 K the model is below the")
    print("   handbook: a perfectly clean surface without native oxide.\n")
    ok = True
    print(f"   {'T, K':>7}{'rho, uOhm*cm':>14}{'eps model':>12}{'handbook':>12}")
    for T in (300.0, 400.0, 600.0, 800.0, 900.0):
        e = float(bare_aluminium_emissivity(T))
        ref = "0.04-0.07" if T >= 600 else "~0.04"
        R[f"bare_eps.T{T:.0f}"] = e
        if T >= 600:
            ok &= 0.04 <= e <= 0.07
        print(f"   {T:>7.0f}{float(aluminium_resistivity(T))*1e8:>14.2f}{e:>12.3f}{ref:>12}")
    for T in (1000.0, 1500.0, 2000.0, 2740.0):
        R[f"bare_eps.T{T:.0f}"] = float(bare_aluminium_emissivity(T))
        R[f"resistivity.T{T:.0f}"] = float(aluminium_resistivity(T)) * 1e8
        print(f"   {T:>7.0f}{float(aluminium_resistivity(T))*1e8:>14.2f}"
              f"{float(bare_aluminium_emissivity(T)):>12.3f}{'liquid':>12}")
    x = 5e-7 * 100 * 2000          # rho = 5e-7 Ohm*m = 50 uOhm*cm, T = 2000 K
    same = abs(float(parker_abbott_emissivity(5e-7, 2000.0))
               - (0.766 * x ** 0.5 - (0.309 - 0.0889 * np.log(x)) * x
                  - 0.0175 * x ** 1.5)) < 1e-12
    ok &= same
    print(f"\n   -> {'OK' if ok else 'FAIL'}: in the solid phase it matches the handbook;")
    print("      the liquid has 2.4-5 times the resistivity of the solid at 900 K,")
    print("      and eps ~0.10-0.22,")
    print("      not the 0.05 of polished solid metal.\n")
    assert ok, "the metal eps estimate does not match the handbook"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
