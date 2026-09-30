"""Step 4 checks. Run: python verify_step4.py  (or pytest)

1. The mean of the angular distribution is exactly 0.25 (and it is the shape
   factor).
2. Closed-form vaporization rate versus numerical integration.
3. Convergence in the number of bands.
4. Energy balance of the per-band model: input = radiation + storage +
   evaporation.
5. Reproducing the ORSAT criterion.
6. Nesting and mass conservation PER BAND: evaporated <= molten <= mass.
7. Validity of the lumped model.
8. Clausius-Clapeyron versus the Al vapour-pressure table (CRC).
9. A plate radiates from two sides: equilibrium T versus the analytic value.
10. The flash regularization does not affect the result.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
import reentry.ablation as ablation
import run_step5 as S
from reentry.ablation import (RHO_ALUMINIUM, Aluminium, surface_thermal_model,
                              thermal_diffusion_depth)
from reentry.results import Recorder
from reentry.heating import (SHAPE_FACTOR_TUMBLING, SIGMA_SB, hot_wall_factor,
                             local_flux_fraction, vaporization_rate,
                             vaporizing_area_fraction)

ATM = MSISAtmosphere()
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
INTACT = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=0.5)
THIN = Vehicle.plate(1.0, 1.0e-3, Cd=1.5)
R = Recorder("verify_step4")


def test_angular_average():
    print("1. MEAN OF THE ANGULAR DISTRIBUTION")
    val, _ = quad(lambda th: local_flux_fraction(th) * np.sin(th) / 2, 0, np.pi)
    print(f"   <cos> over the full sphere = {val:.10f}   (analytic 1/4)")
    print(f"   SHAPE_FACTOR_TUMBLING = {SHAPE_FACTOR_TUMBLING}, "
          f"the standard demise-code range is 0.25-0.30")
    R["angular_mean"] = val
    ok = abs(val - 0.25) < 1e-10
    print(f"   -> {'OK' if ok else 'FAIL'}: the shape factor is the mean of THIS")
    print("      distribution, so we are not adding new physics, just no longer")
    print("      collapsing the physics that was already there.\n")
    assert ok, "the mean of cos is not 1/4"


def test_vaporization_closed_form():
    print("2. CLOSED-FORM VAPORIZATION RATE")
    print("   mdot = A*q*(1-C)^2/(4*L),  C = eps*sigma*T_boil^4/q\n")
    A, Tb, Lv = 4.0, 2740.0, 10.5e6
    ok = True
    print(f"   {'q, W/cm²':>11}{'eps':>7}{'C':>8}{'numerical':>14}{'closed':>14}"
          f"{'difference':>12}")
    for q, eps in ((8.5e5, 0.20), (1.5e6, 0.30), (3.0e6, 0.10)):
        C = eps * SIGMA_SB * Tb ** 4 / q
        if C >= 1:
            continue
        th_c = np.arccos(C)
        num, _ = quad(lambda th: (q * np.cos(th) - eps * SIGMA_SB * Tb ** 4)
                      * np.sin(th) / 2, 0, th_c)
        num = A * num / Lv
        clo = float(vaporization_rate(q, A, eps, Tb, Lv))
        d = abs(num / clo - 1)
        ok &= d < 1e-9
        print(f"   {q/1e4:>11.0f}{eps:>7.2f}{C:>8.3f}{num:>14.6e}"
              f"{clo:>14.6e}{d:>12.1e}")
    print(f"\n   -> {'OK' if ok else 'FAIL'}: the square in (1-C) is not a typo:")
    print("      near the threshold both the area and the excess go to zero.\n")
    assert ok, "the closed form disagrees with the integral"


def test_band_convergence():
    print("3. CONVERGENCE IN THE NUMBER OF BANDS")
    tr = integrate(THIN, ENTRY, ATM)
    mat = Aluminium(emissivity=0.10)
    print(f"   {'bands':>8}{'melt, %':>13}{'evaporated, %':>16}")
    vals = []
    for n in (12, 24, 48, 96):
        r = surface_thermal_model(tr, THIN, mat, wall_thickness=1e-3, n_bands=n)
        vals.append(r["f_vap"])
        print(f"   {n:>8}{100*r['f_melt']:>12.2f}%{100*r['f_vap']:>15.2f}%")
    spread = 100 * (max(vals) - min(vals))
    R["bands.spread_pp"] = spread
    ok = spread < 0.5
    print(f"\n   -> {'OK' if ok else 'FAIL'}: spread of the evaporated fraction "
          f"{spread:.2f} p.p. with 24 bands (the working value).\n")
    assert ok, "no convergence in the number of bands"


def test_energy_balance():
    print("4. ENERGY BALANCE OF THE PER-BAND MODEL")
    print("   Input (incl. oxidation heat) = re-radiation + storage + evaporation")
    print("   + blocked by blowing.")
    print("   The strongest test: it catches any error in the right-hand side.")
    print("   The audit is accumulated INSIDE the same ODE system, not recomputed")
    print("   separately; otherwise the test would check a copy of the code.\n")
    ok = True
    for label, veh, tw, kw in (("whole object", INTACT, None, {}),
                               ("1 mm plate", THIN, 1e-3, {}),
                               ("1 mm plate, blowing 0.6", THIN, 1e-3,
                                dict(blowing_eta=0.6)),
                               ("1 mm plate, oxidation 1.0", THIN, 1e-3,
                                dict(oxidation_eta=1.0)),
                               ("1 mm plate, local boiling", THIN, 1e-3,
                                dict(boil_pressure="local"))):
        tr = integrate(veh, ENTRY, ATM)
        for e in (0.05, 0.20):
            r = surface_thermal_model(tr, veh, Aluminium(emissivity=e),
                                      wall_thickness=tw, **kw)
            print(f"   {label}, eps={e:.2f}:")
            print(f"     input        {r['E_in']:.4e} J"
                  + (f"  (of which oxidation {r['E_ox']:.4e})" if r["E_ox"] else ""))
            print(f"     radiated     {r['E_rad']:.4e}")
            print(f"     stored       {r['E_stored']:.4e}")
            print(f"     evaporation  {r['E_vap']:.4e}")
            print(f"     blocked      {r['E_block']:.4e}")
            print(f"     residual     {r['energy_residual']:+.2e}")
            ok &= abs(r["energy_residual"]) < 1e-6
            if kw.get("oxidation_eta"):
                ok &= r["E_ox"] > 0
            worst = max(abs(r["energy_residual"]), R.data.get("energy_residual_max", 0.0))
            R["energy_residual_max"] = worst
    print(f"\n   -> {'OK' if ok else 'FAIL'}\n")
    assert ok, "the energy balance does not close"


def test_orsat_criterion():
    print("5. REPRODUCING THE ORSAT CRITERION")
    mat = Aluminium()
    print(f"   ORSAT generic aluminum, heat of ablation   0.9345 MJ/kg")
    print(f"   ours: c_p(solid)*(T_melt-T0) + L_fusion    "
          f"{mat.h_melt_complete/1e6:.4f} MJ/kg")
    d = abs(mat.h_melt_complete / 0.9345e6 - 1)
    print(f"   difference                                 {100*d:.1f}%")
    print(f"   complete evaporation at 1 kPa              "
          f"{mat.h_vapour_complete/1e6:.2f} MJ/kg")
    print(f"   ratio                                      {mat.demise_ratio:.1f}x")
    R["orsat.ours"] = mat.h_melt_complete / 1e6
    R["orsat.diff_pct"] = 100 * d
    R["orsat.vapour"] = mat.h_vapour_complete / 1e6
    R["orsat.ratio"] = mat.demise_ratio
    ok = d < 0.10
    print(f"\n   -> {'OK' if ok else 'FAIL'}: the upper end of the bracket is the")
    print("      DRAMA/ORSAT demise criterion, up to the alloy properties.\n")
    assert ok, "the ORSAT criterion is not reproduced within 10%"


def test_bracket_nested():
    print("6. NESTING AND MASS CONSERVATION PER BAND")
    print("   In every band: evaporated <= molten <= band mass.")
    print("   Checked for each band separately, not for the sum.\n")
    ok = True
    tol = 1e-6
    print(f"   {'object':<24}{'eps':>6}{'max evap/mass':>16}"
          f"{'min (melt-evap)/mass':>22}{'result':>8}")
    frags = S.build_fragments()
    cases = [("1 mm plate (entry)", THIN, integrate(THIN, ENTRY, ATM), 1e-3)]
    cases += [(n, v, tr, tw) for n, v, tr, tw in frags]
    for label, veh, tr, tw in cases:
        for e in (0.05, 0.35):
            r = surface_thermal_model(tr, veh, Aluminium(emissivity=e),
                                      wall_thickness=tw)
            mb = r["m_band"]
            a = float(np.max(r["m_vap_band"] / mb))
            b = float(np.min((r["m_melt_band"] - r["m_vap_band"]) / mb))
            good = (a <= 1 + tol) and (b >= -tol)
            ok &= good
            print(f"   {label:<24}{e:>6.2f}{a:>16.4f}{b:>22.4f}"
                  f"{'OK' if good else 'FAIL':>8}")
    print(f"\n   -> {'OK' if ok else 'FAIL'}\n")
    assert ok, "violated: evaporated <= molten <= band mass"


def test_lumped_validity():
    print("7. VALIDITY OF THE LUMPED MODEL")
    d = thermal_diffusion_depth(303.0)
    print(f"   heating depth over the flight sqrt(alpha*t) = {d*100:.1f} cm")
    print("   (alpha = k/(rho*c_p) = 167/(2700*900) = 6.9e-5 m²/s)\n")
    print(f"   {'object':<20}{'wall':>10}{'active':>12}{'conclusion':>22}")
    for label, veh, tw in (("whole object", INTACT, None),
                           ("1 mm plate", THIN, 1e-3)):
        tr = integrate(veh, ENTRY, ATM)
        r = surface_thermal_model(tr, veh, Aluminium(), wall_thickness=tw)
        note = "heated through" if r["lumped_valid"] else "surface layer only"
        print(f"   {label:<20}{r['wall_thickness']*1e3:>8.1f} mm"
              f"{r['participating']*1e3:>10.1f} mm{note:>22}")
        assert r["lumped_valid"], f"{label}: the wall is thicker than the heating depth"
    R["diffusion_depth_cm"] = d * 100
    print("\n   A wall thinner than 14 cm takes part in full; a thicker one only")
    print("   through its near-surface layer.\n")


def test_clausius_clapeyron():
    print("8. BOILING TEMPERATURE VS PRESSURE: CLAUSIUS-CLAPEYRON VERSUS CRC")
    print("   Metal vapour-pressure table (CRC Handbook) for Al:\n")
    crc = {1.0: 1482.0, 10.0: 1632.0, 100.0: 1817.0, 1e3: 2054.0,
           1e4: 2364.0, 1e5: 2790.0}
    mat = Aluminium()
    ok = True
    print(f"   {'p, Pa':>9}{'CRC, K':>9}{'model, K':>11}{'difference':>12}")
    for p, T in crc.items():
        Tm = float(mat.T_boil_at(p))
        d = Tm / T - 1
        ok &= abs(d) < 0.02
        R[f"crc.p{p:g}.model"] = Tm
        R["crc.max_diff_pct"] = max(abs(100 * d), R.data.get("crc", {}).get("max_diff_pct", 0.0))
        print(f"   {p:>9.0f}{T:>9.0f}{Tm:>11.0f}{100*d:>+11.1f}%")
    print(f"\n   -> {'OK' if ok else 'FAIL'}: at the 0.3-1 kPa stagnation pressure")
    print("      Al boils at ~1900-2050 K, not at 2792 K.\n")
    assert ok, "difference from the CRC table exceeds 2%"


class _ConstantFluxTrajectory:
    """Synthetic trajectory with constant q, V, rho. For analytic checks."""

    def __init__(self, q, V=7000.0, rho=1e-6, t_end=600.0):
        self.t = np.linspace(0.0, t_end, 601)
        self.V = np.full_like(self.t, V)
        self.V_rel = self.V
        self.rho = np.full_like(self.t, rho)
        self.h = np.full_like(self.t, 80e3)
        self._q = q

    def heat_flux(self, vehicle, correlation="sutton-graves"):
        return np.full_like(self.t, self._q)


def test_two_sided_plate():
    print("9. A PLATE RADIATES FROM TWO SIDES")
    print("   Constant flux, below melting. The nose band must reach")
    print("   sides*eps*sigma*T^4 = q*mu1*(1 - h_w/h_0), with sides = 2 for a plate.\n")
    from reentry.heating import SIGMA_SB, hot_wall_factor
    from scipy.optimize import brentq
    q, eps, n = 7.0e3, 0.30, 24          # both T_eq below melting (890 K)
    mu1 = 1.0 - 0.5 / n
    tr = _ConstantFluxTrajectory(q)
    mat = Aluminium(emissivity=eps)
    ok = True
    print(f"   {'object':<16}{'sides':>6}{'T model, K':>13}{'T analytic, K':>16}{'difference':>12}")
    Ts = {}
    # The number of radiating sides is set by GEOMETRY, not by a model flag;
    # otherwise the test would compare the model with itself.
    for label, veh, tw, sides in (("1 mm plate", THIN, 1e-3, 2),
                                  ("shell", Vehicle.compact(1.0, 5.0, Cd=1.5), None, 1)):
        r = surface_thermal_model(tr, veh, mat, wall_thickness=tw, n_bands=n)
        T_an = brentq(lambda T: sides * eps * SIGMA_SB * T ** 4
                      - q * mu1 * float(hot_wall_factor(7000.0, T)), 100, 3000)
        d = r["T_nose_end"] / T_an - 1
        ok &= abs(d) < 1e-3
        Ts[sides] = r["T_nose_end"]
        R[f"two_sided.sides{sides}.diff_pct"] = 100 * d
        print(f"   {label:<16}{sides:>6}{r['T_nose_end']:>13.1f}{T_an:>16.1f}{100*d:>+11.3f}%")
    print(f"\n   ratio T(plate)/T(shell) = {Ts[2]/Ts[1]:.4f}"
          f"  (2^-1/4 = {2**-0.25:.4f} for h_w << h_0)")
    print(f"   -> {'OK' if ok else 'FAIL'}\n")
    assert ok, "the equilibrium T does not match the analytic value"


def test_flash_regularization():
    print("10. THE FLASH REGULARIZATION DOES NOT AFFECT THE RESULT")
    print("   When the pressure drops, the boiling plateau goes down, and the")
    print("   superheated melt flashes with time constant TAU_FLASH. We change it")
    print("   fourfold.\n")
    frags = S.build_fragments()
    name, veh, tr, tw = frags[2]
    mat = Aluminium(surface="oxide")
    vals = {}
    saved = ablation.TAU_FLASH
    try:
        for tau in (0.2, 0.05):
            ablation.TAU_FLASH = tau
            vals[tau] = surface_thermal_model(tr, veh, mat, wall_thickness=tw)["m_vap"]
            print(f"   TAU_FLASH = {tau:4.2f} s -> evaporated {vals[tau]:.4f} kg ({name})")
    finally:
        ablation.TAU_FLASH = saved
    d = abs(vals[0.05] / vals[0.2] - 1)
    R["flash.diff_pct"] = 100 * d
    ok = d < 5e-3
    print(f"\n   difference {100*d:.3f}%  -> {'OK' if ok else 'FAIL'}\n")
    assert ok, "the result depends on TAU_FLASH"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
