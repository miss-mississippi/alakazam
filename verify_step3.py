"""Step 3 checks. Run: python verify_step3.py  (or pytest)

1. Units of the Sutton-Graves constant, from the computed Stardust peak.
2. Energy sanity of the integrated heat load.
3. Earth rotation: limiting cases in inclination and the cost in heat flux.
4. Sutton-Graves versus Detra-Kemp-Riddell: the cost of the exponent.
5. The nose radius does not move the peak altitude.
6. Cauchy's formula: wetted area = 4 * mean projection.
7. Size exponents: the sign for total energy is opposite to that for q_stag.
8. Energy needed to melt and to evaporate versus the energy available.
9. Blowing: closed form versus iteration.
10. Hot wall.
"""

from __future__ import annotations

import numpy as np

from reentry.ablation import Aluminium
from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.results import Recorder
from reentry.heating import (EXP_DKR, K_SUTTON_GRAVES, SHAPE_FACTOR_TUMBLING,
                             blowing_factor, detra_kemp_riddell_shape,
                             hot_wall_factor, sutton_graves)

ATM = MSISAtmosphere()
VEH = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=0.5)
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
R = Recorder("verify_step3")


def test_sutton_graves_units():
    print("1. UNITS OF THE SUTTON-GRAVES CONSTANT")
    print("   Benchmark: Stardust, Rn = 0.23 m, V = 12.6 km/s, rho ~ 3e-4 kg/m^3.")
    print("   Computed peak heating ~1200 W/cm^2.\n")
    q = float(sutton_graves(3.0e-4, 12600.0, 0.23))
    print(f"   the formula with SI inputs gives  {q:.3e}")
    print(f"   if this is W/m^2               -> {q/1e4:8.0f} W/cm^2   <- matches")
    print(f"   if this is W/cm^2              -> {q:8.3e} W/cm^2   "
          f"off by a factor of {q/1200:.0e}")
    print(f"\n   k={K_SUTTON_GRAVES:.4e} with SI inputs gives W/m^2.")
    print("   NASA TFAWS labels it W/cm^2: a typo or a different")
    print("   normalization. The error would cost four orders of magnitude.\n")
    R["stardust_wcm2"] = q / 1e4
    assert 0.5 < (q / 1e4) / 1200.0 < 2.0, f"{q/1e4:.0f} W/cm^2 vs ~1200"


def test_energy_sanity():
    print("2. ENERGY SANITY")
    print("   Heat absorbed by the body cannot exceed its kinetic energy. Most of")
    print("   the energy goes into the shock layer and the wake; a few percent")
    print("   reach the body.\n")
    tr = integrate(VEH, ENTRY, ATM)
    Q = tr.heat_load(VEH)                     # J/m^2 at the stagnation point
    E_kin = 0.5 * VEH.mass * ENTRY.velocity ** 2
    # Upper bound: the stagnation-point flux over the whole frontal area
    E_heat_upper = Q * VEH.area
    frac = E_heat_upper / E_kin
    print(f"   kinetic energy at entry             {E_kin:.3e} J")
    print(f"   heat load at the stagnation point   {Q:.3e} J/m^2")
    print(f"   upper bound (load x full area)      {E_heat_upper:.3e} J")
    print(f"   fraction of kinetic energy          {100*frac:.1f}%")
    print("\n   This is an UPPER BOUND: the stagnation-point correlation is applied")
    print("   to the whole area. The actual fraction is several times smaller.\n")
    R["heat_fraction_pct"] = 100 * frac
    assert 0.001 < frac < 0.5, f"fraction {frac:.3f}"


def test_earth_rotation():
    print("3. EARTH ROTATION")
    print("   v_along = omega*R*cos(i): latitude cancels, only the inclination")
    print("   matters. Limiting cases:\n")
    print(f"   {'i, deg':>9}{'v_corot, m/s':>14}{'peak q, W/cm^2':>17}"
          f"{'peak h, km':>13}{'Q, MJ/m^2':>13}")
    ref = None
    for i in (0.0, 53.0, 90.0, 98.0, 180.0):
        e = EntryState(inclination_deg=i)
        tr = integrate(VEH, e, ATM)
        q = tr.heat_flux(VEH)
        k = int(np.argmax(q))
        if i == 90.0:
            ref = q[k]
        key = f"rotation.i{int(i)}"
        R[f"{key}.v_corot"] = e.corotation_speed
        R[f"{key}.q_peak_wcm2"] = q[k] / 1e4
        R[f"{key}.h_km"] = tr.h[k] / 1e3
        R[f"{key}.Q_MJ"] = tr.heat_load(VEH) / 1e6
        print(f"   {i:>9.0f}{e.corotation_speed:>14.1f}{q[k]/1e4:>17.1f}"
              f"{tr.h[k]/1e3:>13.1f}{tr.heat_load(VEH)/1e6:>13.1f}")

    # Cost of switching rotation on for the base case i=53
    on = integrate(VEH, ENTRY, ATM, earth_rotation=True)
    off = integrate(VEH, ENTRY, ATM, earth_rotation=False)
    q_on, q_off = on.heat_flux(VEH).max(), off.heat_flux(VEH).max()
    print(f"\n   base case i=53: switching rotation on changes the peak flux by "
          f"{100*(q_on/q_off - 1):+.1f}%")
    print(f"   and the heat load by {100*(on.heat_load(VEH)/off.heat_load(VEH)-1):+.1f}%")
    print(f"   the prediction was ~-11% (3.7% in V, q ~ V^3)")
    # i=90 must coincide with rotation switched off
    p90 = integrate(VEH, EntryState(inclination_deg=90.0), ATM)
    d = abs(p90.heat_flux(VEH).max() / q_off - 1.0)
    print("\n   A polar orbit (i=90) must coincide exactly with rotation switched")
    print("   off: a self-consistency test.\n")
    R["rotation.q_change_pct"] = 100 * (q_on / q_off - 1)
    R["rotation.Q_change_pct"] = 100 * (on.heat_load(VEH) / off.heat_load(VEH) - 1)
    assert d < 1e-6, f"i=90 differs from no rotation by {d:.1e}"


def test_correlations():
    print("4. SUTTON-GRAVES VERSUS DETRA-KEMP-RIDDELL")
    print(f"   The only difference is the velocity exponent: 3.0 versus {EXP_DKR}.")
    print("   DKR is normalized to S-G at a reference point, so we compare SHAPE.\n")
    tr = integrate(VEH, ENTRY, ATM)
    q_sg = tr.heat_flux(VEH, "sutton-graves")
    q_dkr = tr.heat_flux(VEH, "dkr")
    i_sg, i_dkr = int(np.argmax(q_sg)), int(np.argmax(q_dkr))
    Q_sg = tr.heat_load(VEH, "sutton-graves")
    Q_dkr = tr.heat_load(VEH, "dkr")
    print(f"   {'':<22}{'Sutton-Graves':>16}{'DKR':>12}{'difference':>12}")
    print(f"   {'peak altitude, km':<22}{tr.h[i_sg]/1e3:>16.2f}"
          f"{tr.h[i_dkr]/1e3:>12.2f}{(tr.h[i_dkr]-tr.h[i_sg])/1e3:>+11.2f} km")
    print(f"   {'peak flux, W/cm^2':<22}{q_sg[i_sg]/1e4:>16.1f}"
          f"{q_dkr[i_dkr]/1e4:>12.1f}{100*(q_dkr[i_dkr]/q_sg[i_sg]-1):>+11.1f}%")
    print(f"   {'heat load, MJ/m^2':<22}{Q_sg/1e6:>16.1f}"
          f"{Q_dkr/1e6:>12.1f}{100*(Q_dkr/Q_sg-1):>+11.1f}%")
    print("\n   The exponent moves the peak altitude by fractions of a kilometre:")
    print("   for the altitude distribution the choice of correlation is a")
    print("   fourth-order effect, weaker even than the MSIS version.\n")
    dh = (tr.h[i_dkr] - tr.h[i_sg]) / 1e3
    dQ = 100 * (Q_dkr / Q_sg - 1)
    R["dkr.h_sg_km"] = tr.h[i_sg] / 1e3
    R["dkr.h_dkr_km"] = tr.h[i_dkr] / 1e3
    R["dkr.dh_km"] = dh
    R["dkr.q_sg_wcm2"] = q_sg[i_sg] / 1e4
    R["dkr.q_dkr_wcm2"] = q_dkr[i_dkr] / 1e4
    R["dkr.dq_pct"] = 100 * (q_dkr[i_dkr] / q_sg[i_sg] - 1)
    R["dkr.Q_sg_MJ"] = Q_sg / 1e6
    R["dkr.Q_dkr_MJ"] = Q_dkr / 1e6
    R["dkr.dQ_pct"] = dQ
    assert abs(dh) < 1.0 and abs(dQ) < 10.0, f"dh={dh:.2f} km, dQ={dQ:.1f}%"


def test_nose_radius():
    print("5. NOSE RADIUS")
    print("   q ~ Rn^-0.5, but Rn does not affect the peak ALTITUDE at all:")
    print("   it is constant along the trajectory and factors out of argmax.\n")
    print(f"   {'Rn, m':>8}{'peak q, W/cm^2':>17}{'peak h, km':>13}")
    heights = []
    for rn in (0.1, 0.25, 0.5, 1.0, 2.0):
        v = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=rn)
        tr = integrate(v, ENTRY, ATM)
        q = tr.heat_flux(v)
        k = int(np.argmax(q))
        heights.append(tr.h[k])
        R[f"nose.rn{rn*100:.0f}cm.q_peak_wcm2"] = q[k] / 1e4
        print(f"   {rn:>8.2f}{q[k]/1e4:>17.1f}{tr.h[k]/1e3:>13.2f}")
    spread = max(heights) - min(heights)
    print(f"\n   The peak altitude is identical (spread {spread:.2e} m). Over our")
    print("   range Rn changes the ABSOLUTE flux fourfold, and hence ablation, but")
    print("   not the injection altitude: Rn belongs to the MASS budget, not the")
    print("   ALTITUDE budget.\n")
    R["nose.h_km"] = heights[0] / 1e3
    R["nose.h_spread_m"] = spread
    assert spread < 1.0, f"peak altitude moves by {spread:.2f} m"


def test_cauchy():
    print("6. WETTED AREA: CAUCHY'S FORMULA")
    print("   For any CONVEX body the projected area averaged over random")
    print("   orientations is 1/4 of the surface area. Our `area` for a tumbling")
    print("   body is exactly the mean projection, so A_wet = 4*A EXACTLY, for")
    print("   any shape, not just a sphere.\n")
    for A in (0.5, 1.0, 2.0):
        v = Vehicle(area=A)
        print(f"   area={A:.1f} m^2  ->  A_wet={v.wetted:.1f} m^2  "
              f"(ratio {v.wetted/A:.1f})")
    # Check on a sphere: A_proj = pi R^2, A_surf = 4 pi R^2
    r_sph = 0.65
    print(f"\n   sphere check, R={r_sph} m: projection {np.pi*r_sph**2:.4f}, "
          f"surface {4*np.pi*r_sph**2:.4f}, ratio {4.0:.1f}")
    print()
    assert abs(Vehicle(area=1.0).wetted - 4.0) < 1e-12


def test_rn_scaling():
    print("7. SIZE EXPONENTS: WHERE THE SIGN FLIPS")
    print("   Expected for a geometrically similar body:")
    print("     q_stag ~ L^-0.5,  P_total ~ L^+1.5,  P/m ~ L^-1.5\n")

    scales = np.array([0.1, 0.2, 0.5, 1.0, 2.0])

    # (a) FROZEN trajectory: the same flux history, only the geometry changes.
    # This checks the pure exponents of the correlation and the geometry.
    tr = integrate(VEH, ENTRY, ATM)
    q_hist, t_hist = tr.rho, tr.t
    V_hist = tr.V_rel
    E_frozen, S_frozen = [], []
    for L in scales:
        v = Vehicle.geometric_family(L)
        q = sutton_graves(q_hist, V_hist, v.nose_radius)
        P = SHAPE_FACTOR_TUMBLING * q * v.wetted
        E = np.trapezoid(P, t_hist)
        E_frozen.append(E); S_frozen.append(E / v.mass)

    def slope(y):
        return float(np.polyfit(np.log(scales), np.log(y), 1)[0])

    print(f"   (a) frozen trajectory, pure geometry:")
    print(f"       total energy exponent        {slope(E_frozen):+.3f}  "
          f"(expected +1.500)")
    print(f"       specific energy exponent     {slope(S_frozen):+.3f}  "
          f"(expected -1.500)")
    R["scaling.frozen_E"] = slope(E_frozen)
    R["scaling.frozen_S"] = slope(S_frozen)
    ok = abs(slope(E_frozen) - 1.5) < 1e-6 and abs(slope(S_frozen) + 1.5) < 1e-6

    # (b) SELF-CONSISTENT: a small body has a smaller beta, decelerates higher
    # and receives LESS total heat. The two effects compete.
    print(f"\n   (b) self-consistent (beta changes with size):")
    print(f"   {'L':>6}{'beta':>9}{'peak h, km':>13}{'E total, MJ':>15}"
          f"{'E/m, MJ/kg':>14}")
    E_sc, S_sc = [], []
    for L in scales:
        v = Vehicle.geometric_family(L)
        t2 = integrate(v, ENTRY, ATM)
        E = t2.absorbed_energy(v); S = t2.specific_energy(v)
        E_sc.append(E); S_sc.append(S)
        R[f"scaling.L{L*10:.0f}.beta"] = v.ballistic_coefficient
        R[f"scaling.L{L*10:.0f}.h_km"] = t2.peak_heating()[0] / 1e3
        R[f"scaling.L{L*10:.0f}.E_MJ"] = E / 1e6
        R[f"scaling.L{L*10:.0f}.Em_MJkg"] = S / 1e6
        print(f"   {L:>6.1f}{v.ballistic_coefficient:>9.1f}"
              f"{t2.peak_heating()[0]/1e3:>13.1f}{E/1e6:>15.1f}{S/1e6:>14.2f}")
    print(f"\n       total energy exponent        {slope(E_sc):+.3f}")
    print(f"       specific energy exponent     {slope(S_sc):+.3f}")
    print("\n   -> The sign in size is OPPOSITE to the sign in q_stag.")
    print("      Small fragments receive far more heat per kilogram: this is the")
    print("      SECOND mechanism by which fragmentation decides the outcome,")
    print("      independent of the altitude rise through beta.\n")
    R["scaling.sc_E"] = slope(E_sc)
    R["scaling.sc_S"] = slope(S_sc)
    assert ok, f"exponents {slope(E_frozen):+.4f}, {slope(S_frozen):+.4f}"
    assert slope(E_sc) > 0 > slope(S_sc), "the sign in size must persist"


def test_demise_energy():
    print("8. IS THERE ENOUGH ENERGY TO MELT AND TO EVAPORATE")
    mat = Aluminium()
    Tb = mat.T_boil_nominal
    print(f"   Al 6061 properties from ablation.Aluminium; boiling at the local")
    print(f"   pressure ~1 kPa: T_boil = {Tb:.0f} K (at 1 atm it would be "
          f"{mat.T_boil_1atm:.0f} K).")
    h_heat = mat.h1
    h_liq = float(mat.h3(Tb)) - mat.h2
    L_vap = float(mat.L_vapour_at(Tb))
    H_total = mat.h_vapour_complete
    print(f"     heating to melting   {h_heat/1e6:5.2f} MJ/kg  (c_p solid {mat.c_p:.0f})")
    print(f"     heat of fusion       {mat.L_fusion/1e6:5.2f}")
    print(f"     heating to boiling   {h_liq/1e6:5.2f}         (c_p liquid {mat.c_p_liquid:.0f})")
    print(f"     heat of vaporization {L_vap/1e6:5.2f}")
    print(f"     TOTAL                {H_total/1e6:5.2f} MJ/kg"
          f"   (to complete melting {mat.h_melt_complete/1e6:.2f})\n")
    print("   The ratio of E/m to these values is the mass fraction that could in")
    print("   principle be melted / evaporated if all absorbed energy went into it")
    print("   (no re-radiation: an UPPER bound).\n")
    R["energy.T_boil"] = Tb
    R["energy.heat_to_melt"] = h_heat / 1e6
    R["energy.fusion"] = mat.L_fusion / 1e6
    R["energy.heat_to_boil"] = h_liq / 1e6
    R["energy.vaporization"] = L_vap / 1e6
    R["energy.total"] = H_total / 1e6
    R["energy.melt_complete"] = mat.h_melt_complete / 1e6
    print(f"   {'L':>6}{'mass, kg':>12}{'E/m, MJ/kg':>14}"
          f"{'meltable':>14}{'vaporizable':>13}")
    Ss = []
    for L in (1.0, 0.5, 0.2, 0.1):
        v = Vehicle.geometric_family(L)
        t2 = integrate(v, ENTRY, ATM)
        S = t2.specific_energy(v, T_wall=Tb)
        Ss.append(S)
        R[f"energy.L{L*10:.0f}.mass"] = v.mass
        R[f"energy.L{L*10:.0f}.Em"] = S / 1e6
        R[f"energy.L{L*10:.0f}.meltable_pct"] = 100 * min(S / mat.h_melt_complete, 1)
        R[f"energy.L{L*10:.0f}.vaporizable_pct"] = 100 * S / H_total
        print(f"   {L:>6.1f}{v.mass:>12.2f}{S/1e6:>14.2f}"
              f"{100*min(S/mat.h_melt_complete, 1):>13.0f}%{100*S/H_total:>12.0f}%")
    print("\n   E/m here includes the hot-wall correction at T_boil, so it is lower")
    print("   than in check 7 (a cold wall there).")
    print("   -> Whole object: about half can be melted, only a few percent")
    print("      evaporated. The 95% demisability of OneWeb/SpaceX (Ferreira 2024)")
    print("      is a MELT criterion and must be compared with the first column.")
    print("      The mass gap is ~2x for melting, and fragmentation closes it:")
    print("      an L = 0.2 fragment melts completely.\n")
    parts = h_heat + mat.L_fusion + h_liq + L_vap
    assert abs(parts / H_total - 1) < 1e-12, "the stages do not add up to the total"
    assert all(a < b for a, b in zip(Ss, Ss[1:])), "E/m must grow with fragmentation"


def test_blowing_closed_form():
    print("9. BLOWING: CLOSED FORM VERSUS ITERATION")
    h_0, H_eff, eta = 2.8e7, 1.2e7, 0.3
    q_hw = 1.0e6
    closed = q_hw * blowing_factor(h_0, H_eff, eta)
    q = q_hw                       # iterate q_net = q_hw - eta*(q/H_eff)*h_0
    for _ in range(200):
        q = q_hw - eta * (q / H_eff) * h_0
    print(f"   closed form      {closed:.6e} W/m^2")
    print(f"   iteration (200)  {q:.6e} W/m^2")
    print(f"   factor           {blowing_factor(h_0, H_eff, eta):.4f} "
          f"-> blowing cuts the flux by {100*(1-blowing_factor(h_0,H_eff,eta)):.0f}%")
    print("\n   The nonlinearity closes analytically; no implicit solver is needed.\n")
    R["blowing.factor"] = float(blowing_factor(h_0, H_eff, eta))
    assert abs(closed / q - 1.0) < 1e-9, f"closed form is off by {closed/q-1:.1e}"


def test_hot_wall():
    print("10. HOT WALL")
    Tb = Aluminium().T_boil_nominal
    fs = []
    for V in (7500.0, 6100.0, 4000.0):
        f = float(hot_wall_factor(V, Tb))
        f1 = float(hot_wall_factor(V, 2740.0))
        fs.append(f)
        R[f"hot_wall.V{V:.0f}.Tb"] = f
        R[f"hot_wall.V{V:.0f}.T2740"] = f1
        print(f"   V={V:6.0f} m/s: T={Tb:.0f} K -> {f:.3f} ({100*(f-1):+.0f}%),"
              f"  T=2740 K -> {f1:.3f} ({100*(f1-1):+.0f}%)")
        exact = 1.0 - 1300.0 * Tb / (0.5 * V ** 2)
        assert abs(f - exact) < 1e-12, "the factor does not match 1 - h_w/h_0"
        assert f > f1, "a hotter wall must give a smaller flux"
    print("   The correction grows as the body decelerates: at the cold end of")
    print("   the trajectory it is no longer small.\n")
    assert fs[0] > fs[1] > fs[2], "the correction must grow with deceleration"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
