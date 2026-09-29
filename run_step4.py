"""Step 4: fragmentation, thermal response, ablation, as a BRACKET rather
than an estimate.

Run:     python run_step4.py

Output:  figures/step4_epsilon.png, figures/step4_bracket.png and tables in
         the console. The comparison with Ferreira is in run_step5.py
         (section D).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_step5 as S
from reentry import Vehicle, integrate
from reentry.ablation import Aluminium, regime_number, surface_thermal_model
from reentry.heating import SIGMA_SB
from reentry.results import Recorder

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

ATM, ENTRY, CD, M0, INTACT = S.ATM, S.ENTRY, S.CD, S.M0, S.INTACT
EPS_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
SCEN_KEY = S.SCEN_KEY
R = Recorder("step4")

# A 1 mm plate of 1 kg flying FROM THE ENTRY POINT at 120 km: an illustration
# of the radiative regime. It is not the "small parts" fragment (that one
# starts from the 78 km breakup altitude at the speed of the whole object),
# so their numbers differ.
THIN = Vehicle.plate(1.0, 1.0e-3, Cd=CD)


def report_criteria():
    mat = Aluminium()
    print("A. TWO DEMISE CRITERIA")
    print(f"   to complete melting      {mat.h_melt_complete/1e6:6.2f} MJ/kg"
          f"   <- ORSAT/DRAMA criterion (theirs is 0.93)")
    print(f"   to complete evaporation  {mat.h_vapour_complete/1e6:6.2f} MJ/kg"
          f"   <- what atmospheric chemistry needs (at 1 kPa)")
    print(f"   ratio                    {mat.demise_ratio:6.1f}x\n")
    R["criteria.melt"] = mat.h_melt_complete / 1e6
    R["criteria.vapour"] = mat.h_vapour_complete / 1e6
    R["criteria.ratio"] = mat.demise_ratio
    R["criteria.h1"] = mat.h1 / 1e6
    R["criteria.h2"] = mat.h2 / 1e6
    print("   Boiling temperature of Al at the surface, at stagnation pressure:")
    for p in (1e2, 3e2, 1e3, 3e3, 101325.0):
        Tb = float(mat.T_boil_at(p))
        R[f"boil.p{p:.0f}.T"] = Tb
        R[f"boil.p{p:.0f}.L"] = float(mat.L_vapour_at(Tb)) / 1e6
        R[f"boil.p{p:.0f}.h3"] = float(mat.h3(Tb)) / 1e6
        print(f"     p = {p:>8.0f} Pa  ->  T_boil = {float(mat.T_boil_at(p)):6.0f} K,"
              f"  L = {float(mat.L_vapour_at(mat.T_boil_at(p)))/1e6:5.2f} MJ/kg")
    from reentry.heating import SHAPE_FACTOR_TUMBLING
    for tag, Tb in (("atm", mat.T_boil_1atm), ("kpa", mat.T_boil_nominal)):
        for e in (0.3, 0.2):
            q = e * SIGMA_SB * Tb ** 4 / SHAPE_FACTOR_TUMBLING / 1e4
            R[f"threshold.{tag}.e{round(e*100):02d}"] = q
    print("   Boiling threshold for the mean flux eps*sigma*T^4/phi, W/cm^2:")
    print(f"     1 atm ({mat.T_boil_1atm:.0f} K): eps=0.3 -> "
          f"{0.3*SIGMA_SB*mat.T_boil_1atm**4/SHAPE_FACTOR_TUMBLING/1e4:.0f}, "
          f"eps=0.2 -> {0.2*SIGMA_SB*mat.T_boil_1atm**4/SHAPE_FACTOR_TUMBLING/1e4:.0f}")
    print(f"     1 kPa ({mat.T_boil_nominal:.0f} K): eps=0.3 -> "
          f"{0.3*SIGMA_SB*mat.T_boil_nominal**4/SHAPE_FACTOR_TUMBLING/1e4:.0f}, "
          f"eps=0.2 -> {0.2*SIGMA_SB*mat.T_boil_nominal**4/SHAPE_FACTOR_TUMBLING/1e4:.0f}")
    print("\n   Between melt and vapour lies the fate of the stripped melt: a droplet")
    print("   may evaporate further, oxidise at its surface and fall as a spherule,")
    print("   or freeze whole. Hence the result is a BRACKET.\n")


def report_epsilon():
    print("B. EMISSIVITY: WHERE IT MATTERS AND WHERE IT DOES NOT")
    print("   Whole object and a 1 mm plate (1 kg), both FROM THE ENTRY POINT at 120 km.\n")
    tr = integrate(INTACT, ENTRY, ATM)
    tr_thin = integrate(THIN, ENTRY, ATM)
    print(f"   {'eps':<19}{'T max whole':>12}{'melt':>7}{'evap.':>7}"
          f"   ||{'T max plate':>12}{'melt':>7}{'evap.':>7}")
    mats = [(f"{e:.2f}", Aluminium(emissivity=float(e))) for e in EPS_GRID]
    mats += [(name, m) for name, m in S.SCENARIOS.items()]
    for label, m in mats:
        r = surface_thermal_model(tr, INTACT, m)
        rt = surface_thermal_model(tr_thin, THIN, m, wall_thickness=1.0e-3)
        key = SCEN_KEY[label] if label in SCEN_KEY else f"e{round(float(label)*100):03d}"
        R[f"eps.{key}.whole_Tmax"] = r["T_max"]
        R[f"eps.{key}.whole_melt_pct"] = 100 * r["f_melt"]
        R[f"eps.{key}.whole_vap_pct"] = 100 * r["f_vap"]
        R[f"eps.{key}.plate_Tmax"] = rt["T_max"]
        R[f"eps.{key}.plate_melt_pct"] = 100 * rt["f_melt"]
        R[f"eps.{key}.plate_vap_pct"] = 100 * rt["f_vap"]
        print(f"   {label:<19}{r['T_max']:>12.0f}{100*r['f_melt']:>6.0f}%"
              f"{100*r['f_vap']:>6.0f}%   ||{rt['T_max']:>12.0f}"
              f"{100*rt['f_melt']:>6.0f}%{100*rt['f_vap']:>6.0f}%")
    print("\n   Whole object: eps barely matters. The 16 mm wall has no time to")
    print("   reach radiative equilibrium, so the problem is ENERGY-LIMITED. Melt")
    print("   stays at or below 50%: the leeward half of the shell is not heated")
    print("   in the model. Plate: reaches equilibrium quickly, and there eps matters.\n")


def report_fragments(mat: Aluminium, frags):
    print(f"C. FRAGMENT LIST, scenario: {mat.label()}")
    print(f"   {'fragment':<24}{'m, kg':>7}{'beta':>6}{'Rn, cm':>8}"
          f"{'h break':>9}{'h q-peak':>10}{'T_boil, K':>11}{'melt':>7}{'evap.':>7}")
    run = S.run_model(frags, mat)
    sk = SCEN_KEY[mat.label()]
    fk = {"solar panels": "panels", "primary structure": "structure",
          "small parts, MLI": "mli"}
    tot_melt = tot_vap = 0.0
    for name, veh, tr, r, f_al in run["per"]:
        tot_melt += r["m_melt"]; tot_vap += r["m_vap"]
        k = f"fragments.{sk}.{fk[name]}"
        R[f"{k}.mass"] = veh.mass
        R[f"{k}.beta"] = veh.ballistic_coefficient
        R[f"{k}.rn_cm"] = 100 * veh.nose_radius
        R[f"{k}.h_break_km"] = tr.h[0] / 1e3
        R[f"{k}.h_peak_q_km"] = tr.peak_heating()[0] / 1e3
        R[f"{k}.q_peak_wcm2"] = float(tr.heat_flux(veh).max()) / 1e4
        R[f"{k}.Tb_min"] = r["T_boil"].min()
        R[f"{k}.Tb_max"] = r["T_boil"].max()
        R[f"{k}.melt_pct"] = 100 * r["f_melt"]
        R[f"{k}.vap_pct"] = 100 * r["f_vap"]
        print(f"   {name:<24}{veh.mass:>7.1f}{veh.ballistic_coefficient:>6.0f}"
              f"{100*veh.nose_radius:>8.1f}{tr.h[0]/1e3:>9.0f}"
              f"{tr.peak_heating()[0]/1e3:>10.1f}"
              f"{r['T_boil'].min():>6.0f}-{r['T_boil'].max():<4.0f}"
              f"{100*r['f_melt']:>6.0f}%{100*r['f_vap']:>6.0f}%")
    R[f"fragments.{sk}.total.melt_kg"] = tot_melt
    R[f"fragments.{sk}.total.melt_pct"] = 100 * tot_melt / M0
    R[f"fragments.{sk}.total.melt_al"] = run["melt"]
    R[f"fragments.{sk}.total.vap_kg"] = tot_vap
    R[f"fragments.{sk}.total.vap_pct"] = 100 * tot_vap / M0
    R[f"fragments.{sk}.total.vap_al"] = run["total"]
    print(f"\n   TOTAL for the {M0:.0f} kg object:")
    print(f"     molten                {tot_melt:6.1f} kg ({100*tot_melt/M0:.0f}%)"
          f"  -> Al {run['melt']:5.1f} kg   UPPER BOUND")
    print(f"     evaporated in place   {tot_vap:6.1f} kg ({100*tot_vap/M0:.0f}%)"
          f"  -> Al {run['total']:5.1f} kg\n")


def report_oxide_growth(frags):
    print("E. eps AS A TRAJECTORY: OXIDE FILM GROWTH")
    print("   eps(t,T) = eps_bare(T) + (eps_film(T) - eps_bare(T))(1-exp(-sqrt(t/tau)))")
    print("   t is the time SINCE BREAKUP: the fragment surface is fresh. Fragments")
    print("   are heated in the first ~10-40 s after breakup.\n")
    print(f"   {'tau, s':>9}{'eps(10 s)':>11}{'eps(30 s)':>11}{'Al evap., kg':>14}{'yield':>8}")
    T = np.array([S.T_OPER])
    for tau in (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0):
        mat = Aluminium(tau_oxide=tau)
        s = S.summarize(S.run_model(frags, mat))
        k = f"growth.tau{tau:.0f}"
        R[f"{k}.eps10"] = float(mat.emissivity_at(T, 10.0)[0])
        R[f"{k}.eps30"] = float(mat.emissivity_at(T, 30.0)[0])
        R[f"{k}.total"] = s["total"]
        R[f"{k}.yield_pct"] = 100 * s["total"] / S.M_AL_TOTAL
        print(f"   {tau:>9.0f}{float(mat.emissivity_at(T, 10.0)[0]):>11.3f}"
              f"{float(mat.emissivity_at(T, 30.0)[0]):>11.3f}{s['total']:>14.1f}"
              f"{100*s['total']/S.M_AL_TOTAL:>7.0f}%")
    print("\n   The hypothesis does not narrow the uncertainty: the unknown is renamed")
    print("   from 'which eps' to 'how fast the film grows', but the whole spread now")
    print("   fits between the two scenarios. Lab request: measure eps AS A FUNCTION")
    print("   OF oxide THICKNESS.\n")


def report_regime():
    print("F. WHAT SEPARATES THE REGIMES")
    print("   Not the heating depth (both 1 mm and 16 mm are much thinner than")
    print("   14 cm) but the mass per unit WETTED area. Both faces of a plate are")
    print("   wetted, so the mass per unit wetted area is rho*t/2.\n")
    print(f"   {'object':<18}{'equiv. thickness':>17}{'needed':>11}"
          f"{'available':>13}{'Pi':>8}{'regime':>16}")
    for label, veh in (("whole object", INTACT), ("1 mm plate", THIN)):
        tr = integrate(veh, ENTRY, ATM)
        r = regime_number(tr, veh, Aluminium())
        k = "regime.whole" if veh is INTACT else "regime.plate"
        R[f"{k}.equiv_mm"] = r["equiv_thickness"] * 1e3
        R[f"{k}.need_MJ"] = r["need"] / 1e6
        R[f"{k}.avail_MJ"] = r["available"] / 1e6
        R[f"{k}.Pi"] = r["Pi"]
        print(f"   {label:<18}{r['equiv_thickness']*1e3:>14.1f} mm"
              f"{r['need']/1e6:>8.1f} MJ{r['available']/1e6:>10.1f} MJ"
              f"{r['Pi']:>8.2f}{r['regime']:>16}")
    print()


def figure_epsilon(path="figures/step4_epsilon.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    mat = Aluminium()
    Tb = mat.T_boil_nominal
    q_typ = 8.5e5

    ax = axes[0]
    theta = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.degrees(theta), np.cos(theta), lw=2.0, color="k",
            label=r"$q(\theta)/q_{stag}=\cos\theta$")
    for name, c in (("bare melt", "tab:red"), ("active oxide film", "tab:blue")):
        e = S.eps_oper(S.SCENARIOS[name], Tb)
        C = 2 * e * SIGMA_SB * Tb ** 4 / q_typ
        if C < 1:
            ax.fill_between(np.degrees(theta), C, np.cos(theta),
                            where=np.cos(theta) > C, alpha=0.18, color=c)
        ax.axhline(min(C, 1.05), color=c, ls="--", lw=1.1,
                   label=f"plate boiling threshold, {name}")
    ax.set_xlabel("angle from the stagnation point, deg"); ax.set_ylabel(r"$q/q_{stag}$")
    ax.set_title(f"the distribution, not the mean\nq = 85 W/cm², T_boil = {Tb:.0f} K",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.2); ax.set_ylim(0, 1.1)

    tr_thin = integrate(THIN, ENTRY, ATM)
    tr_int = integrate(INTACT, ENTRY, ATM)
    eps = EPS_GRID
    res_t = [surface_thermal_model(tr_thin, THIN, Aluminium(emissivity=float(e)),
                                   wall_thickness=1.0e-3) for e in eps]
    res_i = [surface_thermal_model(tr_int, INTACT, Aluminium(emissivity=float(e)))
             for e in eps]
    for ax, res, title in ((axes[1], res_t, "1 mm plate (from the entry point)"),
                           (axes[2], res_i, "whole object (16 mm wall)")):
        mel = [100 * r["f_melt"] for r in res]
        vap = [100 * r["f_vap"] for r in res]
        ax.fill_between(eps, vap, mel, color="tab:purple", alpha=0.20)
        ax.plot(eps, mel, lw=2.0, marker="o", ms=3.5, label="molten")
        ax.plot(eps, vap, lw=2.0, ls="--", marker="s", ms=3.5,
                label="evaporated in place")
        for name, c in (("bare melt", "tab:red"), ("active oxide film", "tab:blue")):
            ax.axvline(S.eps_oper(S.SCENARIOS[name], Tb), color=c, lw=1.0, ls=":")
        ax.set_xlabel(r"constant $\varepsilon$"); ax.set_ylabel("mass fraction, %")
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8); ax.set_ylim(-2, 102)

    fig.suptitle("Emissivity decides for thin-walled parts and not for massive "
                 "ones (dotted lines: scenarios at T_boil)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


def figure_bracket(frags, path="figures/step4_bracket.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    cases = [("bare melt", S.SCENARIOS["bare melt"]),
             ("film growth tau = 30 s", Aluminium(tau_oxide=30.0)),
             ("active oxide film", S.SCENARIOS["active oxide film"])]
    for ax, (label, mat) in zip(axes, cases):
        run = S.run_model(frags, mat)
        hist = S.histogram(run)
        ax.barh(S.CENTERS, hist, height=1.8, color="tab:blue")
        ax.axhspan(70, 80, color="seagreen", alpha=0.15)
        ax.set_xlabel("evaporated Al per layer, kg")
        ax.set_ylabel("altitude, km")
        ax.set_title(f"{label}: total {run['total']:.1f} kg Al", fontsize=9.5)
        ax.set_ylim(40, 100)
    fig.suptitle("Altitude distribution of evaporated aluminium: "
                 "its shape barely depends on the surface scenario", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


if __name__ == "__main__":
    print()
    report_criteria()
    report_epsilon()
    frags = S.build_fragments()
    for mat in S.SCENARIOS.values():
        report_fragments(mat, frags)
    report_oxide_growth(frags)
    report_regime()
    figure_epsilon()
    figure_bracket(frags)
    print()
    R.save(__file__)
