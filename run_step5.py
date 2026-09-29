"""Step 5: the main result, the altitude distribution of injected aluminium.

Run:     python run_step5.py

Output:  figures/step5_main.png, figures/step5_sensitivity.png,
         figures/step5_experiment.png and tables in the console.
"""

from __future__ import annotations

import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.ablation import Aluminium, surface_thermal_model
from reentry.results import Recorder
from reentry.emissivity import (LAM_GRID, bare_aluminium_emissivity, eps_film,
                                fit_film_edge, oxide_film_emissivity,
                                planck_weight, required_band,
                                synthetic_oxide_spectrum, total_emissivity)

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.22,
    "axes.spines.top": False, "axes.spines.right": False,
})

ATM = MSISAtmosphere()
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
M0, CD, F_AL = 175.0, 1.5, 0.30
INTACT = Vehicle(mass=M0, area=1.0, Cd=CD, nose_radius=0.5)
M_AL_TOTAL = M0 * F_AL
MW_AL_IN_OXIDE = 2 * 26.98 / (2 * 26.98 + 3 * 16.00)

# Atmospheric layer boundaries. The ones that matter here: above the
# stratopause a particle settles for years, below it reaches the stratosphere
# almost at once.
STRATOPAUSE = 50.0     # km
MESOPAUSE = 85.0       # km
OBS_LO, OBS_HI = 70.0, 80.0
T_OPER = 2000.0        # K, working temperature: Al boils at ~0.5-1 kPa

FRAGMENTS = [
    # name, mass fraction, separation altitude, kind, parameter
    ("solar panels", 0.10, 95.0e3, "plate", 1.5e-3),
    ("primary structure", 0.50, 78.0e3, "compact", 55.0),
    ("small parts, MLI", 0.40, 78.0e3, "plate", 1.0e-3),
]

# DISTRIBUTION OF Al OVER FRAGMENTS at the same total fraction of 30%.
# The thermal model treats every fragment as aluminium; the Al fraction says
# which part of the evaporated mass is aluminium. The actual bill of materials
# of a specific vehicle is unknown, so this is an axis, not a constant.
AL_SPLITS = {
    "uniform (base)": (0.30, 0.30, 0.30),
    "all Al in primary structure": (0.0, 0.60, 0.0),
    "all Al in thin-walled parts": (0.60, 0.0, 0.60),
}
BASE_SPLIT = AL_SPLITS["uniform (base)"]

# Two surface scenarios. The base is "active oxide film": it is the scenario
# that can be measured in the lab, and it gives the smaller (conservative)
# mass.
SCENARIOS = {
    "active oxide film": Aluminium(surface="oxide"),
    "bare melt": Aluminium(surface="bare"),
}
BASE = SCENARIOS["active oxide film"]

SCEN_KEY = {"active oxide film": "film", "bare melt": "bare"}
SENS_KEY = {
    "surface scenario (film/bare)": "surface",
    "film curve shape (48 sets)": "film_shape",
    "film growth tau 1-10000 s": "growth",
    "vapour blowing, eta 0-0.6": "blowing",
    "orientation: stable / tumbling": "orientation",
    "Al distribution over fragments": "al_split",
    "breakup altitude ±10 km": "breakup",
    "plate thickness x2 / /2": "thickness",
    "thin-walled mass fraction 25-75%": "thin_fraction",
    "entry angle -1...-3°": "entry_angle",
    "orbit inclination 0-180°": "inclination",
}
R = Recorder("step5")

BINS = np.arange(30.0, 102.0, 2.0)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])


def build_fragments(fragments=FRAGMENTS, h_shift: float = 0.0,
                    entry: EntryState = ENTRY):
    """-> [(name, Vehicle, trajectory, wall thickness or None)]."""
    tr0 = integrate(INTACT, entry, ATM, h_stop=55.0e3)
    out = []
    for name, f_mass, h_break, kind, param in fragments:
        m = M0 * f_mass
        veh = (Vehicle.plate(m, param, Cd=CD) if kind == "plate"
               else Vehicle.compact(m, param, Cd=CD))
        t_wall = param if kind == "plate" else None
        hb = max(h_break + h_shift, 58.0e3)
        tr = integrate(veh, tr0.state_at_altitude(hb, entry.inclination_deg), ATM)
        out.append((name, veh, tr, t_wall))
    return out


def run_model(frags, mat: Aluminium, al_split=BASE_SPLIT, **model_kw) -> dict:
    """One run of the thermal model over all fragments.

    Returns the total evaporated Al mass, the molten Al and the RAW series
    (altitude, kg Al); both the histogram and the median are computed from it.
    """
    hs, dms, per = [], [], []
    melt_al = 0.0
    for (name, veh, tr, t_wall), f_al in zip(frags, al_split):
        r = surface_thermal_model(tr, veh, mat, wall_thickness=t_wall, **model_kw)
        melt_al += r["m_melt"] * f_al
        hs.append(r["h"][:-1] / 1e3)
        dms.append(np.diff(r["m_vap_series"]) * f_al)
        per.append((name, veh, tr, r, f_al))
    h = np.concatenate(hs)
    dm = np.concatenate(dms)
    order = np.argsort(h)
    return dict(h=h[order], dm=dm[order], total=float(dm.sum()),
                melt=melt_al, per=per)


def histogram(run: dict) -> np.ndarray:
    hist = np.zeros_like(CENTERS)
    idx = np.digitize(run["h"], BINS) - 1
    ok = (idx >= 0) & (idx < len(CENTERS))
    np.add.at(hist, idx[ok], run["dm"][ok])
    return hist


def summarize(run: dict) -> dict:
    """The median comes from the RAW (altitude, mass) series, not from the
    histogram: a binned median depends on the bin width (it moved by 1.5 km
    for 0.5-5 km bins)."""
    h, dm, tot = run["h"], run["dm"], run["total"]
    if tot <= 0:
        return dict(total=0.0, median=np.nan, above_strat=np.nan,
                    above_meso=np.nan, melt=run["melt"])
    c = np.cumsum(dm)
    return dict(total=tot, median=float(np.interp(0.5 * c[-1], c, h)),
                above_strat=float(dm[h >= STRATOPAUSE].sum() / tot),
                above_meso=float(dm[h >= MESOPAUSE].sum() / tot),
                melt=run["melt"])


def eps_oper(mat: Aluminium, T: float = T_OPER) -> float:
    return float(mat.emissivity_at(np.array([T]))[0])


def film_shape_extremes(T: float = T_OPER):
    """Film curve shapes with the smallest and largest eps(T) from the sweep
    over 48 sets (analysis_step5b.E2)."""
    rows = []
    for es, el, w in itertools.product((0.03, 0.05, 0.07, 0.10),
                                       (0.85, 0.92, 1.00), (1.0, 2.0, 3.0, 4.0)):
        try:
            lc = fit_film_edge(es, el, w)
        except ValueError:
            continue
        rows.append(((es, el, w),
                     total_emissivity(LAM_GRID, eps_film(LAM_GRID, es, lc, el, w), T)))
    rows.sort(key=lambda r: r[1])
    return rows[0], rows[-1]


def report_main():
    print("A. MAIN RESULT: ALUMINIUM INJECTION BY ALTITUDE")
    print(f"   object {M0:.0f} kg, {100*F_AL:.0f}% Al = {M_AL_TOTAL:.1f} kg Al,"
          f" Al spread uniformly over fragments")
    print(f"   entry 120 km / 7500 m/s / {ENTRY.gamma_deg:+.1f}°, "
          f"i={ENTRY.inclination_deg:.0f}°; boiling at local pressure\n")
    frags = build_fragments()
    runs = {}
    print(f"   {'scenario':<19}{'eps(2000K)':>11}{'evap. Al, kg':>14}{'yield':>7}"
          f"{'Al2O3/kg sat':>14}{'median, km':>12}{'>50 km':>8}{'>85 km':>8}"
          f"{'molten Al':>11}")
    for name, mat in SCENARIOS.items():
        run = run_model(frags, mat)
        s = summarize(run)
        runs[name] = (run, s)
        k = f"scenarios.{SCEN_KEY[name]}"
        R[f"{k}.eps2000"] = eps_oper(mat)
        R[f"{k}.total"] = s["total"]
        R[f"{k}.yield_pct"] = 100 * s["total"] / M_AL_TOTAL
        R[f"{k}.al2o3_per_kg"] = s["total"] / MW_AL_IN_OXIDE / M0
        R[f"{k}.median"] = s["median"]
        R[f"{k}.above_strat_pct"] = 100 * s["above_strat"]
        R[f"{k}.above_meso_pct"] = 100 * s["above_meso"]
        R[f"{k}.melt"] = s["melt"]
        R[f"{k}.melt_pct"] = 100 * s["melt"] / M_AL_TOTAL
        print(f"   {name:<19}{eps_oper(mat):>11.3f}{s['total']:>14.1f}"
              f"{100*s['total']/M_AL_TOTAL:>6.0f}%"
              f"{s['total']/MW_AL_IN_OXIDE/M0:>14.3f}{s['median']:>12.1f}"
              f"{100*s['above_strat']:>7.0f}%{100*s['above_meso']:>7.0f}%"
              f"{s['melt']:>11.1f}")
    a, b = (runs[k][1]["total"] for k in SCENARIOS)
    R["scenarios.ratio"] = max(a, b) / min(a, b)
    print(f"\n   The scenarios differ by {max(a, b)/min(a, b):.2f}x in mass.")
    print("   Boiling happens at 1750-2050 K; there the film eps is pinned by")
    print("   the alpha-Al2O3 reference point at 1800 K, and the bare-metal eps")
    print("   from resistivity is already ~0.18. The scenarios have converged.\n")

    print("   Constant eps (for the figure and comparison with the first version):")
    print(f"   {'eps':>6}{'evap. Al, kg':>14}{'yield':>7}{'median, km':>12}"
          f"{'molten Al':>11}")
    sweep = {}
    for e in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        run = run_model(frags, Aluminium(emissivity=e))
        s = summarize(run)
        sweep[e] = (run, s)
        k = f"sweep.e{round(e*100):03d}"
        R[f"{k}.total"] = s["total"]
        R[f"{k}.yield_pct"] = 100 * s["total"] / M_AL_TOTAL
        R[f"{k}.median"] = s["median"]
        R[f"{k}.melt"] = s["melt"]
        print(f"   {e:>6.2f}{s['total']:>14.1f}{100*s['total']/M_AL_TOTAL:>6.0f}%"
              f"{s['median']:>12.1f}{s['melt']:>11.1f}")
    print()
    return frags, runs, sweep


def report_changes(frags):
    """Decomposition: what each corrected item changed relative to the first
    version (4.4-7.7 kg at eps 0.19-0.29 and 15.4 kg at eps 0.05)."""
    print("Z. WHAT CHANGED RELATIVE TO THE FIRST VERSION (kg of evaporated Al)")
    print("   first version: eps 0.05 -> 15.4, eps 0.19 -> 7.7, eps 0.29 -> 4.4\n")
    rows = [
        ("+ plate with 2 sides, band <= its own mass,\n"
         "     melt by maximum (old properties)",
         lambda e: Aluminium.legacy(emissivity=e)),
        ("+ new Al properties, boiling at 1 atm (2792 K)",
         lambda e: Aluminium(emissivity=e, boil_at_local_pressure=False)),
        ("+ boiling at local pressure (base)",
         lambda e: Aluminium(emissivity=e)),
    ]
    print(f"   {'step':<50}{'0.05':>7}{'0.19':>7}{'0.29':>7}")
    for (label, make), key in zip(rows, ("plates_fixed", "new_props_1atm", "local_pressure")):
        vals = [run_model(frags, make(e))["total"] for e in (0.05, 0.19, 0.29)]
        for e, v in zip((5, 19, 29), vals):
            R[f"changes.{key}.e{e:03d}"] = v
        first, *rest = label.split("\n")
        print(f"   {first:<50}" + "".join(f"{v:>7.1f}" for v in vals))
        for line in rest:
            print(f"   {line}")
    film_1atm = run_model(frags, Aluminium(surface="oxide",
                                           boil_at_local_pressure=False))["total"]
    R["changes.film_1atm"] = film_1atm
    print(f"\n   active oxide film with boiling at 1 atm: {film_1atm:.1f} kg Al"
          " (base at local pressure is in section A)")
    print()


def report_sensitivity(frags):
    print("B. SENSITIVITY (base: active oxide film, Al uniform)")
    base = summarize(run_model(frags, BASE))
    rows = []

    def add(label, variants):
        ss = [summarize(v) for v in variants]
        tots = [s["total"] for s in ss]
        meds = [s["median"] for s in ss if np.isfinite(s["median"])]
        rows.append((label, min(tots), max(tots),
                     (max(meds) - min(meds)) if len(meds) > 1 else 0.0))

    add("surface scenario (film/bare)",
        [run_model(frags, m) for m in SCENARIOS.values()])
    lo_shape, hi_shape = film_shape_extremes()
    add("film curve shape (48 sets)",
        [run_model(frags, Aluminium(surface="oxide", film_shape=sh))
         for sh, _ in (lo_shape, hi_shape)])
    add("film growth tau 1-10000 s",
        [run_model(frags, Aluminium(tau_oxide=t)) for t in (1.0, 1e4)])
    add("vapour blowing, eta 0-0.6",
        [run_model(frags, BASE, blowing_eta=e) for e in (0.0, 0.6)])
    add("orientation: stable / tumbling",
        [run_model(frags, BASE, flux_mode=m) for m in ("cos", "uniform")])
    add("Al distribution over fragments",
        [run_model(frags, BASE, al_split=sp) for sp in AL_SPLITS.values()])
    add("breakup altitude ±10 km",
        [run_model(build_fragments(h_shift=d), BASE) for d in (-10e3, 10e3)])
    thick = [("solar panels", 0.10, 95e3, "plate", 3.0e-3),
             ("primary structure", 0.50, 78e3, "compact", 55.0),
             ("small parts, MLI", 0.40, 78e3, "plate", 2.0e-3)]
    thin = [("solar panels", 0.10, 95e3, "plate", 0.75e-3),
            ("primary structure", 0.50, 78e3, "compact", 55.0),
            ("small parts, MLI", 0.40, 78e3, "plate", 0.5e-3)]
    add("plate thickness x2 / /2",
        [run_model(build_fragments(f), BASE) for f in (thick, thin)])
    heavy = [("solar panels", 0.05, 95e3, "plate", 1.5e-3),
             ("primary structure", 0.75, 78e3, "compact", 55.0),
             ("small parts, MLI", 0.20, 78e3, "plate", 1.0e-3)]
    light = [("solar panels", 0.20, 95e3, "plate", 1.5e-3),
             ("primary structure", 0.25, 78e3, "compact", 55.0),
             ("small parts, MLI", 0.55, 78e3, "plate", 1.0e-3)]
    add("thin-walled mass fraction 25-75%",
        [run_model(build_fragments(f), BASE) for f in (heavy, light)])
    add("entry angle -1...-3°",
        [run_model(build_fragments(entry=EntryState(gamma_deg=g,
                                                    inclination_deg=53.0)), BASE)
         for g in (-1.0, -3.0)])
    add("orbit inclination 0-180°",
        [run_model(build_fragments(entry=EntryState(gamma_deg=-1.5,
                                                    inclination_deg=i)), BASE)
         for i in (0.0, 180.0)])

    print(f"   {'factor':<38}{'Al, kg: from':>13}{'to':>7}{'ratio':>7}{'median, km':>12}")
    for label, lo, hi, dmed in sorted(rows, key=lambda r: -(r[2] - r[1])):
        k = f"sensitivity.{SENS_KEY[label]}"
        R[f"{k}.lo"] = lo
        R[f"{k}.hi"] = hi
        R[f"{k}.ratio"] = hi / lo if lo > 1e-3 else None
        R[f"{k}.median_span"] = dmed
        ratio = f"{hi/lo:>6.1f}x" if lo > 1e-3 else "     ∞"
        print(f"   {label:<38}{lo:>13.1f}{hi:>7.1f}{ratio}{dmed:>12.1f}")
    print(f"\n   Base: {base['total']:.1f} kg Al, median {base['median']:.1f} km, "
          f"{100*base['above_strat']:.0f}% above the stratopause, "
          f"{100*base['above_meso']:.0f}% above the mesopause.")
    print(f"   Film shapes: eps(2000 K) from {lo_shape[1]:.3f} to {hi_shape[1]:.3f}.\n")
    R["sensitivity.base.total"] = base["total"]
    R["sensitivity.base.median"] = base["median"]
    R["film_shape.eps2000_min"] = lo_shape[1]
    R["film_shape.eps2000_max"] = hi_shape[1]
    return rows, base


def report_ferreira(runs):
    print("D. COMPARISON WITH FERREIRA et al. 2024 (GRL, 10.1029/2024GL109280)")
    print("   Theirs: 250 kg, 30% Al = 75 kg Al. Their oxidation MD oxidises")
    print("   24.0 kg Al (32%), forming 29.8 kg of AlO clusters; 51.0 kg Al")
    print("   remain as unoxidised clusters. So 32% is an OUTPUT OF THEIR MODEL,")
    print("   not an assumption, and it is computed at 2200 K, 86 km.")
    f_ferr = 24.0 / 75.0
    print(f"   In our normalization (52.5 kg Al): {f_ferr*M_AL_TOTAL:.1f} kg Al "
          f"to oxide, {24.0/250:.3f} kg Al per kg of satellite.\n")
    print(f"   {'scenario':<42}{'Al to vapour, kg':>17}{'Al share':>10}{'per kg sat':>12}")
    frags = runs["frags"]
    for name, mat in SCENARIOS.items():
        for sp_name, sp in AL_SPLITS.items():
            if sp_name.startswith("all Al in primary"):
                continue
            s = summarize(run_model(frags, mat, al_split=sp))
            spk = "uniform" if sp_name.startswith("uniform") else "thin"
            k = f"ferreira.{SCEN_KEY[name]}_{spk}"
            R[f"{k}.total"] = s["total"]
            R[f"{k}.pct"] = 100 * s["total"] / M_AL_TOTAL
            R[f"{k}.per_kg"] = s["total"] / M0
            lbl = f"{name}, {sp_name.split(' (')[0]}"
            print(f"   {lbl:<42}{s['total']:>17.1f}{100*s['total']/M_AL_TOTAL:>9.0f}%"
                  f"{s['total']/M0:>12.3f}")
    R["ferreira.theirs.total"] = f_ferr * M_AL_TOTAL
    R["ferreira.theirs.pct"] = 100 * f_ferr
    R["ferreira.theirs.per_kg"] = 24.0 / 250
    print(f"   {'Ferreira (oxidised)':<42}{f_ferr*M_AL_TOTAL:>17.1f}"
          f"{100*f_ferr:>9.0f}%{24.0/250:>12.3f}")
    print("\n   Our 16-23% lie BELOW their 32% with uniform Al; 32% is reached")
    print("   when Al is concentrated in thin-walled parts. Different quantities")
    print("   are compared: ours is evaporated Al (all vapour will oxidise), theirs")
    print("   is the oxidised fraction assuming that all Al is ablated.")
    print("   This is order-of-magnitude agreement, not validation.\n")


def report_experiment():
    print("C. WHAT TO MEASURE, IN TERMS OF INSTRUMENT CAPABILITIES")
    print("   Direct high-temperature emissometry is not needed. Spectral")
    print("   REFLECTANCE at room temperature is enough:")
    print("     eps(lam) = 1 - R(lam)          (Kirchhoff, opaque sample)")
    print("     eps(T)   = int eps(lam) B(lam,T) dlam / int B(lam,T) dlam\n")
    print(f"   {'T, K':>7}{'Wien peak, um':>16}{'95% of energy, um':>22}")
    for T in (1500.0, 1800.0, 2000.0, 2200.0):
        lo, hi, pk = required_band(T)
        R[f"band.T{T:.0f}.wien_um"] = pk * 1e6
        R[f"band.T{T:.0f}.lo_um"] = lo * 1e6
        R[f"band.T{T:.0f}.hi_um"] = hi * 1e6
        print(f"   {T:>7.0f}{pk*1e6:>16.2f}{lo*1e6:>14.2f} - {hi*1e6:<6.2f}")
    lo_all, _, _ = required_band(2200.0)
    _, hi_all, _ = required_band(1500.0)
    R["band.union_lo_um"] = lo_all * 1e6
    R["band.union_hi_um"] = hi_all * 1e6
    print(f"\n   IN TOTAL the 1500-2200 K working range needs {lo_all*1e6:.2f}-"
          f"{hi_all*1e6:.1f} um.")
    print("   That is UV-Vis-NIR PLUS mid-IR FTIR, both with an integrating")
    print("   sphere: the total (specular + diffuse) R is needed.\n")


def figure_main(runs, sweep, path="figures/step5_main.png"):
    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    colors = {"active oxide film": "tab:blue", "bare melt": "tab:red"}
    hmax = 0.0
    for name, (run, s) in runs["scen"].items():
        hist = histogram(run)
        hmax = max(hmax, hist.max())
        ax.step(hist, CENTERS, where="mid", lw=2.0, color=colors[name],
                label=f"{name}: {s['total']:.1f} kg")
    ax.axhspan(OBS_LO, OBS_HI, color="seagreen", alpha=0.16, zorder=0)
    ax.axhline(STRATOPAUSE, color="crimson", ls="--", lw=1.3)
    ax.axhline(MESOPAUSE, color="grey", ls=":", lw=1.1)
    ax.text(hmax * 0.98, STRATOPAUSE + 1.5, "stratopause 50 km", fontsize=7.5,
            color="crimson", ha="right")
    ax.text(hmax * 0.98, MESOPAUSE + 1.2, "mesopause 85 km", fontsize=7.5,
            color="grey", ha="right")
    ax.text(hmax * 0.98, 66.5, "observed breakup\n70–80 km",
            fontsize=7.5, color="seagreen", ha="right", va="center")
    ax.set_xlabel("injected Al per 2 km layer, kg")
    ax.set_ylabel("altitude, km")
    ax.set_title("Altitude distribution of Al injection\n"
                 f"object {M0:.0f} kg, {100*F_AL:.0f}% Al", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(42, 100)

    ax = fig.add_subplot(gs[1])
    eps = sorted(sweep)
    tot = [sweep[e][1]["total"] for e in eps]
    melt = [sweep[e][1]["melt"] for e in eps]
    ax.fill_between(eps, tot, melt, color="tab:purple", alpha=0.15,
                    label="melt/vapour bracket")
    ax.plot(eps, melt, lw=2.0, marker="o", ms=3.5, label="molten")
    ax.plot(eps, tot, lw=2.0, ls="--", marker="s", ms=3.5,
            label="evaporated in place")
    for name, (run, s) in runs["scen"].items():
        e = eps_oper(SCENARIOS[name])
        ax.plot([e], [s["total"]], "D", ms=8, color=colors[name],
                label=f"{name}, ε(2000 K)={e:.2f}")
    f_ferr = 24.0 / 75.0 * M_AL_TOTAL
    ax.axhline(f_ferr, color="crimson", ls=":", lw=1.4)
    ax.text(0.055, f_ferr + 0.7, f"Ferreira: 32% of Al oxidised = {f_ferr:.1f} kg",
            fontsize=7.5, color="crimson", ha="left")
    ax.set_xlabel(r"constant $\varepsilon$"); ax.set_ylabel("Al, kg")
    ax.set_title("Mass: surface scenarios\nagainst the ε sweep", fontsize=10)
    ax.legend(frameon=False, fontsize=7.2, loc="center right",
              bbox_to_anchor=(1.0, 0.66))

    ax = fig.add_subplot(gs[2])
    med = [sweep[e][1]["median"] for e in eps]
    ax.plot(eps, med, lw=2.0, marker="o", ms=3.5, color="tab:blue",
            label="median altitude, km")
    ax.axhspan(OBS_LO, OBS_HI, color="seagreen", alpha=0.16)
    ax.set_ylim(66, 84)
    ax.set_xlabel(r"constant $\varepsilon$"); ax.set_ylabel("altitude, km")
    ax.set_title("The injection median does not depend on ε:\n"
                 "the breakup altitude sets it", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("Step 5: aluminium injection during satellite re-entry: "
                 "where and how much", fontsize=12, y=1.0)
    fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


def figure_sensitivity(rows, base, path="figures/step5_sensitivity.png"):
    rows = sorted(rows, key=lambda r: (r[2] - r[1]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))
    ax = axes[0]
    for k, (_, lo, hi, _) in enumerate(rows):
        ax.plot([lo, hi], [k, k], lw=6, color="tab:orange", solid_capstyle="butt")
    ax.axvline(base["total"], color="k", lw=1.0, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("evaporated Al, kg (base is dashed)")
    ax.set_title("MASS budget", fontsize=10)
    ax = axes[1]
    ax.barh(y, [r[3] for r in rows], color="tab:blue")
    ax.set_yticks(y, [""] * len(rows))
    ax.set_xlabel("spread of median altitude, km")
    ax.set_title("ALTITUDE budget", fontsize=10)
    fig.suptitle("Mass and altitude are controlled by different parameters",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


def figure_experiment(path="figures/step5_experiment.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    lam = np.geomspace(0.2e-6, 30e-6, 1200)

    ax = axes[0]
    for T, c in ((1500.0, "tab:blue"), (2000.0, "tab:orange"),
                 (2740.0, "crimson")):
        w = planck_weight(lam, T)
        ax.semilogx(lam * 1e6, w / w.max(), lw=1.9, color=c, label=f"{T:.0f} K")
        lo, hi, _ = required_band(T)
        ax.axvspan(lo * 1e6, hi * 1e6, color=c, alpha=0.06)
    ax.axvspan(0.22, 1.4, color="tab:green", alpha=0.13)
    ax.axvspan(2.5, 15.0, color="tab:purple", alpha=0.13)
    ax.text(0.5, 0.55, "UV-Vis-NIR\n(available, to 1.4)", fontsize=8,
            color="tab:green", rotation=90)
    ax.text(5.0, 0.55, "mid-IR FTIR\n(needs a sphere)", fontsize=8,
            color="tab:purple", rotation=90)
    ax.set_xlabel("wavelength, um"); ax.set_ylabel("Planck weight, norm.")
    ax.set_title("Band the instrument must cover", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    Ts = np.linspace(900, 2800, 60)
    ax.plot(Ts, oxide_film_emissivity()(Ts), lw=2.0, color="tab:blue",
            label="active oxide film (α-Al₂O₃)")
    ax.plot(Ts, bare_aluminium_emissivity(Ts), lw=2.0, color="tab:red",
            label="bare metal (from resistivity)")
    for d, c in ((0.1e-6, "0.7"), (1.0e-6, "0.5"), (5.0e-6, "0.3")):
        sp = synthetic_oxide_spectrum(lam, d)
        ax.plot(Ts, [total_emissivity(lam, sp, T) for T in Ts], lw=1.0,
                ls=":", color=c, label=f"placeholder, {d*1e6:.1f} um film")
    ax.axvspan(1750, 2050, color="tab:orange", alpha=0.12)
    ax.text(1900, 0.02, "boiling\nat 0.3–1 kPa", fontsize=7.5, ha="center")
    ax.set_xlabel("temperature, K"); ax.set_ylabel(r"total $\varepsilon(T)$")
    ax.set_title("Spectrum → ε(T) chain\n(measured data plug in here)", fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    ax.axis("off")
    ax.text(0.0, 1.0, "WHAT TO MEASURE", fontsize=11, weight="bold", va="top")
    txt = (
        "Samples: Al 6061, a series in oxide thickness\n"
        "  native (~3 nm) and thermal (up to ~0.2 um);\n"
        "  thicker: anodizing / PEO / ALD-Al2O3\n\n"
        "Oxide thickness:\n"
        "  ellipsometry (up to ~1 um)\n"
        "  SEM cross-section (thicker)\n\n"
        "Reflectance (the key part):\n"
        "  UV-Vis-NIR to 1.4 um, integrating sphere\n"
        "  FTIR 2.5–15 um, gold integrating sphere\n"
        "  TOTAL R is needed, not specular, not ATR\n\n"
        "Processing: eps(lam)=1−R(lam), then Planck\n"
        "  weighting at 1500–2200 K\n\n"
        "What it covers: the active-film scenario.\n"
        "Bare melt cannot be measured this way."
    )
    ax.text(0.0, 0.92, txt, fontsize=8.2, va="top", family="monospace")

    fig.suptitle("Bridge to the experiment: reflectance measured at room "
                 "temperature gives ε at the working one", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


if __name__ == "__main__":
    print()
    frags, scen, sweep = report_main()
    report_changes(frags)
    rows, base = report_sensitivity(frags)
    report_ferreira({"frags": frags})
    report_experiment()
    figure_main({"scen": scen}, sweep)
    figure_sensitivity(rows, base)
    figure_experiment()
    print()
    R.save(__file__)
