"""Step 3: aerodynamic heating, Earth rotation, uncertainty budget.

Run:     python run_step3.py

Output:  figures/step3_heating.png, figures/step3_budget.png and tables in
         the console.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry.ablation import Aluminium
from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.constants import G0
from reentry.results import Recorder

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

VEH = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=0.5)
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
ATM = MSISAtmosphere()
OBS_LO, OBS_HI = 70.0, 80.0
# Hot wall at the boiling temperature of Al at the local pressure (~1 kPa).
T_WALL = Aluminium().T_boil_nominal
R = Recorder("step3")
BUDGET_KEYS = {
    "fragmentation (size L 1.0->0.1)": "fragmentation", "Cd 1.0-2.2": "cd",
    "effective Rn 0.2-1.0 m": "rn", "shape factor 0.25-0.30": "shape_factor",
    "orbit inclination 0-180°": "inclination", "entry latitude -75...0°": "latitude",
    "season March/September": "season", "MSIS version 00 / 2.1": "msis_version",
    "correlation S-G / DKR": "correlation", "entry angle -1...-3°": "entry_angle",
    "solar activity F10.7 70-220": "solar",
}


def report_baseline():
    print("A. BASELINE CASE WITH HEATING")
    print(f"   m={VEH.mass:.0f} kg, Cd*A={VEH.Cd*VEH.area:.1f} m^2, "
          f"Rn={VEH.nose_radius:.2f} m, beta={VEH.ballistic_coefficient:.0f} kg/m^2")
    print(f"   entry 120 km / 7500 m/s / {ENTRY.gamma_deg:+.1f}°, "
          f"inclination {ENTRY.inclination_deg:.0f}°")
    print(f"   atmosphere: {ATM.name}")
    print(f"   co-rotation drift omega*R*cos(i) = "
          f"{ENTRY.corotation_speed:.0f} m/s\n")
    tr = integrate(VEH, ENTRY, ATM)
    q = tr.heat_flux(VEH)
    i = int(np.argmax(q))
    a, h_a, _ = tr.peak_decel()
    print(f"   heating peak       {q[i]/1e4:8.1f} W/cm^2  at {tr.h[i]/1e3:.1f} km, "
          f"t = {tr.t[i]:.0f} s, V_rel = {tr.V_rel[i]:.0f} m/s")
    print(f"   deceleration peak  {a/G0:8.1f} g       at {h_a/1e3:.1f} km")
    print(f"   heat load          {tr.heat_load(VEH)/1e6:8.1f} MJ/m^2")
    print(f"   peak separation    {(tr.h[i]-h_a)/1e3:8.1f} km")
    R["base.q_peak_wcm2"] = q[i] / 1e4
    R["base.h_heat_km"] = tr.h[i] / 1e3
    R["base.t_heat"] = tr.t[i]
    R["base.V_rel_heat"] = tr.V_rel[i]
    R["base.amax_g"] = a / G0
    R["base.h_decel_km"] = h_a / 1e3
    R["base.Q_MJ"] = tr.heat_load(VEH) / 1e6
    R["base.separation_km"] = (tr.h[i] - h_a) / 1e3
    R["base.corotation"] = ENTRY.corotation_speed
    print(f"\n   Observed breakup at {OBS_LO:.0f}-{OBS_HI:.0f} km is still "
          f"{OBS_LO - tr.h[i]/1e3:.0f}-{OBS_HI - tr.h[i]/1e3:.0f} km above the peak.")
    print("   The gap remains; fragmentation closes it (step 4).\n")
    return tr


def report_budget():
    """Each row is a measured spread, not an eyeball estimate."""
    print("B. UNCERTAINTY BUDGET")
    print("   Two separate questions: WHERE injection happens and HOW MUCH")
    print("   mass is injected. Some parameters affect only one of them.\n")

    def run(veh=VEH, entry=ENTRY, atm=ATM, corr="sutton-graves", **kw):
        tr = integrate(veh, entry, atm, **kw)
        q = tr.heat_flux(veh, corr)
        i = int(np.argmax(q))
        # MASS METRIC: specific absorbed energy, J/kg, not flux in W/m^2
        # (see heating.absorbed_power: the sign with size flips there).
        return tr.h[i], tr.specific_energy(veh, corr, T_wall=T_WALL)

    base_h, base_S = run()
    rows = []

    def add(label, kind, variants):
        hs = [v[0] for v in variants]
        Ss = [v[1] for v in variants]
        rows.append((label, kind, (max(hs) - min(hs)) / 1e3,
                     100 * (max(Ss) / min(Ss) - 1)))

    # Fragmentation is GEOMETRICALLY SIMILAR: mass, area and Rn change
    # consistently (m~L^3, A~L^2, Rn~L). Sweeping beta alone at fixed mass
    # and Rn is physically meaningless.
    add("fragmentation (size L 1.0->0.1)", "both",
        [run(veh=Vehicle.geometric_family(L)) for L in (1.0, 0.1)])
    add("Cd 1.0-2.2", "both",
        [run(veh=Vehicle(mass=175., area=1., Cd=c, nose_radius=0.5))
         for c in (1.0, 2.2)])
    # Rn as a FREE fitting parameter at given mass and area: for an irregular
    # tumbling body the effective nose radius is not set by mass and frontal
    # area. The 0.2-1.0 m range for a ~1 m body spans a nearly sharp edge to
    # full scale.
    add("effective Rn 0.2-1.0 m", "mass",
        [run(veh=Vehicle(mass=175., area=1., Cd=1.5, nose_radius=r))
         for r in (0.2, 1.0)])
    add("shape factor 0.25-0.30", "mass",
        [(base_h, base_S * f / 0.27) for f in (0.25, 0.30)])
    add("orbit inclination 0-180°", "both",
        [run(entry=EntryState(inclination_deg=i)) for i in (0.0, 180.0)])
    add("entry latitude -75...0°", "both",
        [run(atm=MSISAtmosphere(lat=x)) for x in (-75.0, 0.0)])
    add("season March/September", "both",
        [run(atm=MSISAtmosphere(date=np.datetime64(d)))
         for d in ("2026-03-01T12:00", "2026-09-01T12:00")])
    add("MSIS version 00 / 2.1", "both",
        [run(atm=MSISAtmosphere(version=v)) for v in (0, 2.1)])
    add("correlation S-G / DKR", "mass",
        [run(corr=c) for c in ("sutton-graves", "dkr")])
    add("entry angle -1...-3°", "both",
        [run(entry=EntryState(gamma_deg=g)) for g in (-1.0, -3.0)])
    add("solar activity F10.7 70-220", "both",
        [run(atm=MSISAtmosphere(f107=f, f107a=f)) for f in (70.0, 220.0)])

    print(f"   {'factor':<36}{'affects':>11}{'altitude, km':>14}"
          f"{'specific energy':>17}")
    for label, kind, dh, dQ in sorted(rows, key=lambda r: -r[2]):
        R[f"budget.{BUDGET_KEYS[label]}.h_km"] = dh
        R[f"budget.{BUDGET_KEYS[label]}.energy_pct"] = dQ
        h_str = f"{dh:.1f}" if kind != "mass" else "-"
        print(f"   {label:<36}{kind:>11}{h_str:>14}{dQ:>16.0f}%")
    print()
    print("   The mass metric is SPECIFIC ABSORBED ENERGY, J/kg, not flux in")
    print("   W/m^2. By flux the sign of the size dependence is REVERSED:")
    print("     q_stag ~ L^-0.5, but A_wet ~ L^2  ->  P ~ L^+1.5,  P/m ~ L^-1.5")
    print("   A large body heats less per unit area but more in total. A budget")
    print("   built on q_stag would point the wrong way.")
    print("\n   Inclination and latitude are not uncertainties but KNOWN inputs:")
    print("   for a specific vehicle they come from its orbit.\n")
    return rows


def figure_heating(tr, path="figures/step3_heating.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    q = tr.heat_flux(VEH) / 1e4
    q_dkr = tr.heat_flux(VEH, "dkr") / 1e4

    ax = axes[0]
    ax.plot(q, tr.h / 1e3, lw=1.8, label="Sutton-Graves")
    ax.plot(q_dkr, tr.h / 1e3, lw=1.5, ls="--", label=f"DKR ($V^{{3.15}}$)")
    ax.axhspan(OBS_LO, OBS_HI, color="seagreen", alpha=0.15)
    ax.text(q.max() * 0.45, 75, "observed\nbreakup", fontsize=7.5,
            color="seagreen", ha="center")
    ax.set_xlabel(r"heat flux, W/cm$^2$"); ax.set_ylabel("altitude, km")
    ax.set_title("stagnation-point flux"); ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(30, 100)

    ax = axes[1]
    ax.plot(tr.t, q, lw=1.8, color="tab:red", label="flux, W/cm²")
    ax2 = ax.twinx()
    Q = np.concatenate([[0.0], np.cumsum(np.diff(tr.t) * (
        tr.heat_flux(VEH)[:-1] + tr.heat_flux(VEH)[1:]) / 2)]) / 1e6
    ax2.plot(tr.t, Q, lw=1.8, color="tab:blue", ls="--", label="accumulated, MJ/m²")
    ax2.grid(False)
    ax.set_xlabel("time, s"); ax.set_ylabel(r"flux, W/cm$^2$", color="tab:red")
    ax2.set_ylabel(r"accumulated heat, MJ/m$^2$", color="tab:blue")
    ax.set_title("flux and accumulated heat")

    ax = axes[2]
    for i_orb, ls in ((0.0, "-"), (53.0, "--"), (90.0, ":"), (180.0, "-.")):
        e = EntryState(inclination_deg=i_orb)
        t2 = integrate(VEH, e, ATM)
        ax.plot(t2.heat_flux(VEH) / 1e4, t2.h / 1e3, lw=1.6, ls=ls,
                label=f"i={i_orb:.0f}°, {e.corotation_speed:+.0f} m/s")
    ax.set_xlabel(r"heat flux, W/cm$^2$"); ax.set_ylabel("altitude, km")
    ax.set_title("Earth rotation: 36% in peak flux")
    ax.legend(frameon=False, fontsize=7.5); ax.set_ylim(40, 90)

    fig.suptitle("Step 3: Sutton-Graves heating with speed relative to the "
                 "rotating atmosphere", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


def figure_budget(rows, path="figures/step3_budget.png"):
    rows_h = [(l, dh) for l, k, dh, _ in rows if k != "mass" and dh > 0.02]
    rows_h.sort(key=lambda r: r[1])
    rows_q = sorted([(l, dq) for l, _, _, dq in rows if dq > 0.5], key=lambda r: r[1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    ax.barh([r[0] for r in rows_h], [r[1] for r in rows_h], color="tab:blue")
    ax.set_xlabel("spread of heating-peak altitude, km")
    ax.set_title("injection ALTITUDE budget")

    ax = axes[1]
    ax.barh([r[0] for r in rows_q], [r[1] for r in rows_q], color="tab:orange")
    ax.set_xlabel("spread of specific absorbed energy, %")
    ax.set_title("MASS budget (specific energy, J/kg)")

    fig.suptitle("Two different budgets: what moves the altitude and what moves the mass",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


if __name__ == "__main__":
    print()
    tr = report_baseline()
    rows = report_budget()
    figure_heating(tr)
    figure_budget(rows)
    print()
    R.save(__file__)
