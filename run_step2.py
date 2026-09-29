"""Step 2: NRLMSIS instead of the exponential placeholder.

Run:     python run_step2.py

Output:  figures/step2_atmosphere.png, figures/step2_trajectory.png and
         tables in the console.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import (EntryState, ExponentialAtmosphere, MSISAtmosphere,
                     Vehicle, integrate)
from reentry.constants import G0
from reentry.results import Recorder

# Steps 1-2 are defined for a NON-ROTATING Earth: atmospheric rotation is
# introduced in step 3. integrate() turns it on by default, so it is switched
# off explicitly here; otherwise the step 1-2 numbers do not reproduce.
NO_ROT = dict(earth_rotation=False)
R = Recorder("step2")


plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

VEH = Vehicle(mass=175.0, area=1.0, Cd=1.5)     # Cd = 1.5: tumbling body
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)


def table_density():
    print("A. DENSITY: PLACEHOLDER VS MSIS")
    exp = ExponentialAtmosphere()
    m00 = MSISAtmosphere(version=0)
    m21 = MSISAtmosphere(version=2.1)
    print(f"   {'h, km':>7}{'exp':>12}{'MSISE-00':>12}{'MSIS 2.1':>12}"
          f"{'exp/00':>9}{'21/00':>9}{'local H, km':>13}")
    for h in (40, 50, 60, 70, 80, 90, 100, 110, 120):
        e = float(exp.density(h * 1e3))
        a = float(m00.density(h * 1e3))
        b = float(m21.density(h * 1e3))
        H = float(m00.scale_height(h * 1e3)) / 1e3
        R[f"density.h{h}.exp"] = e
        R[f"density.h{h}.m00"] = a
        R[f"density.h{h}.m21"] = b
        R[f"density.h{h}.exp_over_00"] = e / a
        R[f"density.h{h}.m21_over_00"] = b / a
        R[f"density.h{h}.H_km"] = H
        print(f"   {h:>7}{e:>12.3e}{a:>12.3e}{b:>12.3e}"
              f"{e/a:>9.2f}{b/a:>9.2f}{H:>13.2f}")
    Hs = m00.scale_height(np.arange(40e3, 121e3, 1e3)) / 1e3
    R["density.H_min_km"] = Hs.min()
    R["density.H_max_km"] = Hs.max()
    R["density.H_spread_pct"] = 100 * (Hs.max() - Hs.min()) / 7.2
    print(f"\n   local scale height at 40-120 km: {Hs.min():.1f}-{Hs.max():.1f} km")
    print(f"   the placeholder used a single 7.2 km constant -> "
          f"spread {100*(Hs.max()-Hs.min())/7.2:.0f}%")
    print("   MSIS 2.1 vs MSISE-00: difference up to "
          f"{max(abs(1-float(m21.density(h*1e3))/float(m00.density(h*1e3))) for h in range(40,121,10))*100:.0f}%\n")
    return exp, m00, m21


def table_trajectory(exp, m00, m21):
    print("B. WHAT CHANGED IN THE TRAJECTORY")
    print(f"   {'atmosphere':<14}{'t, s':>9}{'range, km':>12}"
          f"{'h decel, km':>13}{'max g':>9}{'h heating, km':>15}")
    out = {}
    for label, atm in (("exponential", exp), ("MSISE-00", m00), ("MSIS 2.1", m21)):
        tr = integrate(VEH, ENTRY, atm, **NO_ROT)
        a, h_a, _ = tr.peak_decel()
        h_q, _, _ = tr.peak_heating()
        out[label] = tr
        key = {"exponential": "exp", "MSISE-00": "m00", "MSIS 2.1": "m21"}[label]
        R[f"traj.{key}.t"] = tr.t[-1]
        R[f"traj.{key}.range_km"] = tr.s[-1] / 1e3
        R[f"traj.{key}.h_decel_km"] = h_a / 1e3
        R[f"traj.{key}.amax_g"] = a / G0
        R[f"traj.{key}.h_heat_km"] = h_q / 1e3
        print(f"   {label:<14}{tr.t[-1]:>9.0f}{tr.s[-1]/1e3:>12.0f}"
              f"{h_a/1e3:>13.1f}{a/G0:>9.1f}{h_q/1e3:>15.1f}")
    da = (out["MSISE-00"].peak_decel()[1] - out["exponential"].peak_decel()[1]) / 1e3
    dq = (out["MSISE-00"].peak_heating()[0] - out["exponential"].peak_heating()[0]) / 1e3
    dt = out["MSISE-00"].t[-1] - out["exponential"].t[-1]
    print(f"\n   MSIS shift relative to the placeholder: deceleration peak {da:+.1f} km, "
          f"heating peak {dq:+.1f} km, time {dt:+.0f} s\n")
    return out


def table_solar():
    print("C. SOLAR ACTIVITY AND A GEOMAGNETIC STORM")
    print("   F10.7: ~70 solar minimum, ~140 moderate, ~220 maximum")
    print("   Ap: 4 quiet, 80 strong storm\n")
    print(f"   {'scenario':<22}{'rho(120km)':>12}{'rho(70km)':>12}"
          f"{'t, s':>8}{'h decel':>10}{'h heat':>10}")
    cases = [
        ("minimum F10.7=70", dict(f107=70., f107a=70., ap=4.)),
        ("moderate F10.7=140", dict(f107=140., f107a=140., ap=4.)),
        ("maximum F10.7=220", dict(f107=220., f107a=220., ap=4.)),
        ("storm Ap=80", dict(f107=140., f107a=140., ap=80.)),
    ]
    res = {}
    for label, kw in cases:
        atm = MSISAtmosphere(version=0, **kw)
        tr = integrate(VEH, ENTRY, atm, **NO_ROT)
        a, h_a, _ = tr.peak_decel()
        h_q, _, _ = tr.peak_heating()
        res[label] = (float(atm.density(120e3)), float(atm.density(70e3)),
                      tr.t[-1], h_a, h_q)
        print(f"   {label:<22}{res[label][0]:>12.2e}{res[label][1]:>12.2e}"
              f"{tr.t[-1]:>8.0f}{h_a/1e3:>10.1f}{h_q/1e3:>10.1f}")
    lo, hi = res["minimum F10.7=70"], res["maximum F10.7=220"]
    R["solar.rho120_ratio"] = hi[0] / lo[0]
    R["solar.rho70_ratio"] = hi[1] / lo[1]
    R["solar.h_heat_shift_km"] = (hi[4] - lo[4]) / 1e3
    print(f"\n   min->max F10.7: density at 120 km x{hi[0]/lo[0]:.1f}, "
          f"at 70 km x{hi[1]/lo[1]:.2f}")
    print(f"   and the heating-peak altitude moves by only "
          f"{(hi[4]-lo[4])/1e3:+.1f} km\n")
    return res


def table_geography():
    print("D. LATITUDE AND SEASON")
    print("   SPOUA (South Pacific Ocean Uninhabited Area) ~ -40 deg\n")
    print(f"   {'scenario':<28}{'rho(70km)':>12}{'h heating, km':>15}")
    for label, kw in [
        ("SPOUA, September (base)", dict(lat=-40., date=np.datetime64("2026-09-01T12:00"))),
        ("SPOUA, March", dict(lat=-40., date=np.datetime64("2026-03-01T12:00"))),
        ("equator, September", dict(lat=0., date=np.datetime64("2026-09-01T12:00"))),
        ("polar, September", dict(lat=-75., date=np.datetime64("2026-09-01T12:00"))),
    ]:
        atm = MSISAtmosphere(version=0, **kw)
        tr = integrate(VEH, ENTRY, atm, **NO_ROT)
        h_q, _, _ = tr.peak_heating()
        key = {"SPOUA, September (base)": "base", "SPOUA, March": "march",
               "equator, September": "equator", "polar, September": "polar"}[label]
        R[f"geo.{key}.h_heat_km"] = h_q / 1e3
        print(f"   {label:<28}{float(atm.density(70e3)):>12.2e}{h_q/1e3:>15.1f}")
    print()


def figure_atmosphere(exp, m00, m21, path="figures/step2_atmosphere.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    hh = np.linspace(20e3, 140e3, 600)

    ax = axes[0]
    ax.semilogx(exp.density(hh), hh / 1e3, lw=1.6, ls="--", label="exponential (step 1)")
    ax.semilogx(m00.density(hh), hh / 1e3, lw=1.8, label="NRLMSISE-00")
    ax.set_xlabel(r"density, kg/m$^3$"); ax.set_ylabel("altitude, km")
    ax.set_title("density profile"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(exp.density(hh) / m00.density(hh), hh / 1e3, lw=1.8, ls="--",
            label="exponential / MSISE-00")
    ax.plot(m21.density(hh) / m00.density(hh), hh / 1e3, lw=1.8,
            label="MSIS 2.1 / MSISE-00")
    ax.axvline(1.0, color="k", lw=0.8)
    ax.axhspan(40, 90, color="seagreen", alpha=0.12)
    ax.text(2.2, 65, "ablation zone", fontsize=8, color="seagreen")
    ax.set_xlabel("ratio to MSISE-00"); ax.set_ylabel("altitude, km")
    ax.set_xlim(0.5, 4.0)
    ax.set_title("where the placeholder was off"); ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(m00.scale_height(hh) / 1e3, hh / 1e3, lw=1.8, label="MSISE-00, local")
    ax.axvline(7.2, color="crimson", ls="--", lw=1.3, label="placeholder, 7.2 km")
    ax.set_xlabel("scale height, km"); ax.set_ylabel("altitude, km")
    ax.set_title(r"$H = -1/(d\ln\rho/dh)$ is not a constant")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Step 2: a real atmosphere instead of a one-parameter exponential",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


def figure_trajectory(trs, path="figures/step2_trajectory.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    styles = {"exponential": dict(ls="--", color="tab:red"),
              "MSISE-00": dict(ls="-", color="tab:blue"),
              "MSIS 2.1": dict(ls=":", color="tab:green")}
    for label, tr in trs.items():
        st = styles[label]
        axes[0].plot(tr.t, tr.h / 1e3, lw=1.7, label=label, **st)
        axes[1].plot(tr.decel / G0, tr.h / 1e3, lw=1.7, label=label, **st)
        q = tr.heat_flux_shape / tr.heat_flux_shape.max()
        axes[2].plot(q, tr.h / 1e3, lw=1.7, label=label, **st)

    axes[0].set_xlabel("time, s"); axes[0].set_ylabel("altitude, km")
    axes[0].set_title("h(t)")
    axes[1].set_xlabel("deceleration, g"); axes[1].set_ylabel("altitude, km")
    axes[1].set_title("deceleration vs altitude")
    axes[2].set_xlabel(r"heating $\sqrt{\rho}V^3$, normalized")
    axes[2].set_ylabel("altitude, km")
    axes[2].set_title("heat-flux shape")
    axes[2].axhspan(70, 80, color="seagreen", alpha=0.15)
    axes[2].text(0.35, 75, "observed breakup", fontsize=7.5, color="seagreen")
    axes[2].set_ylim(30, 100)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Trajectory on a real atmosphere: there is a shift, but a small one; "
                 "the placeholder was accurate where it matters", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


if __name__ == "__main__":
    print()
    exp, m00, m21 = table_density()
    trs = table_trajectory(exp, m00, m21)
    table_solar()
    table_geography()
    figure_atmosphere(exp, m00, m21)
    figure_trajectory(trs)
    print()
    R.save(__file__)
