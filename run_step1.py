"""Step 1: exponential atmosphere + trajectory integration from 120 km.

Run:     python run_step1.py

Output:  figures/step1_trajectory.png, figures/step1_gamma_sweep.png and a
         comparison table in the console.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import EntryState, ExponentialAtmosphere, Vehicle, allen_eggers, integrate
from reentry.constants import G0
from reentry.results import Recorder

# Steps 1-2 are defined for a NON-ROTATING Earth: atmospheric rotation is
# introduced in step 3. integrate() turns it on by default, so it is switched
# off explicitly here; otherwise the step 1-2 numbers do not reproduce.
NO_ROT = dict(earth_rotation=False)
R = Recorder("step1")


plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def print_report(traj, ae, vehicle, entry, atmosphere):
    a_num, h_num, v_num = traj.peak_decel()

    print()
    print("=" * 68)
    print("STEP 1: TRAJECTORY, EXPONENTIAL ATMOSPHERE (PLACEHOLDER)")
    print("=" * 68)
    print(f"  mass               {vehicle.mass:8.1f} kg")
    print(f"  Cd * A             {vehicle.Cd * vehicle.area:8.2f} m^2")
    print(f"  beta = m/(Cd A)    {vehicle.ballistic_coefficient:8.1f} kg/m^2")
    print(f"  entry              h={entry.altitude/1e3:.0f} km, "
          f"V={entry.velocity:.0f} m/s, gamma={entry.gamma_deg:+.2f} deg")
    print(f"  atmosphere         {atmosphere.name}, "
          f"rho0={atmosphere.rho0} kg/m^3, H={atmosphere.H/1e3:.1f} km")
    print()
    print(f"  stop:              {traj.stop_reason}")
    print(f"  duration:          {traj.t[-1]:8.1f} s")
    print(f"  downrange:         {traj.s[-1]/1e3:8.0f} km")
    i_peak = int(np.argmax(traj.decel))
    print(f"  gamma at peak:     {traj.gamma_deg[i_peak]:+8.2f} deg "
          f"(initially {entry.gamma_deg:+.2f})")
    print(f"  gamma at end:      {traj.gamma_deg[-1]:+8.2f} deg")
    print(f"  V at end:          {traj.V[-1]:8.0f} m/s "
          f"at {traj.h[-1]/1e3:.1f} km")
    print()
    print("  ALLEN-EGGERS COMPARISON (gamma is not constant and the first")
    print("  ~100 s are drag-free, so A-E with gamma0 is not expected to match)")
    print(f"    {'quantity':<26}{'numerical':>12}{'Allen-Eggers':>15}{'dev.':>10}")
    rows = [
        ("max deceleration, g", a_num / G0, ae["a_max"] / G0),
        ("peak altitude, km", h_num / 1e3, ae["h_at_peak"] / 1e3),
        ("speed at peak, m/s", v_num, ae["V_at_peak"]),
    ]
    R["base.t_end"] = traj.t[-1]
    R["base.range_km"] = traj.s[-1] / 1e3
    R["base.amax_g"] = a_num / G0
    R["base.h_peak_km"] = h_num / 1e3
    R["base.V_peak"] = v_num
    R["base.gamma_peak_deg"] = traj.gamma_deg[i_peak]
    R["base.gamma_end_deg"] = traj.gamma_deg[-1]
    for key, (label, num, ana) in zip(("amax_g", "h_km", "V"), rows):
        R[f"ae.{key}"] = ana
        R[f"ae.{key}_dev_pct"] = 100.0 * (num - ana) / ana
    for label, num, ana in rows:
        dev = 100.0 * (num - ana) / ana
        print(f"    {label:<26}{num:>12.2f}{ana:>15.2f}{dev:>9.0f}%")
    print("=" * 68)
    print()


def figure_main(traj, ae, atmosphere, entry, path="figures/step1_trajectory.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    t = traj.t
    h_km = traj.h / 1e3

    ax = axes[0, 0]
    ax.plot(t, h_km, lw=1.6)
    ax.set_xlabel("time, s"); ax.set_ylabel("altitude, km")
    ax.set_title("h(t)")

    ax = axes[0, 1]
    ax.plot(t, traj.V / 1e3, lw=1.6)
    ax.axhline(ae["V_at_peak"] / 1e3, color="crimson", ls="--", lw=1,
               label=r"A-E: $V_0/\sqrt{e}$")
    ax.set_xlabel("time, s"); ax.set_ylabel("speed, km/s")
    ax.set_title("V(t)"); ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 2]
    ax.plot(t, traj.gamma_deg, lw=1.6)
    ax.axhline(entry.gamma_deg, color="grey", ls=":", lw=1, label="$\\gamma_0$")
    ax.set_xlabel("time, s"); ax.set_ylabel("flight-path angle, deg")
    ax.set_title(r"$\gamma(t)$ goes from $-1.5°$ to $-21°$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(traj.V / 1e3, h_km, lw=1.6)
    ax.set_xlabel("speed, km/s"); ax.set_ylabel("altitude, km")
    ax.set_title("V(h): phase portrait")

    ax = axes[1, 1]
    a_num, h_num, _ = traj.peak_decel()
    ax.plot(traj.decel / G0, h_km, lw=1.6)
    ax.plot(ae["a_max"] / G0, ae["h_at_peak"] / 1e3, "x", color="crimson",
            ms=9, mew=2, label="Allen-Eggers")
    ax.plot(a_num / G0, h_num / 1e3, "o", mfc="none", color="tab:blue",
            ms=9, mew=1.6, label="numerical")
    ax.set_xlabel("aerodynamic deceleration, g"); ax.set_ylabel("altitude, km")
    ax.set_title("deceleration vs altitude"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 2]
    hh = np.linspace(30e3, 120e3, 300)
    ax.semilogx(atmosphere.density(hh), hh / 1e3, lw=1.6, label="placeholder")
    # U.S. Standard Atmosphere 1976 reference points: how far off the placeholder is
    ref_h = np.array([40, 60, 80, 100, 120])
    ref_rho = np.array([4.0e-3, 3.1e-4, 1.85e-5, 5.6e-7, 2.2e-8])
    ax.semilogx(ref_rho, ref_h, "s", color="crimson", ms=5, label="USSA-76")
    ax.set_xlabel(r"density, kg/m$^3$"); ax.set_ylabel("altitude, km")
    ax.set_title("placeholder is accurate where ablation happens")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Step 1: 3-DOF entry trajectory, exponential atmosphere "
                 "(placeholder), constant mass", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved: {path}")


def figure_gamma_sweep(vehicle, atmosphere, path="figures/step1_gamma_sweep.png"):
    """Sensitivity to the entry angle: -1...-3 deg are different regimes."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    print("  sensitivity to the entry angle:")
    print(f"    {'gamma0':>8}{'time, s':>12}{'downrange, km':>16}"
          f"{'max g':>10}{'peak h, km':>13}")

    for g0 in (-1.0, -1.5, -2.0, -3.0):
        entry = EntryState(gamma_deg=g0)
        traj = integrate(vehicle, entry, atmosphere, **NO_ROT)
        a, hp, _ = traj.peak_decel()
        lbl = f"$\\gamma_0$={g0:+.1f}°"
        axes[0].plot(traj.t, traj.h / 1e3, lw=1.5, label=lbl)
        axes[1].plot(traj.t, traj.V / 1e3, lw=1.5, label=lbl)
        axes[2].plot(traj.decel / G0, traj.h / 1e3, lw=1.5, label=lbl)
        key = f"gamma_sweep.g{abs(g0)*10:.0f}"
        R[f"{key}.t"] = traj.t[-1]
        R[f"{key}.range_km"] = traj.s[-1] / 1e3
        R[f"{key}.amax_g"] = a / G0
        R[f"{key}.h_km"] = hp / 1e3
        print(f"    {g0:>+8.1f}{traj.t[-1]:>12.0f}{traj.s[-1]/1e3:>16.0f}"
              f"{a/G0:>10.1f}{hp/1e3:>13.1f}")

    axes[0].set_xlabel("time, s"); axes[0].set_ylabel("altitude, km")
    axes[0].set_title("h(t)")
    axes[1].set_xlabel("time, s"); axes[1].set_ylabel("speed, km/s")
    axes[1].set_title("V(t)")
    axes[2].set_xlabel("deceleration, g"); axes[2].set_ylabel("altitude, km")
    axes[2].set_title("deceleration vs altitude")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sensitivity to the entry angle: flight time differs by 1.6x "
                 "while the peak altitude barely moves:\nthe gravity turn erases "
                 r"$\gamma_0$ before drag turns on", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"\n  saved: {path}")


def main():
    vehicle = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)
    atmosphere = ExponentialAtmosphere()

    traj = integrate(vehicle, entry, atmosphere, h_stop=30e3, **NO_ROT)
    ae = allen_eggers(vehicle, entry)

    print_report(traj, ae, vehicle, entry, atmosphere)
    figure_main(traj, ae, atmosphere, entry)
    print()
    figure_gamma_sweep(vehicle, atmosphere)
    print()
    R.save(__file__)


if __name__ == "__main__":
    main()
