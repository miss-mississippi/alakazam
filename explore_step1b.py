"""Step 1 diagnostics: where injection actually happens. Run: python explore_step1b.py

NOT an implementation of step 4: what-if runs on the existing trajectory model
to check three things before changing the build plan:

  A. Where HEATING peaks (not deceleration). The altitude of the
     sqrt(rho)*V^3 peak depends neither on the Sutton-Graves k nor on the
     nose radius, so it can be computed already.
  B. Spread over Cd = 1.0 / 1.5 / 2.2.
  C. Whether fragmentation closes the 25-30 km gap to the observed 70-80 km.

Observational reference: ATV-1 broke up at 74 km, Cygnus OA6 at 70 km,
Cluster II (SALSA) at 80 km (Ferreira, UNOOSA/IAF 2024). SESAM defaults:
panel separation at 95 km, main breakup at 78 km (Lips, SDC6).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import EntryState, ExponentialAtmosphere, Vehicle, integrate
from reentry.constants import G0
from reentry.results import Recorder

# Steps 1-2 are defined for a NON-ROTATING Earth: atmospheric rotation is
# introduced in step 3. integrate() turns it on by default, so it is switched
# off explicitly here; otherwise the step 1-2 numbers do not reproduce.
NO_ROT = dict(earth_rotation=False)
R = Recorder("step1b")


ATM = ExponentialAtmosphere()
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)

OBSERVED_LO, OBSERVED_HI = 70.0, 80.0   # km, observed breakup events


def part_a_heating_peak():
    print("A. WHERE HEATING PEAKS, NOT DECELERATION")
    print("   q ~ sqrt(rho)*V^3 against a ~ rho*V^2: speed weighs more,")
    print("   density less, so the heating peak is higher and earlier.\n")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    tr = integrate(veh, ENTRY, ATM, **NO_ROT)
    a, h_a, v_a = tr.peak_decel()
    h_q, v_q, t_q = tr.peak_heating()
    print(f"   deceleration peak {h_a/1e3:6.1f} km   V = {v_a:5.0f} m/s   "
          f"({a/G0:.1f} g)")
    print(f"   heating peak      {h_q/1e3:6.1f} km   V = {v_q:5.0f} m/s   "
          f"t = {t_q:.0f} s")
    print(f"   separation        {(h_q - h_a)/1e3:6.1f} km\n")
    R["peaks.decel_h_km"] = h_a / 1e3
    R["peaks.decel_V"] = v_a
    R["peaks.decel_g"] = a / G0
    R["peaks.heat_h_km"] = h_q / 1e3
    R["peaks.heat_V"] = v_q
    R["peaks.separation_km"] = (h_q - h_a) / 1e3
    return h_q


def part_b_cd_sweep():
    print("B. SPREAD OVER Cd")
    print("   1.0: blunt body in continuum, lower bound")
    print("   1.5: randomly tumbling irregular body")
    print("   2.2: free-molecular limit / very irregular body\n")
    print(f"   {'Cd':>5}{'beta, kg/m^2':>14}{'h decel, km':>14}"
          f"{'h heating, km':>16}{'max g':>9}")
    out = {}
    for Cd in (1.0, 1.5, 2.2):
        veh = Vehicle(mass=175.0, area=1.0, Cd=Cd)
        tr = integrate(veh, ENTRY, ATM, **NO_ROT)
        a, h_a, _ = tr.peak_decel()
        h_q, _, _ = tr.peak_heating()
        out[Cd] = (h_a, h_q)
        key = f"cd.cd{Cd*10:.0f}"
        R[f"{key}.beta"] = veh.ballistic_coefficient
        R[f"{key}.h_decel_km"] = h_a / 1e3
        R[f"{key}.h_heat_km"] = h_q / 1e3
        print(f"   {Cd:>5.1f}{veh.ballistic_coefficient:>14.1f}"
              f"{h_a/1e3:>14.1f}{h_q/1e3:>16.1f}{a/G0:>9.1f}")
    span_a = (out[2.2][0] - out[1.0][0]) / 1e3
    span_q = (out[2.2][1] - out[1.0][1]) / 1e3
    R["cd.span_decel_km"] = span_a
    R["cd.span_heat_km"] = span_q
    print(f"\n   altitude spread: deceleration {span_a:+.1f} km, "
          f"heating {span_q:+.1f} km")
    print("   -> the Cd uncertainty costs less than the gap to 70-80 km\n")
    return out


def fragment_run(beta_ratio: float, h_frag: float, Cd: float = 1.5):
    """Intact object down to h_frag, then fragments with beta / beta_ratio.

    For geometrically similar fragmentation m ~ L^3 and A ~ L^2, so
    beta = m/(Cd A) ~ L. Pieces ten times smaller in linear size have ten
    times smaller beta.
    """
    intact = Vehicle(mass=175.0, area=1.0, Cd=Cd)
    tr0 = integrate(intact, ENTRY, ATM, h_stop=h_frag, **NO_ROT)
    restart = tr0.state_at_altitude(h_frag)

    # Keep the mass, cut beta through the area: A_eff = A * beta_ratio
    frag = Vehicle(mass=175.0, area=1.0 * beta_ratio, Cd=Cd)
    tr1 = integrate(frag, restart, ATM, **NO_ROT)
    return tr0, tr1, frag


def part_c_fragmentation():
    print("C. DOES FRAGMENTATION CLOSE THE GAP")
    print(f"   observations: breakup at {OBSERVED_LO:.0f}-{OBSERVED_HI:.0f} km "
          "(ATV-1 74, Cygnus OA6 70, SALSA 80)")
    print("   SESAM defaults: panels 95 km, main body 78 km\n")

    print("   C1. Breakup at 78 km, different fragmentation (Cd = 1.5):")
    print(f"   {'fragment beta':>14}{'ratio':>12}{'h heating, km':>16}"
          f"{'in window?':>12}")
    for ratio in (1, 2, 5, 10, 20, 50):
        _, tr1, frag = fragment_run(ratio, 78e3)
        h_q, _, _ = tr1.peak_heating()
        inside = OBSERVED_LO <= h_q / 1e3 <= OBSERVED_HI
        clipped = "  <- peak AT breakup" if h_q > 77.9e3 else ""
        R[f"frag.r{ratio}.beta"] = frag.ballistic_coefficient
        R[f"frag.r{ratio}.h_heat_km"] = h_q / 1e3
        print(f"   {frag.ballistic_coefficient:>14.1f}{'1/' + str(ratio):>12}"
              f"{h_q/1e3:>16.1f}{'yes' if inside else '':>12}{clipped}")

    print("\n   C2. Fragmentation by 10, different breakup altitudes:")
    print(f"   {'h breakup, km':>15}{'h heating, km':>16}{'shift':>10}")
    base = None
    for hf in (95e3, 84e3, 78e3, 70e3):
        _, tr1, _ = fragment_run(10, hf)
        h_q, _, _ = tr1.peak_heating()
        if base is None:
            base = h_q
        R[f"breakup.h{hf/1e3:.0f}.h_heat_km"] = h_q / 1e3
        print(f"   {hf/1e3:>15.0f}{h_q/1e3:>16.1f}{(h_q-base)/1e3:>+10.1f}")
    print()

    # Inverse problem: which fragment beta puts the heating peak mid-window
    target = 0.5 * (OBSERVED_LO + OBSERVED_HI) * 1e3
    ratios = np.logspace(0, 2.3, 40)
    heights = []
    for r in ratios:
        _, tr1, _ = fragment_run(float(r), 78e3)
        heights.append(tr1.peak_heating()[0])
    heights = np.array(heights)
    i = int(np.argmin(np.abs(heights - target)))
    beta_needed = 175.0 / (1.5 * ratios[i])
    beta_intact = 175.0 / 1.5
    factor = beta_intact / beta_needed
    R["inverse.beta_needed"] = beta_needed
    R["inverse.factor"] = factor
    R["inverse.n_fragments"] = factor ** 3
    print(f"   C3. Inverse problem: for the heating peak to sit at {target/1e3:.0f} km,")
    print(f"       a fragment needs beta ~ {beta_needed:.1f} kg/m^2 against "
          f"{beta_intact:.1f} for the intact body,")
    print(f"       i.e. a {factor:.1f}x drop in beta.")
    print(f"       For geometrically similar fragmentation beta ~ L and L ~ N^(-1/3),")
    print(f"       so N ~ {factor**3:.0f} equal fragments would be needed.")
    print(f"       That is a lot. Real fragmentation gives a spectrum of beta, not")
    print(f"       equal pieces: panels and MLI have beta ~1-10, structure ~30-80.\n")

    print("   IMPORTANT CAVEAT: the sqrt(rho)*V^3 peak is an UPPER BOUND on the")
    print("   injection altitude, not the injection itself. Vaporization needs")
    print("   ACCUMULATED heat: heating to ~900 K, then melting, then boiling")
    print("   (~2000 K at the local stagnation pressure, 2792 K at 1 atm).")
    print("   The mass will be injected below the flux peak; step 4 shows how far.")
    print("   So tuning beta to put the flux peak exactly at 70-80 km would be")
    print("   a mistake: the mass itself would then end up too low.\n")
    return ratios, heights


def figure(ratios, heights, path="figures/step1b_fragmentation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # 1: heating vs deceleration shape for the intact object
    ax = axes[0]
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.5)
    tr = integrate(veh, ENTRY, ATM, **NO_ROT)
    q = tr.heat_flux_shape / tr.heat_flux_shape.max()
    a = tr.decel / tr.decel.max()
    ax.plot(q, tr.h / 1e3, lw=1.7, label=r"heating $\sqrt{\rho}V^3$")
    ax.plot(a, tr.h / 1e3, lw=1.7, ls="--", label=r"deceleration $\rho V^2$")
    ax.axhspan(OBSERVED_LO, OBSERVED_HI, color="seagreen", alpha=0.15)
    ax.text(0.5, (OBSERVED_LO + OBSERVED_HI) / 2, "observed\nbreakup",
            fontsize=7.5, color="seagreen", ha="center", va="center")
    ax.set_xlabel("normalized to maximum"); ax.set_ylabel("altitude, km")
    ax.set_title("intact object: both peaks\nfar below observations", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(30, 120)

    # 2: fragment trajectories for different fragmentation
    ax = axes[1]
    for ratio, color in zip((1, 5, 20, 50),
                            ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        _, tr1, frag = fragment_run(ratio, 78e3)
        qq = tr1.heat_flux_shape / tr1.heat_flux_shape.max()
        ax.plot(qq, tr1.h / 1e3, lw=1.6, color=color,
                label=fr"$\beta$={frag.ballistic_coefficient:.0f}")
    ax.axhspan(OBSERVED_LO, OBSERVED_HI, color="seagreen", alpha=0.15)
    ax.axhline(78, color="k", ls=":", lw=1)
    ax.text(0.02, 79, "breakup at 78 km", fontsize=7.5)
    ax.set_xlabel("heating, normalized"); ax.set_ylabel("altitude, km")
    ax.set_title("fragments after breakup at 78 km", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_ylim(30, 100)

    # 3: heating-peak altitude as a function of fragment beta
    ax = axes[2]
    betas = 175.0 / (1.5 * ratios)
    ax.semilogx(betas, heights / 1e3, lw=1.8)
    ax.axhspan(OBSERVED_LO, OBSERVED_HI, color="seagreen", alpha=0.15,
               label="observations 70-80 km")
    ax.plot(175 / 1.5, heights[0] / 1e3, "o", color="crimson", ms=7,
            label="intact object")
    ax.set_xlabel(r"fragment $\beta$, kg/m$^2$")
    ax.set_ylabel("heating-peak altitude, km")
    ax.set_title(r"slope $H\ln\beta$, then a plateau" "\n" r"at the breakup altitude", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Diagnostics: fragmentation as the mechanism that raises the injection altitude",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


if __name__ == "__main__":
    print()
    part_a_heating_peak()
    part_b_cd_sweep()
    ratios, heights = part_c_fragmentation()
    figure(ratios, heights)
    R.save(__file__)
