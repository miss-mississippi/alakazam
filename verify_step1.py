"""Step 1 checks. Run: python verify_step1.py  (or pytest)

1. Kepler test: without drag the specific energy and angular momentum are
   conserved. A direct test of the ODE right-hand side.
2. Convergence: tighten rtol and max_step; the answer must not move.
3. Allen-Eggers: with gamma0 it does not match, with the actual gamma at the
   peak it matches in altitude. So it is the gamma = const assumption that
   fails, not the equations.
4. Sensitivity to beta: h* ~ H ln(beta), doubling beta lowers the peak by
   H ln2.
"""

from __future__ import annotations

import numpy as np

from reentry import EntryState, ExponentialAtmosphere, Vehicle, allen_eggers, integrate
from reentry.constants import G0, H_SCALE_FIT, MU_EARTH, R_EARTH
from reentry.results import Recorder

# Steps 1-2 are defined for a non-rotating Earth; integrate() turns rotation
# on by default, so it is switched off explicitly here.
NO_ROT = dict(earth_rotation=False)
R = Recorder("verify_step1")


class Vacuum:
    """Atmosphere with zero density, for the Kepler test."""
    name = "vacuum"
    rho0 = 1.225

    def density(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def scale_height(self, h):
        return 7.2e3


def test_kepler():
    print("1. KEPLER TEST (drag switched off)")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)
    traj = integrate(veh, entry, Vacuum(), h_stop=30e3, v_stop=0.0, rtol=1e-11,
                     **NO_ROT)

    r = R_EARTH + traj.h
    energy = 0.5 * traj.V ** 2 - MU_EARTH / r          # specific energy, J/kg
    angmom = traj.V * r * np.cos(traj.gamma)            # specific angular momentum, m^2/s

    de = np.ptp(energy) / abs(energy[0])
    dl = np.ptp(angmom) / abs(angmom[0])
    print(f"   specific energy drift        {de:.3e}  (relative)")
    print(f"   angular momentum drift       {dl:.3e}\n")
    R["kepler.energy_drift"] = de
    R["kepler.angmom_drift"] = dl
    assert de < 1e-9 and dl < 1e-9, f"drift {de:.1e}, {dl:.1e}"


def test_convergence():
    print("2. CONVERGENCE IN STEP AND TOLERANCE")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(gamma_deg=-1.5)
    atm = ExponentialAtmosphere()
    print(f"   {'rtol':>8}{'max_step':>10}{'t_end, s':>12}"
          f"{'peak h, km':>13}{'max g':>10}")
    hs, ts = [], []
    for rtol, ms in ((1e-6, 5.0), (1e-8, 2.0), (1e-10, 0.5), (1e-12, 0.25)):
        tr = integrate(veh, entry, atm, rtol=rtol, max_step=ms, **NO_ROT)
        a, hp, _ = tr.peak_decel()
        hs.append(hp); ts.append(tr.t[-1])
        print(f"   {rtol:>8.0e}{ms:>10.2f}{tr.t[-1]:>12.2f}"
              f"{hp/1e3:>13.3f}{a/G0:>10.3f}")
    print()
    R["convergence.h_peak_km"] = hs[-1] / 1e3
    R["convergence.h_spread_m"] = max(hs) - min(hs)
    assert max(hs) - min(hs) < 1.0, f"peak altitude moves by {max(hs)-min(hs):.2f} m"
    assert max(ts) - min(ts) < 0.05, f"time moves by {max(ts)-min(ts):.3f} s"


def test_allen_eggers():
    print("3. ALLEN-EGGERS: why the direct comparison does not match")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(gamma_deg=-1.5)
    atm = ExponentialAtmosphere()
    traj = integrate(veh, entry, atm, **NO_ROT)

    i = int(np.argmax(traj.decel))
    gamma_peak_deg = traj.gamma_deg[i]
    a_num, h_num, v_num = traj.peak_decel()

    ae0 = allen_eggers(veh, entry)
    ae1 = allen_eggers(veh, EntryState(gamma_deg=gamma_peak_deg))

    print(f"   gamma at the start             {entry.gamma_deg:+7.2f} deg")
    print(f"   gamma at peak deceleration     {gamma_peak_deg:+7.2f} deg\n")
    print(f"   {'quantity':<24}{'numerical':>11}{'A-E(g0)':>11}{'A-E(g peak)':>13}")
    print(f"   {'max deceleration, g':<24}{a_num/G0:>11.2f}"
          f"{ae0['a_max']/G0:>11.2f}{ae1['a_max']/G0:>13.2f}")
    print(f"   {'peak altitude, km':<24}{h_num/1e3:>11.2f}"
          f"{ae0['h_at_peak']/1e3:>11.2f}{ae1['h_at_peak']/1e3:>13.2f}")
    print(f"   {'speed at peak, m/s':<24}{v_num:>11.0f}"
          f"{ae0['V_at_peak']:>11.0f}{'--':>13}\n")
    R["ae.gamma_peak_deg"] = gamma_peak_deg
    R["ae.num.amax_g"] = a_num / G0
    R["ae.num.h_km"] = h_num / 1e3
    R["ae.num.V"] = v_num
    R["ae.g0.amax_g"] = ae0["a_max"] / G0
    R["ae.g0.h_km"] = ae0["h_at_peak"] / 1e3
    R["ae.g0.V"] = ae0["V_at_peak"]
    R["ae.gpeak.amax_g"] = ae1["a_max"] / G0
    R["ae.gpeak.h_km"] = ae1["h_at_peak"] / 1e3
    R["ae.amax_dev_pct"] = 100 * (a_num / ae0["a_max"] - 1)
    dh = abs(h_num - ae1["h_at_peak"])
    assert dh < 0.5e3, f"with the actual gamma the altitude is off by {dh/1e3:.2f} km"
    assert abs(h_num - ae0["h_at_peak"]) > 5e3, "a mismatch was expected with gamma0"


def test_beta_sensitivity():
    print("4. SENSITIVITY TO THE BALLISTIC COEFFICIENT")
    shift = H_SCALE_FIT * np.log(2) / 1e3
    print(f"   prediction: h* ~ H*ln(beta), doubling beta lowers the peak by {shift:.1f} km")
    entry = EntryState(gamma_deg=-1.5)
    atm = ExponentialAtmosphere()
    print(f"   {'beta, kg/m^2':>14}{'peak h, km':>13}{'max g':>10}")
    hs, gs = [], []
    for k, area in enumerate((4.0, 2.0, 1.0, 0.5)):
        veh = Vehicle(mass=175.0, area=area, Cd=1.0)
        tr = integrate(veh, entry, atm, **NO_ROT)
        a, hp, _ = tr.peak_decel()
        delta = "" if not hs else f"   ({(hp - hs[-1])/1e3:+.1f} km)"
        print(f"   {veh.ballistic_coefficient:>14.1f}{hp/1e3:>13.2f}{a/G0:>10.2f}{delta}")
        R[f"beta.{k}.beta"] = veh.ballistic_coefficient
        R[f"beta.{k}.h_km"] = hp / 1e3
        R[f"beta.{k}.amax_g"] = a / G0
        hs.append(hp); gs.append(a)
    print()
    steps = -np.diff(hs) / 1e3
    for k, s in enumerate(steps):
        R[f"beta_step.{k}"] = -s
    assert np.all(np.abs(steps - shift) < 0.3), f"steps {steps} vs {shift:.2f}"
    assert max(gs) / min(gs) - 1 < 0.10, "peak g must not depend on beta"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
