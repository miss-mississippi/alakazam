"""Entry trajectory: planar equations of motion and integration.

Coordinates: polar, inertial, state [V, gamma, h, s]. Spherical Earth.
Atmospheric rotation enters only through the air-relative speed in drag and
heating (earth_rotation). There is no lift or side force, so the motion is
STRICTLY planar: a consequence of the assumptions, not an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import H_SCALE_FIT, MU_EARTH, R_EARTH, RHO0_SEA_LEVEL
from .vehicle import EntryState, Vehicle

# State vector indices
I_V, I_GAMMA, I_H, I_S = 0, 1, 2, 3


def eom(t, y, vehicle: Vehicle, atmosphere, v_corot: float = 0.0):
    """Right-hand side.

        dV/dt     = -(1/2) rho V^2 Cd A / m  -  (mu/r^2) sin(gamma)
        dgamma/dt = cos(gamma) * ( V/r  -  mu/(r^2 V) )
        dh/dt     = V sin(gamma)
        ds/dt     = V cos(gamma) * R_E/r

    The second equation is the core of the problem. V/r is the centrifugal
    term, mu/(r^2 V) the gravity term. At near-orbital speed they almost
    cancel, so gamma evolves slowly and the trajectory is sensitive to gamma0.
    """
    V, gamma, h, s = y

    r = R_EARTH + h
    g = MU_EARTH / (r * r)
    rho = atmosphere.density(h)

    # Drag and heating depend on the speed RELATIVE TO THE ATMOSPHERE, which
    # rotates with the Earth. v_corot = omega*R*cos(i), see
    # EntryState.corotation_speed. At i = 53 it is 280 m/s: 3.7% in V and
    # ~11% in heat flux, since q ~ V^3.
    V_rel = V - v_corot

    drag_decel = 0.5 * rho * V_rel * V_rel * vehicle.Cd * vehicle.area / vehicle.mass

    dV = -drag_decel - g * np.sin(gamma)
    dgamma = np.cos(gamma) * (V / r - g / V)
    dh = V * np.sin(gamma)
    ds = V * np.cos(gamma) * R_EARTH / r

    return np.array([dV, dgamma, dh, ds])


def _make_events(h_stop: float, v_stop: float):
    def hit_floor(t, y, *args):
        return y[I_H] - h_stop
    hit_floor.terminal = True
    hit_floor.direction = -1

    def too_slow(t, y, *args):
        return y[I_V] - v_stop
    too_slow.terminal = True
    too_slow.direction = -1

    return [hit_floor, too_slow]


@dataclass
class TrajectoryResult:
    t: np.ndarray
    V: np.ndarray
    gamma: np.ndarray
    h: np.ndarray
    s: np.ndarray
    rho: np.ndarray
    decel: np.ndarray          # total aerodynamic deceleration, m/s^2
    stop_reason: str
    raw: object                # solve_ivp object, if dense_output is needed
    V_rel: np.ndarray = None   # speed relative to the rotating atmosphere

    @property
    def gamma_deg(self):
        return np.rad2deg(self.gamma)

    def peak_decel(self):
        """(a_max [m/s^2], h [m], V [m/s]) at maximum deceleration."""
        i = int(np.argmax(self.decel))
        return self.decel[i], self.h[i], self.V[i]

    @property
    def heat_flux_shape(self):
        """sqrt(rho)*V^3: the SHAPE of the Sutton-Graves heat flux, no constants.

        q = k*sqrt(rho/Rn)*V^3, and k, Rn are constant along the trajectory.
        So the peak-heating ALTITUDE depends neither on k, nor on the nose
        radius, nor on units: it can be found without a heating correlation.
        """
        V = self.V_rel if self.V_rel is not None else self.V
        return np.sqrt(self.rho) * V ** 3

    def heat_flux(self, vehicle, correlation="sutton-graves") -> np.ndarray:
        """Stagnation-point heat flux along the trajectory, W/m^2."""
        from .heating import detra_kemp_riddell_shape, sutton_graves
        V = self.V_rel if self.V_rel is not None else self.V
        f = sutton_graves if correlation == "sutton-graves" else detra_kemp_riddell_shape
        return f(self.rho, V, vehicle.nose_radius)

    def heat_load(self, vehicle, correlation="sutton-graves") -> float:
        """Integrated stagnation-point heat load, J/m^2.

        NOTE: this is NOT a proxy for ablated mass. Mass is set by the total
        absorbed energy absorbed_energy(), which depends on the area, and the
        area scales with Rn. In q_stag the sign of the Rn dependence is the
        opposite of the correct one. Kept as a diagnostic.
        """
        return float(np.trapezoid(self.heat_flux(vehicle, correlation), self.t))

    def absorbed_power(self, vehicle, correlation="sutton-graves",
                       T_wall=None) -> np.ndarray:
        """Total absorbed power along the trajectory, W."""
        from .heating import absorbed_power, hot_wall_factor
        q = self.heat_flux(vehicle, correlation)
        if T_wall is not None:
            V = self.V_rel if self.V_rel is not None else self.V
            q = q * hot_wall_factor(V, T_wall)
        return absorbed_power(q, vehicle.wetted)

    def absorbed_energy(self, vehicle, correlation="sutton-graves",
                        T_wall=None) -> float:
        """Total energy absorbed during entry, J. THIS is the ablated-mass proxy."""
        return float(np.trapezoid(
            self.absorbed_power(vehicle, correlation, T_wall), self.t))

    def specific_energy(self, vehicle, correlation="sutton-graves",
                        T_wall=None) -> float:
        """Absorbed energy per kilogram, J/kg.

        The quantity to compare with the effective enthalpy of ablation: it
        decides whether the material vaporizes or only heats up.
        """
        return self.absorbed_energy(vehicle, correlation, T_wall) / vehicle.mass

    def peak_heating(self):
        """(h [m], V [m/s], t [s]) at the maximum of sqrt(rho)*V^3."""
        i = int(np.argmax(self.heat_flux_shape))
        return self.h[i], self.V[i], self.t[i]

    def state_at_altitude(self, h_target: float, inclination_deg: float = 53.0):
        """EntryState at a given altitude: the restart point after breakup.

        Inclination must be passed explicitly: it cannot be recovered from
        the state [V, gamma, h, s], and the co-rotation drift depends on it.
        """
        i = int(np.argmin(np.abs(self.h - h_target)))
        return EntryState(altitude=float(self.h[i]),
                          velocity=float(self.V[i]),
                          gamma_deg=float(self.gamma_deg[i]),
                          inclination_deg=inclination_deg)


def integrate(
    vehicle: Vehicle,
    entry: EntryState,
    atmosphere,
    h_stop: float = 30.0e3,
    v_stop: float = 300.0,
    earth_rotation: bool = True,
    t_max: float = 3000.0,
    max_step: float = 2.0,
    rtol: float = 1e-8,
) -> TrajectoryResult:
    """Integrate the entry from the initial conditions to h_stop or v_stop.

    max_step = 2 s is a safeguard, not accuracy. During the first ~100 s
    drag is negligible, the integrator grows its step to hundreds of seconds
    and risks jumping over the onset of drag near 100 km.

    v_stop = 300 m/s: below Mach ~1 the hypersonic physics (including the
    Sutton-Graves correlation) no longer applies.

    atol is per component: speed in m/s, angle in rad, distances in m. A
    single scalar for quantities of different scales would be wrong.
    """
    y0 = entry.to_vector()
    atol = np.array([1e-3, 1e-9, 1e-3, 1e-3])
    v_corot = entry.corotation_speed if earth_rotation else 0.0

    sol = solve_ivp(
        eom,
        t_span=(0.0, t_max),
        y0=y0,
        args=(vehicle, atmosphere, v_corot),
        method="DOP853",
        events=_make_events(h_stop, v_stop),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True,
    )

    # Resample onto a uniform fine grid via dense_output. Without this the
    # argmax over solver nodes depends on max_step: at max_step = 5 the peak
    # deceleration altitude moves by ~1 km purely from grid resolution.
    t = np.arange(0.0, sol.t[-1], 0.1)
    if t.size == 0 or t[-1] < sol.t[-1]:
        t = np.append(t, sol.t[-1])
    V, gamma, h, s = sol.sol(t)

    rho = atmosphere.density(h)
    V_rel = V - v_corot
    decel = 0.5 * rho * V_rel ** 2 * vehicle.Cd * vehicle.area / vehicle.mass

    if sol.t_events[0].size:
        reason = f"reached altitude {h_stop/1e3:.0f} km"
    elif sol.t_events[1].size:
        reason = f"speed dropped to {v_stop:.0f} m/s"
    else:
        reason = f"time limit {t_max:.0f} s reached"

    return TrajectoryResult(t, V, gamma, h, s, rho, decel, reason, sol, V_rel)


# ---------------------------------------------------------------------------
# Allen-Eggers analytics: for cross-checking only, not used in the model.
# ---------------------------------------------------------------------------

def allen_eggers(vehicle: Vehicle, entry: EntryState,
                 rho0: float = RHO0_SEA_LEVEL, H: float = H_SCALE_FIT) -> dict:
    """Classic closed-form solution (Allen & Eggers, NACA TR-1381, 1958).

    Assumptions: gamma = const, drag >> gravity, exponential atmosphere,
    constant ballistic coefficient.

    rho0 and H are passed EXPLICITLY rather than taken from the atmosphere
    object: A-E is defined only for an exponential atmosphere, and the
    NRLMSISE-00 wrapper has no rho0 of its own. The check is deliberately
    tied to the placeholder; on a real atmosphere the formula does not apply.

    Our case violates the FIRST TWO assumptions: gamma changes several-fold,
    and the first 100 s of flight are essentially drag-free. So the peak
    altitude with gamma0 is expected to be off, and it should agree once the
    actual gamma at the peak is substituted. A discrepancy that survives that
    substitution would indicate a bug.

    Three targets:
      V*    = V0/sqrt(e) = 0.6065*V0   - independent of EVERYTHING
      a_max = V0^2 sin|gamma| / (2 e H) - independent of beta
      rho*  = beta sin|gamma| / H      -> h* = -H ln(rho*/rho0)
    """
    V0 = entry.velocity
    sin_g = abs(np.sin(entry.gamma_rad))
    beta = vehicle.ballistic_coefficient

    v_at_peak = V0 / np.sqrt(np.e)
    a_max = V0 ** 2 * sin_g / (2 * np.e * H)
    rho_at_peak = beta * sin_g / H
    h_at_peak = -H * np.log(rho_at_peak / rho0)

    return {
        "V_at_peak": v_at_peak,
        "a_max": a_max,
        "rho_at_peak": rho_at_peak,
        "h_at_peak": h_at_peak,
    }
