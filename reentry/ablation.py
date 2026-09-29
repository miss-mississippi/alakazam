"""Thermal response and ablation: two bounds instead of one estimate.

MAIN ARCHITECTURAL DECISION. The model does NOT produce a single vaporized
mass. It produces a bracket:

  UPPER BOUND: molten mass (by the maximum enthalpy reached during flight).
  The ORSAT/DRAMA criterion: absorbed energy against heating to the melting
  point plus the heat of fusion (ORSAT uses a heat of ablation of
  934.5 kJ/kg for generic aluminium). This is everything that could in
  principle become oxide.

  VAPORIZED IN PLACE: mass vaporized under the assumption that the melt is
  RETAINED on the fragment until it boils (an oxide skin allows this). Not a
  strict lower bound: if the melt is stripped by the flow, less vaporizes in
  place, and the fate of the droplets is not modelled.

Between the two lies the fate of melt stripped into the flow. A droplet may
vaporize further along the trajectory, oxidize only at its surface and fall
out as a millimetre spherule (as meteoroid ablation spherules are found in
deep-sea sediments), or solidify whole. No current model resolves this, which
is why the bracket is a result and not a compromise.

THE BOILING POINT DEPENDS ON PRESSURE. 2792 K is Al boiling at 1 atm. At the
surface of a fragment the pressure is the stagnation pressure, ~0.4-1 kPa at
70-80 km, where Al boils at ~1900-2050 K (Clausius-Clapeyron). Boiling at
1 atm would overestimate the vaporization threshold eps*sigma*T^4 about
threefold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .heating import CP_AIR_HOT, SIGMA_SB, hot_wall_factor, stagnation_pressure

R_GAS = 8.314462618      # J/(mol*K)
M_AL = 0.026982          # kg/mol
P_ATM = 101325.0         # Pa

# Kirchhoff: dL/dT = c_p(vapour) - c_p(liquid) = 20.79 - 31.75 J/(mol*K)
# (monatomic gas 5/2 R; liquid from NIST Shomate). Per kg: -406 J/(kg*K).
DCP_VAPOUR = (2.5 * R_GAS - 31.751) / M_AL

K_ALUMINIUM = 167.0      # W/(m*K), thermal conductivity of Al 6061 (~10%)
RHO_ALUMINIUM = 2700.0   # kg/m^3

# Relaxation of superheated melt. When the pressure drops, so does T_boil,
# and a band ends up above the new boiling plateau. The excess enthalpy goes
# into vaporization (flashing) with this time constant. Physically this is
# instantaneous; 0.2 s is a numerical regularization with no effect on the
# result (verify_step4).
TAU_FLASH = 0.2          # s


@dataclass
class Aluminium:
    """Al 6061 properties, each with a source and an uncertainty.

    THERMOPHYSICS
      c_p            effective solid heat capacity from 300 K to T_melt: the
                     mean of NIST Shomate for Al(s) (c_p rises from 899 at
                     300 K to 1190 at 890 K); conserves enthalpy exactly.
      c_p_liquid     31.75 J/(mol*K) = 1177 J/(kg*K), NIST Shomate for Al(l).
      T_melt         6061: solidus 855 K, liquidus 925 K (ASM); a single
                     plateau in the middle of the range. 933 K is pure Al.
      L_fusion       397 kJ/kg (pure Al), ~5%.

    VAPORIZATION
      T_boil_1atm    2792 K (CRC Handbook; some references give 2740-2743 K).
      L_vapour       294 kJ/mol = 10.90 MJ/kg at T_boil_1atm (CRC).
      T_boil_at(p)   Clausius-Clapeyron anchored at (2792 K, 1 atm).
                     Against the CRC vapour-pressure table: 100 Pa -> 1817 K,
                     1 kPa -> 2054 K, 10 kPa -> 2364 K, within 2%
                     (verify_step4).
      L_vapour_at(T) Kirchhoff correction, +0.3 MJ/kg at 2000 K.

    EMISSIVITY: the main axis for thin-walled fragments.
      surface = "constant"  eps = emissivity at any T (sweeps).
      surface = "oxide"     optically active film: eps(T) of one spectral
                            curve pinned to alpha-Al2O3 (~0.32 at 2000 K).
      surface = "bare"      no film: eps(T) of the bare metal from its
                            resistivity (~0.17 at 2000 K).
      tau_oxide             film growth: eps(t, T) goes from "bare" to "oxide"
                            as 1 - exp(-sqrt(t/tau)), t is the time since the
                            surface was created (breakup).
    """

    c_p: float = 1038.0
    c_p_liquid: float = 1177.0
    T_melt: float = 890.0
    L_fusion: float = 0.397e6
    T_initial: float = 300.0

    T_boil_1atm: float = 2792.0
    L_vapour: float = 10.90e6
    boil_at_local_pressure: bool = True
    p_nominal: float = 1.0e3      # Pa, for summary estimates (energy, Pi)

    emissivity: float = 0.10
    surface: str = "constant"
    tau_oxide: float | None = None
    film_shape: tuple = (0.05, 0.95, 1.6)   # eps_short, eps_long, width

    @classmethod
    def legacy(cls, **kw):
        """First-version properties: c_p = 900 throughout, T_melt = 933 K,
        boiling at 2740 K at any pressure, L = 10.5 MJ/kg. Only for the
        "before -> after" comparison."""
        base = dict(c_p=900.0, c_p_liquid=900.0, T_melt=933.0,
                    T_boil_1atm=2740.0, L_vapour=10.5e6,
                    boil_at_local_pressure=False)
        base.update(kw)
        return cls(**base)

    # --- boiling -----------------------------------------------------------

    def T_boil_at(self, p):
        """Boiling temperature at pressure p, K."""
        if not self.boil_at_local_pressure:
            return np.full_like(np.asarray(p, dtype=float), self.T_boil_1atm)
        p = np.maximum(np.asarray(p, dtype=float), 1e-6)
        inv = 1.0 / self.T_boil_1atm - R_GAS / (M_AL * self.L_vapour) * np.log(p / P_ATM)
        return 1.0 / inv

    def L_vapour_at(self, T_boil):
        """Heat of vaporization at boiling temperature T_boil, J/kg."""
        if not self.boil_at_local_pressure:
            return np.full_like(np.asarray(T_boil, dtype=float), self.L_vapour)
        return self.L_vapour + DCP_VAPOUR * (np.asarray(T_boil) - self.T_boil_1atm)

    @property
    def T_boil_nominal(self) -> float:
        """Boiling temperature at p_nominal (1 kPa), for summary estimates."""
        return float(self.T_boil_at(self.p_nominal))

    # --- enthalpy ----------------------------------------------------------

    @property
    def h1(self) -> float:
        """Onset of melting, J/kg."""
        return self.c_p * (self.T_melt - self.T_initial)

    @property
    def h2(self) -> float:
        """End of melting, J/kg."""
        return self.h1 + self.L_fusion

    def h3(self, T_boil):
        """Onset of boiling at a given T_boil, J/kg."""
        return self.h2 + self.c_p_liquid * (np.asarray(T_boil) - self.T_melt)

    @property
    def h_melt_complete(self) -> float:
        """Energy per kilogram to complete melting, J/kg.

        ORSAT criterion: 934.5 kJ/kg for generic aluminium (NTRS 20140016958).
        """
        return self.h2

    @property
    def h_vapour_complete(self) -> float:
        """Energy per kilogram to complete vaporization at p_nominal, J/kg."""
        Tb = self.T_boil_nominal
        return float(self.h3(Tb) + self.L_vapour_at(Tb))

    @property
    def demise_ratio(self) -> float:
        """How many times vaporization costs more than melting."""
        return self.h_vapour_complete / self.h_melt_complete

    # --- emissivity --------------------------------------------------------

    def emissivity_at(self, T, t_since_surface: float = 0.0):
        """Total hemispherical eps at temperature T (array)."""
        from .emissivity import bare_aluminium_emissivity, oxide_film_emissivity
        T = np.asarray(T, dtype=float)
        if self.tau_oxide is not None:
            g = 1.0 - np.exp(-np.sqrt(max(t_since_surface, 0.0) / self.tau_oxide))
            e_b = bare_aluminium_emissivity(T)
            return e_b + (oxide_film_emissivity(*self.film_shape)(T) - e_b) * g
        if self.surface == "oxide":
            return oxide_film_emissivity(*self.film_shape)(T)
        if self.surface == "bare":
            return bare_aluminium_emissivity(T)
        return np.full_like(T, self.emissivity)

    def label(self) -> str:
        if self.tau_oxide is not None:
            return f"film growth tau={self.tau_oxide:g} s"
        return {"oxide": "active oxide film", "bare": "bare melt"}.get(
            self.surface, f"eps={self.emissivity:.2f}")


def temperature_from_enthalpy(h_s, mat: Aluminium, T_boil=None):
    """T(h_s): a monotonic piecewise function with PLATEAUS at phase changes.

    This is how phase changes enter the ODE right-hand side without
    branching: the state is specific enthalpy and temperature is derived
    from it.

        h1 = c_p(solid)*(T_melt - T0)              onset of melting
        h2 = h1 + L_fusion                         end of melting
        h3 = h2 + c_p(liquid)*(T_boil - T_melt)    onset of boiling

    T_boil is the boiling temperature at the current local pressure.
    """
    if T_boil is None:
        T_boil = mat.T_boil_nominal
    h1, h2 = mat.h1, mat.h2
    h3 = mat.h3(T_boil)
    h = np.asarray(h_s, dtype=float)
    return np.where(
        h < h1, mat.T_initial + h / mat.c_p,
        np.where(h < h2, mat.T_melt,
                 np.where(h < h3, mat.T_melt + (h - h2) / mat.c_p_liquid, T_boil)))


# ---------------------------------------------------------------------------
# Regime criteria
# ---------------------------------------------------------------------------

def thermal_diffusion_depth(t_flight: float, k=K_ALUMINIUM,
                            rho=RHO_ALUMINIUM, c_p=900.0) -> float:
    """Heat penetration depth over the flight, m: sqrt(alpha*t), alpha = k/(rho*c_p).

    What this criterion SAYS: temperature has time to equalize through the
    thickness, so a lumped-in-thickness model is valid. For Al alpha = 6.9e-5,
    which gives 14 cm over 300 s.

    What it does NOT say: it does NOT separate the energy-limited and
    radiative regimes. Both 1 mm and 16 mm are much thinner than 14 cm, so the
    lumped model is valid for both, yet they behave oppositely. What separates
    them is the areal heat capacity, see regime_number().
    """
    return float(np.sqrt(k / (rho * c_p) * t_flight))


def areal_mass(vehicle, t_flight: float) -> float:
    """Mass per unit WETTED area taking part in heating, kg/m^2.

    m / A_wet, but not more than rho * penetration depth. For a plate of
    thickness t this is rho*t/2: the wetted area is both faces.
    """
    return min(vehicle.mass / vehicle.wetted,
               RHO_ALUMINIUM * thermal_diffusion_depth(t_flight))


def regime_number(traj, vehicle, material: Aluminium) -> dict:
    """Pi = available energy / energy needed to reach boiling.

    THIS is the regime discriminator, not the penetration depth.

        Pi = phi * int(q_stag dt)  /  (m'' * h_boil)

    m'' is the mass per unit wetted area (rho*t/2 for a plate, m/A_wet for a
    shell), h_boil is the enthalpy from T0 to the onset of boiling at the
    nominal pressure of 1 kPa.

    Pi >> 1: the body reaches radiative equilibrium and eps decides the rest.
    Pi << 1: not enough energy even to heat up; the problem is ENERGY-LIMITED
             and eps barely matters because re-radiation is small.
    """
    from .heating import SHAPE_FACTOR_TUMBLING
    t_flight = float(traj.t[-1] - traj.t[0])
    m_area = areal_mass(vehicle, t_flight)
    h_boil = float(material.h3(material.T_boil_nominal))
    need = m_area * h_boil
    avail = SHAPE_FACTOR_TUMBLING * traj.heat_load(vehicle)
    return {"Pi": avail / need, "areal_mass": m_area,
            "equiv_thickness": m_area / RHO_ALUMINIUM,
            "need": need, "available": avail,
            "regime": "radiative" if avail > need else "energy-limited"}


# ---------------------------------------------------------------------------
# PER-BAND surface model: one consistent scheme
# ---------------------------------------------------------------------------

def surface_thermal_model(traj, vehicle, material: Aluminium,
                          wall_thickness: float | None = None,
                          n_bands: int = 24,
                          correlation: str = "sutton-graves",
                          flux_mode: str = "cos",
                          blowing_eta: float = 0.0,
                          n_out: int = 1200) -> dict:
    """Per-band energy balance over the angular flux distribution.

    The surface is split into bands by the angle theta from the stagnation
    point. Each band:
      - receives q_stag * cos(theta) * (1 - h_w/h_0)
      - re-radiates sides * eps(T) * sigma * T^4 at ITS OWN temperature
        (sides = 2 for a plate: the back face radiates too)
      - has its own participating mass rho*t_wall*dA
      - passes phase changes through plateaus of T(h_s); the boiling plateau
        sits at T_boil(p_stag(t)) and moves down as the pressure drops
      - vaporizes no more than its own mass: a boiled-off band disappears
        (burn-through) and stops receiving and radiating

    Molten and vaporized masses come from ONE temperature field. Melt is
    counted by the MAXIMUM enthalpy of a band over the flight: a band that
    melted and then cooled has still melted. Vaporization happens only above
    the boiling plateau, so in every band vaporized <= molten <= band mass by
    construction, without a forced max().

    flux_mode:
      "cos"      stable orientation: the exposed half of the surface is
                 heated with a cos(theta) distribution, no free parameters.
                 For a shell the leeward half of the mass does not take part,
                 so shell melt is at most 50%.
      "uniform"  fast tumbling: every element of the whole wetted surface
                 receives the mean flux 0.25*q_stag. The limit where the
                 tumbling period is much shorter than the wall's thermal time
                 constant (~3-5 s for a 1 mm plate).

    blowing_eta: flux blockage by vapour injection in boiling bands,
      q_vap = (q_in - rad) / (1 + eta*(h_0 - h_w)/L). 0 disables it (baseline).

    Wall thickness: the plate thickness for a plate; for a compact fragment an
    equivalent shell m/(rho*A_wet). Capped by the penetration depth.
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline

    t0, t1 = float(traj.t[0]), float(traj.t[-1])
    V = traj.V_rel if traj.V_rel is not None else traj.V
    q_spline = CubicSpline(traj.t, traj.heat_flux(vehicle, correlation))
    V_spline = CubicSpline(traj.t, V)
    h_spline = CubicSpline(traj.t, traj.h)
    lnp_spline = CubicSpline(
        traj.t, np.log(np.maximum(stagnation_pressure(traj.rho, V), 1e-9)))

    depth = thermal_diffusion_depth(t1 - t0)
    if flux_mode == "cos":
        if wall_thickness is None:
            wall_thickness = vehicle.mass / (RHO_ALUMINIUM * vehicle.wetted)
        heated_area = vehicle.wetted / 2.0
        sides = float(vehicle.radiating_sides)
        # Bands in theta of EQUAL AREA (equal steps in cos(theta)).
        mu_edges = np.linspace(1.0, 0.0, n_bands + 1)
        flux_frac = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    elif flux_mode == "uniform":
        # The whole wetted surface at the mean flux; both faces of a plate
        # are already part of the wetted area, so each element radiates from
        # one side and the mass per unit area is m/A_wet.
        wall_thickness = vehicle.mass / (RHO_ALUMINIUM * vehicle.wetted)
        heated_area = vehicle.wetted
        sides = 1.0
        flux_frac = np.full(n_bands, 0.25)
    else:
        raise ValueError(f"flux_mode: {flux_mode!r}")

    t_eff = min(wall_thickness, depth)          # participating thickness
    lumped_valid = wall_thickness <= depth
    dA = heated_area / n_bands
    m_band = RHO_ALUMINIUM * t_eff * dA
    n = n_bands
    mat = material
    h1 = mat.h1

    def rhs(t, y):
        h_s = y[:n]
        m_v = y[n:2 * n]
        T_b = float(mat.T_boil_at(np.exp(float(lnp_spline(t)))))
        L_v = float(mat.L_vapour_at(T_b))
        h3 = float(mat.h3(T_b))
        T = temperature_from_enthalpy(h_s, mat, T_b)
        V_now = float(V_spline(t))
        q = max(float(q_spline(t)), 0.0) * hot_wall_factor(V_now, T)
        eps = mat.emissivity_at(T, t - t0)

        # A boiled-off band disappears. Smoothed over 1% of the band mass.
        alive = np.clip((m_band - m_v) / (0.01 * m_band), 0.0, 1.0)
        q_in = q * flux_frac * alive
        rad = sides * eps * SIGMA_SB * T ** 4 * alive
        net = q_in - rad

        # The "heating / boiling" switch is SMOOTHED over a 1% window of h3:
        # a hard discontinuity in the right-hand side makes LSODA 200 times
        # slower.
        # CLAMP ON HEATING ONLY: a boiling band whose input falls below its
        # re-radiation must cool down, otherwise it would radiate energy it
        # does not have.
        w = np.clip((h_s - h3) / (0.01 * h3), 0.0, 1.0)
        pos = np.maximum(net, 0.0)
        neg = np.minimum(net, 0.0)

        # Blowing: part of the excess flux in a boiling band is blocked by vapour.
        dh_gas = max(0.5 * V_now ** 2 - CP_AIR_HOT * T_b, 0.0)
        evap = w * pos / (1.0 + blowing_eta * dh_gas / L_v)
        block = w * pos - evap

        # Flashing of superheated melt when T_boil drops with pressure.
        flash = np.maximum(h_s - 1.01 * h3, 0.0) / TAU_FLASH * alive

        dh = (neg + (1.0 - w) * pos) / (RHO_ALUMINIUM * t_eff) - flash
        dm = evap * dA / L_v + m_band * flash / L_v

        # Audit inside the same ODE system. The split is exact:
        # q_in - rad = [neg + (1-w)pos] + evap + block, and m_band*flash
        # moves from storage to vaporization.
        return np.concatenate([dh, dm, [
            float(np.sum(q_in) * dA),
            float(np.sum(rad) * dA),
            float(np.sum(evap) * dA + np.sum(m_band * flash)),
            float(np.sum(block) * dA)]])

    y0 = np.zeros(2 * n + 4)
    atol = np.concatenate([np.full(n, 1e-1), np.full(n, 1e-7 * max(m_band, 1e-9)),
                           np.full(4, 1e-1)])
    t_out = np.linspace(t0, t1, n_out)
    sol = solve_ivp(rhs, (t0, t1), y0, method="LSODA", t_eval=t_out,
                    rtol=1e-6, atol=atol, max_step=5.0)

    H = sol.y[:n]
    MV = sol.y[n:2 * n]
    h_end = H[:, -1]
    m_vap_band = MV[:, -1]
    E_in, E_rad, E_vap, E_block = (float(sol.y[2 * n + k, -1]) for k in range(4))
    E_stored = float(np.sum(h_end * m_band))
    residual = (E_in - E_rad - E_stored - E_vap - E_block) / E_in if E_in > 0 else 0.0

    melt_frac_band = np.clip((H.max(axis=1) - h1) / mat.L_fusion, 0.0, 1.0)
    m_melt_band = melt_frac_band * m_band
    m_vap_series = MV.sum(axis=0)
    p_series = np.exp(lnp_spline(sol.t))
    Tb_series = mat.T_boil_at(p_series)
    T_end = temperature_from_enthalpy(h_end, mat, float(Tb_series[-1]))

    m_melt = float(m_melt_band.sum())
    m_vap = float(m_vap_band.sum())
    return {
        "m_melt": m_melt, "m_vap": m_vap,
        "f_melt": m_melt / vehicle.mass, "f_vap": m_vap / vehicle.mass,
        "T_max": float(temperature_from_enthalpy(H.max(), mat,
                                                 float(Tb_series.max()))),
        "T_nose_end": float(T_end[0]),
        "wall_thickness": wall_thickness, "participating": t_eff,
        "lumped_valid": lumped_valid, "diffusion_depth": depth,
        "m_band": m_band, "m_vap_band": m_vap_band, "m_melt_band": m_melt_band,
        "m_vap_series": m_vap_series, "t": sol.t, "h": h_spline(sol.t),
        "p_stag": p_series, "T_boil": Tb_series,
        "E_in": E_in, "E_rad": E_rad, "E_stored": E_stored, "E_vap": E_vap,
        "E_block": E_block, "energy_residual": residual,
    }
