"""Stagnation-point aerodynamic heating.

UNITS OF THE SUTTON-GRAVES CONSTANT are the main trap of this module.

The NASA TFAWS Aerothermodynamics Course gives k = 1.7415e-4 for Earth and
labels the result W/cm^2. Direct substitution contradicts this: for Stardust
(Rn = 0.23 m, V = 12.6 km/s, rho ~ 3e-4 kg/m^3, computed peak ~1200 W/cm^2)
the formula with all inputs in SI gives 1.26e7. That is 1260 W/cm^2 only if
the output is read as W/m^2. In W/cm^2 it would be 1.26e7 W/cm^2, off by four
orders of magnitude.

Conclusion: k = 1.7415e-4 with SI inputs (rho kg/m^3, Rn m, V m/s) gives
q in W/M^2. Checked in verify_step3.test_sutton_graves_units().

A variant of the constant, 1.83e-4, appears in open implementations; it is
about 5% higher and is kept as an estimate of the correlation's uncertainty.
"""

from __future__ import annotations

import numpy as np

# Sutton-Graves for air / Earth.
# Source: Sutton K., Graves R.A., NASA TR R-376 (1971); value as reproduced
# in the NASA TFAWS Aerothermodynamics Course.
# SI inputs -> W/m^2 output. Correlation uncertainty ~10-20%, plus ~5% spread
# between variants of the constant (1.7415e-4 vs 1.83e-4).
K_SUTTON_GRAVES = 1.7415e-4

# Normalization point for Detra-Kemp-Riddell: circular speed at the surface.
V_REF_DKR = 7925.0          # m/s
RHO_REF_DKR = 1.225         # kg/m^3
EXP_DKR = 3.15              # velocity exponent


def sutton_graves(rho, V, nose_radius: float, k: float = K_SUTTON_GRAVES):
    """Stagnation-point heat flux, W/m^2.

        q = k * sqrt(rho / Rn) * V^3

    rho: kg/m^3, V: m/s, nose_radius: m.

    What the correlation carries:
      - It is for the STAGNATION POINT of a spherical nose. Over the surface
        the flux is distributed with cos(theta) (local_flux_fraction), and in
        integral estimates with the shape factor SHAPE_FACTOR_TUMBLING.
      - CONVECTIVE flux only. Shock-layer radiative heating is negligible
        below 8 km/s (it becomes important around 10 km/s), so this is valid
        for entry from orbit.
      - Cold wall. The hot-wall correction is a separate factor,
        hot_wall_factor(), not applied here.
    """
    return k * np.sqrt(np.asarray(rho) / nose_radius) * np.asarray(V) ** 3


def detra_kemp_riddell_shape(rho, V, nose_radius: float, q_ref_pair=None):
    """DKR as a SHAPE, not as an absolute value.

        q ~ Rn^-0.5 * rho^0.5 * V^3.15

    WHY SHAPE ONLY. The absolute DKR constant could not be taken from a
    reliable primary source: open implementations use different
    normalizations and exponents (3.15 vs 3.25), and the original
    Detra-Kemp-Riddell (1957) paper was not freely available. Inventing a
    constant for a nicer second curve makes no sense.

    The substantive part of the comparison does not depend on the constant.
    The question DKR answers is how sensitive the result is to the VELOCITY
    EXPONENT, 3.0 vs 3.15. So DKR is normalized to Sutton-Graves at one
    reference point (V_REF_DKR, RHO_REF_DKR) and only the shape is compared:
    where the peak moves and how the integrated heat load changes.
    """
    rho = np.asarray(rho)
    V = np.asarray(V)
    # Normalization: matches Sutton-Graves at the reference point
    q_ref = sutton_graves(RHO_REF_DKR, V_REF_DKR, nose_radius)
    return q_ref * np.sqrt(rho / RHO_REF_DKR) * (V / V_REF_DKR) ** EXP_DKR


# --- Geometry: from stagnation-point flux to total power ------------------

# Mean flux over the wetted surface as a fraction of the stagnation flux.
# The standard shortcut in demise tools for a randomly oriented body is
# 0.25-0.3. We take the middle. Relative uncertainty +-10%.
SHAPE_FACTOR_TUMBLING = 0.27

# Air enthalpy at 2500-3000 K including dissociation. The frozen c_p of air
# rises from 1005 J/(kg*K) at 300 K to ~1300 at 3000 K.
CP_AIR_HOT = 1300.0   # J/(kg*K), uncertainty ~15%


def hot_wall_factor(V, T_wall: float, cp_air: float = CP_AIR_HOT):
    """Hot-wall correction: q_hot = q_cold * (1 - h_w/h_0).

    Sutton-Graves gives a COLD-WALL flux. A real wall is hot, and the
    enthalpy difference that drives heat into the body is smaller.

        h_0 = V^2/2 + h_air   (total stagnation enthalpy)
        h_w = cp_air * T_wall

    At V = 7500 m/s: h_0 ~ 2.8e7 J/kg. At T ~ 2050 K (Al boiling at a
    stagnation pressure of ~1 kPa): h_w ~ 2.7e6, factor ~0.90, minus 10%.
    At 2740 K (boiling at 1 atm) it would be minus 13%.

    The enthalpy is that of the AIR at the wall, not of the metal: the flux
    is driven by the gas enthalpy difference, not by the heat stored in the
    material. An estimate via the metal (cp*dT for Al ~ 2.5e6) gives -9%,
    the same order; the difference between the two routes is the
    uncertainty of the correction.
    """
    V = np.asarray(V, dtype=float)
    h_0 = 0.5 * V ** 2
    h_w = cp_air * T_wall
    return np.clip(1.0 - h_w / h_0, 0.0, 1.0)


# Stagnation pressure behind a normal shock at M -> infinity (Rayleigh pitot
# formula, gamma = 1.4): p0 = 0.92 * rho * V^2.
PITOT_FACTOR = 0.92


def stagnation_pressure(rho, V):
    """Stagnation-point pressure, Pa. It sets the boiling temperature of Al
    at the surface (ablation.Aluminium.T_boil_at)."""
    return PITOT_FACTOR * np.asarray(rho) * np.asarray(V) ** 2


def blowing_factor(h_0, H_eff: float, eta: float = 0.3):
    """Flux blockage by injected ablation products, IN CLOSED FORM.

    Vaporizing material enters the boundary layer and blocks part of the
    flux (transpiration cooling). The classic linear correction:

        q_net = q_hw - eta * mdot'' * h_0

    and the ablation rate is itself proportional to the flux:

        mdot'' = q_net / H_eff

    Substituting one into the other:

        q_net = q_hw - eta * (q_net/H_eff) * h_0
        q_net * (1 + eta*h_0/H_eff) = q_hw
        q_net = q_hw / (1 + eta*h_0/H_eff)

    IMPORTANT FOR THE THERMAL MODEL: the nonlinearity closes analytically.
    Both the ablation law and the blockage are linear in mdot'', so no
    implicit solver is needed at each step; dividing by the factor suffices.
    Returns 1/(1 + eta*h_0/H_eff).

    WHICH eta. In a laminar layer blowing blocks the flux MORE than in a
    turbulent one: turbulent mixing weakens the effect (hypersonic
    transpiration-cooling experiments, AIAA J. 10.2514/1.J053053). At
    70-80 km the layer is laminar. The specific eta value was not verified
    against a primary source, so blowing is a SENSITIVITY AXIS (eta 0.3-0.6),
    not part of the baseline. With eta = 0.3, h_0 = 2.8e7 and H_eff ~ 1.2e7
    the factor is ~0.59.

    In the per-band model (ablation.surface_thermal_model, blowing_eta) only
    the excess flux in BOILING bands is blocked: vapour is injected where
    evaporation happens and nowhere else.
    """
    return 1.0 / (1.0 + eta * np.asarray(h_0, dtype=float) / H_eff)


def absorbed_power(q_stag, wetted_area: float,
                   shape_factor: float = SHAPE_FACTOR_TUMBLING):
    """Total thermal power absorbed by the body, W.

    THIS, NOT q_stag, DETERMINES THE ABLATED MASS.

        P = shape_factor * q_stag * A_wet

    Why the sign with respect to Rn flips. For a geometrically similar body:

        q_stag     ~ Rn^(-1/2)      (correlation)
        A_wet      ~ Rn^2           (geometry)
        P          ~ Rn^(+3/2)      <- GROWS with size
        m          ~ Rn^3
        P/m        ~ Rn^(-3/2)      <- specific load FALLS with size

    By specific flux a large body heats less, by total energy more. Building
    a mass budget on q_stag in W/m^2 is a sign error, not just a magnitude
    error.

    Hence the second reason fragmentation decides the outcome. Fragments do
    not only decelerate higher because of a smaller beta; they also receive
    an order of magnitude more heat PER KILOGRAM. Two mechanisms, same
    direction.
    """
    return shape_factor * np.asarray(q_stag) * wetted_area


# --- Angular flux distribution and a local vaporization criterion --------

SIGMA_SB = 5.670374419e-8   # W/(m^2*K^4), Stefan-Boltzmann constant (CODATA)


def local_flux_fraction(theta):
    """q(theta)/q_stag over the surface of a blunt body.

        f(theta) = cos(theta)   for theta <= 90 deg,  0 behind the shoulder

    WHERE THE SHAPE COMES FROM. The Newtonian pressure distribution on a
    sphere is p/p_stag = cos^2(theta), and laminar heating near the
    stagnation point scales as the square root of the velocity gradient,
    hence a law linear in cos(theta) (Lees, Jet Propulsion 26, 1956).

    NO FREE PARAMETERS. This matters: the alternative, splitting the body
    into a "nose" and "the rest", introduces a new unknown (the nose area
    fraction) that directly multiplies the vaporized mass. Here the geometry
    is fixed.

    SELF-CHECK. The mean over the full sphere surface:

        <f> = (1/4pi) * int_0^(pi/2) cos(th) * 2pi sin(th) dth = 1/4

    exactly 0.25, inside the standard 0.25-0.30 shape-factor band used by
    demise tools. So no new physics is introduced; physics that was already
    there stops being averaged away: SHAPE_FACTOR_TUMBLING is the mean of
    THIS distribution.

    IDEALIZATION. A tumbling irregular body has no axisymmetric flux
    distribution. But an idealization without a free parameter is better
    than an arbitrary split with one.
    """
    theta = np.asarray(theta, dtype=float)
    return np.where(theta <= np.pi / 2, np.cos(theta), 0.0)


def vaporizing_area_fraction(q_stag, emissivity: float, T_boil: float):
    """Fraction of the FULL surface where the local flux exceeds what
    radiation can remove at the boiling temperature.

    Local radiative equilibrium: eps*sigma*T(th)^4 = q_stag*cos(th).
    Boiling where cos(th) > C, with

        C = eps*sigma*T_boil^4 / q_stag

    Fraction of the full sphere:

        A_frac = (1 - cos(th_c))/2 = (1 - C)/2      for C < 1,  else 0

    WHY IT MATTERS. The problem is a threshold one with T^4: evaporation
    happens where the flux is above average, and the average does not see
    that place. With a single mean temperature at eps = 0.3 and
    q = 85 W/cm^2 the equilibrium is 1916 K: below boiling at 1 atm (2792 K)
    but close to boiling at the local pressure (~1900-2050 K).

    This closed form is for checks and estimates; the model itself uses
    ablation.surface_thermal_model (pressure-dependent T_boil, two-sided
    plates).
    """
    q = np.asarray(q_stag, dtype=float)
    C = np.where(q > 0, emissivity * SIGMA_SB * T_boil ** 4 / np.maximum(q, 1e-30),
                 np.inf)
    return np.where(C < 1.0, (1.0 - C) / 2.0, 0.0)


def vaporization_rate(q_stag, wetted_area: float, emissivity: float,
                      T_boil: float, L_vapour: float):
    """Mass loss rate by vaporization, kg/s, IN CLOSED FORM.

    The flux in excess of what radiation removes goes into vaporization:

        q_vap(th) = q_stag*cos(th) - eps*sigma*T_boil^4,   where cos(th) > C

    Integrating over the sphere surface:

        mdot = (A/L_vap) * (1/2) * int_0^(th_c) [q_stag*cos - eps*sig*T^4] sin dth

    Substituting eps*sigma*T_boil^4 = q_stag*C and integrating gives

        mdot = A * q_stag * (1 - C)^2 / (4 * L_vap)

    The square in (1-C) is not a typo: near the threshold evaporation turns
    on smoothly because both the area and the excess flux go to zero.
    Checked against numerical integration in verify_step4.
    """
    q = np.asarray(q_stag, dtype=float)
    C = emissivity * SIGMA_SB * T_boil ** 4 / np.maximum(q, 1e-30)
    excess = np.where(C < 1.0, (1.0 - C) ** 2, 0.0)
    return wetted_area * q * excess / (4.0 * L_vapour)
