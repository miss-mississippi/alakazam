"""Emissivity: surface models and the bridge to a laboratory measurement.

WHY THIS MODULE. For thin-walled fragments the vaporized mass depends on the
total emissivity of the surface at the operating temperature. The operating
temperature is the boiling point of Al at the LOCAL stagnation pressure,
~1900-2050 K (see ablation.Aluminium.T_boil_at), not 2740 K at 1 atm.

Two surface models, both as functions of temperature:

  oxide_film_emissivity(T)      Al2O3 film on metal: one spectral curve eps(lam)
                                fitted to the alpha-Al2O3 reference point
                                (0.35 at 1800 K) and Planck-weighted.
  bare_aluminium_emissivity(T)  bare metal without a film: an estimate from
                                electrical resistivity (Parker-Abbott).

The route to a measurement is standard and works with room-temperature optics:

  1. Measure the SPECTRAL REFLECTANCE R(lambda) of a sample at room
     temperature. It must be the total (specular + diffuse)
     directional-hemispherical R: an integrating sphere, gold-coated in the
     mid-IR.
  2. For an OPAQUE sample, Kirchhoff's law gives eps(lambda) = 1 - R(lambda).
  3. The total emissivity at temperature T follows from Planck weighting:

         eps(T) = int eps(lam) B(lam,T) dlam / int B(lam,T) dlam

This gives eps AT THE OPERATING TEMPERATURE from a measurement AT ROOM
TEMPERATURE.

HONEST LIMITATIONS, stated up front:
  - Optical constants themselves depend on temperature. The method misses it.
  - The real surface during entry is MELT with a film, while the sample is an
    oxidized solid. The measurement constrains the "active film" scenario and
    the thickness threshold; the "bare melt" scenario cannot be measured
    this way.
  - Kirchhoff requires opacity. A film thinner than ~1 um is semi-transparent
    in the IR, so what is measured is the "film on metal" system, which is
    what we need, but the result depends on the substrate, which must be
    fixed.
  - The sphere gives eps near normal incidence, the model needs the
    HEMISPHERICAL value. For metals the hemispherical value is 10-30% above
    normal, for dielectrics slightly below.
"""

from __future__ import annotations

import numpy as np

H_PLANCK = 6.62607015e-34    # J*s (SI, exact)
C_LIGHT = 2.99792458e8       # m/s (exact)
K_BOLTZ = 1.380649e-23       # J/K (exact)
SIGMA_SB = 5.670374419e-8    # W/(m^2*K^4)
WIEN_B = 2.897771955e-3      # m*K, Wien displacement constant


def planck_spectral_radiance(wavelength_m, T: float):
    """Blackbody spectral radiance B(lambda,T), W/(m^2*sr*m)."""
    lam = np.asarray(wavelength_m, dtype=float)
    a = 2.0 * H_PLANCK * C_LIGHT ** 2 / lam ** 5
    x = H_PLANCK * C_LIGHT / (lam * K_BOLTZ * T)
    # Clamp the exponent: exp overflows at very short wavelengths, where the
    # contribution is zero anyway.
    return np.where(x < 700.0, a / np.expm1(np.minimum(x, 700.0)), 0.0)


def planck_weight(wavelength_m, T: float):
    """Normalized Planck weight: int w dlam = 1 on the given grid."""
    lam = np.asarray(wavelength_m, dtype=float)
    B = planck_spectral_radiance(lam, T)
    return B / np.trapezoid(B, lam)


def total_emissivity(wavelength_m, eps_lambda, T: float) -> float:
    """Total emissivity at temperature T.

        eps(T) = int eps(lam) B(lam,T) dlam / int B(lam,T) dlam

    wavelength_m : m, increasing grid
    eps_lambda   : spectral emissivity, 0..1
                   (from a measurement: eps = 1 - R for an opaque sample)
    """
    lam = np.asarray(wavelength_m, dtype=float)
    e = np.asarray(eps_lambda, dtype=float)
    B = planck_spectral_radiance(lam, T)
    return float(np.trapezoid(e * B, lam) / np.trapezoid(B, lam))


def total_emissivity_from_reflectance(wavelength_m, reflectance, T: float) -> float:
    """Same, but the input is a MEASURED reflectance.

    Exactly the function the instrument data will go into: an array of
    wavelengths and an array of R taken with an integrating sphere.
    """
    return total_emissivity(wavelength_m, 1.0 - np.asarray(reflectance), T)


def required_band(T: float, coverage: float = 0.95,
                  lam_lo: float = 1e-7, lam_hi: float = 1e-3):
    """Wavelength range carrying a given fraction of the Planck energy.

    Answers a direct experimental question: which spectral range must the
    instrument cover so that the integral does not drift.

    Returns (lam_min, lam_max, lam_peak) in metres.
    """
    lam = np.geomspace(lam_lo, lam_hi, 20000)
    B = planck_spectral_radiance(lam, T)
    cdf = np.concatenate([[0.0], np.cumsum(np.diff(lam) * (B[:-1] + B[1:]) / 2)])
    cdf /= cdf[-1]
    tail = (1.0 - coverage) / 2.0
    lo = float(np.interp(tail, cdf, lam))
    hi = float(np.interp(1.0 - tail, cdf, lam))
    return lo, hi, WIEN_B / T


# ---------------------------------------------------------------------------
# "Active film" scenario: spectral curve of a film on metal
# ---------------------------------------------------------------------------

LAM_GRID = np.geomspace(0.2e-6, 100e-6, 40000)   # m, weighting grid

# Reference total eps of alpha-Al2O3 (compilation of metal and oxide
# emissivities, White Rose eprints 133266): 0.83 at 300 K, 0.35 at 1800 K.
ALUMINA_REF = {300.0: 0.83, 1800.0: 0.35}

# Default shape. eps_short, eps_long and width are chosen by hand; lam_c is
# fitted to ONE reference point (1800 K). Shape freedom is assessed by a
# 48-set sweep in analysis_step5b.
FILM_SHAPE_DEFAULT = dict(eps_short=0.05, eps_long=0.95, width=1.6)


def eps_film(lam, eps_short=0.05, lam_c=3.98e-6, eps_long=0.95, width=1.6):
    """eps(lam) of an oxide film on metal.

    Transparent region: the METAL under the film is seen, eps ~ eps_short.
    Phonon region (multiphonon absorption of Al2O3): eps ~ eps_long.
    The edge lam_c is fitted to a reference point, see fit_film_edge().
    """
    return eps_short + (eps_long - eps_short) / (1.0 + (lam_c / lam) ** width)


def fit_film_edge(eps_short=0.05, eps_long=0.95, width=1.6,
                  T_ref: float = 1800.0, eps_ref: float = ALUMINA_REF[1800.0]):
    """lam_c at which the curve gives the reference eps(T_ref). Metres."""
    from scipy.optimize import brentq
    return float(brentq(lambda lc: total_emissivity(
        LAM_GRID, eps_film(LAM_GRID, eps_short, lc, eps_long, width), T_ref)
        - eps_ref, 0.3e-6, 40e-6))


class TabulatedEmissivity:
    """eps(T) from a table with linear interpolation.

    The thermal ODE right-hand side calls eps(T) hundreds of thousands of
    times, while an honest Planck weighting costs an integral over 40 000
    points. So the curve is tabulated once on a 300-3500 K grid (25 K step;
    eps(T) is smooth, interpolation error < 3e-5 over the whole range).
    """

    def __init__(self, func_T, T_lo=300.0, T_hi=3500.0, dT=25.0, label=""):
        self.T = np.arange(T_lo, T_hi + dT, dT)
        self.eps = np.array([func_T(T) for T in self.T])
        self.label = label

    def __call__(self, T):
        return np.interp(np.asarray(T, dtype=float), self.T, self.eps)


_OXIDE_CACHE: dict = {}


def oxide_film_emissivity(eps_short=0.05, eps_long=0.95, width=1.6):
    """"Optically active film" scenario: eps(T) of one spectral curve.

    The curve is pinned to the alpha-Al2O3 reference point at 1800 K; the
    second reference point (300 K) is predicted, see analysis_step5b part E.
    Above 2345 K (melting of Al2O3) this is an extrapolation, but at the
    boiling point of Al at the local pressure (~2000 K) the film is still
    solid.
    """
    key = (eps_short, eps_long, width)
    if key not in _OXIDE_CACHE:
        lc = fit_film_edge(eps_short, eps_long, width)
        spec = eps_film(LAM_GRID, eps_short, lc, eps_long, width)
        _OXIDE_CACHE[key] = TabulatedEmissivity(
            lambda T: total_emissivity(LAM_GRID, spec, T), label="oxide")
    return _OXIDE_CACHE[key]


# ---------------------------------------------------------------------------
# "Bare melt" scenario: metal without an optically active film
# ---------------------------------------------------------------------------

def aluminium_resistivity(T, T_melt: float = 933.5):
    """Electrical resistivity of pure Al, Ohm*m.

    Solid: 2.65 uOhm*cm at 293 K -> ~10.9 at melting, linear.
    Liquid: ~24.2 uOhm*cm at melting, slope ~0.0145 uOhm*cm/K.
    Orders of magnitude from recommended data for Al (Desai et al., J. Phys.
    Chem. Ref. Data 13, 1131, 1984); above ~1500 K the liquid branch is an
    extrapolation. Solid 6061 has a higher resistivity (~4 uOhm*cm at 293 K),
    so for it eps here is underestimated.
    """
    T = np.asarray(T, dtype=float)
    solid = 2.65 + (10.9 - 2.65) * (T - 293.0) / (T_melt - 293.0)
    liquid = 24.2 + 0.0145 * (T - T_melt)
    return np.where(T < T_melt, solid, liquid) * 1e-8


def parker_abbott_emissivity(rho_e_ohm_m, T):
    """Total hemispherical eps of a metal from its resistivity (Parker &
    Abbott, NASA SP-55, 1965), a generalization of the Hagen-Rubens relation:

        eps = 0.766 x^0.5 - (0.309 - 0.0889 ln x) x - 0.0175 x^1.5,
        x = rho_e[Ohm*cm] * T[K]

    Checked against reference data: for polished solid Al at 600-900 K it
    gives 0.045-0.068 (handbook: 0.04-0.07), see verify_step5.
    """
    x = np.asarray(rho_e_ohm_m, dtype=float) * 100.0 * np.asarray(T, dtype=float)
    return 0.766 * np.sqrt(x) - (0.309 - 0.0889 * np.log(x)) * x - 0.0175 * x ** 1.5


def bare_aluminium_emissivity(T):
    """"No film" scenario: eps(T) of bare Al from its resistivity.

    An ESTIMATE, not a measurement: no direct data on the total eps of liquid
    Al above ~1500 K were found. Gives ~0.10 at melting and ~0.17 at 2000 K.
    The handbook value 0.05 is polished SOLID Al at 300-900 K; the liquid
    metal's resistivity is 2.4-5 times that of the solid at 900 K, and eps
    grows roughly as its square root.
    """
    T = np.asarray(T, dtype=float)
    return parker_abbott_emissivity(aluminium_resistivity(T), T)


def synthetic_oxide_spectrum(wavelength_m, thickness_m: float,
                             eps_metal: float = 0.05,
                             eps_oxide: float = 0.60,
                             lam_transition: float = 3.0e-6):
    """PLACEHOLDER until real data exist. Not to be presented as a measurement.

    Simplest "film on metal" model: the film is opaque where its optical
    thickness is large, i.e. at short wavelengths relative to
    lam_transition ~ thickness; at long wavelengths the metal shows through.

        eps(lam) = eps_metal + (eps_oxide - eps_metal) / (1 + (lam/(delta*k))^2)

    Needed for exactly one thing: checking that the chain
    "spectrum -> Planck weighting -> eps(T) -> model" works end to end
    BEFORE samples exist. The numbers themselves mean nothing.
    """
    lam = np.asarray(wavelength_m, dtype=float)
    lam_c = max(thickness_m, 1e-9) * (lam_transition / 1e-6) * 1e6 * 1e-6
    return eps_metal + (eps_oxide - eps_metal) / (1.0 + (lam / lam_c) ** 2)
