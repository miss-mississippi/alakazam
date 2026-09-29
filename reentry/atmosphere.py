"""Atmosphere models.

Step 1: exponential placeholder (ExponentialAtmosphere).
Step 2: NRLMSISE-00 / NRLMSIS 2.x via pymsis (MSISAtmosphere).

Both expose the same interface, .density(h) and .scale_height(h), so the
trajectory code does not distinguish between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import H_SCALE_FIT, RHO0_SEA_LEVEL


class ExponentialAtmosphere:
    """rho(h) = rho0 * exp(-h/H).

    Placeholder. Provides nothing but density, which is all step 1 needs
    (Sutton-Graves also needs only rho).

    Expected accuracy against U.S. Standard Atmosphere 1976 with
    rho0 = 1.225, H = 7.2 km:

        120 km : ~3.2x too high
        100 km : ~2x too high
         80 km : ~1.0
         60 km : ~1.0
         40 km : ~1.2x too high

    The model is accurate exactly where ablation happens and wrong where
    deceleration is negligible. Nothing should be tuned against it.
    """

    name = "exponential (placeholder)"

    def __init__(self, rho0: float = RHO0_SEA_LEVEL, H: float = H_SCALE_FIT):
        self.rho0 = rho0
        self.H = H

    def density(self, h):
        """Density, kg/m^3. h is geometric altitude, m."""
        # Clamp from below: otherwise exp(-h/H) overflows if the integrator
        # tries a step below the surface.
        h_clipped = np.maximum(h, 0.0)
        return self.rho0 * np.exp(-h_clipped / self.H)

    def scale_height(self, h):
        """Local scale height, m. Constant by definition for the exponential."""
        return self.H


class MSISAtmosphere:
    """NRLMSISE-00 / NRLMSIS 2.x via pymsis, tabulated and splined.

    WHY pymsis IS NOT CALLED FROM THE ODE RIGHT-HAND SIDE.
    solve_ivp calls density() tens of thousands of times per run. A direct
    MSIS call at every step costs seconds per trajectory and hours per sweep.
    We tabulate once on an altitude grid and interpolate.

    WHY THE SPLINE IS IN log(rho), NOT rho.
    Density spans 6 orders of magnitude over our range; linear interpolation
    in rho is badly wrong in the rarefied part. log(rho) is nearly linear in
    altitude (that is what a scale height means), so it interpolates well.

    WHY CUBIC, NOT LINEAR.
    Linear interpolation in log(rho) gives a piecewise-exponential density:
    continuous, but with kinks in the derivative. An adaptive integrator cuts
    its step at every kink. CubicSpline is C2, so the ODE right-hand side
    stays smooth.

    Environment parameters are physical, not cosmetic:
      f107, f107a : 10.7 cm flux, solar activity index. ~70 at solar minimum,
                    ~140 moderate, ~220 at maximum. Mostly affects the
                    thermosphere (above 100 km), where deceleration is
                    negligible anyway. Checked by a sweep.
      ap          : geomagnetic index. 4 is quiet, 50+ is a storm.
      lat, lon    : controlled deorbits usually target the South Pacific Ocean
                    Uninhabited Area (SPOUA), roughly -40 deg latitude.
      version     : 2.1 = NRLMSIS 2.1 (BASELINE), 0 = NRLMSISE-00 (for
                    comparison with demise tools such as DRAMA and for the
                    uncertainty budget).

                    Why 2.1 is the baseline. In NRLMSISE-00 thermospheric
                    densities were computed independently of the lower layers
                    and the profiles were stitched a posteriori, which gives a
                    kink in d ln(rho)/dh at 72.5 km, right in the ablation zone
                    (see verify_step2.test_model_seams). NRLMSIS 2.0 removed the
                    stitching: the hydrostatic profile is continuous from the
                    ground to the exosphere, and the transition from the mixed
                    region to diffusive separation is continuous from about
                    70 km. Emmert et al. 2021 (Earth and Space Science): "In the
                    mesosphere and below, residual biases and standard
                    deviations are considerably lower than NRLMSISE-00"; new
                    mesospheric and stratospheric temperature data were
                    assimilated, and atomic oxygen extends down to 50 km.
                    The difference from 00 is 9-16% exactly at 50-90 km.
    """

    def __init__(
        self,
        date: "datetime | np.datetime64" = np.datetime64("2026-09-01T12:00"),
        lat: float = -40.0,
        lon: float = -140.0,
        f107: float = 140.0,
        f107a: float = 140.0,
        ap: float = 4.0,
        version: float = 2.1,
        h_max: float = 200.0e3,
        dh: float = 250.0,
    ):
        import pymsis  # local import: step 1 works without pymsis

        self.date = date
        self.lat, self.lon = lat, lon
        self.f107, self.f107a, self.ap = f107, f107a, ap
        self.version = version
        self.name = f"NRLMSISE-00" if version == 0 else f"NRLMSIS {version}"
        self.name += (f" (F10.7={f107:.0f}, Ap={ap:.0f}, "
                      f"lat={lat:+.0f}, {str(date)[:10]})")

        self._h_grid = np.arange(0.0, h_max + dh, dh)
        out = pymsis.calculate(
            date, lon, lat, self._h_grid / 1e3,
            f107s=f107, f107as=f107a, aps=[[ap] * 7],
            version=version,
        )
        rho = out[..., pymsis.Variable.MASS_DENSITY].ravel()

        # MSIS can return NaN right at the ground for some versions: trim the
        # grid to valid values instead of substituting a placeholder.
        good = np.isfinite(rho) & (rho > 0)
        if not good.all():
            self._h_grid = self._h_grid[good]
            rho = rho[good]

        self._log_rho = CubicSpline(self._h_grid, np.log(rho))
        self._h_lo, self._h_hi = self._h_grid[0], self._h_grid[-1]

        # Scale height at the top of the grid, for exponential extrapolation
        self._H_top = -1.0 / self._log_rho(self._h_hi, 1)
        self._log_rho_top = float(self._log_rho(self._h_hi))

        # Compatibility with ExponentialAtmosphere: surface density
        self.rho0 = float(np.exp(self._log_rho(self._h_lo)))

    def density(self, h):
        """Density, kg/m^3. h is geometric altitude, m."""
        h = np.asarray(h, dtype=float)
        h_clipped = np.clip(h, self._h_lo, self._h_hi)
        log_rho = self._log_rho(h_clipped)
        # Above the grid: exponential extrapolation with the top scale height.
        above = h > self._h_hi
        if np.any(above):
            log_rho = np.where(
                above,
                self._log_rho_top - (h - self._h_hi) / self._H_top,
                log_rho,
            )
        return np.exp(log_rho)

    def scale_height(self, h):
        """LOCAL scale height H = -1/(d ln rho / dh), m.

        Unlike the exponential model this is not a constant. Its spread over
        altitude measures how far the one-parameter placeholder is from the
        real atmosphere.
        """
        h_clipped = np.clip(np.asarray(h, dtype=float), self._h_lo, self._h_hi)
        return -1.0 / self._log_rho(h_clipped, 1)
