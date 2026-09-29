"""Vehicle parameters and entry initial conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import OMEGA_EARTH, R_EARTH


@dataclass
class Vehicle:
    """Entering object.

    Mass is constant along the trajectory: the feedback of ablation on the
    ballistic coefficient is not closed (a model assumption).

    Cd: at 120-90 km the flow is free-molecular (Kn >> 1) and a convex body
        with diffuse reflection has Cd ~ 2.0-2.2. Below ~70 km the flow is
        continuum and a blunt body has Cd ~ 0.9-1.2. The default is a constant
        1.0: in the free-molecular zone deceleration is negligible anyway
        (D/m ~ 0.01 m/s^2 against g*sin(gamma) ~ 0.25 m/s^2), and all the
        dynamics happen in the continuum.

    A:  frontal area of a 150-200 kg satellite bus is 0.6-1.0 m^2. Without a
        specific vehicle there is an irreducible factor ~2 in
        beta = m/(Cd*A). beta moves the peak ALTITUDE: h* ~ H*ln(beta), and
        doubling beta lowers the peak by H*ln2 ~ 5 km. For the altitude
        histogram this is a first-order effect.
    """

    mass: float = 175.0          # kg, total mass (task range 150-200)
    area: float = 1.0            # m^2, frontal area
    Cd: float = 1.0              # dimensionless
    nose_radius: float = 0.5     # m, effective nose radius.
                                 # A tumbling irregular vehicle has no physical
                                 # Rn: it is a fitting parameter. Averaging over
                                 # the surface via SHAPE_FACTOR_TUMBLING removes
                                 # the problem: the main mass parameter stops
                                 # being unmeasurable.
    wetted_area: float | None = None   # m^2, total wetted surface.
                                       # None -> 4*area (Cauchy), see .wetted
    radiating_sides: int = 1     # how many faces of a heated wall element
                                 # radiate. 1 for a shell: the inner face looks
                                 # into the cavity and has no net emission.
                                 # 2 for a thin plate: the back face radiates at
                                 # the same temperature because a 1 mm plate is
                                 # isothermal through its thickness (Bi ~ 0.002).

    @property
    def ballistic_coefficient(self) -> float:
        """beta = m / (Cd * A), kg/m^2."""
        return self.mass / (self.Cd * self.area)

    @property
    def wetted(self) -> float:
        """Wetted area, m^2. 4*area by default.

        This is NOT a fit. Cauchy's formula: for any convex body the projected
        area averaged over random orientations equals 1/4 of the surface area.
        Our `area` for a tumbling body is exactly that mean projection (the
        same one used for drag), so

            A_wet = 4 * area                                   EXACTLY

        for any convex shape, not only a sphere. Setting it by hand only makes
        sense for strongly non-convex bodies (deployed panels).
        """
        return 4.0 * self.area if self.wetted_area is None else self.wetted_area

    @classmethod
    def plate(cls, mass: float, thickness: float, Cd: float = 1.5,
              rho_material: float = 2700.0, nose_floor: float = 5.0e-3, **kw):
        """Thin plate: solar panel, cover, MLI element.

        WHY A SEPARATE CONSTRUCTOR. Deriving Rn from the area as for a sphere
        (Rn = sqrt(A/pi)) is wrong for a plate and gives nonsense: a panel with
        beta = 4 kg/m^2 would get Rn = 96 cm. A plate has a low beta not
        because it is large but because it is THIN.

        For a tumbling plate, Cauchy's formula gives a mean projection equal to
        a quarter of the surface, and the surface is ~2*A_plate, so the mean
        projection is A/2. Hence

            beta = m/(Cd*A_proj) = rho*A*t/(Cd*A/2) = 2*rho*t/Cd

        beta is set by THICKNESS and nothing else. For Al at Cd = 1.5:
        t = 1 mm -> beta = 3.6;  t = 2 mm -> beta = 7.2;  t = 5 mm -> 18.
        That is exactly the 1-10 band that panels and MLI occupy in fragment
        lists: the thickness parameterization reproduces the known range
        instead of being tuned to it.

        RADIATION FROM BOTH FACES. The plate is isothermal through its
        thickness, so the back face, not exposed to the flow at the moment,
        radiates at the same temperature as the front face. In the per-band
        model the heated half of the wetted surface is one face, and it must
        radiate for two: radiating_sides = 2. One-sided radiation for a plate
        is equivalent to halving eps.

        The effective nose radius is the edge radius, ~t/2, BUT with a floor.
        The floor is required: as Rn -> 0 the correlation gives q -> infinity,
        which is unphysical. A real ablating edge blunts itself to a
        self-consistent radius. The 5 mm floor is a lower estimate of that
        radius and is part of the sensitivity sweep.
        """
        area_plate = mass / (rho_material * thickness)
        area_proj = area_plate / 2.0
        return cls(mass=mass, area=area_proj, Cd=Cd,
                   nose_radius=max(thickness / 2.0, nose_floor),
                   wetted_area=2.0 * area_plate, radiating_sides=2, **kw)

    @classmethod
    def compact(cls, mass: float, beta: float, Cd: float = 1.5, **kw):
        """Compact (non-plate) fragment: structure, brackets.

        Here a sphere equivalent is appropriate: Rn = sqrt(A_proj/pi).
        """
        area = mass / (Cd * beta)
        return cls(mass=mass, area=area, Cd=Cd,
                   nose_radius=float(np.sqrt(area / np.pi)), **kw)

    @classmethod
    def geometric_family(cls, scale: float, mass0: float = 175.0,
                         area0: float = 1.0, nose0: float = 0.5,
                         Cd: float = 1.5, **kw):
        """Geometrically similar body with linear scale `scale`.

        m ~ L^3, A ~ L^2, Rn ~ L. Needed to sweep SIZE honestly: sweeping Rn
        alone at fixed mass and area is physically meaningless, and that is
        exactly where the sign error hides.
        """
        return cls(mass=mass0 * scale ** 3, area=area0 * scale ** 2,
                   nose_radius=nose0 * scale, Cd=Cd, **kw)


@dataclass
class EntryState:
    """Entry initial conditions.

    gamma0 is the flight-path angle to the LOCAL HORIZON, positive upward,
    negative on descent. Given in degrees, converted to radians internally.

    The -1...-3 deg range of the task is not "plus or minus a little" but
    different regimes: at 120 km the circular speed is 7836 m/s against the
    7500 m/s entry speed, so the centrifugal and gravity terms in dgamma/dt
    nearly cancel, and gamma changes several-fold during the flight.
    """

    altitude: float = 120.0e3    # m
    velocity: float = 7500.0     # m/s, INERTIAL (relative to a non-rotating
                                 # Earth). Relative to the air it is V_rel,
                                 # see corotation_speed.
    gamma_deg: float = -1.5      # deg
    inclination_deg: float = 53.0    # orbit inclination, deg

    @property
    def corotation_speed(self) -> float:
        """Speed of the co-rotating atmosphere ALONG THE TRACK, m/s.

        The atmosphere rotates eastward with the Earth at u = omega*R*cos(lat).
        Along the ground track its projection u*sin(A) acts, where A is the
        track azimuth from north. Spherical trigonometry for an orbit of
        inclination i gives sin(A) = cos(i)/cos(lat).

        Latitude cancels:

            v_along = omega*R*cos(lat) * cos(i)/cos(lat) = omega*R*cos(i)

        So the contribution depends ONLY on inclination, not on where the
        object enters. Limiting cases:
          i = 0    prograde equatorial  -> +465 m/s (atmosphere follows)
          i = 53   typical LEO          -> +280 m/s
          i = 90   polar                -> 0
          i = 98   SSO, retrograde      -> -65 m/s (atmosphere opposes)
        """
        return OMEGA_EARTH * R_EARTH * np.cos(np.deg2rad(self.inclination_deg))

    @property
    def gamma_rad(self) -> float:
        return np.deg2rad(self.gamma_deg)

    def to_vector(self) -> np.ndarray:
        """State vector [V, gamma, h, s]."""
        return np.array([self.velocity, self.gamma_rad, self.altitude, 0.0])
