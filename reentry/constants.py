"""Physical constants. Each one comes with a source and an uncertainty.

Project rule: no number enters the code without a line stating its source.
"""

# --- Earth ---
MU_EARTH = 3.986004418e14   # m^3/s^2, geocentric gravitational constant GM.
                            # Source: WGS-84 / IERS Conventions. Relative
                            # uncertainty ~1e-9, exact for our purposes.

R_EARTH = 6371.0e3          # m, mean (volumetric) Earth radius.
                            # Equatorial 6378.137 km, polar 6356.752 km.
                            # A sphere changes g by <0.3% along our trajectory.

OMEGA_EARTH = 7.292115e-5   # rad/s, Earth rotation rate.
                            # Atmospheric co-rotation: V_rel = V - omega*R*cos(i),
                            # up to 465 m/s at the equator, and q ~ V^3 => ~19%
                            # in heat flux (vehicle.EntryState.corotation_speed).

G0 = 9.80665                # m/s^2, standard gravity.
                            # Only for expressing deceleration in "g", not dynamics.

# --- Standard atmosphere (for the exponential placeholder) ---
RHO0_SEA_LEVEL = 1.225      # kg/m^3, sea-level density.
                            # Source: U.S. Standard Atmosphere 1976.

H_SCALE_FIT = 7.2e3         # m. NOTE: not a physical scale height (that ranges
                            # from ~6 km in the troposphere to 60+ km in the
                            # thermosphere) but a one-parameter fit. Accurate to
                            # ~20% at 40-90 km, overestimates density 2-3x above
                            # 100 km. Placeholder used in step 1.
