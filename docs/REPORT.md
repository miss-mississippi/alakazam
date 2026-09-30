# Aluminium injection into the mesosphere by re-entering satellites: full report

**Author:** Beknur Orazbay ([ORCID 0009-0004-1983-3413](https://orcid.org/0009-0004-1983-3413)), Nazarbayev University
**Period of work:** September 2026
**Code:** Python (numpy, scipy, matplotlib, pymsis), repository [alakazam](https://github.com/miss-mississippi/alakazam), archived as [doi:10.5281/zenodo.23053936](https://doi.org/10.5281/zenodo.23053936)
**Status:** the model is complete, 37<!--=checks.total:.0f--> automated checks pass, the experimental part is designed
**Version 2:** errors found in review are fixed (section 18.2); the main result is recomputed
**Version 2.1:** the heat of oxidation and boiling at the side-band pressure are added as sensitivity axes (sections 9.5, 12); the baseline is unchanged

---

## Contents

0. [Summary: what was obtained](#0-summary-what-was-obtained)
1. [Problem statement](#1-problem-statement)
2. [Model architecture and build order](#2-model-architecture-and-build-order)
3. [Step 1. Trajectory](#3-step-1-trajectory)
4. [Diagnosis: the 25–30 km gap and moving fragmentation forward](#4-diagnosis-the-2530-km-gap-and-moving-fragmentation-forward)
5. [Step 2. Real atmosphere](#5-step-2-real-atmosphere)
6. [Step 3. Aerodynamic heating](#6-step-3-aerodynamic-heating)
7. [Mass metric: the sign flips with size](#7-mass-metric-the-sign-flips-with-size)
8. [Step 4. Fragmentation, thermal response, ablation](#8-step-4-fragmentation-thermal-response-ablation)
9. [Step 5. Main result](#9-step-5-main-result)
10. [What the model predicts and what it inherits](#10-what-the-model-predicts-and-what-it-inherits)
11. [Comparison with Ferreira](#11-comparison-with-ferreira)
12. [Uncertainty budgets](#12-uncertainty-budgets)
13. [Emissivity: validation, shape sweep, two scenarios](#13-emissivity-validation-shape-sweep-two-scenarios)
14. [Experimental part and NU instruments](#14-experimental-part-and-nu-instruments)
15. [Full list of assumptions](#15-full-list-of-assumptions)
16. [Constants and their sources](#16-constants-and-their-sources)
17. [Checks: all 37](#17-checks-all-37)
18. [Errors found along the way and how they were caught](#18-errors-found-along-the-way-and-how-they-were-caught)
19. [Anticipated questions and answers](#19-anticipated-questions-and-answers)
20. [Next steps](#20-next-steps)
21. [Repository layout and how to run](#21-repository-layout-and-how-to-run)
22. [References](#22-references)

---

## 0. Summary: what was obtained

**The problem.** Compute how much aluminium evaporates when a satellite enters the atmosphere and at which altitudes it is injected. Published estimates of the atmospheric effect (Ferreira 2024, Barker 2026, Maloney 2025) differ by an order of magnitude because all of them have to guess the inputs. ESA's standard tool (DRAMA/SESAM) does not provide this quantity.

**Main result.** For a 175 kg object with 30% aluminium (52.5 kg Al) spread uniformly over the fragments; controlled deorbit at 120 km / 7500 m/s / −1.5°, inclination 53°:

| surface scenario | ε at ~2000 K | Al evaporated | yield | injection median |
|---|---|---|---|---|
| oxide film optically active | 0.32<!--=step5.scenarios.film.eps2000:.2f--> (0.30<!--=step5.film_shape.eps2000_min:.2f-->–0.33<!--=step5.film_shape.eps2000_max:.2f-->) | **8.5<!--=step5.scenarios.film.total:.1f--> kg** | 16<!--=step5.scenarios.film.yield_pct:.0f-->% | 76.1<!--=step5.scenarios.film.median:.1f--> km |
| no film, bare melt | ≈0.17<!--=step5.scenarios.bare.eps2000:.2f--> (estimate) | **12.0<!--=step5.scenarios.bare.total:.1f--> kg** | 23<!--=step5.scenarios.bare.yield_pct:.0f-->% | 75.9<!--=step5.scenarios.bare.median:.1f--> km |

The molten mass is 32.4<!--=step5.scenarios.film.melt:.1f-->–32.8<!--=step5.scenarios.bare.melt:.1f--> kg Al (62<!--=step5.scenarios.film.melt_pct:.0f-->%), an upper bound. All of the injection happens above the stratopause; about 10<!--=step5.scenarios.film.above_meso_pct:.0f-->% above the mesopause (85 km).

**Seven conclusions that carry the work:**

1. **Demise in DRAMA/ORSAT means melting, not evaporation.** The energy to melt is 1.01<!--=step4.criteria.melt:.2f--> MJ/kg (ORSAT has 0.93; the 8<!--=verify_step4.orsat.diff_pct:.0f-->% difference comes from the alloy properties), to evaporate completely 13.6<!--=step4.criteria.vapour:.1f--> MJ/kg, a factor of 13.4<!--=step4.criteria.ratio:.1f-->. Between "demised" and "became Al₂O₃ nanoparticles" sits a yield factor that DRAMA does not compute.
2. **Aluminium on a fragment surface boils at the stagnation pressure, not at 1 atm.** At 70–80 km that is 0.1–1 kPa and 1750–2050 K instead of 2792 K. The evaporation threshold εσT⁴ drops threefold, and the evaporated mass in the "active oxide film" scenario goes from 0.6<!--=step5.changes.film_1atm:.1f--> to 8.5<!--=step5.scenarios.film.total:.1f--> kg. This is the largest physical correction in the whole work.
3. **The result is a melt/evaporation bracket, not a single number.** Its width (32.5 versus 8.5<!--=step5.scenarios.film.total:.1f-->–12.0<!--=step5.scenarios.bare.total:.1f--> kg Al) is set by the fate of the droplet phase, which nobody computes. The two surface scenarios differ by only a factor of 1.4<!--=step5.scenarios.ratio:.1f-->: at ~2000 K the film ε is pinned by the α-Al₂O₃ reference point (1800 K), and the liquid metal ε is already ~0.17.
4. **The mass is set above all by where the aluminium sits and what fraction of the mass is thin-walled.** At the same total fraction of 30%, the distribution of Al over fragments gives 0<!--=step5.sensitivity.al_split.lo:.0f-->–17<!--=step5.sensitivity.al_split.hi:.0f--> kg, a thin-walled mass fraction of 25–75% gives 4.3<!--=step5.sensitivity.thin_fraction.lo:.1f-->–13.1<!--=step5.sensitivity.thin_fraction.hi:.1f--> kg, and a breakup altitude of ±10 km gives 4.4<!--=step5.sensitivity.breakup.lo:.1f-->–11.1<!--=step5.sensitivity.breakup.hi:.1f--> kg. The surface state (1.4<!--=step5.sensitivity.surface.ratio:.1f-->×) is on a par with vapour blowing (1.7<!--=step5.sensitivity.blowing.ratio:.1f-->×) and orientation (1.5<!--=step5.sensitivity.orientation.ratio:.1f-->×). The heat of Al oxidation, left out of the baseline, can add up to 1.8<!--=step5.sensitivity.oxidation.ratio:.1f-->× at the diffusion limit of the oxygen supply (section 9.5); none of these axes moves the injection altitude.
5. **The injection altitude is inherited, not predicted.** In the 68–83 km window the injection median is ≈ 75.9<!--=step5b.transfer.intercept78:.1f--> + 0.82<!--=step5b.transfer.gain_mid:.2f-->·(H − 78) km, where H is the breakup altitude; at a fixed H the offset is +1.7<!--=step5b.offsets.min:+.1f-->…+2.8<!--=step5b.offsets.max:+.1f--> km across all other parameters. The main altitude uncertainty lies in the observational data.
6. **Comparison with Ferreira: order-of-magnitude agreement.** Ferreira oxidises 32<!--=step5.ferreira.theirs.pct:.0f-->% of the Al (24.0 of 75 kg), an output of their molecular dynamics. We evaporate 16<!--=step5.ferreira.film_uniform.pct:.0f-->–23<!--=step5.ferreira.bare_uniform.pct:.0f-->% with uniform Al and 32<!--=step5.ferreira.film_thin.pct:.0f-->–46<!--=step5.ferreira.bare_thin.pct:.0f-->% if the Al is concentrated in thin-walled parts.
7. **The experiment is still needed, but it does not close the main uncertainty.** It pins down the "active oxide film" scenario and the oxide-thickness threshold. For that the mid-IR total hemispherical reflectance is required (without it the error is −84<!--=step5b.no_midir.T2000.err_pct:.0f-->%): a gold integrating sphere is needed; the ATR accessories in the NU equipment list are unsuitable. The main unknowns for mass are the bill of materials of the fragments and the breakup altitude.

---

## 1. Problem statement

### 1.1 Why this matters

At the end of their life satellites are deorbited and burned up in the atmosphere. The aluminium of the structure evaporates, oxidises to Al₂O₃, and the oxide nanoparticles stay in the mesosphere and stratosphere for years. There they can potentially catalyse ozone destruction and change the radiative balance. With the deployment of megaconstellations (Starlink, OneWeb) the number of re-entries is growing by orders of magnitude.

Key papers on the atmospheric effect:

- **Ferreira et al. 2024, GRL**: an estimate of the Al₂O₃ mass per satellite from molecular dynamics of oxidation, concluding that ozone depletion is possible.
- **Barker et al. 2026, Earth's Future**: modelling in GEOS-Chem.
- **Maloney et al. 2025, JGR**: modelling in WACCM, radiative effect.

Their estimates differ by an order of magnitude because all of them have to guess the inputs: how much mass evaporates and where.

### 1.2 Why not DRAMA

ESA's standard tool is DRAMA, with its SARA/SESAM modules. It does not fit for two reasons:

1. Access through the Space Debris User Portal is uncertain.
2. **The main one:** DRAMA was designed for casualty risk, the risk to people on the ground. It tracks the **surviving** fragments, and the evaporated mass is simply a loss for it. That mass is exactly what we need.

During the work it turned out to be worse than that (section 8.6): DRAMA and ORSAT track not the survivors but the **molten** mass. Their demise criterion is melting, not evaporation. This changes the premise of the work rather than refining it.

### 1.3 What the model must compute

Five links:

1. **Atmosphere**: density as a function of altitude, 120–30 km.
2. **Trajectory**: 3-DOF, point mass, drag and gravity.
3. **Aerodynamic heating**: heat flux at the stagnation point (Sutton–Graves).
4. **Thermal response**: energy balance of heat input, re-radiation, heating to melting, latent heats.
5. **Ablation**: the mass loss rate.

**Main output:** the altitude distribution of evaporated aluminium mass, i.e. how many kg are injected into each layer from 100 to 40 km.

### 1.4 Inputs

| parameter | value | source |
|---|---|---|
| object mass | 175 kg (range 150–200) | task spec |
| aluminium fraction | 30% | Ferreira et al. 2024 |
| initial altitude | 120 km | task spec |
| initial speed | 7500 m/s | task spec |
| entry angle | −1.5° (range −1…−3°) | controlled deorbit |
| orbit inclination | 53° | typical LEO (Starlink-like) |
| entry point | 40° S, 140° W | SPOUA, the South Pacific disposal area |

Al 6061 properties (section 16): melting at 855–925 K (solidus–liquidus; 933 K for pure Al), boiling at 2792 K at 1 atm and at 1750–2050 K at the stagnation pressure at 70–80 km, heat of vaporization 10.9 MJ/kg, heat capacity 1038 J/(kg·K) in the solid phase (effective over 300–890 K) and 1177 in the liquid.

### 1.5 Working principles

- Accuracy is secondary. The goal is the order of magnitude of temperatures and altitudes, not a prediction of a real entry to within a kilometre.
- Every numerical value comes with a source and an order of uncertainty.
- Constants are checked against the literature, not from memory.
- The model is built in steps; after each step it is run and the figures are checked.
- Every check tests an identity or an independent benchmark, not "looks plausible".

---

## 2. Model architecture and build order

### 2.1 The original plan and what became of it

| step | original plan | what was done |
|---|---|---|
| 1 | exponential atmosphere + trajectory | done |
| 2 | NRLMSISE-00 instead of the placeholder | done, NRLMSIS 2.1 chosen as the base |
| 3 | Sutton–Graves heating | done, plus Earth rotation |
| 4 | thermal response and ablation | done, **plus fragmentation, moved here** |
| 5 | altitude distribution of mass | done, plus a bridge to the experiment |
| 6 | README | done, plus this report |

The main change to the plan: **fragmentation was moved from "later, as a parameter" into step 4, before the histogram is built.** Without it the histogram is systematically shifted down by 25–30 km and answers a different question (section 4).

### 2.2 Code modules

| module | contents |
|---|---|
| `reentry/constants.py` | physical constants with sources |
| `reentry/atmosphere.py` | exponential placeholder and NRLMSIS through pymsis |
| `reentry/vehicle.py` | the object, fragments (a plate radiating from two sides / compact), initial conditions, Earth rotation |
| `reentry/trajectory.py` | equations of motion, integration, Allen–Eggers analytics |
| `reentry/heating.py` | Sutton–Graves, DKR as a shape, hot wall, stagnation pressure, blowing, angular distribution, closed-form evaporation |
| `reentry/ablation.py` | Al 6061 properties, boiling at local pressure, per-band thermal model, the Π regime criterion |
| `reentry/emissivity.py` | Planck weighting, ε(T) of the film and of bare metal, the bridge to reflectance measurements |

---

## 3. Step 1. Trajectory

### 3.1 Choice of coordinate system

Three options were considered:

- **Inertial Cartesian (ECI), state `[x, y, z, vx, vy, vz]`.** General, no singularities. But the altitude and the speed relative to the atmosphere are derived quantities, and the entry angle γ, which is an input, is not a state variable at all.
- **Planar "entry equations" in a planet-fixed polar frame, state `[V, γ, h, s]`.** Chosen.
- **Full 3-DOF in spherical coordinates with rotation**: overkill.

Why the second option: V and h are state variables directly, and they are exactly what enters the heating law `q ~ √ρ(h)·V³` and the main output, the mass histogram by altitude. γ₀ is set directly as an initial condition. The price is a singularity at V → 0, which does not matter because integration stops at ~Mach 1.

Without lift, side force and Earth rotation the motion is **strictly planar**: this is not an approximation but a consequence of the assumptions.

The equations are written in the **inertial** frame: V is the speed relative to the non-rotating Earth. Atmospheric rotation enters later (step 3, section 6.3) only through the speed relative to the air, V_rel, in drag and heating. There are no inertial forces from Earth rotation (Coriolis and centrifugal) in this formulation; they appear only in a rotating frame. The V/r term in the second equation is kinematic and comes from the curvature of the trajectory. Steps 1–2 are computed without atmospheric rotation (`earth_rotation=False` explicitly).

### 3.2 Equations of motion

γ is the flight-path angle to the local horizontal, positive upward; r = R⊕ + h:

```
dV/dt  = −(1/2)·ρ(h)·V_rel²·Cd·A/m  −  (μ/r²)·sin γ
dγ/dt  = cos γ · ( V/r  −  μ/(r²·V) )
dh/dt  = V·sin γ
ds/dt  = V·cos γ · R⊕/r
```

The heart of the problem is the second equation. `V/r` is the centrifugal term and `μ/(r²V)` the gravitational one. At 120 km the circular speed is 7836 m/s while the entry is at 7500 m/s, so these terms almost cancel. At γ₀ = −1.5°:

```
dγ/dt = 1.1554e−3 − 1.2609e−3 = −1.05e−4 rad/s = −0.006 °/s
```

A second consequence: at 120 km the drag deceleration `D/m ≈ 0.011 m/s²` against `g·sin γ ≈ 0.25 m/s²`. For the first ~100 seconds the object effectively flies along a Keplerian arc; drag switches on around 100 km.

### 3.3 Numerical scheme

- Integrator `DOP853` (the system is non-stiff; at tight tolerances it is cheaper than RK45).
- `rtol = 1e−8`, `atol` per component (m/s, rad, m, m): a single scalar cannot be used for quantities of different scales.
- `max_step = 2 s` as a safeguard: on the drag-free arc the integrator grows the step to hundreds of seconds and can jump over the onset of drag.
- Two terminal events: h = 30 km and V = 300 m/s (below Mach 1 hypersonic physics does not apply).
- Resampling through `dense_output` onto a uniform 0.1 s grid. Without it, `argmax` over the solver nodes made the peak altitude depend on `max_step` by ~1 km.

### 3.4 Exponential placeholder atmosphere

`ρ = 1.225·exp(−h/7.2 km)`. H = 7.2 km is not a physical scale height but a fit. Accuracy against the U.S. Standard Atmosphere 1976:

| h, km | 40 | 60 | 80 | 100 | 120 |
|---|---|---|---|---|---|
| model / USSA-76 | ×1.2 | ×1.0 | ×1.0 | ×2 | ×3.2 |

The placeholder is accurate exactly where ablation happens (40–90 km) and off by a factor of three where drag is negligible.

### 3.5 Step 1 results

Base case: m = 175 kg, Cd·A = 1.0 m², β = 175 kg/m², entry at 120 km / 7500 m/s / −1.5°.

```
duration to 30 km         303<!--=step1.base.t_end:.0f--> s
downrange                 1878<!--=step1.base.range_km:.0f--> km
max deceleration          9.8<!--=step1.base.amax_g:.1f--> g at 45.3<!--=step1.base.h_peak_km:.1f--> km, at V = 3862<!--=step1.base.V_peak:.0f--> m/s
γ at the peak             −5.3<!--=step1.base.gamma_peak_deg:.1f-->°   (started at −1.5°)
γ at the end              −21.5<!--=step1.base.gamma_end_deg:.1f-->°
```

![Step 1 trajectory](../figures/step1_trajectory.png)

**Comparison with Allen–Eggers (NACA TR-1381, 1958).** The analytics assume γ = const and drag ≫ gravity. Both assumptions are violated here, so a 20–30% mismatch was expected:

| quantity | numerical | A-E with γ₀ | A-E with γ at the peak |
|---|---|---|---|
| max deceleration, g | 9.83<!--=verify_step1.ae.num.amax_g:.2f--> | 3.84<!--=verify_step1.ae.g0.amax_g:.2f--> | 13.49<!--=verify_step1.ae.gpeak.amax_g:.2f--> |
| peak altitude, km | 45.32<!--=verify_step1.ae.num.h_km:.2f--> | 54.45<!--=verify_step1.ae.g0.h_km:.2f--> | 45.40<!--=verify_step1.ae.gpeak.h_km:.2f--> |
| speed at the peak, m/s | 3862<!--=verify_step1.ae.num.V:.0f--> | 4549<!--=verify_step1.ae.g0.V:.0f--> | — |

In peak g the mismatch turned out to be 156<!--=verify_step1.ae.amax_dev_pct:.0f-->%, not 20–30%: γ goes from −1.5° to −5.3<!--=verify_step1.ae.gamma_peak_deg:.1f-->° by the time of the peak, and `a_max ∝ sin|γ|`. With the actual γ at the peak substituted, the altitude formula gives 45.40<!--=verify_step1.ae.gpeak.h_km:.2f--> km against the numerical 45.32<!--=verify_step1.ae.num.h_km:.2f-->: it is the γ = const assumption that fails, not the equations.

**Confirmed predictions:**

- `h* ∝ H·ln β`: doubling β lowers the peak by −5.0<!--=verify_step1.beta_step.0:.1f--> / −5.0<!--=verify_step1.beta_step.1:.1f--> / −5.0<!--=verify_step1.beta_step.2:.1f--> km against the predicted `H·ln 2 = 5.0 km`.

| β, kg/m² | 43.8<!--=verify_step1.beta.0.beta:.1f--> | 87.5<!--=verify_step1.beta.1.beta:.1f--> | 175<!--=verify_step1.beta.2.beta:.0f--> | 350<!--=verify_step1.beta.3.beta:.0f--> |
|---|---|---|---|---|
| peak altitude, km | 55.30<!--=verify_step1.beta.0.h_km:.2f--> | 50.32<!--=verify_step1.beta.1.h_km:.2f--> | 45.32<!--=verify_step1.beta.2.h_km:.2f--> | 40.36<!--=verify_step1.beta.3.h_km:.2f--> |
| max deceleration, g | 9.55<!--=verify_step1.beta.0.amax_g:.2f--> | 9.69<!--=verify_step1.beta.1.amax_g:.2f--> | 9.83<!--=verify_step1.beta.2.amax_g:.2f--> | 9.97<!--=verify_step1.beta.3.amax_g:.2f--> |

- `a_max` does not depend on β: an 8-fold change in β changes the peak g by 4%.

### 3.6 An unexpected result: the entry angle drops out

The altitude of peak deceleration barely depends on the entry angle within the specified range:

| γ₀ | time, s | downrange, km | max g | peak altitude, km |
|---|---|---|---|---|
| −1.0° | 348<!--=step1.gamma_sweep.g10.t:.0f--> | 2200<!--=step1.gamma_sweep.g10.range_km:.0f--> | 9.5<!--=step1.gamma_sweep.g10.amax_g:.1f--> | 45.3<!--=step1.gamma_sweep.g10.h_km:.1f--> |
| −1.5° | 303<!--=step1.gamma_sweep.g15.t:.0f--> | 1878<!--=step1.gamma_sweep.g15.range_km:.0f--> | 9.8<!--=step1.gamma_sweep.g15.amax_g:.1f--> | 45.3<!--=step1.gamma_sweep.g15.h_km:.1f--> |
| −2.0° | 268<!--=step1.gamma_sweep.g20.t:.0f--> | 1629<!--=step1.gamma_sweep.g20.range_km:.0f--> | 10.4<!--=step1.gamma_sweep.g20.amax_g:.1f--> | 45.3<!--=step1.gamma_sweep.g20.h_km:.1f--> |
| −3.0° | 216<!--=step1.gamma_sweep.g30.t:.0f--> | 1276<!--=step1.gamma_sweep.g30.range_km:.0f--> | 11.8<!--=step1.gamma_sweep.g30.amax_g:.1f--> | 45.0<!--=step1.gamma_sweep.g30.h_km:.1f--> |

The flight time changes by a factor of 1.6, the peak altitude does not. The gravity turn erases the initial angle before drag switches on, so the effective γ at the peak is about the same regardless of the input.

For the project this is directly useful: the entry angle drops out of the list of parameters that the **altitude** distribution of injection depends on. Later (section 12) it turned out that it returns to the **mass** budget through the flight duration.

![Entry angle sweep](../figures/step1_gamma_sweep.png)

---

## 4. Diagnosis: the 25–30 km gap and moving fragmentation forward

### 4.1 The problem

Observed breakup of spacecraft during entry:

| spacecraft | breakup altitude |
|---|---|
| ATV-1 | 74 km |
| Cygnus OA6 | 70 km |
| Cluster II (SALSA) | 80 km |

SESAM defaults: solar panels separate at 95 km, the body breaks up at 78 km (Lips et al., SDC6).

For the intact object in the model deceleration peaks at 45 km and heating at 56 km. The gap with the observations is 25–30 km. The cause is not an error but the "intact object" assumption.

This is critical precisely for this problem: the whole question of the project is at which altitude the injection happens. The particle lifetime depends on it, and so does whether particles enter the mesosphere with years of descent ahead or go straight into the stratosphere. A model that puts all the mass at 45 km answers a different question.

### 4.2 Heating peak versus deceleration peak

The altitude of the `√ρ·V³` heating peak can already be computed in step 1: it depends neither on the Sutton–Graves constant nor on the nose radius.

```
deceleration peak   45.3<!--=step1b.peaks.decel_h_km:.1f--> km   V = 3862<!--=step1b.peaks.decel_V:.0f--> m/s   9.8<!--=step1b.peaks.decel_g:.1f--> g
heating peak        56.4<!--=step1b.peaks.heat_h_km:.1f--> km   V = 6258<!--=step1b.peaks.heat_V:.0f--> m/s
separation          11.1<!--=step1b.peaks.separation_km:.1f--> km
```

Heating peaks higher because speed enters to the third power and density to the power 1/2.

### 4.3 The drag coefficient does not close the gap

For a randomly tumbling irregular body in the continuum regime, survivability codes usually take Cd = 1.4–2.0. The base value is 1.5.

| Cd | β, kg/m² | h deceleration, km | h heating, km |
|---|---|---|---|
| 1.0 | 175<!--=step1b.cd.cd10.beta:.0f--> | 45.3<!--=step1b.cd.cd10.h_decel_km:.1f--> | 56.4<!--=step1b.cd.cd10.h_heat_km:.1f--> |
| 1.5 | 117<!--=step1b.cd.cd15.beta:.0f--> | 48.3<!--=step1b.cd.cd15.h_decel_km:.1f--> | 59.4<!--=step1b.cd.cd15.h_heat_km:.1f--> |
| 2.2 | 80<!--=step1b.cd.cd22.beta:.0f--> | 51.0<!--=step1b.cd.cd22.h_decel_km:.1f--> | 62.2<!--=step1b.cd.cd22.h_heat_km:.1f--> |

The whole sweep from 1.0 to 2.2 is worth 5.8<!--=step1b.cd.span_heat_km:.1f--> km, five times less than the gap.

### 4.4 Fragmentation closes it

For geometrically similar fragmentation `m ~ L³`, `A ~ L²`, so `β ~ L`. Breakup at 78 km, fragments with different β (Cd = 1.5):

| fragment β, kg/m² | h of heating peak, km |
|---|---|
| 117<!--=step1b.frag.r1.beta:.0f--> (intact) | 59.4<!--=step1b.frag.r1.h_heat_km:.1f--> |
| 58<!--=step1b.frag.r2.beta:.0f--> | 64.9<!--=step1b.frag.r2.h_heat_km:.1f--> |
| 23<!--=step1b.frag.r5.beta:.0f--> | 72.5<!--=step1b.frag.r5.h_heat_km:.1f--> |
| 12<!--=step1b.frag.r10.beta:.0f--> | **78.0<!--=step1b.frag.r10.h_heat_km:.1f-->: the peak is at the breakup altitude itself** |
| 6<!--=step1b.frag.r20.beta:.0f--> | 78.0<!--=step1b.frag.r20.h_heat_km:.1f--> |
| 2.3<!--=step1b.frag.r50.beta:.1f--> | 78.0<!--=step1b.frag.r50.h_heat_km:.1f--> |

**Below β ≈ 20 kg/m² the dependence saturates.** The heat flux depends only on ρ and V, and at the moment of breakup these are the same for all fragments. A light fragment decelerates faster than the density grows, so its flux falls monotonically from the breakup point. For light fragments the injection altitude is set by the **breakup altitude**, not by β.

Dependence on the breakup altitude for tenfold fragmentation:

| breakup h, km | 95 | 84 | 78 | 70 |
|---|---|---|---|---|
| h of heating peak, km | 76.9<!--=step1b.breakup.h95.h_heat_km:.1f--> | 77.8<!--=step1b.breakup.h84.h_heat_km:.1f--> | 78.0<!--=step1b.breakup.h78.h_heat_km:.1f--> | 70.0<!--=step1b.breakup.h70.h_heat_km:.1f--> |

![Fragmentation diagnosis](../figures/step1b_fragmentation.png)

### 4.5 Two refinements

1. **Not N equal fragments but a spectrum of β.** To put the peak at 75 km one needs β ≈ 17<!--=step1b.inverse.beta_needed:.0f--> against 117 for the intact object, a drop by a factor of 6.7<!--=step1b.inverse.factor:.1f-->. With equal fragmentation that is N ~ 300 fragments, which is a lot. A spectrum is more realistic: panels and MLI at β ~ 1–10 kg/m², the primary structure at ~30–80.
2. **The flux peak is an upper bound on the injection altitude, not the injection itself.** Evaporation needs accumulated heat: warming to ~900 K, melting, boiling (~2000 K at local pressure, section 8.4). The mass goes in below the flux peak.

**Decision:** fragmentation is moved into step 4, before the histogram is built.

---

## 5. Step 2. Real atmosphere

### 5.1 Implementation

NRLMSIS through `pymsis`, tabulated and splined:

- **Tabulation, not a direct call.** `solve_ivp` evaluates the density tens of thousands of times per run. One MSIS call on a 250 m grid up to 200 km.
- **Spline in log ρ, not in ρ.** The density spans 6 orders of magnitude; its logarithm is nearly linear in altitude.
- **Cubic, not linear.** Linear interpolation in log ρ gives a piecewise-exponential density with kinks in the derivative, and the adaptive integrator cuts its step at each of them. A cubic spline is C2-smooth.
- **The profile is frozen at the moment of entry.** In 300 s the object covers ~1800 km, ~16° of arc, of which ~10° in latitude (track azimuth ~52° for i = 53° at 40° S). ρ(70 km) changes by a few percent over that, less than the uncertainty of MSIS itself.

### 5.2 The placeholder turned out better than it deserved

| h, km | exponential | NRLMSISE-00 | exp / MSIS | MSIS 2.1 / 00 | local H, km |
|---|---|---|---|---|---|
| 40 | 4.74e−3 | 3.83e−3 | 1.24 | 1.00 | 6.80 |
| 60 | 2.95e−4 | 2.81e−4 | 1.05 | 0.91 | 7.61 |
| 70 | 7.34e−5 | 7.05e−5 | 1.04 | 0.90 | 6.91 |
| 80 | 1.83e−5 | 1.58e−5 | 1.16 | 0.89 | 6.50 |
| 100 | 1.14e−6 | 5.40e−7 | 2.11 | 1.00 | 5.27 |
| 120 | 7.08e−8 | 1.80e−8 | 3.92 | 0.99 | 7.80 |

The local scale height over 40–120 km ranges from 5.2<!--=step2.density.H_min_km:.1f--> to 8.0<!--=step2.density.H_max_km:.1f--> km, a spread of 39<!--=step2.density.H_spread_pct:.0f-->% around the fitted 7.2 km. Yet the trajectory barely moved:

| atmosphere | t, s | downrange, km | h deceleration | max g | h heating |
|---|---|---|---|---|---|
| exponential | 306<!--=step2.traj.exp.t:.0f--> | 1822<!--=step2.traj.exp.range_km:.0f--> | 48.3<!--=step2.traj.exp.h_decel_km:.1f--> | 9.7<!--=step2.traj.exp.amax_g:.1f--> | 59.4<!--=step2.traj.exp.h_heat_km:.1f--> |
| NRLMSISE-00 | 305<!--=step2.traj.m00.t:.0f--> | 1843<!--=step2.traj.m00.range_km:.0f--> | 47.1<!--=step2.traj.m00.h_decel_km:.1f--> | 9.1<!--=step2.traj.m00.amax_g:.1f--> | 59.6<!--=step2.traj.m00.h_heat_km:.1f--> |
| NRLMSIS 2.1 | 306<!--=step2.traj.m21.t:.0f--> | 1854<!--=step2.traj.m21.range_km:.0f--> | 46.3<!--=step2.traj.m21.h_decel_km:.1f--> | 9.3<!--=step2.traj.m21.amax_g:.1f--> | 58.8<!--=step2.traj.m21.h_heat_km:.1f--> |

Above 90 km, where the placeholder was off by multiples, drag is negligible; in the 40–80 km zone it was off by 4–24%.

![Atmosphere](../figures/step2_atmosphere.png)

![Trajectory in the real atmosphere](../figures/step2_trajectory.png)

### 5.3 Correction: solar activity barely matters at 120 km

In step 1 it was twice claimed that solar activity changes the density at 120 km "by multiples". The check refuted this:

| h, km | 70 | 100 | 120 | 150 | 200 | 300 | 400 |
|---|---|---|---|---|---|---|---|
| ρ(F10.7=220) / ρ(F10.7=70) | 0.99<!--=verify_step2.solar_ratio.70:.2f--> | 0.95<!--=verify_step2.solar_ratio.100:.2f--> | **1.07<!--=verify_step2.solar_ratio.120:.2f-->** | 1.23<!--=verify_step2.solar_ratio.150:.2f--> | 2.18<!--=verify_step2.solar_ratio.200:.2f--> | 6.35<!--=verify_step2.solar_ratio.300:.2f--> | 15.2<!--=verify_step2.solar_ratio.400:.1f--> |

Mechanism: solar heating controls the exospheric temperature, which sets the scale height in the **upper** thermosphere. At 120 km the atmosphere is still governed by the mesosphere below. The claim holds at 300+ km, i.e. for orbital lifetime, but not for entry.

### 5.4 What actually drives the atmospheric spread

Spread of the heating-peak altitude for each factor:

| factor | spread, km |
|---|---|
| latitude −75°…0° | 3.9<!--=verify_step2.env_span_km.lat:.1f--> |
| season March/September | 1.6<!--=verify_step2.env_span_km.season:.1f--> |
| geomagnetic storm Ap 4 → 80 | 0.1<!--=verify_step2.env_span_km.geomag:.1f--> |
| solar activity F10.7 70 → 220 | 0.0<!--=verify_step2.env_span_km.solar:.1f--> |

Geography and season beat solar activity by two orders of magnitude: this is the temperature structure of the mesosphere, not of the thermosphere. In practice the latitude of the entry point must be fixed explicitly, and F10.7 need not be swept at all.

The numbers in this section are for the step 2 configuration (Cd = 1.5, no atmospheric rotation). In the step 3 budget (section 12, with rotation) the same factors give 4.6<!--=step3.budget.latitude.h_km:.1f--> km for latitude and 1.8<!--=step3.budget.season.h_km:.1f--> km for season.

### 5.5 Seams in NRLMSISE-00 and the choice of version 2.1

The spline accuracy check revealed two local error spikes, 1.2e−3 against a background of 4e−7. It turned out not to be the interpolation: NRLMSISE-00 itself has a broken derivative of `ln ρ` at **72.5 km** and **123.4 km**. NRLMSIS 2.1 has no kinks.

The cause is structural. In NRLMSISE-00 the thermospheric densities were computed independently of the lower layers, and the profiles were joined a posteriori; that seam is what shows up. NRLMSIS 2.0 removed the joining: the hydrostatic profile is continuous from the ground to the exosphere, and the transition from the mixed region to diffusive separation is continuous from about 70 km up.

And 2.0 was refitted precisely for the range that matters here. Emmert et al. (2021): *"In the mesosphere and below, residual biases and standard deviations are considerably lower than NRLMSISE-00"*. New mesospheric, stratospheric and tropospheric temperature data were assimilated, and atomic oxygen was extended down to 50 km.

The choice was between a model whose mesosphere is a known weak spot and a model rebuilt for the sake of the mesosphere, while the injection is decided at 50–90 km. One of the 00 seams lies at 72.5 km, right in the ablation zone.

The argument "comparability with the literature" does not hold: Barker computes in GEOS-Chem and Maloney in WACCM, with their own meteorology; they do not use MSIS. MSIS comparability is needed only against DRAMA, for which a switchable `version=0` is enough.

**Decision:** NRLMSIS 2.1 is the base, version 00 is kept switchable, and the difference between them goes into the uncertainty budget as a row (0.9<!--=step3.budget.msis_version.h_km:.1f--> km in altitude) rather than a one-off choice.

---

## 6. Step 3. Aerodynamic heating

### 6.1 The Sutton–Graves correlation

```
q_stag = k · √(ρ/Rn) · V_rel³          k = 1.7415e−4
```

What the correlation carries with it:

- It is for the **stagnation point** of a spherical nose.
- **Convective** flux only. Radiative heating from the shock layer switches on around 10 km/s; for entry from orbit (7.5 km/s) it can be neglected.
- **Cold wall**: the hot-wall correction is introduced separately.

### 6.2 Units of the constant: an error would cost four orders of magnitude

NASA TFAWS gives k = 1.7415e−4 for Earth and labels the result as W/cm². Direct substitution refutes this. The Stardust benchmark (Rn = 0.23 m, V = 12.6 km/s, ρ ≈ 3e−4 kg/m³, computed peak heating ~1200 W/cm², a design/reconstructed value that includes the radiative part; the flux was not measured directly):

```
q = 1.7415e−4 · √(3e−4/0.23) · 12600³ = 1.258e7
   if W/m²  → 1258<!--=verify_step3.stardust_wcm2:.0f--> W/cm²   agrees with the computed value
   if W/cm² → off by a factor of 10⁴
```

**k = 1.7415e−4 with SI inputs gives W/m².** The units were settled from a flight benchmark, not from the label in the source. A variant of the constant, 1.83e−4 (~5% higher), exists; it is taken as an estimate of the uncertainty of the correlation itself.

### 6.3 Earth rotation: the latitude cancels

The atmosphere rotates eastward with the Earth at `u = ω·R·cos(lat)`. Along the track its projection `u·sin A` acts, where A is the track azimuth. From spherical trigonometry for an orbit of inclination i: `sin A = cos i / cos(lat)`. Substituting:

```
v_along = ω·R·cos(lat) · cos i / cos(lat) = ω·R⊕·cos i
V_rel   = V − ω·R⊕·cos i
```

**The contribution depends only on the inclination**, not on where the object enters.

| i | drift, m/s | peak q, W/cm² | peak h, km | Q, MJ/m² |
|---|---|---|---|---|
| 0° (equatorial prograde) | +465<!--=verify_step3.rotation.i0.v_corot:+.0f--> | 92.2<!--=verify_step3.rotation.i0.q_peak_wcm2:.1f--> | 58.3<!--=verify_step3.rotation.i0.h_km:.1f--> | 85.1<!--=verify_step3.rotation.i0.Q_MJ:.1f--> |
| 53° (typical LEO) | +280<!--=verify_step3.rotation.i53.v_corot:+.0f--> | 98.3<!--=verify_step3.rotation.i53.q_peak_wcm2:.1f--> | 58.5<!--=verify_step3.rotation.i53.h_km:.1f--> | 90.8<!--=verify_step3.rotation.i53.Q_MJ:.1f--> |
| 90° (polar) | 0<!--=verify_step3.rotation.i90.v_corot:.0f--> | 108.0<!--=verify_step3.rotation.i90.q_peak_wcm2:.1f--> | 58.8<!--=verify_step3.rotation.i90.h_km:.1f--> | 99.9<!--=verify_step3.rotation.i90.Q_MJ:.1f--> |
| 98° (SSO) | −65<!--=verify_step3.rotation.i98.v_corot:+.0f--> | 110.3<!--=verify_step3.rotation.i98.q_peak_wcm2:.1f--> | 58.8<!--=verify_step3.rotation.i98.h_km:.1f--> | 102.1<!--=verify_step3.rotation.i98.Q_MJ:.1f--> |
| 180° (retrograde) | −465<!--=verify_step3.rotation.i180.v_corot:+.0f--> | 125.3<!--=verify_step3.rotation.i180.q_peak_wcm2:.1f--> | 59.2<!--=verify_step3.rotation.i180.h_km:.1f--> | 116.1<!--=verify_step3.rotation.i180.Q_MJ:.1f--> |

For i = 53° switching rotation on changes the peak flux by −9.0<!--=verify_step3.rotation.q_change_pct:.1f-->% (the estimate before the computation was −11%). The full span over inclination is 36%. Self-consistency test: a polar orbit (i = 90°) coincides exactly with rotation switched off.

### 6.4 Detra–Kemp–Riddell as a shape only

The absolute DKR constant could not be taken from a reliable primary source: open implementations use different normalizations and exponents (3.15 versus 3.25), and the original 1957 paper was not found in open access. There is no point in plugging a number from a random implementation just to get a second curve on the plot.

The substance of the comparison does not depend on the constant: the question is the sensitivity to the **velocity exponent**, 3.0 versus 3.15. DKR is normalized to Sutton–Graves at a reference point, and the shape is compared:

| | Sutton–Graves | DKR | difference |
|---|---|---|---|
| peak altitude, km | 58.49<!--=verify_step3.dkr.h_sg_km:.2f--> | 58.89<!--=verify_step3.dkr.h_dkr_km:.2f--> | +0.40<!--=verify_step3.dkr.dh_km:+.2f--> |
| peak flux, W/cm² | 98.3<!--=verify_step3.dkr.q_sg_wcm2:.1f--> | 94.5<!--=verify_step3.dkr.q_dkr_wcm2:.1f--> | −3.8<!--=verify_step3.dkr.dq_pct:.1f-->% |
| heat load, MJ/m² | 90.8<!--=verify_step3.dkr.Q_sg_MJ:.1f--> | 86.9<!--=verify_step3.dkr.Q_dkr_MJ:.1f--> | −4.4<!--=verify_step3.dkr.dQ_pct:.1f-->% |

The choice of correlation is a fourth-order effect, weaker even than the MSIS version.

### 6.5 The nose radius does not move the peak altitude

| Rn, m | 0.10 | 0.25 | 0.50 | 1.00 | 2.00 |
|---|---|---|---|---|---|
| peak q, W/cm² | 219.8<!--=verify_step3.nose.rn10cm.q_peak_wcm2:.1f--> | 139.0<!--=verify_step3.nose.rn25cm.q_peak_wcm2:.1f--> | 98.3<!--=verify_step3.nose.rn50cm.q_peak_wcm2:.1f--> | 69.5<!--=verify_step3.nose.rn100cm.q_peak_wcm2:.1f--> | 49.1<!--=verify_step3.nose.rn200cm.q_peak_wcm2:.1f--> |
| peak h, km | 58.49<!--=verify_step3.nose.h_km:.2f--> | 58.49<!--=verify_step3.nose.h_km:.2f--> | 58.49<!--=verify_step3.nose.h_km:.2f--> | 58.49<!--=verify_step3.nose.h_km:.2f--> | 58.49<!--=verify_step3.nose.h_km:.2f--> |

The spread in altitude is exactly zero. Rn enters as a `1/√Rn` factor, which is constant along the trajectory and factors out of argmax. So Rn belongs to the **mass** budget, not the **altitude** budget.

### 6.6 Step 3 base case

m = 175 kg, Cd = 1.5, A = 1 m², Rn = 0.5 m, β = 117 kg/m², i = 53°, NRLMSIS 2.1:

```
heating peak        98.3<!--=step3.base.q_peak_wcm2:.1f--> W/cm²  at 58.5<!--=step3.base.h_heat_km:.1f--> km, t = 218<!--=step3.base.t_heat:.0f--> s, V_rel = 6090<!--=step3.base.V_rel_heat:.0f--> m/s
deceleration peak    9.1<!--=step3.base.amax_g:.1f--> g       at 45.8<!--=step3.base.h_decel_km:.1f--> km
heat load           90.8<!--=step3.base.Q_MJ:.1f--> MJ/m²
peak separation     12.7<!--=step3.base.separation_km:.1f--> km
```

Energy check: the heat load times the area is 1.8<!--=verify_step3.heat_fraction_pct:.1f-->% of the object's kinetic energy (an upper bound).

The observed breakup at 70–80 km is still 12–22 km above the heating peak. Neither the atmosphere, nor rotation, nor the choice of correlation closed the gap; only fragmentation does.

![Heating](../figures/step3_heating.png)

---

## 7. Mass metric: the sign flips with size

### 7.1 The error in the first version of the budget

The first version of the uncertainty budget computed the "mass" axis through the stagnation-point heat load, MJ/m². That is an **error in sign**, not just in magnitude.

The ablated mass is set by the total absorbed energy `∫P dt`, where `P = φ·q_stag·A_wet`. For a geometrically similar body:

```
q_stag  ~ L^(−1/2)      correlation
A_wet   ~ L^(+2)        geometry
─────────────────────────────────
P       ~ L^(+3/2)      total power GROWS with size
m       ~ L^(+3)
P/m     ~ L^(−3/2)      specific load FALLS with size
```

Per unit area a large body heats less; in total energy it heats more. The row "Rn 0.1–2 m, 347%" in the first version of the budget pointed the wrong way.

### 7.2 Checking the exponents

On a frozen trajectory (the same flux history, only the geometry changes) the exponents came out as **+1.500 and −1.500**, matching to the sixth digit.

Self-consistently, when β changes with size:

| L | β, kg/m² | peak h, km | E total, MJ | E/m, MJ/kg |
|---|---|---|---|---|
| 0.1 | 11.7<!--=verify_step3.scaling.L1.beta:.1f--> | 74.5<!--=verify_step3.scaling.L1.h_km:.1f--> | 1.0<!--=verify_step3.scaling.L1.E_MJ:.1f--> | 5.55<!--=verify_step3.scaling.L1.Em_MJkg:.2f--> |
| 0.2 | 23.3<!--=verify_step3.scaling.L2.beta:.1f--> | 69.9<!--=verify_step3.scaling.L2.h_km:.1f--> | 3.9<!--=verify_step3.scaling.L2.E_MJ:.1f--> | 2.79<!--=verify_step3.scaling.L2.Em_MJkg:.2f--> |
| 0.5 | 58.3<!--=verify_step3.scaling.L5.beta:.1f--> | 63.5<!--=verify_step3.scaling.L5.h_km:.1f--> | 24.5<!--=verify_step3.scaling.L5.E_MJ:.1f--> | 1.12<!--=verify_step3.scaling.L5.Em_MJkg:.2f--> |
| 1.0 | 116.7<!--=verify_step3.scaling.L10.beta:.1f--> | 58.5<!--=verify_step3.scaling.L10.h_km:.1f--> | 98.1<!--=verify_step3.scaling.L10.E_MJ:.1f--> | 0.56<!--=verify_step3.scaling.L10.Em_MJkg:.2f--> |
| 2.0 | 233.3<!--=verify_step3.scaling.L20.beta:.1f--> | 53.2<!--=verify_step3.scaling.L20.h_km:.1f--> | 391.0<!--=verify_step3.scaling.L20.E_MJ:.1f--> | 0.28<!--=verify_step3.scaling.L20.Em_MJkg:.2f--> |

The exponents become +2.0 and −1.0: the growth of β partly damps the effect, but the sign persists. (This is a cold wall; table 7.5 has the same energy with the hot-wall correction at T_boil, so E/m there is lower by a factor of ~0.8.)

**Hence the second mechanism by which fragmentation decides the outcome.** Fragments do not just decelerate higher because of their smaller β; they also receive an order of magnitude more heat per kilogram. Two independent mechanisms act in the same direction.

### 7.3 Wetted area: Cauchy's formula, not a fit

For any **convex** body the projected area averaged over random orientations equals ¼ of the surface area. The `area` parameter of a tumbling body is exactly the mean projection (the same one used in drag), so:

```
A_wet = 4 · area          exact, for any convex shape
```

### 7.4 A shape factor instead of an unmeasurable nose radius

Sutton–Graves is a correlation for the stagnation point of a hemispherical nose, and a tumbling irregular spacecraft physically has no nose radius: Rn is a fitting parameter for it. The standard workaround in demise codes is the flux averaged over the wetted surface, `φ·q_stag`, with `φ ≈ 0.25–0.30` for a body in random orientation. φ = 0.27 is used. This removes the problem of an unmeasurable parameter: the spread of φ is worth 20%, the spread of the effective Rn 124%.

### 7.5 How much energy is needed to melt and to evaporate

Boiling is at the stagnation pressure of ~1 kPa (T_boil = 2046 K, section 8.4); Al 6061 properties are in section 16.

| stage | MJ/kg |
|---|---|
| heating to melting, 1038 × (890 − 300) | 0.61<!--=verify_step3.energy.heat_to_melt:.2f--> |
| heat of fusion | 0.40<!--=verify_step3.energy.fusion:.2f--> |
| heating to boiling, 1177 × (2046<!--=verify_step3.energy.T_boil:.0f--> − 890) | 1.36<!--=verify_step3.energy.heat_to_boil:.2f--> |
| heat of vaporization at 2046<!--=verify_step3.energy.T_boil:.0f--> K | 11.20<!--=verify_step3.energy.vaporization:.2f--> |
| **total** | **13.57<!--=verify_step3.energy.total:.2f-->** |

The ratio of the received specific energy to the energy for complete melting (1.01 MJ/kg) and for complete evaporation is the mass fraction that could in principle be melted or evaporated if all the energy went into it (no re-radiation: an upper bound). E/m includes the hot-wall correction:

| size L | mass, kg | E/m, MJ/kg | meltable fraction | vaporizable fraction |
|---|---|---|---|---|
| 1.0 (intact object) | 175<!--=verify_step3.energy.L10.mass:.0f--> | 0.46<!--=verify_step3.energy.L10.Em:.2f--> | 45<!--=verify_step3.energy.L10.meltable_pct:.0f-->% | **3<!--=verify_step3.energy.L10.vaporizable_pct:.0f-->%** |
| 0.5 | 21.9<!--=verify_step3.energy.L5.mass:.1f--> | 0.91<!--=verify_step3.energy.L5.Em:.2f--> | 90<!--=verify_step3.energy.L5.meltable_pct:.0f-->% | 7<!--=verify_step3.energy.L5.vaporizable_pct:.0f-->% |
| 0.2 | 1.40<!--=verify_step3.energy.L2.mass:.2f--> | 2.26<!--=verify_step3.energy.L2.Em:.2f--> | 100<!--=verify_step3.energy.L2.meltable_pct:.0f-->% | 17<!--=verify_step3.energy.L2.vaporizable_pct:.0f-->% |
| 0.1 | 0.18<!--=verify_step3.energy.L1.mass:.2f--> | 4.49<!--=verify_step3.energy.L1.Em:.2f--> | 100<!--=verify_step3.energy.L1.meltable_pct:.0f-->% | 33<!--=verify_step3.energy.L1.vaporizable_pct:.0f-->% |

The demisability of an OneWeb/SpaceX-type design is 95% (Ferreira, UNOOSA 2024). But that is a **melting** criterion (section 8.6), and it must be compared with the meltable fraction: the intact object gives 45%, an L = 0.2 fragment 100%. The mass gap is about 2×, and fragmentation closes it, just like the altitude gap. The first version compared 95% with the vaporizable 3% and got "a thirtyfold gap", which compares different quantities.

### 7.6 Hot wall

Sutton–Graves gives the cold-wall flux. The correction:

```
q_hot = q_cold · (1 − h_w/h_0),     h_0 = V²/2,     h_w = c_p,air · T_wall
```

We use the enthalpy of the **air** at the wall, not of the metal: the flux is driven by the enthalpy difference of the gas.

| V, m/s | factor at T = 2046 K (boiling at ~1 kPa) | at T = 2740 K (first version) |
|---|---|---|
| 7500 | 0.905<!--=verify_step3.hot_wall.V7500.Tb:.3f--> (−9%) | 0.873<!--=verify_step3.hot_wall.V7500.T2740:.3f--> (−13%) |
| 6100 | 0.857<!--=verify_step3.hot_wall.V6100.Tb:.3f--> (−14%) | 0.809<!--=verify_step3.hot_wall.V6100.T2740:.3f--> (−19%) |
| 4000 | 0.668<!--=verify_step3.hot_wall.V4000.Tb:.3f--> (−33%) | 0.555<!--=verify_step3.hot_wall.V4000.T2740:.3f--> (−45%) |

An estimate from the metal (`c_p·ΔT ≈ 2.5 MJ/kg`) gives −9% at 7500 m/s; the difference between the two routes is the uncertainty of the correction. The correction grows as the body decelerates.

### 7.7 Blowing of ablation products closes analytically

The evaporating material goes into the boundary layer and blocks part of the flux (transpiration cooling). The classic linear correction is `q_net = q_hw − η·ṁ''·h_0`, and the ablation rate is itself proportional to the flux, `ṁ'' = q_net/H_eff`. Substituting:

```
q_net = q_hw − η·(q_net/H_eff)·h_0
q_net = q_hw / (1 + η·h_0/H_eff)
```

**No implicit solver is needed:** both the ablation law and the blocking are linear in ṁ'', so the nonlinearity reduces to division by a factor. Checked against 200 fixed-point iterations, agreeing to 1e−9. For η = 0.3, h_0 = 2.8e7, H_eff ≈ 1.2e7 the factor is 0.588: blowing cuts the flux by 41%.

**Which η.** The first version labelled "0.3 laminar, 0.6 turbulent". It is the other way round: in a laminar layer blowing blocks the flux more strongly, and turbulent mixing damps the effect (transpiration cooling experiments in hypersonic flow, AIAA J. 10.2514/1.J053053). At 70–80 km the layer is laminar. The specific value of η has not been checked against a primary source, so blowing is **not part of the base case** but a sensitivity axis: η = 0.6 in the per-band model lowers evaporation from 8.5 to 4.9 kg Al (section 12). In the model only the excess flux in boiling bands is blocked: `q_evap = (q_in − rad)/(1 + η·(h_0 − h_w)/L)`.

Blowing has a counterpart of the opposite sign: the oxygen that diffuses through the same boundary layer reacts with the aluminium and releases heat. It is treated in section 9.5.

---

## 8. Step 4. Fragmentation, thermal response, ablation

### 8.1 The peak flux is scale-invariant

For geometrically similar bodies the peak `q_stag` barely depends on size: ~85 W/cm² both for the intact object and for a two-centimetre fragment. The reason is analytic: by Allen–Eggers `ρ* ∝ β ∝ L`, and `Rn ∝ L`, so the scale cancels in `√(ρ/Rn)`.

Consequence: the radiative equilibrium temperature `εσT⁴ = φ·q_stag` is invariant too; at ε = 0.3 it is **~1915 K for any size**.

The problem splits:

- **Melting is energy-limited:** it depends on `E/m ∝ L⁻¹`, i.e. strongly on size.
- **Evaporation is temperature-limited:** it depends on the flux and on the temperature at which the metal boils.

Boiling threshold for the mean flux, `εσT_boil⁴/φ`:

| | ε = 0.3 | ε = 0.2 |
|---|---|---|
| boiling at 1 atm, 2792 K | 383<!--=step4.threshold.atm.e30:.0f--> W/cm² | 255<!--=step4.threshold.atm.e20:.0f--> W/cm² |
| boiling at ~1 kPa, 2046 K | **110<!--=step4.threshold.kpa.e30:.0f--> W/cm²** | **74<!--=step4.threshold.kpa.e20:.0f--> W/cm²** |

With boiling at 1 atm (the first version) the mean flux fell short of boiling by a factor of 4. With boiling at the local pressure (section 8.4) the threshold is at the level of the peak flux. Thin fragments also see more: plates with a 5 mm edge right after breakup at 78 km see a peak flux of 220<!--=step4.fragments.film.panels.q_peak_wcm2:.0f-->–401<!--=step4.fragments.film.mli.q_peak_wcm2:.0f--> W/cm².

### 8.2 A distribution instead of the mean

Evaporation is a threshold problem, and radiation goes as `T⁴`. In such a problem the mean systematically underestimates: evaporation happens where the flux is above the mean, and the mean does not see that place. So the surface is split into bands by the angle from the stagnation point, each with its own energy balance:

```
q(θ)/q_stag = cos θ         (Newtonian pressure + Lees 1956)
```

**There are no free parameters.** The alternative, splitting the body into a "nose" and "the rest", would introduce a new unknown (the nose area fraction) that directly multiplies the evaporated mass.

**Self-check:** the mean of this distribution over the full sphere is

```
⟨cos θ⟩ = (1/4π) ∫₀^(π/2) cos θ · 2π sin θ dθ = 1/4
```

exactly 0.25, i.e. it falls into the standard shape-factor range of 0.25–0.30. No new physics is introduced; the existing physics is just no longer collapsed. It is the same geometric fact as Cauchy's formula, seen from the other side.

**Where this assumption may fail.** The `cos θ` distribution means a stable orientation. If a fragment tumbles faster than it heats up, every surface element sees the mean flux 0.25·q_stag. The thermal time constant of a 1 mm plate at ~2000 K is about 3 s. This limit is checked explicitly (the `flux_mode="uniform"` mode): evaporation drops from 8.5<!--=step5.scenarios.film.total:.1f--> to 5.9<!--=step5.sensitivity.orientation.lo:.1f--> kg Al (section 12). The result depends on this assumption within a factor of 1.5.

### 8.3 Closed-form evaporation

Local radiative equilibrium `n·εσT(θ)⁴ = q_stag·cos θ` (n is the number of radiating sides, section 8.5) gives boiling where `cos θ > C`, with

```
C = n·ε·σ·T_boil⁴ / q_stag
```

The boiling fraction of a full sphere's surface:

```
A_frac = (1 − cos θ_c)/2 = (1 − C)/2
```

The evaporation rate is the integral of the excess flux over the boiling surface. With `dA = 2πR² d(cos θ)` the integral gives `πR²·q·(1 − C)²`, and `πR² = A_wet/4`:

```
ṁ = A_wet · q_stag · (1 − C)² / (4 · L_vap)
```

The square in `(1 − C)` is not a typo: near the threshold both the area and the excess flux go to zero. Both formulas are checked against numerical integration to 1e−9. In the computation they serve as a check; the computation itself is the per-band model.

### 8.4 Per-band thermal model

The state of each band is the specific enthalpy `h_s` and the mass `m_vap` evaporated from it. The temperature is derived from the enthalpy by a monotonic piecewise function with plateaus at the phase transitions, so the phase transitions need no branching in the ODE right-hand side:

```
h₁ = c_p,sol(T_melt − T₀)            = 0.61<!--=step4.criteria.h1:.2f--> MJ/kg    start of melting
h₂ = h₁ + L_fus                      = 1.01<!--=step4.criteria.h2:.2f--> MJ/kg    end of melting
h₃ = h₂ + c_p,liq(T_boil(p) − T_melt) ≈ 2.0–2.4 MJ/kg  start of boiling
```

**The boiling temperature is at the local pressure.** 2792 K is the boiling point of Al at 1 atm. At a fragment surface the pressure is the stagnation pressure `p₀ = 0.92·ρV²` (the Rayleigh pitot formula), and the boiling plateau sits at

```
1/T_boil(p) = 1/T_boil,1atm − (R/(M·L)) · ln(p/p_atm)      (Clausius–Clapeyron)
```

| p, Pa | 100 | 300 | 1 000 | 3 000 | 101 325 |
|---|---|---|---|---|---|
| T_boil, K | 1806<!--=step4.boil.p100.T:.0f--> | 1913<!--=step4.boil.p300.T:.0f--> | 2046<!--=step4.boil.p1000.T:.0f--> | 2185<!--=step4.boil.p3000.T:.0f--> | 2792<!--=step4.boil.p101325.T:.0f--> |
| L_vap, MJ/kg | 11.30<!--=step4.boil.p100.L:.2f--> | 11.26<!--=step4.boil.p300.L:.2f--> | 11.20<!--=step4.boil.p1000.L:.2f--> | 11.15<!--=step4.boil.p3000.L:.2f--> | 10.90<!--=step4.boil.p101325.L:.2f--> |

Against the CRC vapour-pressure table for Al the difference is 0.0–1.4<!--=verify_step4.crc.max_diff_pct:.1f-->% from 1 Pa to 1 atm (check 4.8). On the fragment trajectories boiling happens at 1759<!--=step4.fragments.film.panels.Tb_min:.0f-->–2039<!--=step4.fragments.film.mli.Tb_max:.0f--> K. The heat of vaporization includes the Kirchhoff correction. When the pressure drops as the body decelerates, the plateau goes down and the superheated melt flashes (numerical time constant 0.2 s; at 0.05 s the result changes by 0.004<!--=verify_step4.flash.diff_pct:.3f-->%).

The whole surface boils at the stagnation pressure. On the side bands the pressure is lower (Newtonian `p₀·cos²θ`, consistent with the `cos θ` flux) and so is the boiling point. Boiling each band at its own pressure raises the evaporated mass by 17<!--=step5.side_pressure.film.change_pct:.0f-->% in the film scenario and by 14<!--=step5.side_pressure.bare.change_pct:.0f-->% in the bare-melt one, with the median unchanged (section 9.5): the baseline is on the conservative side.

For each band:

```
P_in  = q_stag(t) · cos θ · (1 − h_w/h_0)
P_out = n · ε(T) · σ · T(h_s)⁴                n = 2 for a plate, 1 for a shell
while h_s < h₃:   dh_s/dt = (P_in − P_out) / (ρ·t_wall)
at    h_s = h₃:   the excess goes into evaporation, dm/dt = (P_in − P_out)·dA / L_vap
m_vap ≤ m_band:   a band that has boiled away disappears (burn-through) and stops receiving and radiating
```

The "heating / boiling" switch is smoothed over a window of 1% of h₃. A hard switch made the right-hand side discontinuous, and LSODA took 14 seconds on a thin plate instead of 0.07: a 200-fold speed-up with no loss of convergence.

The clamp at boiling acts **only on heating**: a band at the boiling point whose input has dropped below its re-radiation must cool down. The first version clamped symmetrically, and the band radiated energy it did not have (section 18).

**The melt is counted by the maximum enthalpy over the flight.** A band that melted and then cooled has not stopped being molten. Evaporation happens only above the boiling plateau, so in every band evaporated ≤ molten ≤ band mass, by construction. Check 4.6 verifies this for every band.

The energy audit is accumulated inside the same ODE system: input = radiated + stored + evaporation + blocked by blowing, with a residual of ~1e−15.

### 8.5 Fragments: plate or compact

Each fragment is either a **plate** (set by its thickness) or **compact** (set by β).

For a tumbling plate, by Cauchy's formula the mean projection is half its area, so:

```
β = m/(Cd·A_proj) = ρ·A·t/(Cd·A/2) = 2ρt/Cd
```

β is set by the thickness and nothing else. For Al at Cd = 1.5:

| thickness, mm | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 |
|---|---|---|---|---|---|
| β, kg/m² | 1.8 | 3.6 | 7.2 | 18.0 | 36.0 |

This is exactly the 1–10 range that panels and MLI have in DRAMA fragment lists: the thickness parametrization **reproduces** the known range rather than being fitted to it.

**A plate radiates from two sides.** A 1 mm plate is heated uniformly through its thickness (Bi ~ 0.002), and its back side radiates at the same temperature as the front. In the per-band model one face is heated, and it has to radiate for two. The first version radiated from one side, which for plates is equivalent to halving ε, precisely for the parts where ε matters (check 4.9 compares the equilibrium temperature with the analytic value to 0.03%). The inner side of a shell (a compact fragment) faces a cavity and does not radiate on balance: one side.

The nose radius of a plate is the edge radius ~t/2, with a 5 mm floor. The floor is required: as Rn → 0 the correlation gives an infinite flux, and a real ablating edge blunts itself.

The sphere-equivalent `Rn = √(A/π)` is wrong for a plate and gives absurd results: a panel with β = 4 would get Rn = 96 cm and would not evaporate at all. A plate has a low β not because it is large but because it is thin.

The model's fragment list:

| fragment | mass fraction | kind | parameter | β | breakup h |
|---|---|---|---|---|---|
| solar panels | 10% | plate | 1.5 mm | 5 | 95 km |
| primary structure | 50% | compact | β = 55 | 55 | 78 km |
| small parts, MLI | 40% | plate | 1.0 mm | 4 | 78 km |

**Fragment material.** The thermal model treats every fragment as aluminium, and the Al fraction of a fragment says which part of the evaporated mass is aluminium. In the base case Al is distributed uniformly (30% in each fragment). In reality Al is concentrated in the primary structure and the skin, while the panels are silicon, glass and CFRP, and MLI is coated polyimide. So the distribution of Al over fragments is a sensitivity axis, and it turned out to be the strongest one for mass (section 12).

### 8.6 Demise in DRAMA means melting

ORSAT uses a specific heat of ablation of **934.5 kJ/kg** for generic aluminum (NTRS 20140016958). Our computation of the energy for complete melting of Al 6061, `c_p,sol·(T_melt − T₀) + L_fus`, gives **1.01<!--=step4.criteria.melt:.2f--> MJ/kg**, an 8<!--=verify_step4.orsat.diff_pct:.0f-->% difference, within the difference in alloy properties (6061 melts over 855–925 K; ORSAT's generic Al is defined more simply). The ORSAT/DRAMA demise criterion is **melting**.

The first version got a 3.4% match because of two compensating errors: c_p = 900 J/(kg·K) over the whole range (in reality it rises to ~1190 near melting) and T_melt = 933 K (pure Al).

Complete evaporation needs **13.6<!--=step4.criteria.vapour:.1f--> MJ/kg** (with boiling at ~1 kPa). The ratio is **13.4<!--=step4.criteria.ratio:.1f-->×**.

Molten aluminium stripped into the flow by shear is not Al₂O₃ nanoparticles. A droplet may evaporate further along the trajectory, may oxidise only at its surface and fall as a millimetre spherule (this is how ablation spherules of meteoroids, molten but not evaporated, are found in deep-sea sediments), or may freeze whole.

Between "demised" and "became nanoparticles" there is a **yield factor** that DRAMA does not compute. Our 13.4<!--=step4.criteria.ratio:.1f-->× in energy is a quantitative bound on how much "demise" can differ from "vapour".

### 8.7 The melt/evaporation bracket

| bound | what it is | what it corresponds to |
|---|---|---|
| **upper** | molten mass (maximum over the flight) | everything that could in principle become oxide; reproduces the DRAMA criterion |
| **evaporation in place** | mass evaporated while the melt is retained | what became vapour if the melt was not stripped |
| **width** | fate of the droplet phase | physics that no current model resolves |

**Evaporation in place is not a strict lower bound.** It is computed assuming the melt stays on the fragment until it boils (an oxide crust allows this: liquid Al in an oxide shell does not spread). If the melt is stripped immediately (and that is the very mechanism by which DRAMA counts melting as demise), less will evaporate in place, and the stripped droplets are not modelled. The honest wording is "evaporation in the melt-retention scenario".

Melt stripping is not deferred physics but **the reason the bracket exists**. Wording for a paper: "We bound the aluminium yield from above by the melt; the width of the bracket is set by the physics of the droplet phase, which nobody computes".

The nesting of the bracket (melt ≥ evaporation) holds by construction in every band. An early version with two independent accountings violated it, giving "melt 0%, evaporation 1.6%" for a thick body; the first version of the per-band model hid the violation behind `max(melt, evaporation)` (section 18).

### 8.8 Two regimes: energy-limited and radiative

The regimes are separated by the **mass per unit wetted area** `m'' = m/A_wet` (both faces of a plate are wetted, so m'' = ρt/2). The dimensionless number is

```
Π = φ·∫q dt / (m'' · h_boil)
```

where `h_boil` is the enthalpy from 300 K to the start of boiling (2.4<!--=step4.boil.p1000.h3:.1f--> MJ/kg at 1 kPa).

| | equiv. thickness m''/ρ | needed to boil | available | Π | regime |
|---|---|---|---|---|---|
| intact object | 16.2<!--=step4.regime.whole.equiv_mm:.1f--> mm | 104<!--=step4.regime.whole.need_MJ:.0f--> MJ/m² | 25<!--=step4.regime.whole.avail_MJ:.0f--> MJ/m² | **0.24<!--=step4.regime.whole.Pi:.2f-->** | energy-limited |
| 1 mm plate | 0.5<!--=step4.regime.plate.equiv_mm:.1f--> mm | 3.2<!--=step4.regime.plate.need_MJ:.1f--> MJ/m² | 42<!--=step4.regime.plate.avail_MJ:.0f--> MJ/m² | **13.2<!--=step4.regime.plate.Pi:.1f-->** | radiative |

(The first version used 1 mm instead of 0.5 mm for the plate and got Π = 7.1; the regime does not change.)

At Π ≪ 1 there is not even enough energy to heat up, re-radiation is small and ε barely matters. At Π ≫ 1 the body reaches radiative equilibrium, and from then on ε matters.

The heating depth `√(α·t) = 14 cm` over the flight (α = k/(ρc_p) = 6.9e−5 m²/s) is a separate check: it says that the temperature has time to equalize through the thickness, i.e. a model lumped through the thickness is legitimate for both walls. It does **not** separate the regimes.

The intact object and a 1 mm plate of 1 kg, **both flying from the 120 km entry point** (an illustration of the regimes; this is not the "small parts" fragment, which starts at 78 km at the speed of the intact object):

| ε | T max intact, K | melt | evaporated | T max plate, K | melt | evaporated |
|---|---|---|---|---|---|---|
| 0.05 | 1551<!--=step4.eps.e005.whole_Tmax:.0f--> | 29<!--=step4.eps.e005.whole_melt_pct:.0f-->% | 0<!--=step4.eps.e005.whole_vap_pct:.0f-->% | 1989<!--=step4.eps.e005.plate_Tmax:.0f--> | 100<!--=step4.eps.e005.plate_melt_pct:.0f-->% | 78<!--=step4.eps.e005.plate_vap_pct:.0f-->% |
| 0.10 | 1539<!--=step4.eps.e010.whole_Tmax:.0f--> | 29<!--=step4.eps.e010.whole_melt_pct:.0f-->% | 0<!--=step4.eps.e010.whole_vap_pct:.0f-->% | 1989<!--=step4.eps.e010.plate_Tmax:.0f--> | 99<!--=step4.eps.e010.plate_melt_pct:.0f-->% | 72<!--=step4.eps.e010.plate_vap_pct:.0f-->% |
| 0.20 | 1516<!--=step4.eps.e020.whole_Tmax:.0f--> | 29<!--=step4.eps.e020.whole_melt_pct:.0f-->% | 0<!--=step4.eps.e020.whole_vap_pct:.0f-->% | 1989<!--=step4.eps.e020.plate_Tmax:.0f--> | 97<!--=step4.eps.e020.plate_melt_pct:.0f-->% | 60<!--=step4.eps.e020.plate_vap_pct:.0f-->% |
| 0.30 | 1496<!--=step4.eps.e030.whole_Tmax:.0f--> | 28<!--=step4.eps.e030.whole_melt_pct:.0f-->% | 0<!--=step4.eps.e030.whole_vap_pct:.0f-->% | 1989<!--=step4.eps.e030.plate_Tmax:.0f--> | 96<!--=step4.eps.e030.plate_melt_pct:.0f-->% | 48<!--=step4.eps.e030.plate_vap_pct:.0f-->% |
| 0.35 | 1487<!--=step4.eps.e035.whole_Tmax:.0f--> | 28<!--=step4.eps.e035.whole_melt_pct:.0f-->% | 0<!--=step4.eps.e035.whole_vap_pct:.0f-->% | 1989<!--=step4.eps.e035.plate_Tmax:.0f--> | 96<!--=step4.eps.e035.plate_melt_pct:.0f-->% | 43<!--=step4.eps.e035.plate_vap_pct:.0f-->% |
| active oxide film, ε(T) | 1468<!--=step4.eps.film.whole_Tmax:.0f--> | 28<!--=step4.eps.film.whole_melt_pct:.0f-->% | 0<!--=step4.eps.film.whole_vap_pct:.0f-->% | 1989<!--=step4.eps.film.plate_Tmax:.0f--> | 96<!--=step4.eps.film.plate_melt_pct:.0f-->% | 44<!--=step4.eps.film.plate_vap_pct:.0f-->% |
| bare melt, ε(T) | 1533<!--=step4.eps.bare.whole_Tmax:.0f--> | 29<!--=step4.eps.bare.whole_melt_pct:.0f-->% | 0<!--=step4.eps.bare.whole_vap_pct:.0f-->% | 1989<!--=step4.eps.bare.plate_Tmax:.0f--> | 100<!--=step4.eps.bare.plate_melt_pct:.0f-->% | 65<!--=step4.eps.bare.plate_vap_pct:.0f-->% |

The melt of the intact object is at most 50% by construction: in the stable-orientation mode the leeward half of the shell does not take part in heating. The plate is limited by the boiling temperature (1989<!--=step4.eps.film.plate_Tmax:.0f--> K, boiling at the local pressure).

**Practical consequence:** ε matters only for thin-walled parts: panels, MLI, skin. The primary structure evaporates nothing at any ε. And it is the thin-walled parts that produce all of the evaporated aluminium.

![Emissivity and distribution](../figures/step4_epsilon.png)

### 8.9 Step 4 result by fragment

"Active oxide film" scenario, 175 kg object, Al uniform:

| fragment | m, kg | β | Rn, cm | breakup h | h of q peak | T_boil, K | melt | evaporated |
|---|---|---|---|---|---|---|---|---|
| solar panels | 17.5<!--=step4.fragments.film.panels.mass:.1f--> | 5<!--=step4.fragments.film.panels.beta:.0f--> | 0.5<!--=step4.fragments.film.panels.rn_cm:.1f--> | 95<!--=step4.fragments.film.panels.h_break_km:.0f--> km | 79.7<!--=step4.fragments.film.panels.h_peak_q_km:.1f--> km | 1759<!--=step4.fragments.film.panels.Tb_min:.0f-->–2033<!--=step4.fragments.film.panels.Tb_max:.0f--> | 96<!--=step4.fragments.film.panels.melt_pct:.0f-->% | 44<!--=step4.fragments.film.panels.vap_pct:.0f-->% |
| primary structure | 87.5<!--=step4.fragments.film.structure.mass:.1f--> | 55<!--=step4.fragments.film.structure.beta:.0f--> | 58.1<!--=step4.fragments.film.structure.rn_cm:.1f--> | 78<!--=step4.fragments.film.structure.h_break_km:.0f--> km | 64.2<!--=step4.fragments.film.structure.h_peak_q_km:.1f--> km | 2036<!--=step4.fragments.film.structure.Tb_min:.0f-->–2341<!--=step4.fragments.film.structure.Tb_max:.0f--> | 28<!--=step4.fragments.film.structure.melt_pct:.0f-->% | 0<!--=step4.fragments.film.structure.vap_pct:.0f-->% |
| small parts, MLI | 70.0<!--=step4.fragments.film.mli.mass:.1f--> | 4<!--=step4.fragments.film.mli.beta:.0f--> | 0.5<!--=step4.fragments.film.mli.rn_cm:.1f--> | 78<!--=step4.fragments.film.mli.h_break_km:.0f--> km | 78.0<!--=step4.fragments.film.mli.h_peak_q_km:.1f--> km | 1778<!--=step4.fragments.film.mli.Tb_min:.0f-->–2039<!--=step4.fragments.film.mli.Tb_max:.0f--> | 96<!--=step4.fragments.film.mli.melt_pct:.0f-->% | 30<!--=step4.fragments.film.mli.vap_pct:.0f-->% |

```
molten                108<!--=step4.fragments.film.total.melt_kg:.0f--> kg (62<!--=step4.fragments.film.total.melt_pct:.0f-->%)  →  32.4<!--=step4.fragments.film.total.melt_al:.1f--> kg Al   upper bound
evaporated in place    28<!--=step4.fragments.film.total.vap_kg:.0f--> kg (16<!--=step4.fragments.film.total.vap_pct:.0f-->%)  →   8.5<!--=step4.fragments.film.total.vap_al:.1f--> kg Al
```

In the "bare melt" scenario: panels 59<!--=step4.fragments.bare.panels.vap_pct:.0f-->%, MLI 43<!--=step4.fragments.bare.mli.vap_pct:.0f-->%; molten 109<!--=step4.fragments.bare.total.melt_kg:.0f--> kg (63<!--=step4.fragments.bare.total.melt_pct:.0f-->%) → 32.8<!--=step4.fragments.bare.total.melt_al:.1f--> kg Al, evaporated 40<!--=step4.fragments.bare.total.vap_kg:.0f--> kg (23<!--=step4.fragments.bare.total.vap_pct:.0f-->%) → 12.0<!--=step4.fragments.bare.total.vap_al:.1f--> kg Al.

The histogram peaks at **72–80 km**, inside the observed 70–80 km breakup band, which remained out of reach in steps 1–3.

![Altitude bracket](../figures/step4_bracket.png)

---

## 9. Step 5. Main result

### 9.1 Altitude distribution of injection

A 175 kg object, 30% Al = 52.5 kg Al, spread uniformly over the fragments. Entry at 120 km / 7500 m/s / −1.5°, i = 53°, NRLMSIS 2.1, the three fragments from section 8.5, boiling at the local pressure.

| scenario | ε at 2000 K | Al evaporated, kg | yield | Al₂O₃ per kg of satellite | median, km | above 50 km | above 85 km | molten Al, kg |
|---|---|---|---|---|---|---|---|---|
| **active oxide film** | 0.322<!--=step5.scenarios.film.eps2000:.3f--> | **8.5<!--=step5.scenarios.film.total:.1f-->** | 16<!--=step5.scenarios.film.yield_pct:.0f-->% | 0.092<!--=step5.scenarios.film.al2o3_per_kg:.3f--> | 76.1<!--=step5.scenarios.film.median:.1f--> | 100<!--=step5.scenarios.film.above_strat_pct:.0f-->% | 10<!--=step5.scenarios.film.above_meso_pct:.0f-->% | 32.4<!--=step5.scenarios.film.melt:.1f--> |
| **bare melt** | 0.173<!--=step5.scenarios.bare.eps2000:.3f--> | **12.0<!--=step5.scenarios.bare.total:.1f-->** | 23<!--=step5.scenarios.bare.yield_pct:.0f-->% | 0.130<!--=step5.scenarios.bare.al2o3_per_kg:.3f--> | 75.9<!--=step5.scenarios.bare.median:.1f--> | 100<!--=step5.scenarios.bare.above_strat_pct:.0f-->% | 11<!--=step5.scenarios.bare.above_meso_pct:.0f-->% | 32.8<!--=step5.scenarios.bare.melt:.1f--> |

For comparison, a constant ε (the same physics):

| ε | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |
|---|---|---|---|---|---|---|---|
| Al evaporated, kg | 15.2<!--=step5.sweep.e005.total:.1f--> | 13.9<!--=step5.sweep.e010.total:.1f--> | 12.6<!--=step5.sweep.e015.total:.1f--> | 11.4<!--=step5.sweep.e020.total:.1f--> | 10.2<!--=step5.sweep.e025.total:.1f--> | 9.0<!--=step5.sweep.e030.total:.1f--> | 7.9<!--=step5.sweep.e035.total:.1f--> |
| yield | 29<!--=step5.sweep.e005.yield_pct:.0f-->% | 26<!--=step5.sweep.e010.yield_pct:.0f-->% | 24<!--=step5.sweep.e015.yield_pct:.0f-->% | 22<!--=step5.sweep.e020.yield_pct:.0f-->% | 19<!--=step5.sweep.e025.yield_pct:.0f-->% | 17<!--=step5.sweep.e030.yield_pct:.0f-->% | 15<!--=step5.sweep.e035.yield_pct:.0f-->% |
| median, km | 75.6<!--=step5.sweep.e005.median:.1f--> | 75.8<!--=step5.sweep.e010.median:.1f--> | 75.8<!--=step5.sweep.e015.median:.1f--> | 75.9<!--=step5.sweep.e020.median:.1f--> | 76.0<!--=step5.sweep.e025.median:.1f--> | 76.1<!--=step5.sweep.e030.median:.1f--> | 76.2<!--=step5.sweep.e035.median:.1f--> |

The stratopause is at 50 km. Above it particles settle for years (mesospheric injection); below it they enter the stratosphere almost at once. In the model all of the injection happens above the stratopause; about 10<!--=step5.scenarios.film.above_meso_pct:.0f-->% above the mesopause (85 km), mostly from the panels, which separate at 95 km. The first version said "100% into the mesosphere"; it is "100% above the stratopause".

The median is computed from the raw (altitude, mass) series, not from the histogram: a binned median depended on the bin width and moved by 1.5 km for 0.5–5 km bins. The histogram is kept only for display.

![Main result](../figures/step5_main.png)

### 9.2 What changed relative to the first version

The first version gave 4.4–7.7 kg Al in the "active oxide film" scenario (ε 0.19–0.29 at 2740 K) and 15.4 kg without the film (ε 0.05). Decomposition, kg of evaporated Al at constant ε:

| step | ε = 0.05 | ε = 0.19 | ε = 0.29 |
|---|---|---|---|
| first version | 15.4 | 7.7 | 4.4 |
| + plate with two sides, band ≤ its own mass, melt by maximum | 11.2<!--=step5.changes.plates_fixed.e005:.1f--> | 2.4<!--=step5.changes.plates_fixed.e019:.1f--> | 0.5<!--=step5.changes.plates_fixed.e029:.1f--> |
| + Al 6061 properties (c_p solid/liquid, T_melt), boiling at 1 atm = 2792 K | 9.8<!--=step5.changes.new_props_1atm.e005:.1f--> | 1.6<!--=step5.changes.new_props_1atm.e019:.1f--> | 0.2<!--=step5.changes.new_props_1atm.e029:.1f--> |
| + boiling at local pressure (base) | 15.2<!--=step5.changes.local_pressure.e005:.1f--> | 11.6<!--=step5.changes.local_pressure.e019:.1f--> | 9.2<!--=step5.changes.local_pressure.e029:.1f--> |

Two errors in the plate model overestimated evaporation by a factor of 1.4–9; ignoring the pressure dependence of boiling underestimated it by a factor of 1.5–45. Then ε went from a constant to a function of temperature (section 13), and the scenarios arrived at 8.5 and 12.0 kg.

### 9.3 The two scenarios have converged

The first formulation presented the answer as two distant scenarios: "active oxide film" (ε 0.19–0.29 at 2740 K) and "bare melt" (ε ≈ 0.05), differing by a factor of 2–3.5. Both anchors moved:

- **The working temperature is ~2000 K, not 2740 K.** There the film ε is set by the α-Al₂O₃ reference point at 1800 K: 0.30–0.33 over all 48 curve shapes (section 13.5). And the Al₂O₃ is still solid there (it melts at 2345 K), so there is no longer any extrapolation beyond the oxide melting point.
- **ε ≈ 0.05 is polished solid Al at 300–900 K.** The liquid metal has 2.4–5 times the resistivity, and the resistivity estimate (Parker–Abbott) gives ~0.10 at melting and ~0.17 at 2000 K (section 13.7).

Result: 8.5<!--=step5.scenarios.film.total:.1f--> versus 12.0<!--=step5.scenarios.bare.total:.1f--> kg, a factor of 1.4. The surface scenario is no longer the main uncertainty.

### 9.4 Oxide film growth: ε as a trajectory

The oxide film grows during the flight. A thin film is optically transparent, and it is effectively the metal underneath that radiates. As the film thickens, ε grows toward the film value. This is implemented as `Aluminium(tau_oxide=τ)` with parabolic growth `δ ~ √t`, now between the two ε(T) curves:

```
ε(t, T) = ε_bare(T) + (ε_film(T) − ε_bare(T)) · (1 − exp(−√(t/τ)))
```

Time t is counted **from the moment of breakup**: the fragment surface is fresh, and fragments are heated in the first ~10–40 s. (The first version labelled the column "ε at the heating peak (~200 s)" but printed ε(200 s), whereas the fragment clock starts at breakup; and it took 0.35 for the film, the value at 1800 K.)

| τ, s | ε(10 s), 2000 K | ε(30 s), 2000 K | Al evaporated, kg | yield |
|---|---|---|---|---|
| 1 | 0.316<!--=step4.growth.tau1.eps10:.3f--> | 0.322<!--=step4.growth.tau1.eps30:.3f--> | 8.7<!--=step4.growth.tau1.total:.1f--> | 16<!--=step4.growth.tau1.yield_pct:.0f-->% |
| 10 | 0.267<!--=step4.growth.tau10.eps10:.3f--> | 0.296<!--=step4.growth.tau10.eps30:.3f--> | 9.6<!--=step4.growth.tau10.total:.1f--> | 18<!--=step4.growth.tau10.yield_pct:.0f-->% |
| 30 | 0.238<!--=step4.growth.tau30.eps10:.3f--> | 0.267<!--=step4.growth.tau30.eps30:.3f--> | 10.2<!--=step4.growth.tau30.total:.1f--> | 19<!--=step4.growth.tau30.yield_pct:.0f-->% |
| 100 | 0.213<!--=step4.growth.tau100.eps10:.3f--> | 0.236<!--=step4.growth.tau100.eps30:.3f--> | 10.8<!--=step4.growth.tau100.total:.1f--> | 21<!--=step4.growth.tau100.yield_pct:.0f-->% |
| 300 | 0.198<!--=step4.growth.tau300.eps10:.3f--> | 0.213<!--=step4.growth.tau300.eps30:.3f--> | 11.2<!--=step4.growth.tau300.total:.1f--> | 21<!--=step4.growth.tau300.yield_pct:.0f-->% |
| 1000 | 0.187<!--=step4.growth.tau1000.eps10:.3f--> | 0.197<!--=step4.growth.tau1000.eps30:.3f--> | 11.6<!--=step4.growth.tau1000.total:.1f--> | 22<!--=step4.growth.tau1000.yield_pct:.0f-->% |
| 10000 | 0.178<!--=step4.growth.tau10000.eps10:.3f--> | 0.181<!--=step4.growth.tau10000.eps30:.3f--> | 11.9<!--=step4.growth.tau10000.total:.1f--> | 23<!--=step4.growth.tau10000.yield_pct:.0f-->% |

The whole spread over τ fits between the two scenarios. The unknown is renamed from "which ε" to "how fast the film grows". Hence the conclusion for the experiment: ε should be measured **as a function of oxide thickness**, not as a single number.

### 9.5 Heat of oxidation: an upper bound

Version 2 left the heat of Al oxidation out with the argument "the film is nanometre-thick, so the contribution is small". That argument is wrong: the heat release is set by how fast oxygen is supplied, not by the thickness of the film. A boiling, constantly renewed melt can keep consuming oxygen while the film stays thin.

The supply is bounded by diffusion through the boundary layer. With a Lewis number of 1 the Reynolds analogy gives one transfer coefficient for heat and mass, `C = q_cold/h_0`, so the oxygen flux to the wall is at most `Y_O·q_cold/h_0`. If all of it reacts at the wall (2Al + 3/2 O₂ → Al₂O₃, ΔH = 34.9 MJ per kg of O):

```
q_ox / q_hw = η · Y_O · ΔH_ox / (h_0 − h_w)          Y_O = 0.231,  η ∈ [0, 1]
```

At η = 1 this is 0.23 · 35 / 27 ≈ 0.3 at orbital speed. The ratio does not depend on altitude or fragment size (both cancel in the transfer coefficient), but it grows as the fragment slows down, because h₀ = V²/2 falls while the heat per kilogram of oxygen does not (T_wall = 2046 K):

| V, m/s | 7500 | 7000 | 6500 | 6000 |
|---|---|---|---|---|
| q_ox / q_hw at η = 1 | 0.32<!--=step5.oxidation.ratio.V7500:.2f--> | 0.37<!--=step5.oxidation.ratio.V7000:.2f--> | 0.44<!--=step5.oxidation.ratio.V6500:.2f--> | 0.53<!--=step5.oxidation.ratio.V6000:.2f--> |

In the model (`oxidation_eta`) each band gets this extra input on its **molten** part only: solid aluminium is passivated by its own oxide. The Al that the oxygen would consume is reported separately and **not** removed from the fragments, so the evaporated mass stays an upper bound.

| scenario | η | Al evaporated, kg | × baseline | median, km | Al oxidized at the wall, kg |
|---|---|---|---|---|---|
| active oxide film | 0 (baseline) | 8.5<!--=step5.oxidation.film.eta000.total:.1f--> | 1.00 | 76.1<!--=step5.oxidation.film.eta000.median:.1f--> | — |
| active oxide film | 0.5 | 12.6<!--=step5.oxidation.film.eta050.total:.1f--> | 1.48<!--=step5.oxidation.film.eta050.factor:.2f--> | 75.9<!--=step5.oxidation.film.eta050.median:.1f--> | 5.2<!--=step5.oxidation.film.eta050.ox_al:.1f--> |
| active oxide film | 1 (upper bound) | 15.4<!--=step5.oxidation.film.eta100.total:.1f--> | 1.80<!--=step5.oxidation.film.eta100.factor:.2f--> | 75.9<!--=step5.oxidation.film.eta100.median:.1f--> | 7.4<!--=step5.oxidation.film.eta100.ox_al:.1f--> |
| bare melt | 0 (baseline) | 12.0<!--=step5.oxidation.bare.eta000.total:.1f--> | 1.00 | 75.9<!--=step5.oxidation.bare.eta000.median:.1f--> | — |
| bare melt | 0.5 | 15.6<!--=step5.oxidation.bare.eta050.total:.1f--> | 1.30<!--=step5.oxidation.bare.eta050.factor:.2f--> | 75.8<!--=step5.oxidation.bare.eta050.median:.1f--> | 4.0<!--=step5.oxidation.bare.eta050.ox_al:.1f--> |
| bare melt | 1 (upper bound) | 18.1<!--=step5.oxidation.bare.eta100.total:.1f--> | 1.51<!--=step5.oxidation.bare.eta100.factor:.2f--> | 75.9<!--=step5.oxidation.bare.eta100.median:.1f--> | 5.8<!--=step5.oxidation.bare.eta100.ox_al:.1f--> |

What this means:

1. **The upper bound is ×1.8<!--=step5.oxidation.film.eta100.factor:.1f--> for the film scenario and ×1.5<!--=step5.oxidation.bare.eta100.factor:.1f--> for bare melt**: larger than the difference between the two surface scenarios (×1.4). A constant +30% of the flux gives less, because the ratio grows during deceleration. The axis goes into the budget (section 12).
2. **The altitude does not move**: the median changes by 0.2<!--=step5.sensitivity.oxidation.median_span:.1f--> km.
3. **The real η lies between 0 and 1.** Part of the reaction happens in the gas: the Al vapour burns in the boundary layer, and that heat is partly carried off by the flow. The oxide skin slows the rest: bulk aluminium ignites only near the melting point of its oxide, ~2300 K (Friedman & Maček 1962), above the 1750–2050 K of the boiling surface here.
4. **The axis is not independent of the surface scenarios.** An intact, optically active oxide film is exactly what passivates the melt (η close to 0); a bare, constantly renewed melt is where oxidation can approach the diffusion limit. The coherent pairs are "film, η ≈ 0" (8.5<!--=step5.oxidation.film.eta000.total:.1f--> kg) and "bare melt, η up to 1" (up to 18.1<!--=step5.oxidation.bare.eta100.total:.1f--> kg).
5. **At the upper bound the wall also makes condensed oxide directly**: 7.4<!--=step5.oxidation.film.eta100.ox_al:.1f--> kg of Al for the film scenario. This is Al₂O₃ in the atmosphere too, but as a surface product (slag, droplets), not as vapour that nucleates nanoparticles. It is a diagnostic, not part of the evaporated mass.

Conclusion 3 holds: the surface state is not the main uncertainty. The heat of oxidation joins the axes of order 1.5–1.8× and does not change the altitude result.

The same run also quantifies the side-band pressure (section 8.4): boiling at the local `p₀·cos²θ` gives 9.9<!--=step5.side_pressure.film.total:.1f--> kg instead of 8.5 (+17<!--=step5.side_pressure.film.change_pct:.0f-->%) for the film and 13.7<!--=step5.side_pressure.bare.total:.1f--> kg instead of 12.0 (+14<!--=step5.side_pressure.bare.change_pct:.0f-->%) for bare melt; the median stays at 76.0<!--=step5.side_pressure.film.median:.1f--> km.

---

## 10. What the model predicts and what it inherits

### 10.1 The objection

An early version claimed: "the injection altitude is robust: 75–76 km at any ε". The claim is true but misleading: ε was never a candidate for controlling the altitude. It showed robustness to a parameter that was not supposed to matter anyway.

A referee's objection would be: "you got injection at 75 km because you set breakup at 78".

### 10.2 The transfer function

Both breakup altitudes from the fragment list are shifted together ("active oxide film" scenario):

| shift, km | body breakup h | panel h | injection median, km | offset, km | Al mass, kg |
|---|---|---|---|---|---|
| −15 | 63 | 80 | 74.0<!--=step5b.transfer.m15.median:.1f--> | −11.0<!--=step5b.transfer.m15.offset:+.1f--> | 2.5<!--=step5b.transfer.m15.mass:.1f--> |
| −10 | 68 | 85 | 67.5<!--=step5b.transfer.m10.median:.1f--> | +0.5<!--=step5b.transfer.m10.offset:+.1f--> | 4.4<!--=step5b.transfer.m10.mass:.1f--> |
| −5 | 73 | 90 | 71.9<!--=step5b.transfer.m5.median:.1f--> | +1.1<!--=step5b.transfer.m5.offset:+.1f--> | 6.5<!--=step5b.transfer.m5.mass:.1f--> |
| 0 | 78 | 95 | 76.1<!--=step5b.transfer.p0.median:.1f--> | +1.9<!--=step5b.transfer.p0.offset:+.1f--> | 8.5<!--=step5b.transfer.p0.mass:.1f--> |
| +5 | 83 | 100 | 79.7<!--=step5b.transfer.p5.median:.1f--> | +3.3<!--=step5b.transfer.p5.offset:+.1f--> | 10.2<!--=step5b.transfer.p5.mass:.1f--> |
| +10 | 88 | 105 | 82.6<!--=step5b.transfer.p10.median:.1f--> | +5.4<!--=step5b.transfer.p10.offset:+.1f--> | 11.1<!--=step5b.transfer.p10.mass:.1f--> |
| +15 | 93 | 110 | 84.8<!--=step5b.transfer.p15.median:.1f--> | +8.2<!--=step5b.transfer.p15.offset:+.1f--> | 11.5<!--=step5b.transfer.p15.mass:.1f--> |

- The transfer gain is **0.82<!--=step5b.transfer.gain_mid:.2f-->** in the plausible 68–83 km window; 0.50<!--=step5b.transfer.gain_all:.2f--> over the full range (the dependence saturates at the edges).
- The mass changes by a factor of **4.5<!--=step5b.transfer.mass_ratio:.1f-->**: the breakup altitude turned out to be the main lever for mass as well.

Offset at a fixed breakup altitude (78 km) across the other parameters:

| parameter | offset, km |
|---|---|
| active oxide film (base) | +1.9<!--=step5b.offsets.film:+.1f--> |
| bare melt | +2.1<!--=step5b.offsets.bare:+.1f--> |
| τ = 1 s / 10 000 s | +1.9<!--=step5b.offsets.tau1:+.1f--> / +2.1<!--=step5b.offsets.tau10000:+.1f--> |
| blowing η = 0.6 | +2.2<!--=step5b.offsets.blowing:+.1f--> |
| fast tumbling | +2.0<!--=step5b.offsets.tumbling:+.1f--> |
| entry angle −1° / −3° | +1.7<!--=step5b.offsets.gamma1:+.1f--> / +2.8<!--=step5b.offsets.gamma3:+.1f--> |

**The offset of +1.7<!--=step5b.offsets.min:+.1f-->…+2.8<!--=step5b.offsets.max:+.1f--> km is robust to everything except the breakup altitude itself.**

### 10.3 An honest statement

The model does not predict the absolute injection altitude; it predicts **the mapping from breakup altitude to injection altitude**:

```
h_inj ≈ 75.9<!--=step5b.transfer.intercept78:.1f--> + 0.82<!--=step5b.transfer.gain_mid:.2f--> · (H − 78) km,        H in the 68–83 km window
```

The first version wrote "with breakup at altitude H the injection happens at H minus ~2 km", a rule with a gain of 1 that holds only near 78 km: at H = 68 the offset is +0.5<!--=step5b.transfer.m10.offset:+.1f--> km, at H = 88 it is +5.4<!--=step5b.transfer.p10.offset:+.1f--> km.

This is a useful result: it lets atmospheric modellers convert **observed** breakup altitudes into injection altitudes. But it is conditional.

This reorders the priorities: **the main altitude uncertainty lies in the observational data on the breakup altitude, not in the model.** It is worth saying so directly; it strengthens the work rather than weakening it.

![Transfer function and validation](../figures/step5b_revision.png)

---

## 11. Comparison with Ferreira

### 11.1 What Ferreira actually computed

Ferreira et al. 2024 (GRL, 10.1029/2024GL109280): a 250 kg satellite, 30% Al = 75 kg Al. Molecular dynamics of Al oxidation in oxygen at 2200 K at an altitude of 86 km (the set-up next to their Figure 1a), with the result scaled up to the satellite through the atom counts of the clusters in their Table 1b. Outcome (their Section 3.3, "Full-Scale and Long-Term Extrapolation", p. 8, and Section 4): **24.0 kg of Al is oxidised (32%)**, into 29.8 kg of oxide clusters; 51.0 kg of Al remain as unoxidised clusters. All of the Al is assumed to be ablated.

So 32% is an **output of their model**, not an assumption.

**What the clusters are.** The paper calls them "AlO particles", but that is a label, not a 1:1 formula. Stoichiometric AlO from 24.0 kg of Al would weigh 38.2 kg, and Al₂O₃ 45.3 kg. The 29.8 kg correspond to 5.8 kg of oxygen, O/Al ≈ 0.4 by atoms: aluminium-rich AlₓOᵧ, consistent with the paper's own description of an oxygen-deficient reaction and an "aluminum-rich" cluster (their Figure 1c). The comparison below is made in kilograms of **aluminium** (24.0 kg, 32%), so the cluster stoichiometry does not enter it.

**The first version's error.** The report took "~30 kg Al₂O₃", converted it to Al using the mass fraction of Al in Al₂O₃ (0.529) and got 15.9 kg Al = 21%. Correct: 24.0 kg Al = 32%. Besides, in one place the first version called Ferreira's yield "an assumption" and in another "an output of molecular dynamics"; the second is right.

### 11.2 Comparison

Normalized to our object (52.5 kg Al), Ferreira's yield is 16.8<!--=step5.ferreira.theirs.total:.1f--> kg Al, 0.096<!--=step5.ferreira.theirs.per_kg:.3f--> kg Al per kg of satellite:

| scenario | Al to vapour, kg | Al share | kg Al per kg of satellite |
|---|---|---|---|
| active oxide film, Al uniform | 8.5<!--=step5.ferreira.film_uniform.total:.1f--> | 16<!--=step5.ferreira.film_uniform.pct:.0f-->% | 0.049<!--=step5.ferreira.film_uniform.per_kg:.3f--> |
| bare melt, Al uniform | 12.0<!--=step5.ferreira.bare_uniform.total:.1f--> | 23<!--=step5.ferreira.bare_uniform.pct:.0f-->% | 0.069<!--=step5.ferreira.bare_uniform.per_kg:.3f--> |
| active oxide film, all Al in thin-walled parts | 17.0<!--=step5.ferreira.film_thin.total:.1f--> | 32<!--=step5.ferreira.film_thin.pct:.0f-->% | 0.097<!--=step5.ferreira.film_thin.per_kg:.3f--> |
| bare melt, all Al in thin-walled parts | 24.0<!--=step5.ferreira.bare_thin.total:.1f--> | 46<!--=step5.ferreira.bare_thin.pct:.0f-->% | 0.137<!--=step5.ferreira.bare_thin.per_kg:.3f--> |
| **Ferreira (oxidised)** | 16.8<!--=step5.ferreira.theirs.total:.1f--> | **32<!--=step5.ferreira.theirs.pct:.0f-->%** | 0.096<!--=step5.ferreira.theirs.per_kg:.3f--> |

1. **The route is completely different.** Ferreira uses molecular dynamics of oxidation; we use the trajectory, the heat balance and the flux distribution over the surface.
2. **With uniform Al our 16<!--=step5.ferreira.film_uniform.pct:.0f-->–23<!--=step5.ferreira.bare_uniform.pct:.0f-->% is below their 32<!--=step5.ferreira.theirs.pct:.0f-->%.** 32% is reached if the Al is concentrated in thin-walled parts. Different quantities are compared: ours is evaporated Al (all the vapour will oxidise), theirs is the oxidised fraction assuming all Al is ablated.
3. **Agreement in temperature.** Ferreira computes at 2200 K, below boiling at 1 atm and close to our boiling at local pressure (1759<!--=step4.fragments.film.panels.Tb_min:.0f-->–2039<!--=step4.fragments.film.mli.Tb_max:.0f--> K).
4. **The upper bound works as a constraint from above.** A melt of 62<!--=step5.scenarios.film.melt_pct:.0f-->% means that physically almost twice as much is possible as Ferreira oxidises. The whole difference sits in the fate of the droplet phase.

### 11.3 Caveats

This must not be presented without them: a different object mass, different entry conditions, different computed quantities. This is **order-of-magnitude agreement, not validation**. The first version interpreted the position of Ferreira's value between the scenarios as "an intermediate film state (ε ≈ 0.12, τ ≈ 300 s)"; that interpretation is withdrawn, since it rested on a wrong conversion (21%) and an outdated film-growth table.

---

## 12. Uncertainty budgets

### 12.1 There are two budgets

Some parameters move the injection altitude and some move the mass. They must not be confused.

**Step 5 budget** (main result; base is "active oxide film", Al uniform, 8.5<!--=step5.sensitivity.base.total:.1f--> kg):

| factor | Al evaporated, kg: from – to | ratio | median, km |
|---|---|---|---|
| **distribution of Al over fragments** | 0.0<!--=step5.sensitivity.al_split.lo:.1f--> – 17.0<!--=step5.sensitivity.al_split.hi:.1f--> | ∞ | 0.0<!--=step5.sensitivity.al_split.median_span:.1f--> |
| **thin-walled mass fraction 25–75%** | 4.3<!--=step5.sensitivity.thin_fraction.lo:.1f--> – 13.1<!--=step5.sensitivity.thin_fraction.hi:.1f--> | 3.1<!--=step5.sensitivity.thin_fraction.ratio:.1f-->× | 0.2<!--=step5.sensitivity.thin_fraction.median_span:.1f--> |
| **breakup altitude ±10 km** | 4.4<!--=step5.sensitivity.breakup.lo:.1f--> – 11.1<!--=step5.sensitivity.breakup.hi:.1f--> | 2.5<!--=step5.sensitivity.breakup.ratio:.1f-->× | **15.1<!--=step5.sensitivity.breakup.median_span:.1f-->** |
| heat of oxidation, η 0–1 (upper bound; not in the baseline) | 8.5<!--=step5.sensitivity.oxidation.lo:.1f--> – 15.4<!--=step5.sensitivity.oxidation.hi:.1f--> | 1.8<!--=step5.sensitivity.oxidation.ratio:.1f-->× | 0.2<!--=step5.sensitivity.oxidation.median_span:.1f--> |
| orbit inclination 0–180° | 7.4<!--=step5.sensitivity.inclination.lo:.1f--> – 12.4<!--=step5.sensitivity.inclination.hi:.1f--> | 1.7<!--=step5.sensitivity.inclination.ratio:.1f-->× | 0.2<!--=step5.sensitivity.inclination.median_span:.1f--> |
| vapour blowing, η 0–0.6 | 4.9<!--=step5.sensitivity.blowing.lo:.1f--> – 8.5<!--=step5.sensitivity.blowing.hi:.1f--> | 1.7<!--=step5.sensitivity.blowing.ratio:.1f-->× | 0.3<!--=step5.sensitivity.blowing.median_span:.1f--> |
| orientation: stable / fast tumbling | 5.9<!--=step5.sensitivity.orientation.lo:.1f--> – 8.5<!--=step5.sensitivity.orientation.hi:.1f--> | 1.5<!--=step5.sensitivity.orientation.ratio:.1f-->× | 0.1<!--=step5.sensitivity.orientation.median_span:.1f--> |
| surface scenario (film / bare) | 8.5<!--=step5.sensitivity.surface.lo:.1f--> – 12.0<!--=step5.sensitivity.surface.hi:.1f--> | 1.4<!--=step5.sensitivity.surface.ratio:.1f-->× | 0.2<!--=step5.sensitivity.surface.median_span:.1f--> |
| film growth τ 1–10⁴ s | 8.7<!--=step5.sensitivity.growth.lo:.1f--> – 11.9<!--=step5.sensitivity.growth.hi:.1f--> | 1.4<!--=step5.sensitivity.growth.ratio:.1f-->× | 0.2<!--=step5.sensitivity.growth.median_span:.1f--> |
| boiling at the local pressure `p₀·cos²θ` | 8.5<!--=step5.sensitivity.side_pressure.lo:.1f--> – 9.9<!--=step5.sensitivity.side_pressure.hi:.1f--> | 1.2<!--=step5.sensitivity.side_pressure.ratio:.1f-->× | 0.1<!--=step5.sensitivity.side_pressure.median_span:.1f--> |
| plate thickness ×2 / ÷2 | 7.8<!--=step5.sensitivity.thickness.lo:.1f--> – 8.8<!--=step5.sensitivity.thickness.hi:.1f--> | 1.1<!--=step5.sensitivity.thickness.ratio:.1f-->× | 2.9<!--=step5.sensitivity.thickness.median_span:.1f--> |
| film curve shape (48 sets) | 8.3<!--=step5.sensitivity.film_shape.lo:.1f--> – 9.0<!--=step5.sensitivity.film_shape.hi:.1f--> | 1.1<!--=step5.sensitivity.film_shape.ratio:.1f-->× | 0.1<!--=step5.sensitivity.film_shape.median_span:.1f--> |
| entry angle −1…−3° | 8.4<!--=step5.sensitivity.entry_angle.lo:.1f--> – 8.5<!--=step5.sensitivity.entry_angle.hi:.1f--> | 1.0<!--=step5.sensitivity.entry_angle.ratio:.1f-->× | 1.1<!--=step5.sensitivity.entry_angle.median_span:.1f--> |

Vapour blowing and the heat of oxidation are two boundary-layer effects of opposite sign that are both left out of the baseline: one blocks heat, the other adds it. Treated together they would partly cancel; a joint model of the vapour flame in the boundary layer is on the list of next steps (section 20).

"Distribution of Al over fragments": at the same total fraction of 30%, all Al in the primary structure (0<!--=step5.sensitivity.al_split.lo:.0f--> kg, since it does not evaporate), uniform (8.5<!--=step5.sensitivity.base.total:.1f--> kg), all Al in thin-walled parts (17.0<!--=step5.sensitivity.al_split.hi:.1f--> kg).

Boiling at 1 atm instead of the local pressure would give 0.6<!--=step5.changes.film_1atm:.1f--> kg instead of 8.5<!--=step5.scenarios.film.total:.1f-->; that is not an uncertainty but corrected physics (section 9.2), so it is not in the budget.

![Two budgets](../figures/step5_sensitivity.png)

**Step 3 budget** (heating peak and specific absorbed energy, intact object, with atmospheric rotation):

| factor | affects | peak altitude, km | specific energy |
|---|---|---|---|
| fragmentation (size L 1.0 → 0.1) | both | **16.0<!--=step3.budget.fragmentation.h_km:.1f-->** | **887<!--=step3.budget.fragmentation.energy_pct:.0f-->%** |
| Cd 1.0–2.2 | both | 5.8<!--=step3.budget.cd.h_km:.1f--> | 49<!--=step3.budget.cd.energy_pct:.0f-->% |
| entry latitude −75…0° | both | 4.6<!--=step3.budget.latitude.h_km:.1f--> | 3<!--=step3.budget.latitude.energy_pct:.0f-->% |
| entry angle −1…−3° | both | 2.1<!--=step3.budget.entry_angle.h_km:.1f--> | 24<!--=step3.budget.entry_angle.energy_pct:.0f-->% |
| season March/September | both | 1.8<!--=step3.budget.season.h_km:.1f--> | 1<!--=step3.budget.season.energy_pct:.0f-->% |
| inclination 0–180° | both | 0.9<!--=step3.budget.inclination.h_km:.1f--> | 43<!--=step3.budget.inclination.energy_pct:.0f-->% |
| MSIS version 00 / 2.1 | both | 0.9<!--=step3.budget.msis_version.h_km:.1f--> | 0<!--=step3.budget.msis_version.energy_pct:.0f-->% |
| correlation S–G / DKR | mass only | — | 4<!--=step3.budget.correlation.energy_pct:.0f-->% |
| solar activity F10.7 70–220 | both | 0.0<!--=step3.budget.solar.h_km:.1f--> | 0<!--=step3.budget.solar.energy_pct:.0f-->% |
| effective Rn 0.2–1.0 m | mass only | — | 124<!--=step3.budget.rn.energy_pct:.0f-->% |
| shape factor φ 0.25–0.30 | mass only | — | 20<!--=step3.budget.shape_factor.energy_pct:.0f-->% |

(Latitude and season are 4.6<!--=step3.budget.latitude.h_km:.1f--> and 1.8<!--=step3.budget.season.h_km:.1f--> km here and 3.9<!--=verify_step2.env_span_km.lat:.1f--> and 1.6<!--=verify_step2.env_span_km.season:.1f--> km in section 5.4: that section uses the step 2 configuration, without atmospheric rotation. Inclination is 43<!--=step3.budget.inclination.energy_pct:.0f-->% here versus 36% in section 6.3: that one is the stagnation-point heat load, this one is the specific energy with the hot-wall correction, which itself depends on speed.)

![Step 3 budget](../figures/step3_budget.png)

### 12.2 The structural conclusion

**A parameter can drop out of one budget and return in another.**

- **The entry angle** dropped out of the altitude budget (2.1 km: the gravity turn erases γ₀ before drag switches on), but it returned to the specific-energy budget of the intact object at 24%: γ₀ controls the flight duration. On the evaporated Al mass of the fragments it has almost no effect (1.0×): fragments start from the breakup altitude.
- **The nose radius** is the opposite case: 124% in mass and **exactly zero** in altitude, because Rn is constant along the trajectory and factors out of argmax.
- **The breakup altitude** is in both budgets at once: 15 km in the median and 2.5× in mass.

A rule for the whole work: **the list of leading parameters must be built separately for each output quantity.** A single ranking of "what matters in this model" would be wrong no matter how carefully it was computed. For a paper this structure of the argument is worth more than any single number.

**What changed in the ranking.** In the first version ε was the main mass axis. After the corrections the main axes are **where the aluminium sits and what fraction of the mass is thin-walled**, i.e. the bill of materials of the fragments of a specific spacecraft. The surface state is one of several axes of order 1.4–1.8×, alongside vapour blowing, orientation and the heat of oxidation.

Inclination and latitude in these tables are not uncertainties but **known inputs**: for a specific spacecraft they come from its orbit.

---

## 13. Emissivity: validation, shape sweep, two scenarios

### 13.1 Where ε matters

The boiling threshold is directly proportional to ε: `q_threshold = n·ε·σ·T_boil⁴/φ`. The equilibrium temperature goes as `ε^(−1/4)`. So for thin-walled fragments (Π ≫ 1) ε is a noticeable axis, and for massive ones (Π ≪ 1) it is not.

### 13.2 The right material

At ~2000 K what radiates is not oxidised **solid** aluminium but **melt**, with or without an oxide film:

| surface | ε | T, K |
|---|---|---|
| polished Al | 0.04–0.07 | 300–900 |
| oxidised Al | 0.07–0.09 | 450–900 |
| lightly oxidised Al | 0.11–0.19 | 473–873 |
| **α-Al₂O₃** | **0.83 → 0.35** | **300 → 1800 (falling)** |

The order of the "oxidised" and "lightly oxidised" rows in the compilation looks swapped (the lightly oxidised one has the larger ε); this should be checked against the primary source. These rows are not used in the computation.

The emissivity of the oxide **falls** with temperature rather than rising.

### 13.3 Method: reflectance at room temperature → ε at the working temperature

Direct emissometry at 1500–2200 K is not needed:

```
ε(λ) = 1 − R(λ)                          Kirchhoff's law, opaque sample
ε(T) = ∫ε(λ)·B(λ,T) dλ / ∫B(λ,T) dλ      Planck weighting
```

The function `emissivity.total_emissivity_from_reflectance(λ, R, T)` takes the arrays straight from the instrument.

**Sample requirement:** `ε = 1 − R` holds only for an **opaque** sample. An oxide film on a metal is opaque thanks to the substrate, so it works. A free-standing film or an oxide powder does not; there `1 − R − T` with a transmittance measurement is needed.

**Normal versus hemispherical.** An integrating sphere gives ε near normal incidence; the model needs the hemispherical value. For metals the hemispherical value is 10–30% above the normal one, for dielectrics slightly below. For a film-on-metal system the correction must be estimated from the measured spectrum (or R measured at several angles).

### 13.4 Checking the method on reference data

Hypothesis: the fall of ε(T) for α-Al₂O₃ is a **Planck weighting** effect, not a change of ε(λ). Aluminium oxide is transparent in the visible and near IR, and its strong phonon bands are in the mid-IR. As the temperature rises the Planck maximum moves to short wavelengths, where the material does not radiate, and the total ε falls.

Spectrum model for a film on metal:

```
ε(λ) = ε_short + (ε_long − ε_short) / (1 + (λ_c/λ)^w)
```

`ε_short = 0.05` (the metal shows through in the transparent region), `ε_long = 0.95` (phonon region), `w = 1.6`. The edge λ_c is fitted **to one point**, ε(1800 K) = 0.35, and comes out at 3.98<!--=step5b.kirchhoff.lam_c_um:.2f--> um (physically sensible: sapphire is transparent up to ~5 um, with multiphonon absorption beyond). The second point is then predicted:

```
ε(300 K) = 0.823<!--=step5b.kirchhoff.e300:.3f-->    against the reference 0.83    error −0.9<!--=step5b.kirchhoff.err_pct:.1f-->%
```

The same curve at the working temperatures:

| T, K | 300 | 800 | 1200 | 1500 | 1800 | 2000 | 2200 | 2740 |
|---|---|---|---|---|---|---|---|---|
| total ε | 0.823<!--=step5b.film_eps.T300:.3f--> | 0.591<!--=step5b.film_eps.T800:.3f--> | 0.467<!--=step5b.film_eps.T1200:.3f--> | 0.401<!--=step5b.film_eps.T1500:.3f--> | 0.350<!--=step5b.film_eps.T1800:.3f--> | **0.322<!--=step5b.film_eps.T2000:.3f-->** | 0.298<!--=step5b.film_eps.T2200:.3f--> | 0.248<!--=step5b.film_eps.T2740:.3f--> |
| Wien peak, um | 9.66 | 3.62 | 2.41 | 1.93 | 1.61 | 1.45 | 1.32 | 1.06 |

Caveat: the curve is fitted to data for bulk α-Al₂O₃ and applied to a film on metal. This is exactly the "oxide film optically active" scenario; how ε depends on the film thickness is a question for the experiment.

### 13.5 Shape sweep: support for the hypothesis, not confirmation

The curve has four shape parameters (`ε_short`, `λ_c`, `ε_long`, `w`), and only `λ_c` is fitted to the reference point. The other three were chosen by hand, and the agreement at the second point could be a consequence of that choice.

A sweep over 48 sets (`ε_short` 0.03–0.10, `ε_long` 0.85–1.00, `w` 1.0–4.0), each time refitting `λ_c` to the same 1800 K point:

| quantity | spread over the sets |
|---|---|
| predicted ε(300 K), reference 0.83 | **0.644<!--=step5b.shape.e300_min:.3f--> – 0.986<!--=step5b.shape.e300_max:.3f-->** (−22<!--=step5b.shape.e300_min_pct:.0f-->% … +19<!--=step5b.shape.e300_max_pct:+.0f-->%) |
| sets within ±10% of the reference | 28<!--=step5b.shape.n_ok300:.0f--> of 48<!--=step5b.shape.n:.0f--> |
| **working ε(2000 K)** | **0.303<!--=step5b.shape.T2000.min:.3f--> – 0.334<!--=step5b.shape.T2000.max:.3f-->, factor 1.10<!--=step5b.shape.T2000.factor:.2f-->** |
| ε(2000 K) only over the sets reproducing 300 K | 0.304<!--=step5b.shape.T2000.ok_min:.3f--> – 0.323<!--=step5b.shape.T2000.ok_max:.3f--> |
| ε(2100 K) | 0.282<!--=step5b.shape.T2100.min:.3f--> – 0.327<!--=step5b.shape.T2100.max:.3f--> |
| for comparison, ε(2740 K) | 0.189<!--=step5b.shape.T2740.min:.3f--> – 0.290<!--=step5b.shape.T2740.max:.3f-->, factor 1.53<!--=step5b.shape.T2740.factor:.2f--> |

**The correct wording:** within a reasonable family of curves there exists a set that reproduces both reference points. This supports the hypothesis "the fall of ε is a weighting effect"; it does not confirm it.

### 13.6 But the working number is robust

At the ~2000 K working temperature the shape freedom barely reaches the number: the edge is pinned by the 1800 K condition, and 2000 K is not far from the anchor. A 10% spread, against a factor of 1.5 at 2740 K and the factor of 5–7 that stood as the main uncertainty in the earliest versions. In the computation the shape sweep gives 8.3<!--=step5.sensitivity.film_shape.lo:.1f-->–9.0<!--=step5.sensitivity.film_shape.hi:.1f--> kg Al (section 12).

A second benefit: at ~2000 K the Al₂O₃ is still solid (it melts at 2345 K). The first version extended the solid-oxide curve to 2740 K, above the oxide's own melting point.

### 13.7 Bare melt: ε ≈ 0.17, not 0.05

The first version's value of 0.05 is polished **solid** Al at 300–900 K. The emissivity of a metal grows with its electrical resistivity (Hagen–Rubens), and liquid Al at melting has a resistivity of ~24 uOhm·cm against ~11 for the solid at 900 K and 2.7 at room temperature.

Estimate of the total hemispherical ε from resistivity (Parker & Abbott, NASA SP-55, 1965):

| T, K | 300 | 600 | 900 | 1000 (liq.) | 1500 | 2000 | 2740 |
|---|---|---|---|---|---|---|---|
| ρ_e, uOhm·cm | 2.7 | 6.6 | 10.5 | 25.2<!--=verify_step5.resistivity.T1000:.1f--> | 32.4<!--=verify_step5.resistivity.T1500:.1f--> | 39.7<!--=verify_step5.resistivity.T2000:.1f--> | 50.4<!--=verify_step5.resistivity.T2740:.1f--> |
| ε | 0.021<!--=verify_step5.bare_eps.T300:.3f--> | 0.045<!--=verify_step5.bare_eps.T600:.3f--> | 0.068<!--=verify_step5.bare_eps.T900:.3f--> | 0.105<!--=verify_step5.bare_eps.T1000:.3f--> | 0.141<!--=verify_step5.bare_eps.T1500:.3f--> | **0.173<!--=verify_step5.bare_eps.T2000:.3f-->** | 0.217<!--=verify_step5.bare_eps.T2740:.3f--> |

Check where reference data exist: polished solid Al at 600–900 K gives 0.045<!--=verify_step5.bare_eps.T600:.3f-->–0.068<!--=verify_step5.bare_eps.T900:.3f--> against the reference 0.04–0.07 (check 5.8). At 300–400 K the formula is below the handbook values (a perfectly clean surface without native oxide).

This is **an estimate, not a measurement**: no direct data on the total ε of liquid Al above ~1500 K could be found, and the liquid resistivity above ~1500 K is an extrapolation. But the order of magnitude is clear: liquid metal at 2000 K is not a shiny polished surface at room temperature.

**Validity at the working wavelengths.** Parker–Abbott is an expansion of the free-electron (Drude) model in its relaxation region, which requires ωτ ≪ 1. For liquid Al at 2000 K (39.7 uOhm·cm) the electron relaxation time is τ ≈ 6·10⁻¹⁶ s, so ωτ ≈ 0.8 at the Wien peak (1.45 um) and ≈ 0.4 at 3 um: the estimate is at the edge of its validity. (A point in its favour: the interband absorption of solid Al near 1.5 eV is a band-structure feature that the liquid largely loses, so the liquid behaves closer to free electrons.)

**Why the result is robust to it anyway.** Evaporation is driven by the heat input, and re-radiation takes only part of it. At 2000 K a plate radiating from both sides loses `2εσT⁴` = 0.31 MW/m² at ε = 0.17 and 0.58 MW/m² at ε = 0.32, against a peak input of 2.2–4.0 MW/m² (220<!--=step4.fragments.film.panels.q_peak_wcm2:.0f-->–401<!--=step4.fragments.film.mli.q_peak_wcm2:.0f--> W/cm²) on thin fragments. Across the whole flight the constant-ε sweep gives the elasticity directly: a sevenfold change in ε (0.05 → 0.35) changes the mass by less than a factor of two (15.2<!--=step5.sweep.e005.total:.1f--> → 7.9<!--=step5.sweep.e035.total:.1f--> kg). Even an ε of the bare melt anywhere between 0.10 and 0.25 keeps the mass between 13.9<!--=step5.sweep.e010.total:.1f--> and 10.2<!--=step5.sweep.e025.total:.1f--> kg.

### 13.8 Hence two scenarios, and close ones

- **active film** → ε(2000 K) ≈ 0.30–0.33 → 8.5 kg Al evaporated (8.3–9.0 depending on the curve shape);
- **no film** → ε(2000 K) ≈ 0.17 (estimate) → 12.0 kg Al evaporated.

The difference between the scenarios is 1.4×. Film growth with any τ gives values in between.

**And this reframes the question for the experiment.** Not "measure ε" but **"at what oxide thickness does the film become optically active"**, with the understanding that the answer moves the mass within 1.4×, not by multiples.

---

## 14. Experimental part and NU instruments

### 14.1 The closed design of the work

The model determines which parameters decide the outcome; the lab part measures one of them. After the corrections the honest wording is: **the measurement pins down the surface scenario (a factor of 1.4) and the oxide-thickness threshold**. The main mass uncertainties, the bill of materials of the fragments and the breakup altitude, are not closed by the lab; they are closed by data on a specific spacecraft and by observations of re-entries.

The answer to "why do you need the facility": to measure ε(T) of the "oxide on Al 6061" system as a function of oxide thickness over the 1500–2200 K working range, one of the quantities the aluminium vapour yield depends on, and the only one that can be measured with room-temperature optics.

### 14.2 The required spectral band

The share of Planck energy carried by each range is computed, not estimated:

| T, K | Wien peak, um | 95% of energy, um |
|---|---|---|
| 1500 | 1.93<!--=step5.band.T1500.wien_um:.2f--> | 1.11<!--=step5.band.T1500.lo_um:.2f--> – 10.86<!--=step5.band.T1500.hi_um:.2f--> |
| 1800 | 1.61<!--=step5.band.T1800.wien_um:.2f--> | 0.92<!--=step5.band.T1800.lo_um:.2f--> – 9.05<!--=step5.band.T1800.hi_um:.2f--> |
| 2000 | 1.45<!--=step5.band.T2000.wien_um:.2f--> | 0.83<!--=step5.band.T2000.lo_um:.2f--> – 8.15<!--=step5.band.T2000.hi_um:.2f--> |
| 2200 | 1.32<!--=step5.band.T2200.wien_um:.2f--> | 0.76<!--=step5.band.T2200.lo_um:.2f--> – 7.41<!--=step5.band.T2200.hi_um:.2f--> |

The union over the 1500–2200 K working range: **0.76<!--=step5.band.union_lo_um:.2f--> – 10.9<!--=step5.band.union_hi_um:.1f--> um**. It is covered by a pair of UV-Vis-NIR and mid-IR FTIR, both with an integrating sphere: the total (specular + diffuse) directional-hemispherical R is needed.

![Bridge to the experiment](../figures/step5_experiment.png)

### 14.3 Measurement protocol

- **Samples:** Al 6061, **thin-walled** (ε matters only for those), a series in oxide thickness.
- **How to get a thickness series.** Thermal oxidation of Al below its melting point is self-limiting at tens to hundreds of nanometres: a furnace can give only the lower end of the series (native ~3 nm → ~0.1–0.2 um). Thicker films come from anodizing (porous anodic oxide, sealing needed), plasma electrolytic oxidation, or deposited Al₂O₃ (ALD, magnetron sputtering). Their structures differ (amorphous/γ rather than α). In the 6061 alloy, magnesium migrates into the surface oxide on heating (MgO, MgAl₂O₄ spinel); this is part of the real surface and should be recorded (EDS/XPS).
- **Oxide thickness:** ellipsometry up to ~1 um; SEM cross-sections for thicker films.
- **Reflectance:** total hemispherical R(λ) in the UV-Vis-NIR and in the mid-IR.
- **Processing:** ε(λ) = 1 − R(λ) → Planck weighting at 1500–2200 K → ε(T) as a function of oxide thickness.
- **Goal:** find the thickness at which ε(2000 K) changes from the bare-metal value (~0.1–0.2) to the film value (~0.3). Whether this transition is a threshold or gradual is not yet known; the film-growth model in section 9.4 assumes it is gradual.

### 14.4 What NU has and what is missing

There is no public list of instruments on the NU Core Facilities website, only categories (84 instruments for analytical chemistry, materials science, electron microscopy, biology). A check of the catalogue gave:

- **UV-Vis-NIR:** Shimadzu UV-2600i with the ISR-2600Plus integrating sphere, range **0.22–1.4 um**, not 2.5.
- **FTIR:** **ATR** only: Bruker Alpha II, Thermo Nicolet iS12.

Share of Planck energy by spectral range:

| T, K | < 1.4 um | 1.4–2.5 | 2.5–5 | > 5 um |
|---|---|---|---|---|
| 1500 | 8.3<!--=step5b.fractions.T1500.lt14:.1f-->% | 35.1<!--=step5b.fractions.T1500.b14_25:.1f-->% | 40.1<!--=step5b.fractions.T1500.b25_5:.1f-->% | 16.6<!--=step5b.fractions.T1500.gt5:.1f-->% |
| 2000 | 22.8<!--=step5b.fractions.T2000.lt14:.1f-->% | 40.6<!--=step5b.fractions.T2000.b14_25:.1f-->% | 28.0<!--=step5b.fractions.T2000.b25_5:.1f-->% | 8.6<!--=step5b.fractions.T2000.gt5:.1f-->% |
| 2200 | 29.1<!--=step5b.fractions.T2200.lt14:.1f-->% | 40.0<!--=step5b.fractions.T2200.b14_25:.1f-->% | 24.1<!--=step5b.fractions.T2200.b25_5:.1f-->% | 6.8<!--=step5b.fractions.T2200.gt5:.1f-->% |

### 14.5 The 1.4–2.5 um gap is harmless, now checked with interference

The 1.4–2.5 um range carries ~35–40% of the Planck energy. The first version checked the interpolation across the gap on its own smooth curve, which is flat there by construction; that checks the arithmetic, not the physics (0.1% error).

The real danger is the **interference fringes** of a transparent film on metal: at the 0.3–5 um thicknesses that are planned to be measured, their period `λ²/(2nd)` falls right into this range. Computation for an Al₂O₃ film (n = 1.65) on Al (Drude, Rakic 1998), error in ε(2000 K) from linear interpolation across the gap:

| film thickness, um | 0.1 | 0.3 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|---|
| absolute error | +0.0002<!--=step5b.interference.d1.err_abs:+.4f--> | −0.0011<!--=step5b.interference.d3.err_abs:+.4f--> | +0.0008<!--=step5b.interference.d5.err_abs:+.4f--> | +0.0010<!--=step5b.interference.d10.err_abs:+.4f--> | +0.0002<!--=step5b.interference.d20.err_abs:+.4f--> | +0.0001<!--=step5b.interference.d50.err_abs:+.4f--> |
| relative to the working ε ≈ 0.32 | 0.0<!--=step5b.interference.d1.err_pct:+.1f-->% | −0.3<!--=step5b.interference.d3.err_pct:+.1f-->% | +0.3<!--=step5b.interference.d5.err_pct:+.1f-->% | +0.3<!--=step5b.interference.d10.err_pct:+.1f-->% | +0.1<!--=step5b.interference.d20.err_pct:+.1f-->% | 0.0<!--=step5b.interference.d50.err_pct:+.1f-->% |

The fringe contrast is small: the Al substrate reflects ~95%, and in the transparent region the film adds hundredths to ε. **There is no need to extend the UV-Vis beyond 1.4 um.** Caveats: the Drude model without the Al interband peak near 0.8 um, and a smooth surface; this should be checked on rough samples.

### 14.6 The mid-IR is required

If the mid-IR is not measured and a constant is extrapolated from 1.4 um:

| T, K | true ε | without mid-IR | error |
|---|---|---|---|
| 1500 | 0.401<!--=step5b.no_midir.T1500.true:.3f--> | 0.050 | −88<!--=step5b.no_midir.T1500.err_pct:.0f-->% |
| 2000 | 0.322<!--=step5b.no_midir.T2000.true:.3f--> | 0.050 | −84<!--=step5b.no_midir.T2000.err_pct:.0f-->% |
| 2200 | 0.298<!--=step5b.no_midir.T2200.true:.3f--> | 0.050 | −83<!--=step5b.no_midir.T2200.err_pct:.0f-->% |

Even if the position of the phonon edge is taken from the literature rather than measured, its 3–6 um uncertainty gives:

| T, K | edge at 3 um | edge at 6 um | spread |
|---|---|---|---|
| 1500 | 0.486<!--=step5b.edge.T1500.edge3:.3f--> | 0.292<!--=step5b.edge.T1500.edge6:.3f--> | +66<!--=step5b.edge.T1500.spread_pct:+.0f-->% |
| 2000 | 0.400<!--=step5b.edge.T2000.edge3:.3f--> | 0.229<!--=step5b.edge.T2000.edge6:.3f--> | +75<!--=step5b.edge.T2000.spread_pct:+.0f-->% |
| 2200 | 0.373<!--=step5b.edge.T2200.edge3:.3f--> | 0.211<!--=step5b.edge.T2200.edge6:.3f--> | +77<!--=step5b.edge.T2200.spread_pct:+.0f-->% |

That is where all of the spectral structure is.

### 14.7 Why ATR is unsuitable

ATR measures the near-surface layer with an evanescent wave through a crystal pressed against the sample:

1. It does not give the sample reflectance at all, neither specular nor hemispherical.
2. The penetration depth is ~0.5–2 um in the mid-IR. But the whole film-on-metal system is needed, because in the transparent region it is the substrate that radiates.
3. Proper optical contact cannot be made with a rigid, rough metal coupon.

### 14.8 What to look for by name

**A gold-coated mid-IR integrating sphere** (for example, an external module for the FTIR). It gives the total directional-hemispherical R, exactly what is needed for ε = 1 − R. It is an accessory to the spectrometer, so its availability must be checked separately from the instrument name; in the catalogue it may be listed as an accessory.

**A DRIFT accessory (Praying Mantis and similar) is only a fallback.** It is designed for powders, does not collect the full hemisphere (off-axis ellipsoidal mirrors) and gives no absolute R without calibration against a standard. The first version put it on a par with the sphere; that is wrong.

If there is no sphere, this is **the main blocker of the experimental part**, and it should be the first item to discuss. The options then are:

1. Find an instrument with a gold sphere in another lab.
2. DRIFT with calibration against a diffuse gold standard and an explicit estimate of the lost specular fraction.
3. Narrow the task: measure only the transparent region and take the phonon-edge position from the literature, honestly carrying ±75%. This is weaker, but it will distinguish bare metal (~0.1–0.2) from an active film (~0.3).

### 14.9 What the measurement closes and what it does not

1. The optical constants themselves depend on temperature, and the method does not capture that. The shape sweep (13.5) shows that this barely affects the working number.
2. The real surface during entry is melt with a film, while a solid oxidised sample is measured. The measurement pins down the "active oxide film" scenario and the transition threshold; the "bare melt" scenario cannot be measured this way.
3. Kirchhoff requires opacity; the sphere gives ε near normal incidence, while the model needs the hemispherical value (13.3).
4. **The measurement does not close the main mass uncertainties**, the distribution of Al over fragments and the breakup altitude (section 12).

---

## 15. Full list of assumptions

### Object physics

1. **Constant Cd = 1.5.** At 120–90 km the flow is free-molecular (Kn ≫ 1, Cd ≈ 2.0–2.2), below ~70 km it is continuum (Cd ≈ 0.9–1.2). We do not interpolate in the Knudsen number: drag in the free-molecular zone is negligible (`D/m ≈ 0.011 m/s²` against `g·sin γ ≈ 0.25`). But Cd also enters the continuum phase, where everything is decided, so it is a sweep parameter. The 1.0–2.2 spread is worth 5.8 km in heating-peak altitude.
2. **Fixed shape and frontal area.** Tumbling is accounted for statistically, through Cauchy's formula.
3. **The nose radius is a fitting parameter.** A tumbling irregular body physically has none. Switching to a surface average through the shape factor removes the sharpness of the problem: the spread of φ is worth 20%, the spread of the effective Rn 124%.
4. **The stagnation-point correlation** is extended over the surface by the `cos θ` angular distribution, not applied to the whole body.
5. **A thermal model lumped through the thickness.** Checked: the heating depth over the flight is `√(α·t) = 14 cm`, and both walls considered (1 and 16 mm) are much thinner.
6. **The mass does not decrease in the trajectory equation.** Ablation is computed along the trajectory; the feedback on β is not closed.
7. **Fragments come in three classes** (panels, primary structure, small parts) with fixed mass fractions. The thin-walled mass fraction is a sweep axis (3.1× in mass).
8. **All fragments are thermally aluminium.** The Al fraction of a fragment only sets which part of the evaporated mass is aluminium. The distribution of Al over fragments at a total of 30% is a sweep axis: 0–17 kg, the strongest one for mass.
9. **Stable orientation** (the `cos θ` distribution). The fast-tumbling limit (mean flux) is checked: 5.9 kg versus 8.5.
10. **Only the windward half of a shell is heated**: the melt of a compact fragment is at most 50%.

### Dynamics

11. **Equations in the inertial frame.** Atmospheric rotation enters through `V_rel = V − ωR⊕·cos i` in drag and heating; there are no Coriolis or centrifugal terms in this formulation. Not included: the cross-track component of the atmosphere velocity (`ωR·cos(lat)·cos A`, ~220 m/s at 40° S; it enters |V_rel| quadratically, ~0.1%) and the out-of-plane deflection of the trajectory. The first version called Coriolis (`2ωV ≈ 1.1 m/s²`) "the largest assumption"; for an inertial formulation that is wrong.
12. **Spherical Earth, μ/r² gravity, no J2** (J2 gives ~10⁻³ of g over 300 s).
13. **No lift or side force** → planar motion.
14. **No wind.**

### Atmosphere

15. **The profile is frozen at the moment of entry.** In 300 s the object covers ~16° of arc, ~10° in latitude, over which ρ(70 km) changes by a few percent, less than the uncertainty of MSIS itself.

### Heating and ablation

16. **Radiative heating from the shock layer is not included**: it switches on around 10 km/s and is negligible for entry from orbit (7.5 km/s).
17. **Boiling at the stagnation pressure over the whole windward surface.** On the sides the pressure is lower (`~cos²θ`), and boiling there happens at a lower temperature, so the assumption underestimates evaporation: by 17<!--=step5.side_pressure.film.change_pct:.0f-->% with the local Newtonian pressure (section 9.5). Evaporation below the boiling point (diffusive, Langmuir) is not modelled.
18. **Vapour blowing is not in the base case**; sensitivity axis η 0–0.6 (8.5 → 4.9 kg).
19. **The heat of Al oxidation is not in the baseline** (31 MJ per kg of Al, 34.9 MJ per kg of O). Its size is set by the oxygen supply through the boundary layer, not by the film thickness. At the diffusion limit it adds 0.32–0.53 of the hot-wall flux and raises evaporation up to ×1.8 (film) and ×1.5 (bare melt); it is a sensitivity axis η 0–1 (section 9.5). The real η is below 1: part of the vapour burns in the gas, and the oxide skin passivates the melt.
20. **An axisymmetric angular distribution** for a tumbling irregular body is an idealization without a free parameter.
21. **The fate of the stripped melt is not modelled**: that is the width of the melt/evaporation bracket. Evaporation in place is computed with the melt retained (section 8.7).
22. **The film ε(T)** is a single spectral curve fitted to bulk α-Al₂O₃; **the bare-metal ε(T)** is an estimate from the resistivity of pure Al.
23. **Breakup altitudes are inputs**, not model results (section 10).

---

## 16. Constants and their sources

| quantity | value | source | uncertainty |
|---|---|---|---|
| μ = GM⊕ | 3.986004418e14 m³/s² | WGS-84 / IERS | ~1e−9 |
| R⊕ | 6371.0 km (mean) | — | oblateness ±21 km → < 0.3% in g |
| ω⊕ | 7.292115e−5 rad/s | — | exact |
| ρ₀ (placeholder) | 1.225 kg/m³ | U.S. Standard Atmosphere 1976 | — |
| H (placeholder) | 7.2 km | fit | see 3.4 |
| Sutton–Graves k | 1.7415e−4 (SI → W/m²) | NASA TR R-376 / TFAWS; units from Stardust | 10–20% for the correlation, +5% for the 1.83e−4 variant |
| DKR exponent | 3.15 | open implementations | constant not found, only the shape is used |
| φ (shape factor) | 0.27 | ⟨cos θ⟩ = 0.25, demise-code range 0.25–0.30 | 20% |
| c_p of air (hot wall) | 1300 J/(kg·K) | frozen c_p at 2500–3000 K | ~15% |
| p₀ / (ρV²) | 0.92 | Rayleigh pitot formula, γ = 1.4, M → ∞ | ~5% (→ ~5 K in T_boil) |
| η (blowing) | 0 in the base case; axis 0–0.6 | a laminar layer is blocked more strongly than a turbulent one (AIAA J. 10.2514/1.J053053) | value not verified |
| Y_O | 0.231 | mass fraction of O in air, well mixed below ~100 km | ~1% |
| ΔH of oxidation | 34.9 MJ per kg of O (31.0 per kg of Al) | ΔfH°(Al₂O₃) = −1675.7 kJ/mol, NIST-JANAF | ~3% at 2000 K |
| η_ox (oxidation) | 0 in the base case; axis 0–1 | fraction of the diffusion-limited O supply reacting at the wall | upper bound at 1 |
| c_p Al 6061, solid | 1038 J/(kg·K) | mean over 300–890 K from NIST Shomate for Al(s) | ~5% |
| c_p Al, liquid | 1177 J/(kg·K) | NIST Shomate for Al(l), 31.75 J/(mol·K) | ~5% |
| k (Al 6061) | 167 W/(m·K) | handbooks give 155–167 | ~7% |
| ρ (Al) | 2700 kg/m³ | — | — |
| melting T (6061) | 890 K | solidus 855, liquidus 925 K (ASM) | interval ±35 K |
| L of fusion | 397 kJ/kg | pure Al | ~5% |
| boiling T at 1 atm | 2792 K | CRC Handbook | — |
| boiling T at p | Clausius–Clapeyron | against the CRC vapour-pressure table: ≤ 1.4<!--=verify_step4.crc.max_diff_pct:.1f-->% | — |
| L of vaporization | 10.90 MJ/kg at 2792 K (294 kJ/mol) | CRC; Kirchhoff correction −406 J/(kg·K) | ~3% |
| ORSAT heat of ablation (Al) | 934.5 kJ/kg | NTRS 20140016958 | — |
| plate edge radius floor | 5 mm | estimate of self-blunting of an ablating edge | sensitivity axis |
| **film ε(T)** | 0.32<!--=step5.scenarios.film.eps2000:.2f--> at 2000 K | film-on-metal curve, λ_c from α-Al₂O₃ ε(1800 K) = 0.35 | 0.30<!--=step5.film_shape.eps2000_min:.2f-->–0.33<!--=step5.film_shape.eps2000_max:.2f--> over 48 shapes |
| **bare-metal ε(T)** | 0.17<!--=step5.scenarios.bare.eps2000:.2f--> at 2000 K | Parker & Abbott from the resistivity of pure Al | estimate, ±30% |
| reference ε of α-Al₂O₃ | 0.83 (300 K), 0.35 (1800 K) | compilation of emissivities of metals and oxides | — |
| Al resistivity | 2.65 (293 K) → 10.9 (solid, T_melt) → 24.2 uOhm·cm (liquid) + 0.0145/K | Desai et al., JPCRD 13, 1131 (1984); the liquid above ~1500 K is an extrapolation | ~10% |
| Al optics (Drude) | ħω_p = 14.98 eV, ħγ = 0.047 eV | Rakić et al. 1998 | only for the interference estimate |

Project rule: no number appears in the code without a line about its source.

---

## 17. Checks: all 37

Principle: every check tests **an identity or an independent benchmark**, not "looks plausible". Each one fails through `assert`; run them with `python -m pytest` (or one file at a time: `python verify_step4.py`). Two checks for the new fixes passed a mutation test: if the plate is made to radiate from one side again, or the limit of evaporation by the band mass is removed, they fail.

Separately, `test_docs.py` (6 tests) checks the documents against the code: the numbers in the README and in this report carry invisible tags `<!--=key:format-->` and are compared with `results/*.json`, which the scripts write; the results themselves are recomputed selectively and cross-checked between scripts. The check immediately found three numbers that had drifted from the code (section 18.2).

### verify_step1.py: trajectory (4<!--=checks.verify_step1:.0f-->)

1. **Kepler test.** Drag switched off: the specific energy `V²/2 − μ/r` and the angular momentum `V·r·cos γ` must be conserved. Drift ~1e−15, machine precision. A direct test of the ODE right-hand side.
2. **Convergence.** rtol from 1e−6 to 1e−12, max_step from 5 to 0.25 s: the peak altitude is 45.322 km in all cases.
3. **Allen–Eggers**: three analytic targets; the mismatch is explained by the violation of the γ = const assumption.
4. **Sensitivity to β**: `h* ∝ H·ln β`, −5.0 km per doubling against the predicted 5.0.

### verify_step2.py: atmosphere (5<!--=checks.verify_step2:.0f-->)

1. Spline accuracy against a direct pymsis call at 20 000 altitudes: away from the model seams the maximum is 1.2e−5, the median 3.7e−7.
2. **Numerical search for model seams**: 72.5 and 123.4 km in NRLMSISE-00, no kinks in 2.1.
3. Convergence in the tabulation step (2000–100 m): the peak altitudes agree to a metre.
4. **F10.7 mechanism**: the sensitivity grows with altitude (×1.07 at 120 km, ×15 at 400 km).
5. Ranking of environmental factors.

### verify_step3.py: heating (10<!--=checks.verify_step3:.0f-->)

1. **Units of k** from the Stardust benchmark: 1258 W/cm² against the computed ~1200.
2. Energy sanity: the absorbed heat is 1.8% of the kinetic energy.
3. **Earth rotation**: i = 90° coincides exactly with rotation switched off.
4. Sutton–Graves versus DKR: +0.40 km, −4.4%.
5. Rn does not move the peak altitude: the spread is exactly 0.
6. **Cauchy's formula**: A_wet = 4·A exactly.
7. **Size exponents** on a frozen trajectory: +1.500 and −1.500 to the sixth digit.
8. Energy to melt and to evaporate versus the energy available, by size (section 7.5).
9. **Blowing: closed form versus 200 iterations**, 1e−9.
10. Hot wall at several speeds at T_boil at local pressure and at 2740 K.

### verify_step4.py: ablation (10<!--=checks.verify_step4:.0f-->)

1. The mean of the angular distribution is exactly 0.25.
2. Closed-form evaporation rate versus numerical integration: 1e−9.
3. Convergence in the number of bands: spread < 0.5 p.p. at 24 bands.
4. **Energy balance**: input = re-radiation + storage + evaporation + blocked by blowing, residual ~1e−15, including with blowing, with the heat of oxidation and with boiling at the local pressure. The audit is accumulated **inside the same ODE system**.
5. Reproducing the ORSAT criterion: 8%.
6. **Nesting and mass conservation per band**: in every band evaporated ≤ molten ≤ band mass. Without a forced `max()`; the first version passed this test by construction.
7. Validity of the lumped model from the heating depth.
8. **Clausius–Clapeyron versus the Al vapour-pressure table (CRC)**: 1 Pa – 1 atm, difference ≤ 1.4%.
9. **A plate radiates from two sides**: the equilibrium temperature of the nose band against the analytic value, 0.03%; the ratio to a shell is 2^(−1/4).
10. **Flash regularization**: changing the time constant from 0.2 to 0.05 s changes evaporation by 0.004%.

### verify_step5.py: main result (8<!--=checks.verify_step5:.0f-->)

1. **Planck → Stefan–Boltzmann**: `π∫B dλ = σT⁴` to 1e−9. An identity without free parameters.
2. A constant ε is returned exactly.
3. Wien displacement: 1e−6.
4. Kirchhoff: the `R → ε → ε(T)` path matches the direct one.
5. **Mass conservation**: the sum over histogram layers = the direct sum over fragments with their Al fractions, 2e−16.
6. Independence of the median from the bin width, a regression test (the median over the raw series does not depend on the bin by construction).
7. The required 0.76–10.9 um band is covered by a UV-Vis-NIR + FTIR pair.
8. **Bare-metal ε from resistivity** against handbook values for polished Al at 600–900 K.

### In addition: analysis_step5b.py

Not checks in the strict sense but analyses with a numerical answer: the breakup-altitude transfer function, validation of the Kirchhoff method on α-Al₂O₃, the curve-shape sweep (48 sets), and the cost of the truncated band on the NU instruments with interference in the film.

---

## 18. Errors found along the way and how they were caught

### 18.1 Before review

Most errors were found by checks rather than by eye, and all of them looked plausible until checked.

| what was wrong | how it was caught | cost had it gone through |
|---|---|---|
| **A band at the boiling point could not cool down**: when the input dropped, it radiated energy it did not have | the energy balance was off by 50% | overestimated evaporation; everything else looked plausible |
| **Mass metric through q_stag in W/m²** | dimensional analysis: `q ~ Rn^(−1/2)`, but `A ~ Rn²` → `P ~ Rn^(+3/2)` | **a sign error**: a large body would "heat less" instead of "more" |
| **Plate Rn from the sphere equivalent** | a panel with β = 4 got Rn = 96 cm | panels would not evaporate at all |
| **Median from the histogram** | bin-width independence test: it moved by 1.5 km | the number would have had to be defended in a paper |
| **Two independent accountings of melt and evaporation** | a thick body gave "melt 0%, evaporation 1.6%" | the bracket did not add up |
| **"Solar activity changes ρ at 120 km by multiples"** | a direct check: ×1.07 | a superfluous sweep axis, a wrong physical picture |
| **Regime criterion through the heating depth** | both 1 mm and 16 mm are much thinner than 14 cm, so it does not separate them | right physics with the wrong justification |
| **"The injection altitude is robust"** | the transfer function | the claim was broader than what was shown |
| **"The Kirchhoff method is confirmed"** | sweep over 48 shape sets: ε(300 K) from 0.644 to 0.986 | the claim was stronger than what was shown |
| **Units of the Sutton–Graves k** (labelled W/cm² in the source) | the Stardust benchmark | an error of four orders of magnitude |

Numerical defects found along the way:

- `argmax` over the solver nodes made the peak altitude depend on `max_step` by ~1 km; fixed by resampling through dense_output.
- A hard "heating / boiling" switch made the right-hand side discontinuous: LSODA took 14 s on a plate instead of 0.07; smoothing over a 1% window gave a 200-fold speed-up with no loss of convergence.
- The `allen_eggers()` function took ρ₀ and H from the atmosphere object, which the MSIS wrapper does not have; the parameters were made explicit.

### 18.2 After review (version 2)

| what was wrong | how it was caught | cost |
|---|---|---|
| **A plate radiated from one side** | code review; now check 4.9 against the analytic value | for plates ε was effectively halved; the film scenario was overestimated 3–9 times |
| **A band evaporated more than its own mass** (the limit was only on the sum) | code review; now check 4.6 per band | panels at ε = 0.05: +4.6 kg beyond their mass; the "no film" scenario was overestimated by 12% |
| **Melt from the final enthalpy**, with nesting enforced through `max()` | code review: a plate at ε = 0.35 gave "melt = evaporation = 1.8%" | the "melt" column falsely dropped with ε (33 → 16 kg); the nesting check was a tautology |
| **Boiling at 1 atm** on a surface where the pressure is 0.1–1 kPa | estimate of the stagnation pressure along the fragment trajectories | the evaporation threshold was overestimated threefold; 0.6 kg instead of 8.5 in the film scenario |
| **Bare-melt ε of 0.05**, the value for polished solid Al | estimate from the resistivity of liquid Al | the "film / no film" scenarios seemed 2–3.5 times apart, in fact 1.4 |
| **The solid Al₂O₃ curve extended to 2740 K**, above its melting point (2345 K) | review | removed: the working temperature is ~2000 K |
| **All fragments are aluminium, ×0.30 uniformly**, without being listed in the assumptions | code review | the strongest mass axis was invisible |
| **Ferreira: 21% instead of 32%**, the ~30 kg of oxide clusters converted as if they were Al₂O₃; "assumption" vs "MD" | reading the paper | the comparison and its interpretation ("ε ≈ 0.12, τ ≈ 300 s") were wrong |
| **Ferreira's clusters called AlO**, as if 1:1 | external review: 29.8 kg from 24.0 kg of Al means O/Al ≈ 0.4; checked against the paper (Section 3.3, Table 1b) | wording only: the comparison is in kg of Al; the place in the paper is now cited |
| **The 95% demisability was compared with the 3% vaporizable fraction** | review: 95% is a melt criterion | "a thirtyfold gap" instead of ~2× |
| **Steps 1–2 did not reproduce**: `earth_rotation=True` became the default in step 3 | running verify_step1: 44.94 km instead of 45.32 | the numbers in section 3 did not match the code |
| **c_p = 900 for the liquid and T_melt = 933 K for the alloy** | review of the properties | the enthalpy to boiling was underestimated by ~24%; the 3.4% match with ORSAT was a cancellation of errors |
| **Plate Π from the full thickness** | review: both faces are wetted | Π = 7.1 instead of 13.2 (same regime) |
| **Film growth: ε(200 s) and ε_ox = 0.35** | review: the fragment clock starts at breakup | a column of the film-growth table did not match the computation |
| **Blowing η: laminar/turbulent labels swapped; blowing described as part of the model but never called** | code review | the assumption described something that did not exist |
| **The 1.4–2.5 um gap was checked on the smooth model** | review: a circular argument | now checked with interference; the conclusion held (≤ 0.3%) |
| **DRIFT on a par with an integrating sphere**; thermal oxide up to 5 um | review of the protocol | a wrong instrument request; an unrealizable sample series |
| **"Coriolis is the main dynamical assumption"** in an inertial formulation | review of the equations | a wrongly named assumption |
| **"H minus ~2 km"** with a transfer gain of 0.77 | review | the rule holds only near 78 km |
| **Nine checks always returned True**: convergence, Allen–Eggers, sensitivity to β, MSIS seams, geography versus F10.7, S–G versus DKR, energy to evaporate, hot wall, validity of the lumped model | conversion to pytest: `return True` cannot fail | each now has a condition through `assert` |
| **The two-sided plate check took the analytic value from the same model flag** | mutation test: with a broken plate the check still passed | the number of sides is now set by geometry, not by a flag |
| **Allen–Eggers altitude with γ at the peak: 45.49 km in the text, 45.40 in the code**; 72.0 and 84.9 km in the transfer function, 400 W/cm², all manual roundings | `check_docs.py` | fixed automatically from the results |
| **Heat of oxidation called small because the film is nanometre-thick** (version 2) | external review: the heat is set by the O supply through the boundary layer, not by the film thickness | an axis of up to ×1.8 was missing from the budget; now section 9.5 |
| **Boiling on the side bands at the stagnation pressure** stated only qualitatively | external review | now quantified: +17% with the local pressure, the baseline is conservative |
| **Validity of the bare-melt ε at the Wien peak not discussed** | external review | ωτ ≈ 0.8, at the edge of the Drude relaxation region; the robustness argument is now explicit (13.7) |
| Small items: "4–16%" with ×1.24 at 40 km; "16° in latitude" (≈10°); Stardust "measured"; a 15.9 kg Ferreira line on the plot for 175 kg; 12 commits instead of 11 | review | — |

---

## 19. Anticipated questions and answers

**"You got injection at 75 km because you set breakup at 78."**
Yes, and that is stated directly. The model does not predict the absolute injection altitude; it maps the breakup altitude H to the injection altitude: `h ≈ 75.9 + 0.82·(H − 78)` km in the 68–83 km window. This makes it possible to convert observed breakup altitudes into injection altitudes. The main altitude uncertainty lies in the observational data.

**"Why does your aluminium boil at 2000 K and not at 2790?"**
Because at a fragment surface the pressure is the stagnation pressure, 0.1–1 kPa at 70–80 km, not 1 atm. By Clausius–Clapeyron (checked against the CRC vapour-pressure table to 1.4%) Al boils there at 1759<!--=step4.fragments.film.panels.Tb_min:.0f-->–2039<!--=step4.fragments.film.mli.Tb_max:.0f--> K. Ferreira, incidentally, computes oxidation at 2200 K.

**"Your validation of the Kirchhoff method is tuned."**
Partly, yes. One parameter out of four is fitted to the reference point. The sweep over 48 shape sets shows that the prediction of the second point ranges from 0.644 to 0.986. That is why the claim is support for the hypothesis, not confirmation. But the working number ε(2000 K) is robust to the choice of shape: 0.303–0.334.

**"Your ε of bare liquid aluminium is made up."**
It is an estimate from electrical resistivity (Parker–Abbott), checked on solid polished Al at 600–900 K (it matches the handbook). We found no direct measurements of the total ε of liquid Al above 1500 K. Even with a ±30% error the difference between the scenarios stays around 1.3–1.6×.

**"Your agreement with Ferreira is a coincidence."**
It is order-of-magnitude agreement, not validation: a different mass, different entry conditions, different computed quantities (theirs is the oxidised fraction under full ablation, ours the evaporated fraction). Their 32% is above our 16–23% with uniform Al and is reached if the Al is concentrated in thin-walled parts.

**"The nose radius of a tumbling body is undefined."**
Agreed. That is why the mass channel is computed through the `cos θ` angular distribution, whose mean is the shape factor φ = 0.25. The spread of φ is worth 20%, that of the effective Rn 124%, and Rn does not affect the altitude at all. The fast-tumbling limit is checked separately: 5.9 kg versus 8.5.

**"Why not DRAMA?"**
The DRAMA/ORSAT demise criterion is melting (0.93–1.01 MJ/kg), not evaporation (13.6 MJ/kg). Atmospheric chemistry needs the evaporated mass, and it costs 13.4 times more energy. The upper end of our bracket reproduces the DRAMA criterion; this is a comparison with it, not a replacement.

**"Why NRLMSIS 2.1 when everyone uses NRLMSISE-00?"**
NRLMSISE-00 has a seam in the density derivative at 72.5 km, right in the ablation zone; 2.1 does not, and 2.x was refitted for the mesosphere. The atmospheric studies (Barker with GEOS-Chem, Maloney with WACCM) do not use MSIS. Version 00 is kept switchable, and the difference is a row in the budget: 0.9 km.

**"Coriolis?"**
The equations are written in the inertial frame, and atmospheric rotation enters through the speed relative to the air. There are no Coriolis terms in this formulation. The cross-track component of the atmosphere velocity is not included; it contributes ~0.1% to the speed magnitude.

**"You ignore the heat of oxidation."**
Not any more. At the diffusion limit of the oxygen supply it adds 0.32–0.53 of the hot-wall flux and raises the evaporated mass up to ×1.8 for the film scenario and ×1.5 for bare melt, without moving the altitude (section 9.5). The real value is below that bound: part of the vapour burns in the gas, and an intact oxide film passivates the melt, which is also why "film" pairs with small oxidation and "bare melt" with large.

**"What decides the mass, if not ε?"**
Where the aluminium sits (0–17 kg at the same 30%), what fraction of the mass is thin-walled (4.3–13.1 kg), and at which altitude the object breaks up (4.4–11.1 kg). All of these are data about a specific spacecraft and its entry.

**"Why do you need our facility?"**
To measure ε(T) of the "oxide on Al 6061" system as a function of oxide thickness at 1500–2200 K, one of the quantities the Al vapour yield depends on, and the only one that can be measured with room-temperature optics. The total hemispherical reflectance over 0.76–11 um is needed: UV-Vis with a sphere (available) and FTIR with a gold sphere for the mid-IR.

---

## 20. Next steps

**Modelling**

- Take the real list of fragments and materials of a specific spacecraft (Starlink/OneWeb-like) instead of three classes with uniform Al; this is the strongest mass axis.
- Evaporation below the boiling point (diffusive/Langmuir) and the `~cos²θ` pressure on the sides.
- A tumbling criterion: compare the rotation period of fragments with the thermal time constant (~3 s for a 1 mm plate).
- Check the blowing η against a primary source and decide whether blowing goes into the base case.
- Close the feedback of ablation on the ballistic coefficient.
- A joint boundary-layer model of vapour blowing and vapour combustion (where the Al flame stands and how much of its heat returns to the wall): it fixes the blowing η and the oxidation η together, and they act in opposite directions.
- Replace the axisymmetric angular distribution with an asymmetric one for the real fragment geometry.

**Observations**

- The main altitude uncertainty lies in the breakup-altitude data. A larger sample of observed re-entries will give more than any refinement of the physics.

**Experiment**

- First of all, find out whether a gold mid-IR integrating sphere is available.
- A series of Al 6061 samples: native and thermal oxide (up to ~0.2 um), anodic/PEO/ALD for thick films; thickness by ellipsometry and SEM; surface composition by EDS/XPS.
- Find the oxide thickness at which ε(2000 K) changes from ~0.1–0.2 to ~0.3.
- A literature search for direct measurements of ε of liquid Al above 1500 K.

**Publication**

- The axis of the paper: DRAMA counts melt, the atmosphere needs vapour; a 13.4× difference in energy; Al boils at the stagnation pressure (~2000 K); the result is a bracket whose width is set by the droplet phase.
- The structural result about the two uncertainty budgets, and that the mass comes down to the bill of materials of the fragments.
- The comparison with Ferreira as order-of-magnitude agreement.

---

## 21. Repository layout and how to run

```
pip install -r requirements-dev.txt

python run_step1.py        # trajectory on the exponential placeholder
python explore_step1b.py   # diagnosis: heating peak, Cd, fragmentation
python run_step2.py        # NRLMSIS: atmosphere, trajectory, environment sweeps
python run_step3.py        # heating, Earth rotation, uncertainty budget
python run_step4.py        # fragmentation, thermal response, ablation, bracket
python run_step5.py        # main result, budget, Ferreira comparison
python analysis_step5b.py  # transfer function, ε validation, shape sweep, instruments

python -m pytest           # 37 checks in verify_step*.py + 6 document tests
python check_docs.py       # numbers in the README and the report against results/*.json
python check_docs.py --fix # rewrite the numbers from results/
```

Every `run_step*.py` and `verify_step*.py` writes its numbers to `results/<name>.json`; after a model change, rerun the scripts, then `check_docs.py --fix` and pytest.

Figures (the `figures/` folder):

| file | what it shows |
|---|---|
| `figures/step1_trajectory.png` | trajectory in the exponential atmosphere, Allen–Eggers comparison |
| `figures/step1_gamma_sweep.png` | entry-angle sweep: the time changes, the peak altitude does not |
| `figures/step1b_fragmentation.png` | heating peak versus deceleration peak, fragments after breakup, saturation in β |
| `figures/step2_atmosphere.png` | MSIS profile versus the placeholder, local scale height |
| `figures/step2_trajectory.png` | trajectory in three atmospheres |
| `figures/step3_heating.png` | heat flux, accumulated heat, Earth rotation by inclination |
| `figures/step3_budget.png` | altitude and mass budgets |
| `figures/step4_epsilon.png` | angular distribution and boiling threshold, plate and intact object versus ε |
| `figures/step4_bracket.png` | altitude distribution of evaporated Al: bare melt, film growth, film |
| `figures/step5_main.png` | main result: distribution by scenario, mass and median versus ε, the Ferreira line |
| `figures/step5_sensitivity.png` | the two step 5 budgets |
| `figures/step5_experiment.png` | required band, ε(T) of the two scenarios, measurement protocol |
| `figures/step5b_revision.png` | transfer function, Planck weighting with interference, validation on α-Al₂O₃ |

Git history: one commit per step and per major revision. What was corrected during the work and why is in section 18 and in the git history; the intermediate wording of the first version is in older commits (`README_history.md`).

---

## 22. References

- Friedman R., Maček A. *Ignition and combustion of aluminium particles in hot ambient gases.* Combustion and Flame 6, 9–19, 1962 (ignition near the melting point of Al₂O₃).
- Chase M.W. *NIST-JANAF Thermochemical Tables*, 4th ed., J. Phys. Chem. Ref. Data Monograph 9, 1998 (ΔfH° of Al₂O₃).
- Allen H.J., Eggers A.J. *A study of the motion and aerodynamic heating of ballistic missiles entering the Earth's atmosphere at high supersonic speeds.* NACA TR-1381, 1958.
- Sutton K., Graves R.A. *A general stagnation-point convective heating equation for arbitrary gas mixtures.* NASA TR R-376, 1971.
- NASA TFAWS Aerothermodynamics Course, 2012 (the value k = 1.7415e−4 for Earth).
- Lees L. *Laminar heat transfer over blunt-nosed bodies at hypersonic flight speeds.* Jet Propulsion 26, 1956.
- Emmert J.T. et al. *NRLMSIS 2.0: A whole-atmosphere empirical model of temperature and neutral species densities.* Earth and Space Science 8, e2020EA001321, 2021.
- Emmert J.T. et al. *NRLMSIS 2.1: An empirical model of nitric oxide incorporated into MSIS.* JGR Space Physics 127, e2022JA030896, 2022.
- Lips T. et al. *Equivalent re-entry breakup altitude and fragment list.* 6th European Conference on Space Debris (SESAM: panels at 95 km, body at 78 km).
- Ferreira J.P. *Space Debris Demise in the Atmosphere: The case of Aluminum.* UNOOSA/IAF, 2024 (observations: ATV-1 74 km, Cygnus OA6 70 km, Cluster II SALSA 80 km; demisability of an OneWeb/SpaceX-type design 95%).
- Ferreira J.P., Huang Z., Nomura K., Wang J. *Potential ozone depletion from satellite demise during atmospheric reentry in the era of mega-constellations.* Geophysical Research Letters 51, e2024GL109280, 2024 (Section 3.3 and Section 4: 24.0 kg of the 75 kg Al oxidised, 32%, into 29.8 kg of Al-rich oxide clusters labelled "AlO"; atom counts in Table 1b; MD set-up at 2,200 K and 86 km next to Figure 1a).
- Barker et al. Earth's Future, 2026 (title and DOI to be confirmed).
- Maloney et al. *Investigating the potential atmospheric accumulation and radiative impact of the coming increase in satellite reentry frequency.* JGR: Atmospheres, 2025.
- NASA ODPO, ORSAT; Kelley R.L., Jarkey D.R. *CubeSat material limits for design for demise*, NTRS 20140016958 (heat of ablation for generic aluminum 934.5 kJ/kg).
- CRC Handbook of Chemistry and Physics: boiling point and heat of vaporization of Al; vapour pressure of metals.
- NIST Chemistry WebBook: Shomate equations for Al (solid, liquid).
- Parker W.J., Abbott G.L. *Theoretical and experimental studies of the total emittance of metals.* NASA SP-55, 1965.
- Desai P.D. et al. *Electrical resistivity of aluminum and manganese.* J. Phys. Chem. Ref. Data 13, 1131, 1984.
- Rakić A.D., Djurišić A.B., Elazar J.M., Majewski M.L. *Optical properties of metallic films for vertical-cavity optoelectronic devices.* Applied Optics 37, 5271, 1998.
- *Variable transpiration cooling effectiveness in laminar and turbulent flows for hypersonic vehicles.* AIAA Journal, 10.2514/1.J053053.
- A compilation of high-temperature emissivities of metals and oxides (White Rose eprints 133266): α-Al₂O₃ 0.83 → 0.35 over 300–1800 K; Engineering ToolBox, emissivity of aluminium surfaces.
