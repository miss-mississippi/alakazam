"""Answers to objections to step 5. Run: python analysis_step5b.py

D. Transfer function "breakup altitude -> injection altitude".
   The model does NOT predict the injection altitude independently: it
   inherits it.
E. Validation of the Kirchhoff method on reference data for alpha-Al2O3,
   BEFORE touching the instrument; the working number at the boiling
   temperature.
F. Cost of the truncated spectral band on the actual NU instruments,
   including interference in the film that the smooth model does not see.
"""

from __future__ import annotations

import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_step5 as S
from reentry import EntryState
from reentry.ablation import Aluminium
from reentry.results import Recorder
from reentry.emissivity import (ALUMINA_REF, LAM_GRID, eps_film, fit_film_edge,
                                planck_spectral_radiance, total_emissivity)

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.22,
    "axes.spines.top": False, "axes.spines.right": False,
})

LAM = LAM_GRID
H_BODY_BASE = 78.0     # km, base breakup altitude of the body
T_WORK = (1800.0, 2000.0, 2100.0)   # K, Al boils at 0.3-2 kPa
R = Recorder("step5b")


def part_d_transfer():
    print("D. THE INJECTION ALTITUDE IS INHERITED, NOT PREDICTED")
    print("   Objection: 'you got injection at 75 km because you set breakup")
    print("   at 78'. We compute the transfer function explicitly.")
    print("   Base: active oxide film.\n")
    print(f"   {'shift, km':>10}{'h break':>11}{'injection median':>18}"
          f"{'offset':>10}{'mass, kg':>11}")
    mat = S.BASE
    hb, med, mass = [], [], []
    for d in (-15e3, -10e3, -5e3, 0.0, 5e3, 10e3, 15e3):
        s = S.summarize(S.run_model(S.build_fragments(h_shift=d), mat))
        b = H_BODY_BASE + d / 1e3
        hb.append(b); med.append(s["median"]); mass.append(s["total"])
        k = f"transfer.{'m' if d < 0 else 'p'}{abs(d)/1e3:.0f}"
        R[f"{k}.h_break"] = b
        R[f"{k}.median"] = s["median"]
        R[f"{k}.offset"] = b - s["median"]
        R[f"{k}.mass"] = s["total"]
        print(f"   {d/1e3:>+10.0f}{b:>11.0f}{s['median']:>18.2f}"
              f"{b - s['median']:>+10.2f}{s['total']:>11.1f}")
    hb, med = np.array(hb), np.array(med)
    gain_all = float(np.polyfit(hb, med, 1)[0])
    m = (hb >= 68) & (hb <= 83)
    fit = np.polyfit(hb[m], med[m], 1)
    gain_mid = float(fit[0])
    med78 = float(np.polyval(fit, 78.0))
    R["transfer.gain_all"] = gain_all
    R["transfer.gain_mid"] = gain_mid
    R["transfer.intercept78"] = med78
    R["transfer.mass_ratio"] = max(mass) / min(mass)
    print(f"\n   transfer gain over the full range       {gain_all:.2f}")
    print(f"   in the plausible 68-83 km window        {gain_mid:.2f}")
    print(f"   -> in the window: h_inj ≈ {med78:.1f} + {gain_mid:.2f}·(H − 78) km")
    print(f"   the mass meanwhile changes by {max(mass)/min(mass):.1f}x:")
    print("   the breakup altitude is the main lever for MASS as well.\n")

    print("   Offset (78 km − injection median) at a FIXED breakup altitude:")
    frags0 = S.build_fragments()
    cases = [("active oxide film", S.BASE, None, {}),
             ("bare melt", S.SCENARIOS["bare melt"], None, {}),
             ("tau=1 s", Aluminium(tau_oxide=1.0), None, {}),
             ("tau=10000 s", Aluminium(tau_oxide=1e4), None, {}),
             ("blowing eta=0.6", S.BASE, None, dict(blowing_eta=0.6)),
             ("tumbling", S.BASE, None, dict(flux_mode="uniform")),
             ("entry angle -1°", S.BASE, EntryState(gamma_deg=-1.0, inclination_deg=53.), {}),
             ("entry angle -3°", S.BASE, EntryState(gamma_deg=-3.0, inclination_deg=53.), {})]
    offs = []
    okeys = ["film", "bare", "tau1", "tau10000", "blowing", "tumbling", "gamma1", "gamma3"]
    for (lbl, mt, ent, kw), ok_ in zip(cases, okeys):
        f = S.build_fragments(entry=ent) if ent else frags0
        s = S.summarize(S.run_model(f, mt, **kw))
        offs.append(H_BODY_BASE - s["median"])
        R[f"offsets.{ok_}"] = H_BODY_BASE - s["median"]
        print(f"     {lbl:<20}{H_BODY_BASE - s['median']:>+8.2f} km")
    offs = np.array(offs)
    R["offsets.min"] = offs.min()
    R["offsets.max"] = offs.max()
    print(f"\n   -> Offset {offs.mean():+.1f} km, spread {offs.min():+.1f}"
          f" ... {offs.max():+.1f} km: robust")
    print("      to everything EXCEPT the breakup altitude itself.")
    print("\n   HONEST STATEMENT: the model does not predict the injection altitude;")
    print("   it maps the breakup altitude to the injection altitude with a gain")
    print("   of ~0.8. The main altitude uncertainty lies in the observations.\n")
    return hb, med, fit


def part_e_kirchhoff():
    print("E. VALIDATING THE METHOD BEFORE THE INSTRUMENT")
    print(f"   Reference for alpha-Al2O3: eps_total = {ALUMINA_REF[300.0]} at 300 K")
    print(f"   and {ALUMINA_REF[1800.0]} at 1800 K. Hypothesis: the drop is a "
          "WEIGHTING effect,")
    print("   not a change of eps(lam) with temperature.")
    print("   Test: ONE fixed eps(lam) must give both numbers.\n")
    lam_c = fit_film_edge()
    e = eps_film(LAM, 0.05, lam_c)
    e300 = total_emissivity(LAM, e, 300.)
    R["kirchhoff.lam_c_um"] = lam_c * 1e6
    R["kirchhoff.e300"] = e300
    R["kirchhoff.err_pct"] = 100 * (e300 / ALUMINA_REF[300.0] - 1)
    print(f"   Phonon-band edge fitted to ONE point (1800 K):"
          f" {lam_c*1e6:.2f} um")
    print(f"   The second point is predicted: eps(300 K) = {e300:.3f}"
          f"  vs the reference {ALUMINA_REF[300.0]}")
    print(f"   Error {100*(e300/ALUMINA_REF[300.0]-1):+.1f}%.\n")
    print(f"   {'T, K':>7}{'eps total':>13}{'Wien peak, um':>16}")
    for T in (300, 800, 1200, 1500, 1800, 2000, 2200, 2740):
        R[f"film_eps.T{T}"] = total_emissivity(LAM, e, T)
        print(f"   {T:>7}{total_emissivity(LAM, e, T):>13.3f}"
              f"{2.8977e-3/T*1e6:>16.2f}")
    print("\n   The working temperature is Al boiling at local pressure, 1750-2050 K,")
    print("   i.e. close to the 1800 K anchor. Extrapolation to 2740 K (above the")
    print("   melting point of Al2O3 itself, 2345 K) is no longer needed.\n")
    return lam_c


def part_e2_shape_sweep():
    """How much the validation and the working number depend on the HAND-
    picked shape.

    eps_film has four parameters. Only lam_c is fitted to the 1800 K
    reference point; eps_short, eps_long and width are chosen by hand. We
    sweep all of them, refitting lam_c to the same 1800 K point each time.
    """
    print("E2. HOW MUCH THE VALIDATION DEPENDS ON THE CHOICE OF SHAPE")
    print("   Sweep over eps_short, eps_long, width; lam_c is refitted each")
    print("   time to the 1800 K reference point.\n")
    rows = []
    for es, el, w in itertools.product((0.03, 0.05, 0.07, 0.10),
                                       (0.85, 0.92, 1.00), (1.0, 2.0, 3.0, 4.0)):
        try:
            lc = fit_film_edge(es, el, w)
        except ValueError:
            continue
        sp = eps_film(LAM, es, lc, el, w)
        rows.append([total_emissivity(LAM, sp, T) for T in (300.,) + T_WORK + (2740.,)])
    r = np.array(rows)
    e300 = r[:, 0]
    ok300 = np.abs(e300 / ALUMINA_REF[300.0] - 1) < 0.10
    print(f"   shape sets: {len(rows)}")
    print(f"   PREDICTED eps(300 K), reference {ALUMINA_REF[300.0]}:")
    print(f"     spread {e300.min():.3f} - {e300.max():.3f}  "
          f"({100*(e300.min()/0.83-1):+.0f}% ... {100*(e300.max()/0.83-1):+.0f}%)")
    print(f"     within ±10% of the reference: {ok300.sum()} of {len(rows)}")
    R["shape.n"] = len(rows)
    R["shape.e300_min"] = e300.min()
    R["shape.e300_max"] = e300.max()
    R["shape.e300_min_pct"] = 100 * (e300.min() / 0.83 - 1)
    R["shape.e300_max_pct"] = 100 * (e300.max() / 0.83 - 1)
    R["shape.n_ok300"] = int(ok300.sum())
    print("\n   -> Wording: within a reasonable family of curves a set EXISTS that")
    print("      reproduces both reference points. This SUPPORTS the hypothesis,")
    print("      it does not confirm it.\n")
    print(f"   {'T, K':>7}{'eps over 48 sets':>22}{'factor':>8}"
          f"{'only those fitting 300 K':>26}")
    for k, T in enumerate(T_WORK + (2740.,), start=1):
        c = r[:, k]
        R[f"shape.T{T:.0f}.min"] = c.min()
        R[f"shape.T{T:.0f}.max"] = c.max()
        R[f"shape.T{T:.0f}.factor"] = c.max() / c.min()
        R[f"shape.T{T:.0f}.ok_min"] = c[ok300].min()
        R[f"shape.T{T:.0f}.ok_max"] = c[ok300].max()
        print(f"   {T:>7.0f}{c.min():>12.3f} - {c.max():.3f}{c.max()/c.min():>8.2f}"
              f"{c[ok300].min():>16.3f} - {c[ok300].max():.3f}")
    print("\n   -> At the working temperature the shape freedom barely reaches the")
    print("      number: the edge is pinned by the 1800 K condition, and 2000 K is")
    print("      close to it. At 2740 K the spread would be 1.5x.\n")
    return r


def _al_index(lam):
    """Complex refractive index of Al from the Drude model (Rakic 1998:
    hbar*wp = 14.98 eV, hbar*gamma = 0.047 eV). Without the interband peak
    near 0.8 um; good enough to estimate interference fringes at 1-5 um."""
    E = 1.23984e-6 / lam
    eps = 1.0 - 14.98 ** 2 / (E ** 2 + 1j * 0.047 * E)
    return np.sqrt(eps)


def film_on_al_emissivity(lam, d, n_film=1.65):
    """eps(lam) = 1 - R of a transparent Al2O3 film of thickness d on Al, at
    normal incidence, with interference. n = 1.65 is amorphous/anodic oxide."""
    r01 = (1.0 - n_film) / (1.0 + n_film)
    N = _al_index(lam)
    r12 = (n_film - N) / (n_film + N)
    ph = np.exp(2j * 2 * np.pi * n_film * d / lam)
    r = (r01 + r12 * ph) / (1.0 + r01 * r12 * ph)
    return 1.0 - np.abs(r) ** 2


def part_f_instruments(lam_c):
    print("F. COST OF THE TRUNCATED BAND ON THE ACTUAL INSTRUMENTS")
    print("   From the NU equipment list:")
    print("     UV-2600i + ISR-2600Plus: 0.22-1.4 um")
    print("     FTIR with ATR only (Bruker Alpha II, Thermo Nicolet iS12)\n")
    print(f"   {'T, K':>7}{'<1.4 um':>11}{'1.4-2.5':>10}{'2.5-5':>9}{'>5 um':>9}")
    for T in (1500., 2000., 2200.):
        B = planck_spectral_radiance(LAM, T)
        tot = np.trapezoid(B, LAM)

        def fr(a, b):
            m = (LAM >= a) & (LAM <= b)
            return np.trapezoid(B[m], LAM[m]) / tot
        R[f"fractions.T{T:.0f}.lt14"] = 100 * fr(0, 1.4e-6)
        R[f"fractions.T{T:.0f}.b14_25"] = 100 * fr(1.4e-6, 2.5e-6)
        R[f"fractions.T{T:.0f}.b25_5"] = 100 * fr(2.5e-6, 5e-6)
        R[f"fractions.T{T:.0f}.gt5"] = 100 * fr(5e-6, 1e-3)
        print(f"   {T:>7.0f}{100*fr(0,1.4e-6):>10.1f}%{100*fr(1.4e-6,2.5e-6):>9.1f}%"
              f"{100*fr(2.5e-6,5e-6):>8.1f}%{100*fr(5e-6,1e-3):>8.1f}%")

    true = eps_film(LAM, 0.05, lam_c)
    gap = (LAM > 1.4e-6) & (LAM < 2.5e-6)
    i0 = int(np.argmax(LAM > 1.4e-6)) - 1
    i1 = int(np.argmax(LAM > 2.5e-6))

    def interp_gap(spec):
        out = spec.copy()
        out[gap] = np.interp(LAM[gap], [LAM[i0], LAM[i1]], [spec[i0], spec[i1]])
        return out

    print("\n   1.4-2.5 um GAP on the SMOOTH model (flat there by construction, so")
    print("   this checks the arithmetic, not the physics):")
    print(f"   {'T, K':>7}{'true':>11}{'interpolated':>15}{'error':>10}")
    for T in (1500., 2000., 2200.):
        a = total_emissivity(LAM, true, T)
        b = total_emissivity(LAM, interp_gap(true), T)
        R[f"gap_smooth.T{T:.0f}.true"] = a
        R[f"gap_smooth.T{T:.0f}.interp"] = b
        R[f"gap_smooth.T{T:.0f}.err_pct"] = 100 * (b / a - 1)
        print(f"   {T:>7.0f}{a:>11.4f}{b:>15.4f}{100*(b/a-1):>+9.2f}%")

    print("\n   GAP WITH INTERFERENCE: a transparent Al2O3 film (n = 1.65) on Al.")
    print("   Fringes with period ~lam^2/(2nd) fall right into 1.4-2.5 um.")
    print("   Absolute error of eps(2000 K) from interpolating the gap, and as a")
    print("   fraction of the working film eps ~0.32:")
    print(f"   {'d, um':>8}{'gap share':>15}{'interpolated':>14}{'error':>9}{'of 0.32':>9}")
    B = planck_spectral_radiance(LAM, 2000.)
    Btot = np.trapezoid(B, LAM)
    worst = 0.0
    for d in (0.1e-6, 0.3e-6, 0.5e-6, 1.0e-6, 2.0e-6, 5.0e-6):
        sp = film_on_al_emissivity(LAM, d)
        a = np.trapezoid((sp * B)[gap], LAM[gap]) / Btot
        b = np.trapezoid((interp_gap(sp) * B)[gap], LAM[gap]) / Btot
        worst = max(worst, abs(b - a))
        R[f"interference.d{d*1e7:.0f}.err_abs"] = b - a
        R[f"interference.d{d*1e7:.0f}.err_pct"] = 100 * (b - a) / 0.32
        print(f"   {d*1e6:>8.1f}{a:>15.4f}{b:>14.4f}{b-a:>+9.4f}{100*(b-a)/0.32:>+8.1f}%")
    R["interference.worst_pct"] = 100 * worst / 0.32
    print(f"   -> worst case {100*worst/0.32:.1f}% of the working eps. The fringe")
    print("      contrast is small: the Al substrate reflects ~95%, and in the")
    print("      transparent region the film adds hundredths to eps. The gap can be")
    print("      interpolated; extending UV-Vis beyond 1.4 um is not needed. Caveat:")
    print("      Drude without the Al interband peak and a smooth surface; roughness")
    print("      must be checked on a sample.\n")

    print("   MID-IR, if it is NOT measured (constant extrapolation from 1.4 um):")
    print(f"   {'T, K':>7}{'true':>11}{'no mid-IR':>14}{'error':>10}")
    for T in (1500., 2000., 2200.):
        a = total_emissivity(LAM, true, T)
        b = total_emissivity(LAM, np.full_like(LAM, 0.05), T)
        R[f"no_midir.T{T:.0f}.true"] = a
        R[f"no_midir.T{T:.0f}.err_pct"] = 100 * (b / a - 1)
        print(f"   {T:>7.0f}{a:>11.3f}{b:>14.3f}{100*(b/a-1):>+9.0f}%")
    print("\n   And if the edge is taken from the literature, not measured (3-6 um):")
    print(f"   {'T, K':>7}{'edge 3 um':>13}{'edge 6 um':>13}{'spread':>10}")
    for T in (1500., 2000., 2200.):
        a = total_emissivity(LAM, eps_film(LAM, 0.05, 3e-6), T)
        b = total_emissivity(LAM, eps_film(LAM, 0.05, 6e-6), T)
        R[f"edge.T{T:.0f}.edge3"] = a
        R[f"edge.T{T:.0f}.edge6"] = b
        R[f"edge.T{T:.0f}.spread_pct"] = 100 * (a / b - 1)
        print(f"   {T:>7.0f}{a:>13.3f}{b:>13.3f}{100*(a/b-1):>+9.0f}%")

    print("\n   CONCLUSIONS FOR THE INSTRUMENTS:")
    print("     1. Mid-IR is REQUIRED: without it the error is ~80-85%.")
    print("     2. The TOTAL directional-hemispherical R is needed: a gold")
    print("        integrating sphere for the mid-IR. DRIFT (Praying Mantis)")
    print("        does not collect the full hemisphere and gives no absolute R;")
    print("        it is only a fallback with calibration against a standard.")
    print("     3. ATR is unsuitable: it measures with an evanescent wave (depth")
    print("        ~0.5-2 um in the mid-IR) through a pressed crystal, gives no")
    print("        reflectance, and optical contact with a rigid coupon cannot be")
    print("        ensured.")
    print("     4. The 1.4-2.5 um gap can be interpolated even with interference.\n")
    return true


def figure(hb, med, fit, lam_c, path="figures/step5b_revision.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    ax = axes[0]
    ax.plot(hb, med, "o-", lw=2.0, ms=5, label="injection median")
    ax.plot(hb, hb, ls=":", color="grey", lw=1.3, label="1:1")
    m = (hb >= 68) & (hb <= 83)
    ax.plot(hb[m], np.polyval(fit, hb[m]), lw=1.4, color="crimson",
            label=f"slope {fit[0]:.2f} in the 68–83 window")
    ax.axvspan(70, 80, color="seagreen", alpha=0.14)
    ax.set_xlabel("prescribed breakup altitude, km")
    ax.set_ylabel("median injection altitude, km")
    ax.set_title("Injection altitude is INHERITED\nfrom the breakup altitude", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    e = eps_film(LAM, 0.05, lam_c)
    ax.semilogx(LAM * 1e6, e, lw=2.0, color="k", label=r"$\varepsilon(\lambda)$, one curve")
    ax.semilogx(LAM * 1e6, film_on_al_emissivity(LAM, 1e-6), lw=0.9,
                color="tab:purple", alpha=0.8, label="1 um film on Al\n(interference)")
    ax.axvspan(1.4, 2.5, color="grey", alpha=0.15)
    for T, c in ((300.0, "tab:blue"), (1800.0, "tab:orange"), (2000.0, "crimson")):
        B = planck_spectral_radiance(LAM, T)
        ax.semilogx(LAM * 1e6, B / B.max() * 0.9, lw=1.2, ls="--", color=c,
                    alpha=0.75, label=f"Planck {T:.0f} K")
    ax.set_xlim(0.3, 40); ax.set_ylim(0, 1.02)
    ax.set_xlabel("wavelength, um"); ax.set_ylabel(r"$\varepsilon$ / Planck, norm.")
    ax.set_title("The drop of ε with temperature is a weighting\neffect; "
                 "grey is the instrument gap", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="center left")

    ax = axes[2]
    Ts = np.linspace(300, 2800, 60)
    ax.plot(Ts, [total_emissivity(LAM, e, T) for T in Ts], lw=2.0, color="k",
            label="model, one ε(λ)")
    ax.plot([300, 1800], [ALUMINA_REF[300.0], ALUMINA_REF[1800.0]], "o", ms=8,
            mfc="none", mew=2, color="crimson", label="α-Al₂O₃ reference")
    ax.axvspan(1750, 2050, color="tab:orange", alpha=0.15)
    ax.text(1900, 0.72, "Al boiling\nat 0.3–1 kPa", fontsize=8, ha="center",
            color="tab:orange")
    e2000 = total_emissivity(LAM, e, 2000.)
    ax.annotate(f"ε(2000 K) = {e2000:.2f}", xy=(2000, e2000),
                xytext=(1300, 0.22), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.set_ylim(0.15, 0.92)
    ax.set_xlabel("temperature, K"); ax.set_ylabel(r"total $\varepsilon(T)$")
    ax.set_title("Validation before the instrument:\nthe second point is predicted",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Step 5 revision: what the model actually predicts "
                 "and what can be measured", fontsize=11.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   saved: {path}")


if __name__ == "__main__":
    print()
    hb, med, fit = part_d_transfer()
    lam_c = part_e_kirchhoff()
    part_e2_shape_sweep()
    part_f_instruments(lam_c)
    figure(hb, med, fit, lam_c)
    print()
    R.save(__file__)
