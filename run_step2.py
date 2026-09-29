"""Шаг 2: NRLMSISE-00 вместо экспоненциальной заглушки.

Запуск:  python run_step2.py

Выход: step2_atmosphere.png, step2_trajectory.png и таблицы в консоли.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import (EntryState, ExponentialAtmosphere, MSISAtmosphere,
                     Vehicle, integrate)
from reentry.constants import G0
from reentry.results import Recorder

# Шаги 1-2 определены для НЕВРАЩАЮЩЕЙСЯ Земли: вращение атмосферы вводится
# на шаге 3. С тех пор integrate() включает его по умолчанию, поэтому
# здесь оно выключено явно — иначе цифры шагов 1-2 не воспроизводятся.
NO_ROT = dict(earth_rotation=False)
R = Recorder("step2")


plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

VEH = Vehicle(mass=175.0, area=1.0, Cd=1.5)     # Cd=1.5 — кувыркающееся тело
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)


def table_density():
    print("A. ПЛОТНОСТЬ: ЗАГЛУШКА ПРОТИВ MSIS")
    exp = ExponentialAtmosphere()
    m00 = MSISAtmosphere(version=0)
    m21 = MSISAtmosphere(version=2.1)
    print(f"   {'h, км':>7}{'exp':>12}{'MSISE-00':>12}{'MSIS 2.1':>12}"
          f"{'exp/00':>9}{'21/00':>9}{'H лок., км':>12}")
    for h in (40, 50, 60, 70, 80, 90, 100, 110, 120):
        e = float(exp.density(h * 1e3))
        a = float(m00.density(h * 1e3))
        b = float(m21.density(h * 1e3))
        H = float(m00.scale_height(h * 1e3)) / 1e3
        R[f"density.h{h}.exp"] = e
        R[f"density.h{h}.m00"] = a
        R[f"density.h{h}.m21"] = b
        R[f"density.h{h}.exp_over_00"] = e / a
        R[f"density.h{h}.m21_over_00"] = b / a
        R[f"density.h{h}.H_km"] = H
        print(f"   {h:>7}{e:>12.3e}{a:>12.3e}{b:>12.3e}"
              f"{e/a:>9.2f}{b/a:>9.2f}{H:>12.2f}")
    Hs = m00.scale_height(np.arange(40e3, 121e3, 1e3)) / 1e3
    R["density.H_min_km"] = Hs.min()
    R["density.H_max_km"] = Hs.max()
    R["density.H_spread_pct"] = 100 * (Hs.max() - Hs.min()) / 7.2
    print(f"\n   локальная шкала высот на 40-120 км: {Hs.min():.1f}-{Hs.max():.1f} км")
    print(f"   заглушка использовала одну константу 7.2 км -> "
          f"разброс {100*(Hs.max()-Hs.min())/7.2:.0f}%")
    print("   MSIS 2.1 против MSISE-00: расхождение до "
          f"{max(abs(1-float(m21.density(h*1e3))/float(m00.density(h*1e3))) for h in range(40,121,10))*100:.0f}%\n")
    return exp, m00, m21


def table_trajectory(exp, m00, m21):
    print("B. ЧТО ИЗМЕНИЛОСЬ В ТРАЕКТОРИИ")
    print(f"   {'атмосфера':<14}{'t, с':>9}{'дальн., км':>12}"
          f"{'h торм., км':>13}{'макс g':>9}{'h нагрева, км':>15}")
    out = {}
    for label, atm in (("экспонента", exp), ("MSISE-00", m00), ("MSIS 2.1", m21)):
        tr = integrate(VEH, ENTRY, atm, **NO_ROT)
        a, h_a, _ = tr.peak_decel()
        h_q, _, _ = tr.peak_heating()
        out[label] = tr
        key = {"экспонента": "exp", "MSISE-00": "m00", "MSIS 2.1": "m21"}[label]
        R[f"traj.{key}.t"] = tr.t[-1]
        R[f"traj.{key}.range_km"] = tr.s[-1] / 1e3
        R[f"traj.{key}.h_decel_km"] = h_a / 1e3
        R[f"traj.{key}.amax_g"] = a / G0
        R[f"traj.{key}.h_heat_km"] = h_q / 1e3
        print(f"   {label:<14}{tr.t[-1]:>9.0f}{tr.s[-1]/1e3:>12.0f}"
              f"{h_a/1e3:>13.1f}{a/G0:>9.1f}{h_q/1e3:>15.1f}")
    da = (out["MSISE-00"].peak_decel()[1] - out["экспонента"].peak_decel()[1]) / 1e3
    dq = (out["MSISE-00"].peak_heating()[0] - out["экспонента"].peak_heating()[0]) / 1e3
    dt = out["MSISE-00"].t[-1] - out["экспонента"].t[-1]
    print(f"\n   сдвиг MSIS относительно заглушки: пик торможения {da:+.1f} км, "
          f"пик нагрева {dq:+.1f} км, время {dt:+.0f} с\n")
    return out


def table_solar():
    print("C. СОЛНЕЧНАЯ АКТИВНОСТЬ И ГЕОМАГНИТНАЯ БУРЯ")
    print("   F10.7: ~70 минимум цикла, ~140 умеренно, ~220 максимум")
    print("   Ap: 4 спокойно, 80 сильная буря\n")
    print(f"   {'сценарий':<22}{'rho(120км)':>12}{'rho(70км)':>12}"
          f"{'t, с':>8}{'h торм.':>10}{'h нагр.':>10}")
    cases = [
        ("минимум F10.7=70", dict(f107=70., f107a=70., ap=4.)),
        ("умеренно F10.7=140", dict(f107=140., f107a=140., ap=4.)),
        ("максимум F10.7=220", dict(f107=220., f107a=220., ap=4.)),
        ("буря Ap=80", dict(f107=140., f107a=140., ap=80.)),
    ]
    res = {}
    for label, kw in cases:
        atm = MSISAtmosphere(version=0, **kw)
        tr = integrate(VEH, ENTRY, atm, **NO_ROT)
        a, h_a, _ = tr.peak_decel()
        h_q, _, _ = tr.peak_heating()
        res[label] = (float(atm.density(120e3)), float(atm.density(70e3)),
                      tr.t[-1], h_a, h_q)
        print(f"   {label:<22}{res[label][0]:>12.2e}{res[label][1]:>12.2e}"
              f"{tr.t[-1]:>8.0f}{h_a/1e3:>10.1f}{h_q/1e3:>10.1f}")
    lo, hi = res["минимум F10.7=70"], res["максимум F10.7=220"]
    R["solar.rho120_ratio"] = hi[0] / lo[0]
    R["solar.rho70_ratio"] = hi[1] / lo[1]
    R["solar.h_heat_shift_km"] = (hi[4] - lo[4]) / 1e3
    print(f"\n   min->max по F10.7: плотность на 120 км x{hi[0]/lo[0]:.1f}, "
          f"на 70 км x{hi[1]/lo[1]:.2f}")
    print(f"   а высота пика нагрева сдвигается всего на "
          f"{(hi[4]-lo[4])/1e3:+.1f} км\n")
    return res


def table_geography():
    print("D. ШИРОТА И СЕЗОН")
    print("   SPOUA (район затопления в южной части Тихого океана) ~ -40 град\n")
    print(f"   {'сценарий':<28}{'rho(70км)':>12}{'h нагрева, км':>15}")
    for label, kw in [
        ("SPOUA, сентябрь (базовый)", dict(lat=-40., date=np.datetime64("2026-09-01T12:00"))),
        ("SPOUA, март", dict(lat=-40., date=np.datetime64("2026-03-01T12:00"))),
        ("экватор, сентябрь", dict(lat=0., date=np.datetime64("2026-09-01T12:00"))),
        ("полярная, сентябрь", dict(lat=-75., date=np.datetime64("2026-09-01T12:00"))),
    ]:
        atm = MSISAtmosphere(version=0, **kw)
        tr = integrate(VEH, ENTRY, atm, **NO_ROT)
        h_q, _, _ = tr.peak_heating()
        key = {"SPOUA, сентябрь (базовый)": "base", "SPOUA, март": "march",
               "экватор, сентябрь": "equator", "полярная, сентябрь": "polar"}[label]
        R[f"geo.{key}.h_heat_km"] = h_q / 1e3
        print(f"   {label:<28}{float(atm.density(70e3)):>12.2e}{h_q/1e3:>15.1f}")
    print()


def figure_atmosphere(exp, m00, m21, path="step2_atmosphere.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    hh = np.linspace(20e3, 140e3, 600)

    ax = axes[0]
    ax.semilogx(exp.density(hh), hh / 1e3, lw=1.6, ls="--", label="экспонента (шаг 1)")
    ax.semilogx(m00.density(hh), hh / 1e3, lw=1.8, label="NRLMSISE-00")
    ax.set_xlabel(r"плотность, кг/м$^3$"); ax.set_ylabel("высота, км")
    ax.set_title("профиль плотности"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(exp.density(hh) / m00.density(hh), hh / 1e3, lw=1.8, ls="--",
            label="экспонента / MSISE-00")
    ax.plot(m21.density(hh) / m00.density(hh), hh / 1e3, lw=1.8,
            label="MSIS 2.1 / MSISE-00")
    ax.axvline(1.0, color="k", lw=0.8)
    ax.axhspan(40, 90, color="seagreen", alpha=0.12)
    ax.text(2.2, 65, "зона абляции", fontsize=8, color="seagreen")
    ax.set_xlabel("отношение к MSISE-00"); ax.set_ylabel("высота, км")
    ax.set_xlim(0.5, 4.0)
    ax.set_title("где заглушка врала"); ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(m00.scale_height(hh) / 1e3, hh / 1e3, lw=1.8, label="MSISE-00, локальная")
    ax.axvline(7.2, color="crimson", ls="--", lw=1.3, label="заглушка, 7.2 км")
    ax.set_xlabel("шкала высот, км"); ax.set_ylabel("высота, км")
    ax.set_title(r"$H = -1/(d\ln\rho/dh)$ — не константа")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Шаг 2: реальная атмосфера вместо однопараметрической экспоненты",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


def figure_trajectory(trs, path="step2_trajectory.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    styles = {"экспонента": dict(ls="--", color="tab:red"),
              "MSISE-00": dict(ls="-", color="tab:blue"),
              "MSIS 2.1": dict(ls=":", color="tab:green")}
    for label, tr in trs.items():
        st = styles[label]
        axes[0].plot(tr.t, tr.h / 1e3, lw=1.7, label=label, **st)
        axes[1].plot(tr.decel / G0, tr.h / 1e3, lw=1.7, label=label, **st)
        q = tr.heat_flux_shape / tr.heat_flux_shape.max()
        axes[2].plot(q, tr.h / 1e3, lw=1.7, label=label, **st)

    axes[0].set_xlabel("время, с"); axes[0].set_ylabel("высота, км")
    axes[0].set_title("h(t)")
    axes[1].set_xlabel("торможение, g"); axes[1].set_ylabel("высота, км")
    axes[1].set_title("перегрузка по высоте")
    axes[2].set_xlabel(r"нагрев $\sqrt{\rho}V^3$, нормирован")
    axes[2].set_ylabel("высота, км")
    axes[2].set_title("форма теплового потока")
    axes[2].axhspan(70, 80, color="seagreen", alpha=0.15)
    axes[2].text(0.35, 75, "наблюдаемое разрушение", fontsize=7.5, color="seagreen")
    axes[2].set_ylim(30, 100)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Траектория на реальной атмосфере: сдвиг есть, но маленький "
                 "— заглушка была точна там, где важно", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


if __name__ == "__main__":
    print()
    exp, m00, m21 = table_density()
    trs = table_trajectory(exp, m00, m21)
    table_solar()
    table_geography()
    figure_atmosphere(exp, m00, m21)
    figure_trajectory(trs)
    print()
    R.save(__file__)
