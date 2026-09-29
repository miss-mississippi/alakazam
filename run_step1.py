"""Шаг 1: экспоненциальная атмосфера + интегрирование траектории от 120 км.

Запуск:  python run_step1.py

Выход: step1_trajectory.png, step1_gamma_sweep.png и сверочная таблица в консоли.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import EntryState, ExponentialAtmosphere, Vehicle, allen_eggers, integrate
from reentry.constants import G0
from reentry.results import Recorder

# Шаги 1-2 определены для НЕВРАЩАЮЩЕЙСЯ Земли: вращение атмосферы вводится
# на шаге 3. С тех пор integrate() включает его по умолчанию, поэтому
# здесь оно выключено явно — иначе цифры шагов 1-2 не воспроизводятся.
NO_ROT = dict(earth_rotation=False)
R = Recorder("step1")


plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def print_report(traj, ae, vehicle, entry, atmosphere):
    a_num, h_num, v_num = traj.peak_decel()

    print()
    print("=" * 68)
    print("ШАГ 1 — ТРАЕКТОРИЯ, ЭКСПОНЕНЦИАЛЬНАЯ АТМОСФЕРА (ЗАГЛУШКА)")
    print("=" * 68)
    print(f"  масса              {vehicle.mass:8.1f} кг")
    print(f"  Cd * A             {vehicle.Cd * vehicle.area:8.2f} м^2")
    print(f"  beta = m/(Cd A)    {vehicle.ballistic_coefficient:8.1f} кг/м^2")
    print(f"  вход               h={entry.altitude/1e3:.0f} км, "
          f"V={entry.velocity:.0f} м/с, gamma={entry.gamma_deg:+.2f} град")
    print(f"  атмосфера          {atmosphere.name}, "
          f"rho0={atmosphere.rho0} кг/м^3, H={atmosphere.H/1e3:.1f} км")
    print()
    print(f"  остановка:         {traj.stop_reason}")
    print(f"  длительность:      {traj.t[-1]:8.1f} с")
    print(f"  дальность:         {traj.s[-1]/1e3:8.0f} км")
    i_peak = int(np.argmax(traj.decel))
    print(f"  gamma в пике:      {traj.gamma_deg[i_peak]:+8.2f} град "
          f"(в начале {entry.gamma_deg:+.2f})")
    print(f"  gamma в конце:     {traj.gamma_deg[-1]:+8.2f} град")
    print(f"  V в конце:         {traj.V[-1]:8.0f} м/с "
          f"на {traj.h[-1]/1e3:.1f} км")
    print()
    print("  СВЕРКА С АЛЛЕНОМ-ЭГГЕРСОМ (ожидание: расхождение 20-30%,")
    print("  т.к. у нас gamma не постоянна и первые ~100 с полёт бездрагвый)")
    print(f"    {'величина':<26}{'численно':>12}{'Аллен-Эггерс':>15}{'откл.':>10}")
    rows = [
        ("макс. торможение, g", a_num / G0, ae["a_max"] / G0),
        ("высота пика, км", h_num / 1e3, ae["h_at_peak"] / 1e3),
        ("скорость в пике, м/с", v_num, ae["V_at_peak"]),
    ]
    R["base.t_end"] = traj.t[-1]
    R["base.range_km"] = traj.s[-1] / 1e3
    R["base.amax_g"] = a_num / G0
    R["base.h_peak_km"] = h_num / 1e3
    R["base.V_peak"] = v_num
    R["base.gamma_peak_deg"] = traj.gamma_deg[i_peak]
    R["base.gamma_end_deg"] = traj.gamma_deg[-1]
    for key, (label, num, ana) in zip(("amax_g", "h_km", "V"), rows):
        R[f"ae.{key}"] = ana
        R[f"ae.{key}_dev_pct"] = 100.0 * (num - ana) / ana
    for label, num, ana in rows:
        dev = 100.0 * (num - ana) / ana
        print(f"    {label:<26}{num:>12.2f}{ana:>15.2f}{dev:>9.0f}%")
    print("=" * 68)
    print()


def figure_main(traj, ae, atmosphere, entry, path="figures/step1_trajectory.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    t = traj.t
    h_km = traj.h / 1e3

    ax = axes[0, 0]
    ax.plot(t, h_km, lw=1.6)
    ax.set_xlabel("время, с"); ax.set_ylabel("высота, км")
    ax.set_title("h(t)")

    ax = axes[0, 1]
    ax.plot(t, traj.V / 1e3, lw=1.6)
    ax.axhline(ae["V_at_peak"] / 1e3, color="crimson", ls="--", lw=1,
               label=r"А-Э: $V_0/\sqrt{e}$")
    ax.set_xlabel("время, с"); ax.set_ylabel("скорость, км/с")
    ax.set_title("V(t)"); ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 2]
    ax.plot(t, traj.gamma_deg, lw=1.6)
    ax.axhline(entry.gamma_deg, color="grey", ls=":", lw=1, label="$\\gamma_0$")
    ax.set_xlabel("время, с"); ax.set_ylabel("угол наклона, град")
    ax.set_title(r"$\gamma(t)$ — уходит от $-1.5°$ к $-21°$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(traj.V / 1e3, h_km, lw=1.6)
    ax.set_xlabel("скорость, км/с"); ax.set_ylabel("высота, км")
    ax.set_title("V(h) — фазовый портрет")

    ax = axes[1, 1]
    a_num, h_num, _ = traj.peak_decel()
    ax.plot(traj.decel / G0, h_km, lw=1.6)
    ax.plot(ae["a_max"] / G0, ae["h_at_peak"] / 1e3, "x", color="crimson",
            ms=9, mew=2, label="Аллен-Эггерс")
    ax.plot(a_num / G0, h_num / 1e3, "o", mfc="none", color="tab:blue",
            ms=9, mew=1.6, label="численно")
    ax.set_xlabel("аэродинамическое торможение, g"); ax.set_ylabel("высота, км")
    ax.set_title("перегрузка по высоте"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 2]
    hh = np.linspace(30e3, 120e3, 300)
    ax.semilogx(atmosphere.density(hh), hh / 1e3, lw=1.6, label="заглушка")
    # реперные точки U.S. Standard Atmosphere 1976 — насколько заглушка врёт
    ref_h = np.array([40, 60, 80, 100, 120])
    ref_rho = np.array([4.0e-3, 3.1e-4, 1.85e-5, 5.6e-7, 2.2e-8])
    ax.semilogx(ref_rho, ref_h, "s", color="crimson", ms=5, label="USSA-76")
    ax.set_xlabel(r"плотность, кг/м$^3$"); ax.set_ylabel("высота, км")
    ax.set_title("заглушка точна там, где идёт абляция")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Шаг 1: 3-DOF траектория входа, экспоненциальная атмосфера "
                 "(заглушка), масса постоянна", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  сохранено: {path}")


def figure_gamma_sweep(vehicle, atmosphere, path="figures/step1_gamma_sweep.png"):
    """Чувствительность к углу входа — диапазон -1...-3 град это разные режимы."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    print("  чувствительность к углу входа:")
    print(f"    {'gamma0':>8}{'время, с':>12}{'дальность, км':>16}"
          f"{'макс g':>10}{'h пика, км':>13}")

    for g0 in (-1.0, -1.5, -2.0, -3.0):
        entry = EntryState(gamma_deg=g0)
        traj = integrate(vehicle, entry, atmosphere, **NO_ROT)
        a, hp, _ = traj.peak_decel()
        lbl = f"$\\gamma_0$={g0:+.1f}°"
        axes[0].plot(traj.t, traj.h / 1e3, lw=1.5, label=lbl)
        axes[1].plot(traj.t, traj.V / 1e3, lw=1.5, label=lbl)
        axes[2].plot(traj.decel / G0, traj.h / 1e3, lw=1.5, label=lbl)
        key = f"gamma_sweep.g{abs(g0)*10:.0f}"
        R[f"{key}.t"] = traj.t[-1]
        R[f"{key}.range_km"] = traj.s[-1] / 1e3
        R[f"{key}.amax_g"] = a / G0
        R[f"{key}.h_km"] = hp / 1e3
        print(f"    {g0:>+8.1f}{traj.t[-1]:>12.0f}{traj.s[-1]/1e3:>16.0f}"
              f"{a/G0:>10.1f}{hp/1e3:>13.1f}")

    axes[0].set_xlabel("время, с"); axes[0].set_ylabel("высота, км")
    axes[0].set_title("h(t)")
    axes[1].set_xlabel("время, с"); axes[1].set_ylabel("скорость, км/с")
    axes[1].set_title("V(t)")
    axes[2].set_xlabel("торможение, g"); axes[2].set_ylabel("высота, км")
    axes[2].set_title("перегрузка по высоте")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Чувствительность к углу входа: время полёта различается в 1.6 раза, "
                 "а высота пика почти не сдвигается —\nгравитационный разворот стирает "
                 r"$\gamma_0$ раньше, чем включается торможение", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"\n  сохранено: {path}")


def main():
    vehicle = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)
    atmosphere = ExponentialAtmosphere()

    traj = integrate(vehicle, entry, atmosphere, h_stop=30e3, **NO_ROT)
    ae = allen_eggers(vehicle, entry)

    print_report(traj, ae, vehicle, entry, atmosphere)
    figure_main(traj, ae, atmosphere, entry)
    print()
    figure_gamma_sweep(vehicle, atmosphere)
    print()
    R.save(__file__)


if __name__ == "__main__":
    main()
