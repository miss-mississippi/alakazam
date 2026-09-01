"""Шаг 3: аэродинамический нагрев, вращение Земли, бюджет неопределённостей.

Запуск:  python run_step3.py

Выход: step3_heating.png, step3_budget.png и таблицы в консоли.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.constants import G0

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

VEH = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=0.5)
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
ATM = MSISAtmosphere()
OBS_LO, OBS_HI = 70.0, 80.0


def report_baseline():
    print("A. БАЗОВЫЙ СЛУЧАЙ С НАГРЕВОМ")
    print(f"   m={VEH.mass:.0f} кг, Cd*A={VEH.Cd*VEH.area:.1f} м^2, "
          f"Rn={VEH.nose_radius:.2f} м, beta={VEH.ballistic_coefficient:.0f} кг/м^2")
    print(f"   вход 120 км / 7500 м/с / {ENTRY.gamma_deg:+.1f}°, "
          f"наклонение {ENTRY.inclination_deg:.0f}°")
    print(f"   атмосфера: {ATM.name}")
    print(f"   соатмосферный снос omega*R*cos(i) = "
          f"{ENTRY.corotation_speed:.0f} м/с\n")
    tr = integrate(VEH, ENTRY, ATM)
    q = tr.heat_flux(VEH)
    i = int(np.argmax(q))
    a, h_a, _ = tr.peak_decel()
    print(f"   пик нагрева        {q[i]/1e4:8.1f} Вт/см^2  на {tr.h[i]/1e3:.1f} км, "
          f"t = {tr.t[i]:.0f} с, V_отн = {tr.V_rel[i]:.0f} м/с")
    print(f"   пик торможения     {a/G0:8.1f} g          на {h_a/1e3:.1f} км")
    print(f"   интегральный поток {tr.heat_load(VEH)/1e6:8.1f} МДж/м^2")
    print(f"   разнос пиков       {(tr.h[i]-h_a)/1e3:8.1f} км")
    print(f"\n   Наблюдаемое разрушение {OBS_LO:.0f}-{OBS_HI:.0f} км всё ещё на "
          f"{OBS_LO - tr.h[i]/1e3:.0f}-{OBS_HI - tr.h[i]/1e3:.0f} км выше пика.")
    print("   Разрыв никуда не делся — его закрывает фрагментация, шаг 4.\n")
    return tr


def report_budget():
    """Каждая строка — измеренный размах, а не оценка на глаз."""
    print("B. БЮДЖЕТ НЕОПРЕДЕЛЁННОСТЕЙ")
    print("   Разделяем два разных вопроса: ГДЕ происходит вброс и СКОЛЬКО")
    print("   массы вбрасывается. Часть параметров влияет только на одно.\n")

    def run(veh=VEH, entry=ENTRY, atm=ATM, corr="sutton-graves", **kw):
        tr = integrate(veh, entry, atm, **kw)
        q = tr.heat_flux(veh, corr)
        i = int(np.argmax(q))
        # МЕТРИКА МАССЫ — удельная поглощённая энергия, Дж/кг, а не поток
        # в Вт/м^2: см. heating.absorbed_power, там знак по размеру
        # переворачивается.
        return tr.h[i], tr.specific_energy(veh, corr, T_wall=2740.0)

    base_h, base_S = run()
    rows = []

    def add(label, kind, variants):
        hs = [v[0] for v in variants]
        Ss = [v[1] for v in variants]
        rows.append((label, kind, (max(hs) - min(hs)) / 1e3,
                     100 * (max(Ss) / min(Ss) - 1)))

    # Фрагментация теперь ГЕОМЕТРИЧЕСКИ ПОДОБНАЯ: масса, площадь и Rn
    # меняются согласованно (m~L^3, A~L^2, Rn~L). Свипировать один beta
    # при фиксированных массе и Rn физически бессмысленно.
    add("фрагментация (размер L 1.0->0.1)", "оба",
        [run(veh=Vehicle.geometric_family(L)) for L in (1.0, 0.1)])
    add("Cd 1.0-2.2", "оба",
        [run(veh=Vehicle(mass=175., area=1., Cd=c, nose_radius=0.5))
         for c in (1.0, 2.2)])
    # Rn как СВОБОДНЫЙ подгоночный параметр при заданных массе и площади:
    # у нерегулярного кувыркающегося тела эффективный радиус затупления не
    # определяется массой и миделем. Диапазон 0.2-1.0 м для тела с
    # характерным размером ~1 м: от почти острой кромки до полного масштаба.
    add("эффективный Rn 0.2-1.0 м", "масса",
        [run(veh=Vehicle(mass=175., area=1., Cd=1.5, nose_radius=r))
         for r in (0.2, 1.0)])
    add("форм-фактор 0.25-0.30", "масса",
        [(base_h, base_S * f / 0.27) for f in (0.25, 0.30)])
    add("наклонение орбиты 0-180°", "оба",
        [run(entry=EntryState(inclination_deg=i)) for i in (0.0, 180.0)])
    add("широта входа -75...0°", "оба",
        [run(atm=MSISAtmosphere(lat=x)) for x in (-75.0, 0.0)])
    add("сезон март/сентябрь", "оба",
        [run(atm=MSISAtmosphere(date=np.datetime64(d)))
         for d in ("2026-03-01T12:00", "2026-09-01T12:00")])
    add("версия MSIS 00 / 2.1", "оба",
        [run(atm=MSISAtmosphere(version=v)) for v in (0, 2.1)])
    add("корреляция С-Г / DKR", "масса",
        [run(corr=c) for c in ("sutton-graves", "dkr")])
    add("угол входа -1...-3°", "оба",
        [run(entry=EntryState(gamma_deg=g)) for g in (-1.0, -3.0)])
    add("солнечная активность F10.7 70-220", "оба",
        [run(atm=MSISAtmosphere(f107=f, f107a=f)) for f in (70.0, 220.0)])

    print(f"   {'фактор':<36}{'влияет на':>11}{'высота, км':>13}"
          f"{'удельная энергия':>18}")
    for label, kind, dh, dQ in sorted(rows, key=lambda r: -r[2]):
        h_str = f"{dh:.1f}" if kind != "масса" else "—"
        print(f"   {label:<36}{kind:>11}{h_str:>13}{dQ:>17.0f}%")
    print()
    print("   Метрика массы — УДЕЛЬНАЯ ПОГЛОЩЁННАЯ ЭНЕРГИЯ, Дж/кг, а не поток")
    print("   в Вт/м^2. По потоку знак зависимости от размера ОБРАТНЫЙ:")
    print("     q_stag ~ L^-0.5, но A_омыв ~ L^2  ->  P ~ L^+1.5,  P/m ~ L^-1.5")
    print("   Крупное тело по удельному потоку греется слабее, а по полной")
    print("   энергии сильнее. Бюджет по q_stag указывал бы не в ту сторону.")
    print("\n   Наклонение и широта — не неопределённости, а ИЗВЕСТНЫЕ входы:")
    print("   для конкретного аппарата они берутся из его орбиты.\n")
    return rows


def figure_heating(tr, path="step3_heating.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    q = tr.heat_flux(VEH) / 1e4
    q_dkr = tr.heat_flux(VEH, "dkr") / 1e4

    ax = axes[0]
    ax.plot(q, tr.h / 1e3, lw=1.8, label="Саттон-Грейвс")
    ax.plot(q_dkr, tr.h / 1e3, lw=1.5, ls="--", label=f"DKR ($V^{{3.15}}$)")
    ax.axhspan(OBS_LO, OBS_HI, color="seagreen", alpha=0.15)
    ax.text(q.max() * 0.45, 75, "наблюдаемое\nразрушение", fontsize=7.5,
            color="seagreen", ha="center")
    ax.set_xlabel(r"тепловой поток, Вт/см$^2$"); ax.set_ylabel("высота, км")
    ax.set_title("поток в точке торможения"); ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(30, 100)

    ax = axes[1]
    ax.plot(tr.t, q, lw=1.8, color="tab:red", label="поток, Вт/см²")
    ax2 = ax.twinx()
    Q = np.concatenate([[0.0], np.cumsum(np.diff(tr.t) * (
        tr.heat_flux(VEH)[:-1] + tr.heat_flux(VEH)[1:]) / 2)]) / 1e6
    ax2.plot(tr.t, Q, lw=1.8, color="tab:blue", ls="--", label="накоплено, МДж/м²")
    ax2.grid(False)
    ax.set_xlabel("время, с"); ax.set_ylabel(r"поток, Вт/см$^2$", color="tab:red")
    ax2.set_ylabel(r"накопленное тепло, МДж/м$^2$", color="tab:blue")
    ax.set_title("поток и накопленное тепло")

    ax = axes[2]
    for i_orb, ls in ((0.0, "-"), (53.0, "--"), (90.0, ":"), (180.0, "-.")):
        e = EntryState(inclination_deg=i_orb)
        t2 = integrate(VEH, e, ATM)
        ax.plot(t2.heat_flux(VEH) / 1e4, t2.h / 1e3, lw=1.6, ls=ls,
                label=f"i={i_orb:.0f}°, {e.corotation_speed:+.0f} м/с")
    ax.set_xlabel(r"тепловой поток, Вт/см$^2$"); ax.set_ylabel("высота, км")
    ax.set_title("вращение Земли: 36% по пику потока")
    ax.legend(frameon=False, fontsize=7.5); ax.set_ylim(40, 90)

    fig.suptitle("Шаг 3: нагрев по Саттону–Грейвсу со скоростью относительно "
                 "вращающейся атмосферы", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


def figure_budget(rows, path="step3_budget.png"):
    rows_h = [(l, dh) for l, k, dh, _ in rows if k != "масса" and dh > 0.02]
    rows_h.sort(key=lambda r: r[1])
    rows_q = sorted([(l, dq) for l, _, _, dq in rows if dq > 0.5], key=lambda r: r[1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    ax.barh([r[0] for r in rows_h], [r[1] for r in rows_h], color="tab:blue")
    ax.set_xlabel("размах высоты пика нагрева, км")
    ax.set_title("бюджет по ВЫСОТЕ вброса")

    ax = axes[1]
    ax.barh([r[0] for r in rows_q], [r[1] for r in rows_q], color="tab:orange")
    ax.set_xlabel("размах удельной поглощённой энергии, %")
    ax.set_title("бюджет по МАССЕ (удельная энергия, Дж/кг)")

    fig.suptitle("Два разных бюджета: что двигает высоту и что двигает массу",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


if __name__ == "__main__":
    print()
    tr = report_baseline()
    rows = report_budget()
    figure_heating(tr)
    figure_budget(rows)
    print()
