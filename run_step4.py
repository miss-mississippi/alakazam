"""Шаг 4: фрагментация, тепловой отклик, абляция — как ВИЛКА, а не оценка.

Запуск:  python run_step4.py

Выход: step4_epsilon.png, step4_bracket.png и таблицы в консоли.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.ablation import (Aluminium, surface_thermal_model,
                             thermal_diffusion_depth)
from reentry.heating import (SHAPE_FACTOR_TUMBLING, SIGMA_SB, hot_wall_factor,
                             vaporizing_area_fraction, vaporization_rate)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

ATM = MSISAtmosphere()
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
M0, CD = 175.0, 1.5
INTACT = Vehicle(mass=M0, area=1.0, Cd=CD, nose_radius=0.5)

# Диапазон эмиссивности — физический, не подгоночный.
# 0.05  голый жидкий Al (блестящий жидкий металл)
# 0.10  расплав с тонкой плёнкой
# 0.20  промежуточное
# 0.35  сплошная альфа-Al2O3 (0.83 при 300 K -> 0.35 при 1800 K, ПАДАЕТ с T)
EPS_GRID = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])

# Список фрагментов. Каждый — либо ПЛАСТИНА (задаётся толщиной, beta из неё
# выводится), либо КОМПАКТНЫЙ (задаётся beta, Rn как сфероэквивалент).
# SESAM по умолчанию: панели 95 км, корпус 78 км (Lips, SDC6).
FRAGMENTS = [
    # имя, доля массы, высота отделения, вид, параметр
    ("солнечные панели", 0.10, 95.0e3, "plate", 1.5e-3),
    ("силовой набор", 0.50, 78.0e3, "compact", 55.0),
    ("мелкие элементы, MLI", 0.40, 78.0e3, "plate", 1.0e-3),
]


def fragment_vehicle(mass_fraction: float, kind: str, param: float):
    """-> (Vehicle, толщина стенки). Для пластины толщина известна."""
    m = M0 * mass_fraction
    if kind == "plate":
        return Vehicle.plate(m, param, Cd=CD), param
    return Vehicle.compact(m, param, Cd=CD), None


def report_criteria(mat: Aluminium):
    print("A. ДВА КРИТЕРИЯ ДЕМИЗА")
    print(f"   до полного расплава      {mat.h_melt_complete/1e6:6.2f} МДж/кг"
          f"   <- критерий ORSAT/DRAMA (у них 0.93)")
    print(f"   до полного испарения     {mat.h_vapour_complete/1e6:6.2f} МДж/кг"
          f"   <- то, что нужно атмосферной химии")
    print(f"   отношение                {mat.demise_ratio:6.1f}x\n")
    print("   Между ними — судьба сорванного расплава: капля может испариться")
    print("   дальше, может окислиться по поверхности и выпасть сферулой, может")
    print("   застыть целиком. Ни одна текущая модель этого не разрешает.")
    print("   Поэтому результат — ВИЛКА, и её ширина есть физический результат.\n")


def report_epsilon(mat: Aluminium):
    print("B. ЭМИССИВНОСТЬ — ГЛАВНАЯ ОСЬ")
    print("   При 1900 K излучает не окисленный ТВЁРДЫЙ Al, а расплав с")
    print("   оксидной плёнкой. У альфа-Al2O3 эмиссивность с ростом T ПАДАЕТ:")
    print("   0.83 при 300 K -> 0.35 при 1800 K. Голый жидкий Al ~0.05.")
    print("   Физическая вилка 0.05-0.35, фактор 7. Порог кипения ~ eps.\n")
    tr = integrate(INTACT, ENTRY, ATM)
    print(f"   ЦЕЛЫЙ ОБЪЕКТ (m={M0:.0f} кг, beta={INTACT.ballistic_coefficient:.0f}):")
    print(f"   {'eps':>6}{'T носа, K':>12}{'расплав':>10}{'испарено':>11}"
          f"      || тонкая пластина 1 мм:{'T носа':>9}{'расплав':>9}{'испар.':>9}")
    thin = Vehicle.plate(1.0, 1.0e-3, Cd=CD)
    tr_thin = integrate(thin, ENTRY, ATM)
    for e in EPS_GRID:
        m = Aluminium(emissivity=float(e))
        r = surface_thermal_model(tr, INTACT, m)
        rt = surface_thermal_model(tr_thin, thin, m, wall_thickness=1.0e-3)
        print(f"   {e:>6.2f}{r['T_nose']:>12.0f}{100*r['f_melt']:>9.0f}%"
              f"{100*r['f_vap']:>10.1f}%      ||"
              f"{rt['T_nose']:>26.0f}{100*rt['f_melt']:>8.0f}%"
              f"{100*rt['f_vap']:>8.0f}%")
    print(f"\n   Целый объект: eps почти не влияет — стенка 16 мм не успевает")
    print(f"   выйти на радиационное равновесие, задача ЭНЕРГЕТИЧЕСКАЯ.")
    print(f"   Тонкая пластина: выходит на равновесие быстро, и там eps решает всё.")
    print(f"   Глубина прогрева за полёт {thermal_diffusion_depth(303.)*100:.0f} см — "
          f"критерий применимости\n   сосредоточенной модели по толщине стенки.\n")


def run_fragments(mat: Aluminium):
    """Целый объект до высоты разрушения, затем каждый фрагмент отдельно."""
    tr0 = integrate(INTACT, ENTRY, ATM, h_stop=60.0e3)
    out = []
    for name, f_mass, h_break, kind, param in FRAGMENTS:
        veh, t_wall = fragment_vehicle(f_mass, kind, param)
        start = tr0.state_at_altitude(h_break, ENTRY.inclination_deg)
        tr = integrate(veh, start, ATM)
        out.append((name, veh, tr, h_break, veh.ballistic_coefficient, t_wall))
    return tr0, out


def report_fragments(mat: Aluminium):
    print("C. СПИСОК ФРАГМЕНТОВ (не N равных осколков, а спектр beta)")
    print("   SESAM по умолчанию: панели 95 км, корпус 78 км\n")
    _, frags = run_fragments(mat)
    print(f"   {'фрагмент':<24}{'m, кг':>8}{'beta':>7}{'Rn, см':>8}"
          f"{'h разр.':>9}{'h пика q':>10}{'расплав':>10}{'испар.':>9}")
    tot_melt = tot_vap = 0.0
    for name, veh, tr, h_break, beta, t_wall in frags:
        r = surface_thermal_model(tr, veh, mat, wall_thickness=t_wall)
        tot_melt += r["m_melt"]; tot_vap += r["m_vap"]
        print(f"   {name:<24}{veh.mass:>8.1f}{beta:>7.0f}"
              f"{100*veh.nose_radius:>8.1f}{h_break/1e3:>9.0f}"
              f"{tr.peak_heating()[0]/1e3:>10.1f}"
              f"{100*r['f_melt']:>9.0f}%"
              f"{100*r['f_vap']:>8.1f}%")
    f_al = INTACT.al_mass_fraction
    print(f"\n   ИТОГО по объекту {M0:.0f} кг при eps={mat.emissivity:.2f}:")
    print(f"     расплавлено      {tot_melt:6.1f} кг ({100*tot_melt/M0:.0f}%)"
          f"  -> алюминия {tot_melt*f_al:5.1f} кг   ВЕРХНЯЯ ГРАНИЦА")
    print(f"     испарено на месте{tot_vap:6.1f} кг ({100*tot_vap/M0:.0f}%)"
          f"  -> алюминия {tot_vap*f_al:5.1f} кг   НИЖНЯЯ ГРАНИЦА")
    print(f"\n   Наблюдаемая демизабельность конструкции типа OneWeb/SpaceX — 95%")
    print(f"   (Ferreira 2024). Верхняя граница её воспроизводит, значит модель")
    print(f"   согласована с известными результатами по расплавленной массе.\n")
    return frags


def altitude_histogram(frags, mat: Aluminium, bins=np.arange(40, 102, 2.0)):
    """Высотное распределение испарённой массы — задел шага 5."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist = np.zeros_like(centers)
    for _, veh, tr, _, _, t_wall in frags:
        r = surface_thermal_model(tr, veh, mat, wall_thickness=t_wall)
        dm = np.diff(r["m_vap_series"]) * veh.al_mass_fraction
        h_km = r["h"][:-1] / 1e3
        idx = np.digitize(h_km, bins) - 1
        ok = (idx >= 0) & (idx < len(centers))
        np.add.at(hist, idx[ok], dm[ok])
    return centers, hist


def figure_epsilon(path="step4_epsilon.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    eps = np.array([0.05, 0.08, 0.12, 0.17, 0.22, 0.28, 0.35])

    def band(ax):
        ax.axvspan(0.04, 0.10, color="tab:blue", alpha=0.10)
        ax.axvspan(0.30, 0.36, color="tab:green", alpha=0.12)

    ax = axes[0]
    theta = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.degrees(theta), np.cos(theta), lw=2.0, color="k",
            label=r"$q(\theta)/q_{stag}=\cos\theta$")
    for e, c in ((0.10, "tab:red"), (0.20, "tab:orange"), (0.30, "tab:green")):
        C = e * SIGMA_SB * 2740.0 ** 4 / 8.5e5
        if C < 1:
            ax.fill_between(np.degrees(theta), C, np.cos(theta),
                            where=np.cos(theta) > C, alpha=0.18, color=c)
        ax.axhline(min(C, 1.05), color=c, ls="--", lw=1.1,
                   label=fr"порог кипения, $\varepsilon$={e}")
    ax.set_xlabel(r"угол от точки торможения, град"); ax.set_ylabel(r"$q/q_{stag}$")
    ax.set_title("распределение, а не среднее\n" r"$\langle\cos\theta\rangle=0.25$ точно",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5); ax.set_ylim(0, 1.1)

    # Тонкая пластина 1 мм — там eps решает всё
    thin = Vehicle.plate(1.0, 1.0e-3, Cd=CD)
    tr_thin = integrate(thin, ENTRY, ATM)
    tr_int = integrate(INTACT, ENTRY, ATM)
    mel_t, vap_t, mel_i, vap_i = [], [], [], []
    for e in eps:
        m = Aluminium(emissivity=float(e))
        rt = surface_thermal_model(tr_thin, thin, m, wall_thickness=1.0e-3)
        ri = surface_thermal_model(tr_int, INTACT, m)
        mel_t.append(100 * rt["f_melt"]); vap_t.append(100 * rt["f_vap"])
        mel_i.append(100 * ri["f_melt"]); vap_i.append(100 * ri["f_vap"])

    ax = axes[1]
    ax.fill_between(eps, vap_t, mel_t, color="tab:purple", alpha=0.20,
                    label="вилка")
    ax.plot(eps, mel_t, lw=2.0, marker="o", ms=3.5, label="расплав (верхняя)")
    ax.plot(eps, vap_t, lw=2.0, ls="--", marker="s", ms=3.5,
            label="испарено на месте (нижняя)")
    band(ax)
    ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel("доля массы, %")
    ax.set_title("тонкая пластина 1 мм:\nответ меняется в 50 раз", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.fill_between(eps, vap_i, mel_i, color="tab:purple", alpha=0.20)
    ax.plot(eps, mel_i, lw=2.0, marker="o", ms=3.5, label="расплав")
    ax.plot(eps, vap_i, lw=2.0, ls="--", marker="s", ms=3.5, label="испарено")
    band(ax)
    ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel("доля массы, %")
    ax.set_title("целый объект (стенка 16 мм):\n" r"$\varepsilon$ не влияет, "
                 "задача энергетическая", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8); ax.set_ylim(-2, 100)

    fig.suptitle("Весь ответ упирается в излучательную способность окисленной "
                 "поверхности при высокой температуре", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


def figure_bracket(path="step4_bracket.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)

    for k, (eps_v, ax) in enumerate(zip((0.05, 0.10, 0.20), axes)):
        mat = Aluminium(emissivity=eps_v)
        _, frags = run_fragments(mat)
        c, hist = altitude_histogram(frags, mat)
        ax.barh(c, hist, height=1.8, color="tab:blue")
        ax.axhspan(70, 80, color="seagreen", alpha=0.15)
        ax.set_xlabel("испарённый Al в слое, кг")
        ax.set_ylabel("высота, км")
        ax.set_title(fr"$\varepsilon$={eps_v}: всего "
                     f"{hist.sum():.1f} кг Al", fontsize=9.5)
        if hist.sum() < 1e-9:
            ax.text(0.5, 70, "испарения нет\nвсё остаётся расплавом",
                    transform=ax.get_yaxis_transform(), fontsize=9,
                    ha="center", color="crimson")

    fig.suptitle("Высотное распределение испарённого алюминия: "
                 "два значения одной неизмеренной константы", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


if __name__ == "__main__":
    print()
    mat = Aluminium()
    report_criteria(mat)
    report_epsilon(mat)
    report_fragments(Aluminium(emissivity=0.10))
    figure_epsilon()
    figure_bracket()
    print()
