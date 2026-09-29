"""Шаг 4: фрагментация, тепловой отклик, абляция — как ВИЛКА, а не оценка.

Запуск:  python run_step4.py

Выход: step4_epsilon.png, step4_bracket.png и таблицы в консоли.
Сверка с Ferreira — в run_step5.py (раздел D).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_step5 as S
from reentry import Vehicle, integrate
from reentry.ablation import Aluminium, regime_number, surface_thermal_model
from reentry.heating import SIGMA_SB
from reentry.results import Recorder

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

ATM, ENTRY, CD, M0, INTACT = S.ATM, S.ENTRY, S.CD, S.M0, S.INTACT
EPS_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
SCEN_KEY = {"плёнка активна": "film", "голый расплав": "bare"}
R = Recorder("step4")

# Пластина 1 мм массой 1 кг, летящая ОТ ТОЧКИ ВХОДА 120 км — иллюстрация
# радиационного режима. Это не фрагмент "мелкие элементы" (тот стартует с
# высоты разрушения 78 км со скоростью целого объекта), поэтому цифры у них
# разные.
THIN = Vehicle.plate(1.0, 1.0e-3, Cd=CD)


def report_criteria():
    mat = Aluminium()
    print("A. ДВА КРИТЕРИЯ ДЕМИЗА")
    print(f"   до полного расплава      {mat.h_melt_complete/1e6:6.2f} МДж/кг"
          f"   <- критерий ORSAT/DRAMA (у них 0.93)")
    print(f"   до полного испарения     {mat.h_vapour_complete/1e6:6.2f} МДж/кг"
          f"   <- то, что нужно атмосферной химии (при 1 кПа)")
    print(f"   отношение                {mat.demise_ratio:6.1f}x\n")
    R["criteria.melt"] = mat.h_melt_complete / 1e6
    R["criteria.vapour"] = mat.h_vapour_complete / 1e6
    R["criteria.ratio"] = mat.demise_ratio
    R["criteria.h1"] = mat.h1 / 1e6
    R["criteria.h2"] = mat.h2 / 1e6
    print("   Температура кипения Al на поверхности — при давлении торможения:")
    for p in (1e2, 3e2, 1e3, 3e3, 101325.0):
        Tb = float(mat.T_boil_at(p))
        R[f"boil.p{p:.0f}.T"] = Tb
        R[f"boil.p{p:.0f}.L"] = float(mat.L_vapour_at(Tb)) / 1e6
        R[f"boil.p{p:.0f}.h3"] = float(mat.h3(Tb)) / 1e6
        print(f"     p = {p:>8.0f} Па  ->  T_кип = {float(mat.T_boil_at(p)):6.0f} K,"
              f"  L = {float(mat.L_vapour_at(mat.T_boil_at(p)))/1e6:5.2f} МДж/кг")
    from reentry.heating import SHAPE_FACTOR_TUMBLING
    for tag, Tb in (("atm", mat.T_boil_1atm), ("kpa", mat.T_boil_nominal)):
        for e in (0.3, 0.2):
            q = e * SIGMA_SB * Tb ** 4 / SHAPE_FACTOR_TUMBLING / 1e4
            R[f"threshold.{tag}.e{round(e*100):02d}"] = q
    print("   Порог кипения при среднем потоке eps*sigma*T^4/phi, Вт/см^2:")
    print(f"     1 атм ({mat.T_boil_1atm:.0f} K): eps=0.3 -> "
          f"{0.3*SIGMA_SB*mat.T_boil_1atm**4/SHAPE_FACTOR_TUMBLING/1e4:.0f}, "
          f"eps=0.2 -> {0.2*SIGMA_SB*mat.T_boil_1atm**4/SHAPE_FACTOR_TUMBLING/1e4:.0f}")
    print(f"     1 кПа ({mat.T_boil_nominal:.0f} K): eps=0.3 -> "
          f"{0.3*SIGMA_SB*mat.T_boil_nominal**4/SHAPE_FACTOR_TUMBLING/1e4:.0f}, "
          f"eps=0.2 -> {0.2*SIGMA_SB*mat.T_boil_nominal**4/SHAPE_FACTOR_TUMBLING/1e4:.0f}")
    print("\n   Между расплавом и паром — судьба сорванного расплава: капля может")
    print("   испариться дальше, окислиться по поверхности и выпасть сферулой,")
    print("   застыть целиком. Поэтому результат — ВИЛКА.\n")


def report_epsilon():
    print("B. ЭМИССИВНОСТЬ: ГДЕ ОНА РЕШАЕТ, А ГДЕ НЕТ")
    print("   Целый объект и пластина 1 мм (1 кг), обе ОТ ТОЧКИ ВХОДА 120 км.\n")
    tr = integrate(INTACT, ENTRY, ATM)
    tr_thin = integrate(THIN, ENTRY, ATM)
    print(f"   {'eps':<16}{'T макс целого':>14}{'расплав':>9}{'испар.':>8}"
          f"   ||{'T макс пластины':>16}{'расплав':>9}{'испар.':>8}")
    mats = [(f"{e:.2f}", Aluminium(emissivity=float(e))) for e in EPS_GRID]
    mats += [(name, m) for name, m in S.SCENARIOS.items()]
    for label, m in mats:
        r = surface_thermal_model(tr, INTACT, m)
        rt = surface_thermal_model(tr_thin, THIN, m, wall_thickness=1.0e-3)
        key = SCEN_KEY[label] if label in SCEN_KEY else f"e{round(float(label)*100):03d}"
        R[f"eps.{key}.whole_Tmax"] = r["T_max"]
        R[f"eps.{key}.whole_melt_pct"] = 100 * r["f_melt"]
        R[f"eps.{key}.whole_vap_pct"] = 100 * r["f_vap"]
        R[f"eps.{key}.plate_Tmax"] = rt["T_max"]
        R[f"eps.{key}.plate_melt_pct"] = 100 * rt["f_melt"]
        R[f"eps.{key}.plate_vap_pct"] = 100 * rt["f_vap"]
        print(f"   {label:<16}{r['T_max']:>14.0f}{100*r['f_melt']:>8.0f}%"
              f"{100*r['f_vap']:>7.0f}%   ||{rt['T_max']:>16.0f}"
              f"{100*rt['f_melt']:>8.0f}%{100*rt['f_vap']:>7.0f}%")
    print("\n   Целый объект: eps почти не влияет — стенка 16 мм не успевает")
    print("   выйти на радиационное равновесие, задача ЭНЕРГЕТИЧЕСКАЯ. Расплав")
    print("   не больше 50%: подветренная половина оболочки в модели не греется.")
    print("   Пластина: выходит на равновесие быстро, и там eps важна.\n")


def report_fragments(mat: Aluminium, frags):
    print(f"C. СПИСОК ФРАГМЕНТОВ, сценарий: {mat.label()}")
    print(f"   {'фрагмент':<24}{'m, кг':>7}{'beta':>6}{'Rn, см':>8}"
          f"{'h разр.':>9}{'h пика q':>10}{'T_кип, K':>11}{'расплав':>9}{'испар.':>8}")
    run = S.run_model(frags, mat)
    sk = SCEN_KEY[mat.label()]
    fk = {"солнечные панели": "panels", "силовой набор": "structure",
          "мелкие элементы, MLI": "mli"}
    tot_melt = tot_vap = 0.0
    for name, veh, tr, r, f_al in run["per"]:
        tot_melt += r["m_melt"]; tot_vap += r["m_vap"]
        k = f"fragments.{sk}.{fk[name]}"
        R[f"{k}.mass"] = veh.mass
        R[f"{k}.beta"] = veh.ballistic_coefficient
        R[f"{k}.rn_cm"] = 100 * veh.nose_radius
        R[f"{k}.h_break_km"] = tr.h[0] / 1e3
        R[f"{k}.h_peak_q_km"] = tr.peak_heating()[0] / 1e3
        R[f"{k}.q_peak_wcm2"] = float(tr.heat_flux(veh).max()) / 1e4
        R[f"{k}.Tb_min"] = r["T_boil"].min()
        R[f"{k}.Tb_max"] = r["T_boil"].max()
        R[f"{k}.melt_pct"] = 100 * r["f_melt"]
        R[f"{k}.vap_pct"] = 100 * r["f_vap"]
        print(f"   {name:<24}{veh.mass:>7.1f}{veh.ballistic_coefficient:>6.0f}"
              f"{100*veh.nose_radius:>8.1f}{tr.h[0]/1e3:>9.0f}"
              f"{tr.peak_heating()[0]/1e3:>10.1f}"
              f"{r['T_boil'].min():>6.0f}-{r['T_boil'].max():<4.0f}"
              f"{100*r['f_melt']:>8.0f}%{100*r['f_vap']:>7.0f}%")
    R[f"fragments.{sk}.total.melt_kg"] = tot_melt
    R[f"fragments.{sk}.total.melt_pct"] = 100 * tot_melt / M0
    R[f"fragments.{sk}.total.melt_al"] = run["melt"]
    R[f"fragments.{sk}.total.vap_kg"] = tot_vap
    R[f"fragments.{sk}.total.vap_pct"] = 100 * tot_vap / M0
    R[f"fragments.{sk}.total.vap_al"] = run["total"]
    print(f"\n   ИТОГО по объекту {M0:.0f} кг:")
    print(f"     расплавлено        {tot_melt:6.1f} кг ({100*tot_melt/M0:.0f}%)"
          f"  -> Al {run['melt']:5.1f} кг   ВЕРХНЯЯ ГРАНИЦА")
    print(f"     испарено на месте  {tot_vap:6.1f} кг ({100*tot_vap/M0:.0f}%)"
          f"  -> Al {run['total']:5.1f} кг\n")


def report_oxide_growth(frags):
    print("E. eps КАК ТРАЕКТОРИЯ: РОСТ ОКСИДНОЙ ПЛЁНКИ")
    print("   eps(t,T) = eps_голый(T) + (eps_плёнка(T) - eps_голый(T))(1-exp(-sqrt(t/tau)))")
    print("   t — время С МОМЕНТА РАЗРУШЕНИЯ: поверхность фрагмента новая. Нагрев")
    print("   фрагментов идёт в первые ~10-40 с после разрушения.\n")
    print(f"   {'tau, с':>9}{'eps(10 с)':>11}{'eps(30 с)':>11}{'Al исп., кг':>13}{'выход':>8}")
    T = np.array([S.T_OPER])
    for tau in (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0):
        mat = Aluminium(tau_oxide=tau)
        s = S.summarize(S.run_model(frags, mat))
        k = f"growth.tau{tau:.0f}"
        R[f"{k}.eps10"] = float(mat.emissivity_at(T, 10.0)[0])
        R[f"{k}.eps30"] = float(mat.emissivity_at(T, 30.0)[0])
        R[f"{k}.total"] = s["total"]
        R[f"{k}.yield_pct"] = 100 * s["total"] / S.M_AL_TOTAL
        print(f"   {tau:>9.0f}{float(mat.emissivity_at(T, 10.0)[0]):>11.3f}"
              f"{float(mat.emissivity_at(T, 30.0)[0]):>11.3f}{s['total']:>13.1f}"
              f"{100*s['total']/S.M_AL_TOTAL:>7.0f}%")
    print("\n   Гипотеза не сужает неопределённость: неизвестное переименовалось")
    print("   из 'какое eps' в 'как быстро растёт плёнка', но весь размах теперь")
    print("   укладывается между двумя сценариями. Лабораторный запрос: мерить eps")
    print("   КАК ФУНКЦИЮ ТОЛЩИНЫ оксида.\n")


def report_regime():
    print("F. ЧТО РАЗДЕЛЯЕТ РЕЖИМЫ")
    print("   Не глубина прогрева (и 1 мм, и 16 мм много тоньше 14 см), а масса")
    print("   на единицу ОМЫВАЕМОЙ площади. У пластины омываются обе грани,")
    print("   поэтому на единицу омываемой площади приходится rho*t/2.\n")
    print(f"   {'объект':<18}{'эквив. толщина':>16}{'нужно':>12}"
          f"{'доступно':>12}{'Pi':>8}{'режим':>16}")
    for label, veh in (("целый объект", INTACT), ("пластина 1 мм", THIN)):
        tr = integrate(veh, ENTRY, ATM)
        r = regime_number(tr, veh, Aluminium())
        k = "regime.whole" if veh is INTACT else "regime.plate"
        R[f"{k}.equiv_mm"] = r["equiv_thickness"] * 1e3
        R[f"{k}.need_MJ"] = r["need"] / 1e6
        R[f"{k}.avail_MJ"] = r["available"] / 1e6
        R[f"{k}.Pi"] = r["Pi"]
        print(f"   {label:<18}{r['equiv_thickness']*1e3:>13.1f} мм"
              f"{r['need']/1e6:>8.1f} МДж{r['available']/1e6:>8.1f} МДж"
              f"{r['Pi']:>8.2f}{r['regime']:>16}")
    print()


def figure_epsilon(path="figures/step4_epsilon.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    mat = Aluminium()
    Tb = mat.T_boil_nominal
    q_typ = 8.5e5

    ax = axes[0]
    theta = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.degrees(theta), np.cos(theta), lw=2.0, color="k",
            label=r"$q(\theta)/q_{stag}=\cos\theta$")
    for name, c in (("голый расплав", "tab:red"), ("плёнка активна", "tab:blue")):
        e = S.eps_oper(S.SCENARIOS[name], Tb)
        C = 2 * e * SIGMA_SB * Tb ** 4 / q_typ
        if C < 1:
            ax.fill_between(np.degrees(theta), C, np.cos(theta),
                            where=np.cos(theta) > C, alpha=0.18, color=c)
        ax.axhline(min(C, 1.05), color=c, ls="--", lw=1.1,
                   label=f"порог кипения пластины, {name}")
    ax.set_xlabel("угол от точки торможения, град"); ax.set_ylabel(r"$q/q_{stag}$")
    ax.set_title(f"распределение, а не среднее\nq = 85 Вт/см², T_кип = {Tb:.0f} K",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.2); ax.set_ylim(0, 1.1)

    tr_thin = integrate(THIN, ENTRY, ATM)
    tr_int = integrate(INTACT, ENTRY, ATM)
    eps = EPS_GRID
    res_t = [surface_thermal_model(tr_thin, THIN, Aluminium(emissivity=float(e)),
                                   wall_thickness=1.0e-3) for e in eps]
    res_i = [surface_thermal_model(tr_int, INTACT, Aluminium(emissivity=float(e)))
             for e in eps]
    for ax, res, title in ((axes[1], res_t, "пластина 1 мм (от точки входа)"),
                           (axes[2], res_i, "целый объект (стенка 16 мм)")):
        mel = [100 * r["f_melt"] for r in res]
        vap = [100 * r["f_vap"] for r in res]
        ax.fill_between(eps, vap, mel, color="tab:purple", alpha=0.20)
        ax.plot(eps, mel, lw=2.0, marker="o", ms=3.5, label="расплав")
        ax.plot(eps, vap, lw=2.0, ls="--", marker="s", ms=3.5,
                label="испарено на месте")
        for name, c in (("голый расплав", "tab:red"), ("плёнка активна", "tab:blue")):
            ax.axvline(S.eps_oper(S.SCENARIOS[name], Tb), color=c, lw=1.0, ls=":")
        ax.set_xlabel(r"постоянная $\varepsilon$"); ax.set_ylabel("доля массы, %")
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8); ax.set_ylim(-2, 102)

    fig.suptitle("Эмиссивность решает для тонкостенных элементов и не решает "
                 "для массивных (пунктир — сценарии при T_кип)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


def figure_bracket(frags, path="figures/step4_bracket.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    cases = [("голый расплав", S.SCENARIOS["голый расплав"]),
             ("рост плёнки tau = 30 с", Aluminium(tau_oxide=30.0)),
             ("плёнка активна", S.SCENARIOS["плёнка активна"])]
    for ax, (label, mat) in zip(axes, cases):
        run = S.run_model(frags, mat)
        hist = S.histogram(run)
        ax.barh(S.CENTERS, hist, height=1.8, color="tab:blue")
        ax.axhspan(70, 80, color="seagreen", alpha=0.15)
        ax.set_xlabel("испарённый Al в слое, кг")
        ax.set_ylabel("высота, км")
        ax.set_title(f"{label}: всего {run['total']:.1f} кг Al", fontsize=9.5)
        ax.set_ylim(40, 100)
    fig.suptitle("Высотное распределение испарённого алюминия: "
                 "форма от сценария поверхности почти не зависит", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


if __name__ == "__main__":
    print()
    report_criteria()
    report_epsilon()
    frags = S.build_fragments()
    for mat in S.SCENARIOS.values():
        report_fragments(mat, frags)
    report_oxide_growth(frags)
    report_regime()
    figure_epsilon()
    figure_bracket(frags)
    print()
    R.save(__file__)
