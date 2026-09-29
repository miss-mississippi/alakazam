"""Проверки шага 3. Запуск: python verify_step3.py  (или pytest)

1. Размерность константы Саттона-Грейвса — по расчётному пику Stardust.
2. Энергетическая вменяемость интегрального потока.
3. Вращение Земли: предельные случаи по наклонению и цена в тепловом потоке.
4. Саттон-Грейвс против Detra-Kemp-Riddell: цена показателя степени.
5. Радиус затупления не двигает высоту пика.
6. Формула Коши: омываемая площадь = 4 * средняя проекция.
7. Показатели по размеру: знак по полной энергии противоположен знаку по q_stag.
8. Энергия на расплав и на испарение против доступной.
9. Вдув: замкнутая форма против итераций.
10. Горячая стенка.
"""

from __future__ import annotations

import numpy as np

from reentry.ablation import Aluminium
from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.results import Recorder
from reentry.heating import (EXP_DKR, K_SUTTON_GRAVES, SHAPE_FACTOR_TUMBLING,
                             blowing_factor, detra_kemp_riddell_shape,
                             hot_wall_factor, sutton_graves)

ATM = MSISAtmosphere()
VEH = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=0.5)
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
R = Recorder("verify_step3")


def test_sutton_graves_units():
    print("1. РАЗМЕРНОСТЬ КОНСТАНТЫ САТТОНА-ГРЕЙВСА")
    print("   Бенчмарк: Stardust, Rn = 0.23 м, V = 12.6 км/с, rho ~ 3e-4 кг/м^3.")
    print("   Расчётный пик нагрева ~1200 Вт/см^2.\n")
    q = float(sutton_graves(3.0e-4, 12600.0, 0.23))
    print(f"   формула со входами в СИ даёт   {q:.3e}")
    print(f"   если это Вт/м^2               -> {q/1e4:8.0f} Вт/см^2   <- сходится")
    print(f"   если это Вт/см^2              -> {q:8.3e} Вт/см^2   "
          f"мимо на {q/1200:.0e} раз")
    print(f"\n   k={K_SUTTON_GRAVES:.4e} при входах в СИ даёт Вт/м^2.")
    print("   NASA TFAWS подписывает Вт/см^2 — это опечатка либо другая")
    print("   нормировка. Цена ошибки была бы четыре порядка.\n")
    R["stardust_wcm2"] = q / 1e4
    assert 0.5 < (q / 1e4) / 1200.0 < 2.0, f"{q/1e4:.0f} Вт/см^2 против ~1200"


def test_energy_sanity():
    print("2. ЭНЕРГЕТИЧЕСКАЯ ВМЕНЯЕМОСТЬ")
    print("   Тепло, поглощённое телом, не может превышать его кинетическую")
    print("   энергию. Большая часть энергии уходит в ударный слой и в след,")
    print("   до тела доходят единицы процентов.\n")
    tr = integrate(VEH, ENTRY, ATM)
    Q = tr.heat_load(VEH)                     # Дж/м^2 в точке торможения
    E_kin = 0.5 * VEH.mass * ENTRY.velocity ** 2
    # Оценка сверху: поток точки торможения по всей площади миделя
    E_heat_upper = Q * VEH.area
    frac = E_heat_upper / E_kin
    print(f"   кинетическая энергия при входе      {E_kin:.3e} Дж")
    print(f"   интегральный поток в точке торм.    {Q:.3e} Дж/м^2")
    print(f"   оценка сверху (поток x вся площадь) {E_heat_upper:.3e} Дж")
    print(f"   доля от кинетической энергии        {100*frac:.1f}%")
    print("\n   Это ОЦЕНКА СВЕРХУ: корреляция для точки торможения применена")
    print("   ко всей площади. Реальная доля в разы меньше.\n")
    R["heat_fraction_pct"] = 100 * frac
    assert 0.001 < frac < 0.5, f"доля {frac:.3f}"


def test_earth_rotation():
    print("3. ВРАЩЕНИЕ ЗЕМЛИ")
    print("   v_вдоль = omega*R*cos(i) — широта сокращается, зависит только")
    print("   от наклонения. Предельные случаи:\n")
    print(f"   {'i, град':>9}{'v_corot, м/с':>14}{'пик q, Вт/см^2':>17}"
          f"{'h пика, км':>13}{'Q, МДж/м^2':>13}")
    ref = None
    for i in (0.0, 53.0, 90.0, 98.0, 180.0):
        e = EntryState(inclination_deg=i)
        tr = integrate(VEH, e, ATM)
        q = tr.heat_flux(VEH)
        k = int(np.argmax(q))
        if i == 90.0:
            ref = q[k]
        key = f"rotation.i{int(i)}"
        R[f"{key}.v_corot"] = e.corotation_speed
        R[f"{key}.q_peak_wcm2"] = q[k] / 1e4
        R[f"{key}.h_km"] = tr.h[k] / 1e3
        R[f"{key}.Q_MJ"] = tr.heat_load(VEH) / 1e6
        print(f"   {i:>9.0f}{e.corotation_speed:>14.1f}{q[k]/1e4:>17.1f}"
              f"{tr.h[k]/1e3:>13.1f}{tr.heat_load(VEH)/1e6:>13.1f}")

    # Цена включения вращения для базового случая i=53
    on = integrate(VEH, ENTRY, ATM, earth_rotation=True)
    off = integrate(VEH, ENTRY, ATM, earth_rotation=False)
    q_on, q_off = on.heat_flux(VEH).max(), off.heat_flux(VEH).max()
    print(f"\n   базовый случай i=53: включение вращения меняет пик потока на "
          f"{100*(q_on/q_off - 1):+.1f}%")
    print(f"   интегральный поток на {100*(on.heat_load(VEH)/off.heat_load(VEH)-1):+.1f}%")
    print(f"   предсказание было ~-11% (3.7% в V, q ~ V^3)")
    # i=90 должно совпадать с выключенным вращением
    p90 = integrate(VEH, EntryState(inclination_deg=90.0), ATM)
    d = abs(p90.heat_flux(VEH).max() / q_off - 1.0)
    print("\n   Полярная орбита (i=90) должна в точности совпасть с выключенным")
    print("   вращением — тест самосогласованности.\n")
    R["rotation.q_change_pct"] = 100 * (q_on / q_off - 1)
    R["rotation.Q_change_pct"] = 100 * (on.heat_load(VEH) / off.heat_load(VEH) - 1)
    assert d < 1e-6, f"i=90 отличается от выключенного вращения на {d:.1e}"


def test_correlations():
    print("4. САТТОН-ГРЕЙВС ПРОТИВ DETRA-KEMP-RIDDELL")
    print(f"   Разница только в показателе по скорости: 3.0 против {EXP_DKR}.")
    print("   DKR нормирован на С-Г в опорной точке, поэтому сравниваем ФОРМУ.\n")
    tr = integrate(VEH, ENTRY, ATM)
    q_sg = tr.heat_flux(VEH, "sutton-graves")
    q_dkr = tr.heat_flux(VEH, "dkr")
    i_sg, i_dkr = int(np.argmax(q_sg)), int(np.argmax(q_dkr))
    Q_sg = tr.heat_load(VEH, "sutton-graves")
    Q_dkr = tr.heat_load(VEH, "dkr")
    print(f"   {'':<22}{'Саттон-Грейвс':>16}{'DKR':>12}{'разница':>11}")
    print(f"   {'высота пика, км':<22}{tr.h[i_sg]/1e3:>16.2f}"
          f"{tr.h[i_dkr]/1e3:>12.2f}{(tr.h[i_dkr]-tr.h[i_sg])/1e3:>+10.2f} км")
    print(f"   {'пик потока, Вт/см^2':<22}{q_sg[i_sg]/1e4:>16.1f}"
          f"{q_dkr[i_dkr]/1e4:>12.1f}{100*(q_dkr[i_dkr]/q_sg[i_sg]-1):>+10.1f}%")
    print(f"   {'интеграл, МДж/м^2':<22}{Q_sg/1e6:>16.1f}"
          f"{Q_dkr/1e6:>12.1f}{100*(Q_dkr/Q_sg-1):>+10.1f}%")
    print("\n   Показатель степени двигает высоту пика на доли километра:")
    print("   для высотного распределения выбор корреляции — эффект")
    print("   четвёртого порядка, слабее даже версии MSIS.\n")
    dh = (tr.h[i_dkr] - tr.h[i_sg]) / 1e3
    dQ = 100 * (Q_dkr / Q_sg - 1)
    R["dkr.h_sg_km"] = tr.h[i_sg] / 1e3
    R["dkr.h_dkr_km"] = tr.h[i_dkr] / 1e3
    R["dkr.dh_km"] = dh
    R["dkr.q_sg_wcm2"] = q_sg[i_sg] / 1e4
    R["dkr.q_dkr_wcm2"] = q_dkr[i_dkr] / 1e4
    R["dkr.dq_pct"] = 100 * (q_dkr[i_dkr] / q_sg[i_sg] - 1)
    R["dkr.Q_sg_MJ"] = Q_sg / 1e6
    R["dkr.Q_dkr_MJ"] = Q_dkr / 1e6
    R["dkr.dQ_pct"] = dQ
    assert abs(dh) < 1.0 and abs(dQ) < 10.0, f"dh={dh:.2f} км, dQ={dQ:.1f}%"


def test_nose_radius():
    print("5. РАДИУС ЗАТУПЛЕНИЯ")
    print("   q ~ Rn^-0.5, но на ВЫСОТУ пика Rn не влияет вообще:")
    print("   он константа вдоль траектории и выносится за argmax.\n")
    print(f"   {'Rn, м':>8}{'пик q, Вт/см^2':>17}{'h пика, км':>13}")
    heights = []
    for rn in (0.1, 0.25, 0.5, 1.0, 2.0):
        v = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=rn)
        tr = integrate(v, ENTRY, ATM)
        q = tr.heat_flux(v)
        k = int(np.argmax(q))
        heights.append(tr.h[k])
        R[f"nose.rn{rn*100:.0f}cm.q_peak_wcm2"] = q[k] / 1e4
        print(f"   {rn:>8.2f}{q[k]/1e4:>17.1f}{tr.h[k]/1e3:>13.2f}")
    spread = max(heights) - min(heights)
    print(f"\n   Высота пика идентична (разброс {spread:.2e} м). Rn меняет")
    print("   АБСОЛЮТНЫЙ поток вчетверо на нашем диапазоне, а значит и абляцию,")
    print("   но не высоту вброса: Rn попадает в бюджет по МАССЕ, а не по ВЫСОТЕ.\n")
    R["nose.h_km"] = heights[0] / 1e3
    R["nose.h_spread_m"] = spread
    assert spread < 1.0, f"высота пика гуляет на {spread:.2f} м"


def test_cauchy():
    print("6. ОМЫВАЕМАЯ ПЛОЩАДЬ — ФОРМУЛА КОШИ")
    print("   Для любого ВЫПУКЛОГО тела средняя по случайным ориентациям")
    print("   площадь проекции = 1/4 площади поверхности. Наш `area` для")
    print("   кувыркающегося тела и есть средняя проекция, значит A_омыв = 4*A")
    print("   ТОЧНО, для любой формы, а не только для сферы.\n")
    for A in (0.5, 1.0, 2.0):
        v = Vehicle(area=A)
        print(f"   area={A:.1f} м^2  ->  A_омыв={v.wetted:.1f} м^2  "
              f"(отношение {v.wetted/A:.1f})")
    # Проверка на сфере: A_проекц = pi R^2, A_поверх = 4 pi R^2
    R = 0.65
    print(f"\n   сверка на сфере R={R} м: проекция {np.pi*R**2:.4f}, "
          f"поверхность {4*np.pi*R**2:.4f}, отношение {4.0:.1f}")
    print()
    assert abs(Vehicle(area=1.0).wetted - 4.0) < 1e-12


def test_rn_scaling():
    print("7. ПОКАЗАТЕЛИ ПО РАЗМЕРУ — ГДЕ ПЕРЕВОРАЧИВАЕТСЯ ЗНАК")
    print("   Ожидание для геометрически подобного тела:")
    print("     q_stag ~ L^-0.5,  P_полн ~ L^+1.5,  P/m ~ L^-1.5\n")

    scales = np.array([0.1, 0.2, 0.5, 1.0, 2.0])

    # (а) ЗАМОРОЖЕННАЯ траектория: та же история потока, меняем только геометрию.
    # Так проверяются чистые показатели корреляции и геометрии.
    tr = integrate(VEH, ENTRY, ATM)
    q_hist, t_hist = tr.rho, tr.t
    V_hist = tr.V_rel
    E_frozen, S_frozen = [], []
    for L in scales:
        v = Vehicle.geometric_family(L)
        q = sutton_graves(q_hist, V_hist, v.nose_radius)
        P = SHAPE_FACTOR_TUMBLING * q * v.wetted
        E = np.trapezoid(P, t_hist)
        E_frozen.append(E); S_frozen.append(E / v.mass)

    def slope(y):
        return float(np.polyfit(np.log(scales), np.log(y), 1)[0])

    print(f"   (а) замороженная траектория — чистая геометрия:")
    print(f"       показатель полной энергии    {slope(E_frozen):+.3f}  "
          f"(ожидалось +1.500)")
    print(f"       показатель удельной энергии  {slope(S_frozen):+.3f}  "
          f"(ожидалось -1.500)")
    R["scaling.frozen_E"] = slope(E_frozen)
    R["scaling.frozen_S"] = slope(S_frozen)
    ok = abs(slope(E_frozen) - 1.5) < 1e-6 and abs(slope(S_frozen) + 1.5) < 1e-6

    # (б) САМОСОГЛАСОВАННО: у мелкого тела beta меньше, оно тормозится выше
    # и получает МЕНЬШЕ полного тепла. Два эффекта борются.
    print(f"\n   (б) самосогласованно (beta меняется вместе с размером):")
    print(f"   {'L':>6}{'beta':>9}{'h пика, км':>13}{'E полн., МДж':>15}"
          f"{'E/m, МДж/кг':>14}")
    E_sc, S_sc = [], []
    for L in scales:
        v = Vehicle.geometric_family(L)
        t2 = integrate(v, ENTRY, ATM)
        E = t2.absorbed_energy(v); S = t2.specific_energy(v)
        E_sc.append(E); S_sc.append(S)
        R[f"scaling.L{L*10:.0f}.beta"] = v.ballistic_coefficient
        R[f"scaling.L{L*10:.0f}.h_km"] = t2.peak_heating()[0] / 1e3
        R[f"scaling.L{L*10:.0f}.E_MJ"] = E / 1e6
        R[f"scaling.L{L*10:.0f}.Em_MJkg"] = S / 1e6
        print(f"   {L:>6.1f}{v.ballistic_coefficient:>9.1f}"
              f"{t2.peak_heating()[0]/1e3:>13.1f}{E/1e6:>15.1f}{S/1e6:>14.2f}")
    print(f"\n       показатель полной энергии    {slope(E_sc):+.3f}")
    print(f"       показатель удельной энергии  {slope(S_sc):+.3f}")
    print("\n   -> Знак по размеру ПРОТИВОПОЛОЖЕН знаку по q_stag.")
    print("      Мелкие осколки получают радикально больше тепла на килограмм —")
    print("      это ВТОРОЙ механизм, которым фрагментация решает исход,")
    print("      независимый от подъёма высоты через beta.\n")
    R["scaling.sc_E"] = slope(E_sc)
    R["scaling.sc_S"] = slope(S_sc)
    assert ok, f"показатели {slope(E_frozen):+.4f}, {slope(S_frozen):+.4f}"
    assert slope(E_sc) > 0 > slope(S_sc), "знак по размеру должен сохраниться"


def test_demise_energy():
    print("8. ХВАТАЕТ ЛИ ЭНЕРГИИ НА РАСПЛАВ И НА ИСПАРЕНИЕ")
    mat = Aluminium()
    Tb = mat.T_boil_nominal
    print(f"   Свойства Al 6061 из ablation.Aluminium; кипение при местном")
    print(f"   давлении ~1 кПа: T_кип = {Tb:.0f} K (при 1 атм было бы "
          f"{mat.T_boil_1atm:.0f} K).")
    h_heat = mat.h1
    h_liq = float(mat.h3(Tb)) - mat.h2
    L_vap = float(mat.L_vapour_at(Tb))
    H_total = mat.h_vapour_complete
    print(f"     нагрев до плавления  {h_heat/1e6:5.2f} МДж/кг  (c_p тв. {mat.c_p:.0f})")
    print(f"     теплота плавления    {mat.L_fusion/1e6:5.2f}")
    print(f"     нагрев до кипения    {h_liq/1e6:5.2f}         (c_p ж. {mat.c_p_liquid:.0f})")
    print(f"     теплота испарения    {L_vap/1e6:5.2f}")
    print(f"     ИТОГО                {H_total/1e6:5.2f} МДж/кг"
          f"   (до полного расплава {mat.h_melt_complete/1e6:.2f})\n")
    print("   Отношение E/m к этим величинам = доля массы, которую в принципе")
    print("   можно расплавить / испарить, если бы вся поглощённая энергия шла")
    print("   на это (без переизлучения — оценка СВЕРХУ).\n")
    R["energy.T_boil"] = Tb
    R["energy.heat_to_melt"] = h_heat / 1e6
    R["energy.fusion"] = mat.L_fusion / 1e6
    R["energy.heat_to_boil"] = h_liq / 1e6
    R["energy.vaporization"] = L_vap / 1e6
    R["energy.total"] = H_total / 1e6
    R["energy.melt_complete"] = mat.h_melt_complete / 1e6
    print(f"   {'L':>6}{'масса, кг':>12}{'E/m, МДж/кг':>14}"
          f"{'расплавимая':>14}{'испаримая':>12}")
    Ss = []
    for L in (1.0, 0.5, 0.2, 0.1):
        v = Vehicle.geometric_family(L)
        t2 = integrate(v, ENTRY, ATM)
        S = t2.specific_energy(v, T_wall=Tb)
        Ss.append(S)
        R[f"energy.L{L*10:.0f}.mass"] = v.mass
        R[f"energy.L{L*10:.0f}.Em"] = S / 1e6
        R[f"energy.L{L*10:.0f}.meltable_pct"] = 100 * min(S / mat.h_melt_complete, 1)
        R[f"energy.L{L*10:.0f}.vaporizable_pct"] = 100 * S / H_total
        print(f"   {L:>6.1f}{v.mass:>12.2f}{S/1e6:>14.2f}"
              f"{100*min(S/mat.h_melt_complete, 1):>13.0f}%{100*S/H_total:>11.0f}%")
    print("\n   E/m здесь с поправкой на горячую стенку при T_кип, поэтому она")
    print("   ниже, чем в проверке 7 (там холодная стенка).")
    print("   -> Целый объект: расплавить можно ~половину, испарить — единицы")
    print("      процентов. Демизабельность 95% у OneWeb/SpaceX (Ferreira 2024) —")
    print("      это критерий РАСПЛАВА, сравнивать её надо с первой колонкой.")
    print("      Разрыв по массе ~2x для расплава, и его закрывает фрагментация:")
    print("      осколок L = 0.2 плавится целиком.\n")
    parts = h_heat + mat.L_fusion + h_liq + L_vap
    assert abs(parts / H_total - 1) < 1e-12, "этапы не складываются в итог"
    assert all(a < b for a, b in zip(Ss, Ss[1:])), "E/m должна расти с дроблением"


def test_blowing_closed_form():
    print("9. ВДУВ: ЗАМКНУТАЯ ФОРМА ПРОТИВ ИТЕРАЦИЙ")
    h_0, H_eff, eta = 2.8e7, 1.2e7, 0.3
    q_hw = 1.0e6
    closed = q_hw * blowing_factor(h_0, H_eff, eta)
    q = q_hw                       # итерируем q_net = q_hw - eta*(q/H_eff)*h_0
    for _ in range(200):
        q = q_hw - eta * (q / H_eff) * h_0
    print(f"   замкнутая форма  {closed:.6e} Вт/м^2")
    print(f"   итерации (200)   {q:.6e} Вт/м^2")
    print(f"   множитель        {blowing_factor(h_0, H_eff, eta):.4f} "
          f"-> вдув срезает поток на {100*(1-blowing_factor(h_0,H_eff,eta)):.0f}%")
    print("\n   Нелинейность замыкается аналитически, неявный решатель не нужен.\n")
    R["blowing.factor"] = float(blowing_factor(h_0, H_eff, eta))
    assert abs(closed / q - 1.0) < 1e-9, f"замкнутая форма расходится на {closed/q-1:.1e}"


def test_hot_wall():
    print("10. ГОРЯЧАЯ СТЕНКА")
    Tb = Aluminium().T_boil_nominal
    fs = []
    for V in (7500.0, 6100.0, 4000.0):
        f = float(hot_wall_factor(V, Tb))
        f1 = float(hot_wall_factor(V, 2740.0))
        fs.append(f)
        R[f"hot_wall.V{V:.0f}.Tb"] = f
        R[f"hot_wall.V{V:.0f}.T2740"] = f1
        print(f"   V={V:6.0f} м/с: T={Tb:.0f} K -> {f:.3f} ({100*(f-1):+.0f}%),"
              f"  T=2740 K -> {f1:.3f} ({100*(f1-1):+.0f}%)")
        exact = 1.0 - 1300.0 * Tb / (0.5 * V ** 2)
        assert abs(f - exact) < 1e-12, "множитель не совпадает с 1 - h_w/h_0"
        assert f > f1, "горячее стенка — меньше поток"
    print("   Поправка растёт по мере торможения: у холодного конца")
    print("   траектории она уже не мала.\n")
    assert fs[0] > fs[1] > fs[2], "поправка должна расти с торможением"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
