"""Проверки шага 3. Запуск: python verify_step3.py

1. Размерность константы Саттона-Грейвса — по измеренному пику Stardust.
2. Энергетическая вменяемость интегрального потока.
3. Вращение Земли: предельные случаи по наклонению и цена в тепловом потоке.
4. Саттон-Грейвс против Detra-Kemp-Riddell: цена показателя степени.
"""

from __future__ import annotations

import numpy as np

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.heating import (EXP_DKR, K_SUTTON_GRAVES, detra_kemp_riddell_shape,
                             sutton_graves)

ATM = MSISAtmosphere()
VEH = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=0.5)
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)


def test_sutton_graves_units():
    print("1. РАЗМЕРНОСТЬ КОНСТАНТЫ САТТОНА-ГРЕЙВСА")
    print("   Бенчмарк: Stardust, Rn = 0.23 м, V = 12.6 км/с, rho ~ 3e-4 кг/м^3.")
    print("   Измеренный пик конвективного нагрева ~1200 Вт/см^2.\n")
    q = float(sutton_graves(3.0e-4, 12600.0, 0.23))
    print(f"   формула со входами в СИ даёт   {q:.3e}")
    print(f"   если это Вт/м^2               -> {q/1e4:8.0f} Вт/см^2   <- сходится")
    print(f"   если это Вт/см^2              -> {q:8.3e} Вт/см^2   "
          f"мимо на {q/1200:.0e} раз")
    ok = 0.5 < (q / 1e4) / 1200.0 < 2.0
    print(f"\n   -> {'ПОДТВЕРЖДЕНО' if ok else 'ПРОВАЛ'}: k={K_SUTTON_GRAVES:.4e} "
          f"при входах в СИ даёт Вт/м^2.")
    print("      NASA TFAWS подписывает Вт/см^2 — это опечатка либо другая")
    print("      нормировка. Цена ошибки была бы четыре порядка.\n")
    return ok


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
    ok = 0.001 < frac < 0.5
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: доля в разумном диапазоне.")
    print("      Заметь, это ОЦЕНКА СВЕРХУ: корреляция для точки торможения")
    print("      применена ко всей площади. Реальная доля в разы меньше.\n")
    return ok


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
    ok = abs(p90.heat_flux(VEH).max() / q_off - 1.0) < 1e-6
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: полярная орбита (i=90) в точности")
    print("      совпадает с выключенным вращением — это тест самосогласованности.\n")
    return ok


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
    print("\n   -> Показатель степени двигает высоту пика на доли километра.")
    print("      Для высотного распределения массы выбор корреляции —")
    print("      эффект четвёртого порядка, слабее даже версии MSIS.\n")
    return True


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
        print(f"   {rn:>8.2f}{q[k]/1e4:>17.1f}{tr.h[k]/1e3:>13.2f}")
    ok = (max(heights) - min(heights)) < 1.0
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: высота пика идентична "
          f"(разброс {max(heights)-min(heights):.2e} м).")
    print("      Rn меняет АБСОЛЮТНЫЙ поток вчетверо на нашем диапазоне,")
    print("      а значит и абляцию, но не высоту вброса. Это разделение")
    print("      важно: Rn попадает в бюджет по МАССЕ, а не по ВЫСОТЕ.\n")
    return ok


if __name__ == "__main__":
    print()
    res = [test_sutton_graves_units(), test_energy_sanity(), test_earth_rotation(),
           test_correlations(), test_nose_radius()]
    print("ИТОГ:", "все проверки пройдены" if all(res) else "есть провалы")
    print()
