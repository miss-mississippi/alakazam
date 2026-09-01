"""Проверки шага 4. Запуск: python verify_step4.py

1. Среднее углового распределения = 0.25 точно (и оно же есть форм-фактор).
2. Замкнутая формула скорости испарения против численного интегрирования.
3. Сходимость по числу поясов.
4. Энергетический баланс поэлементной модели: приход = излучение + запас + испарение.
5. Воспроизведение критерия ORSAT.
6. Вложенность вилки: расплав >= испарение всегда.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.ablation import (RHO_ALUMINIUM, Aluminium, surface_thermal_model,
                              temperature_from_enthalpy, thermal_diffusion_depth)
from reentry.heating import (SHAPE_FACTOR_TUMBLING, SIGMA_SB, hot_wall_factor,
                             local_flux_fraction, vaporization_rate,
                             vaporizing_area_fraction)

ATM = MSISAtmosphere()
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
INTACT = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=0.5)
THIN = Vehicle.plate(1.0, 1.0e-3, Cd=1.5)


def test_angular_average():
    print("1. СРЕДНЕЕ УГЛОВОГО РАСПРЕДЕЛЕНИЯ")
    val, _ = quad(lambda th: local_flux_fraction(th) * np.sin(th) / 2, 0, np.pi)
    print(f"   <cos> по полной сфере = {val:.10f}   (аналитика 1/4)")
    print(f"   SHAPE_FACTOR_TUMBLING = {SHAPE_FACTOR_TUMBLING}, "
          f"стандартная полоса демиз-кодов 0.25-0.30")
    ok = abs(val - 0.25) < 1e-10
    print(f"   -> {'OK' if ok else 'ПРОВАЛ'}: форм-фактор есть среднее ЭТОГО")
    print("      распределения, то есть мы не вводим новую физику, а")
    print("      перестаём схлопывать имевшуюся.\n")
    return ok


def test_vaporization_closed_form():
    print("2. ЗАМКНУТАЯ ФОРМУЛА СКОРОСТИ ИСПАРЕНИЯ")
    print("   mdot = A*q*(1-C)^2/(4*L),  C = eps*sigma*T_кип^4/q\n")
    A, Tb, Lv = 4.0, 2740.0, 10.5e6
    ok = True
    print(f"   {'q, Вт/см²':>11}{'eps':>7}{'C':>8}{'численно':>14}{'замкнуто':>14}"
          f"{'разница':>11}")
    for q, eps in ((8.5e5, 0.20), (1.5e6, 0.30), (3.0e6, 0.10)):
        C = eps * SIGMA_SB * Tb ** 4 / q
        if C >= 1:
            continue
        th_c = np.arccos(C)
        num, _ = quad(lambda th: (q * np.cos(th) - eps * SIGMA_SB * Tb ** 4)
                      * np.sin(th) / 2, 0, th_c)
        num = A * num / Lv
        clo = float(vaporization_rate(q, A, eps, Tb, Lv))
        d = abs(num / clo - 1)
        ok &= d < 1e-9
        print(f"   {q/1e4:>11.0f}{eps:>7.2f}{C:>8.3f}{num:>14.6e}"
              f"{clo:>14.6e}{d:>11.1e}")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: квадрат по (1-C) не опечатка —")
    print("      у порога одновременно стремятся к нулю и площадь, и избыток.\n")
    return ok


def test_band_convergence():
    print("3. СХОДИМОСТЬ ПО ЧИСЛУ ПОЯСОВ")
    tr = integrate(THIN, ENTRY, ATM)
    mat = Aluminium(emissivity=0.10)
    print(f"   {'поясов':>8}{'расплав, %':>13}{'испарено, %':>14}")
    vals = []
    for n in (12, 24, 48, 96):
        r = surface_thermal_model(tr, THIN, mat, wall_thickness=1e-3, n_bands=n)
        vals.append(r["f_vap"])
        print(f"   {n:>8}{100*r['f_melt']:>12.2f}%{100*r['f_vap']:>13.2f}%")
    spread = 100 * (max(vals) - min(vals))
    ok = spread < 0.5
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: разброс испарённой доли "
          f"{spread:.2f} п.п. при 24 поясах (рабочее значение).\n")
    return ok


def test_energy_balance():
    print("4. ЭНЕРГЕТИЧЕСКИЙ БАЛАНС ПОЭЛЕМЕНТНОЙ МОДЕЛИ")
    print("   Приход = переизлучение + запас + испарение.")
    print("   Самый сильный тест: ловит любую ошибку в правой части.")
    print("   Аудит накапливается ВНУТРИ той же системы ОДУ, а не пересчитывается")
    print("   отдельно, иначе тест проверял бы копию кода, а не сам код.\n")
    ok = True
    for label, veh, tw in (("целый объект", INTACT, None),
                           ("пластина 1 мм", THIN, 1e-3)):
        tr = integrate(veh, ENTRY, ATM)
        for e in (0.05, 0.20):
            r = surface_thermal_model(tr, veh, Aluminium(emissivity=e),
                                      wall_thickness=tw)
            print(f"   {label}, eps={e:.2f}:")
            print(f"     приход       {r['E_in']:.4e} Дж")
            print(f"     излучено     {r['E_rad']:.4e}")
            print(f"     запасено     {r['E_stored']:.4e}")
            print(f"     на испарение {r['E_vap']:.4e}")
            print(f"     невязка      {r['energy_residual']:+.2e}")
            ok &= abs(r["energy_residual"]) < 1e-3
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}\n")
    return ok


def test_orsat_criterion():
    print("5. ВОСПРОИЗВЕДЕНИЕ КРИТЕРИЯ ORSAT")
    mat = Aluminium()
    print(f"   ORSAT generic aluminum, heat of ablation   0.9345 МДж/кг")
    print(f"   наш расчёт c_p*(T_пл-T0) + L_пл            "
          f"{mat.h_melt_complete/1e6:.4f} МДж/кг")
    d = abs(mat.h_melt_complete / 0.9345e6 - 1)
    print(f"   расхождение                                {100*d:.1f}%")
    print(f"   полное испарение                           "
          f"{mat.h_vapour_complete/1e6:.2f} МДж/кг")
    print(f"   отношение                                  {mat.demise_ratio:.1f}x")
    ok = d < 0.10
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: верхняя граница вилки — это ровно")
    print("      критерий демиза DRAMA/ORSAT, а не наша выдумка.\n")
    return ok


def test_bracket_nested():
    print("6. ВЛОЖЕННОСТЬ ВИЛКИ")
    print("   Испарение — стадия после плавления, значит расплав >= испарение")
    print("   ВСЕГДА. Раньше два независимых учёта это нарушали.\n")
    ok = True
    print(f"   {'объект':<16}{'eps':>7}{'расплав':>10}{'испарено':>11}{'порядок':>10}")
    for label, veh, tw in (("целый", INTACT, None), ("пластина 1мм", THIN, 1e-3)):
        tr = integrate(veh, ENTRY, ATM)
        for e in (0.05, 0.15, 0.35):
            r = surface_thermal_model(tr, veh, Aluminium(emissivity=e),
                                      wall_thickness=tw)
            good = r["m_melt"] >= r["m_vap"] - 1e-12
            ok &= good
            print(f"   {label:<16}{e:>7.2f}{100*r['f_melt']:>9.1f}%"
                  f"{100*r['f_vap']:>10.1f}%{'OK' if good else 'ПРОВАЛ':>10}")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}\n")
    return ok


def test_lumped_validity():
    print("7. ПРИМЕНИМОСТЬ СОСРЕДОТОЧЕННОЙ МОДЕЛИ")
    d = thermal_diffusion_depth(303.0)
    print(f"   глубина прогрева за полёт sqrt(alpha*t) = {d*100:.1f} см")
    print("   (alpha = k/(rho*c_p) = 167/(2700*900) = 6.9e-5 м²/с)\n")
    print(f"   {'объект':<20}{'стенка':>10}{'участвует':>12}{'вывод':>22}")
    for label, veh, tw in (("целый объект", INTACT, None),
                           ("пластина 1 мм", THIN, 1e-3)):
        tr = integrate(veh, ENTRY, ATM)
        r = surface_thermal_model(tr, veh, Aluminium(), wall_thickness=tw)
        note = "прогревается насквозь" if r["lumped_valid"] else "только поверхн. слой"
        print(f"   {label:<20}{r['wall_thickness']*1e3:>8.1f} мм"
              f"{r['participating']*1e3:>10.1f} мм{note:>22}")
    print("\n   -> Критерий количественный, а не на глаз: стенка тоньше 14 см")
    print("      участвует целиком, толще — только приповерхностный слой.\n")
    return True


if __name__ == "__main__":
    print()
    res = [test_angular_average(), test_vaporization_closed_form(),
           test_band_convergence(), test_energy_balance(),
           test_orsat_criterion(), test_bracket_nested(), test_lumped_validity()]
    print("ИТОГ:", "все проверки пройдены" if all(res) else "есть провалы")
    print()
