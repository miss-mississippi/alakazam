"""Проверки шага 4. Запуск: python verify_step4.py  (или pytest)

1. Среднее углового распределения = 0.25 точно (и оно же есть форм-фактор).
2. Замкнутая формула скорости испарения против численного интегрирования.
3. Сходимость по числу поясов.
4. Энергетический баланс поэлементной модели: приход = излучение + запас + испарение.
5. Воспроизведение критерия ORSAT.
6. Вложенность и сохранение массы ПО ПОЯСАМ: испарено <= расплавлено <= масса.
7. Применимость сосредоточенной модели.
8. Клапейрон-Клаузиус против таблицы давления паров Al (CRC).
9. Пластина излучает двумя сторонами: равновесная T против аналитики.
10. Регуляризация вскипания не влияет на результат.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
import reentry.ablation as ablation
import run_step5 as S
from reentry.ablation import (RHO_ALUMINIUM, Aluminium, surface_thermal_model,
                              thermal_diffusion_depth)
from reentry.results import Recorder
from reentry.heating import (SHAPE_FACTOR_TUMBLING, SIGMA_SB, hot_wall_factor,
                             local_flux_fraction, vaporization_rate,
                             vaporizing_area_fraction)

ATM = MSISAtmosphere()
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
INTACT = Vehicle(mass=175.0, area=1.0, Cd=1.5, nose_radius=0.5)
THIN = Vehicle.plate(1.0, 1.0e-3, Cd=1.5)
R = Recorder("verify_step4")


def test_angular_average():
    print("1. СРЕДНЕЕ УГЛОВОГО РАСПРЕДЕЛЕНИЯ")
    val, _ = quad(lambda th: local_flux_fraction(th) * np.sin(th) / 2, 0, np.pi)
    print(f"   <cos> по полной сфере = {val:.10f}   (аналитика 1/4)")
    print(f"   SHAPE_FACTOR_TUMBLING = {SHAPE_FACTOR_TUMBLING}, "
          f"стандартная полоса демиз-кодов 0.25-0.30")
    R["angular_mean"] = val
    ok = abs(val - 0.25) < 1e-10
    print(f"   -> {'OK' if ok else 'ПРОВАЛ'}: форм-фактор есть среднее ЭТОГО")
    print("      распределения, то есть мы не вводим новую физику, а")
    print("      перестаём схлопывать имевшуюся.\n")
    assert ok, "среднее cos не равно 1/4"


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
    assert ok, "замкнутая форма расходится с интегралом"


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
    R["bands.spread_pp"] = spread
    ok = spread < 0.5
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: разброс испарённой доли "
          f"{spread:.2f} п.п. при 24 поясах (рабочее значение).\n")
    assert ok, "нет сходимости по числу поясов"


def test_energy_balance():
    print("4. ЭНЕРГЕТИЧЕСКИЙ БАЛАНС ПОЭЛЕМЕНТНОЙ МОДЕЛИ")
    print("   Приход = переизлучение + запас + испарение + блокировано вдувом.")
    print("   Самый сильный тест: ловит любую ошибку в правой части.")
    print("   Аудит накапливается ВНУТРИ той же системы ОДУ, а не пересчитывается")
    print("   отдельно, иначе тест проверял бы копию кода, а не сам код.\n")
    ok = True
    for label, veh, tw, kw in (("целый объект", INTACT, None, {}),
                               ("пластина 1 мм", THIN, 1e-3, {}),
                               ("пластина 1 мм, вдув 0.6", THIN, 1e-3,
                                dict(blowing_eta=0.6))):
        tr = integrate(veh, ENTRY, ATM)
        for e in (0.05, 0.20):
            r = surface_thermal_model(tr, veh, Aluminium(emissivity=e),
                                      wall_thickness=tw, **kw)
            print(f"   {label}, eps={e:.2f}:")
            print(f"     приход       {r['E_in']:.4e} Дж")
            print(f"     излучено     {r['E_rad']:.4e}")
            print(f"     запасено     {r['E_stored']:.4e}")
            print(f"     на испарение {r['E_vap']:.4e}")
            print(f"     блок. вдувом {r['E_block']:.4e}")
            print(f"     невязка      {r['energy_residual']:+.2e}")
            ok &= abs(r["energy_residual"]) < 1e-6
            worst = max(abs(r["energy_residual"]), R.data.get("energy_residual_max", 0.0))
            R["energy_residual_max"] = worst
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}\n")
    assert ok, "энергобаланс не замыкается"


def test_orsat_criterion():
    print("5. ВОСПРОИЗВЕДЕНИЕ КРИТЕРИЯ ORSAT")
    mat = Aluminium()
    print(f"   ORSAT generic aluminum, heat of ablation   0.9345 МДж/кг")
    print(f"   наш расчёт c_p(тв)*(T_пл-T0) + L_пл        "
          f"{mat.h_melt_complete/1e6:.4f} МДж/кг")
    d = abs(mat.h_melt_complete / 0.9345e6 - 1)
    print(f"   расхождение                                {100*d:.1f}%")
    print(f"   полное испарение при 1 кПа                 "
          f"{mat.h_vapour_complete/1e6:.2f} МДж/кг")
    print(f"   отношение                                  {mat.demise_ratio:.1f}x")
    R["orsat.ours"] = mat.h_melt_complete / 1e6
    R["orsat.diff_pct"] = 100 * d
    R["orsat.vapour"] = mat.h_vapour_complete / 1e6
    R["orsat.ratio"] = mat.demise_ratio
    ok = d < 0.10
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: верхняя граница вилки — это")
    print("      критерий демиза DRAMA/ORSAT с точностью до свойств сплава.\n")
    assert ok, "критерий ORSAT не воспроизводится в пределах 10%"


def test_bracket_nested():
    print("6. ВЛОЖЕННОСТЬ И СОХРАНЕНИЕ МАССЫ ПО ПОЯСАМ")
    print("   В каждом поясе: испарено <= расплавлено <= масса пояса.")
    print("   Проверяется по каждому поясу отдельно, а не по сумме.\n")
    ok = True
    tol = 1e-6
    print(f"   {'объект':<24}{'eps':>6}{'макс исп/масса':>16}"
          f"{'мин (расп-исп)/масса':>22}{'итог':>7}")
    frags = S.build_fragments()
    cases = [("пластина 1 мм (вход)", THIN, integrate(THIN, ENTRY, ATM), 1e-3)]
    cases += [(n, v, tr, tw) for n, v, tr, tw in frags]
    for label, veh, tr, tw in cases:
        for e in (0.05, 0.35):
            r = surface_thermal_model(tr, veh, Aluminium(emissivity=e),
                                      wall_thickness=tw)
            mb = r["m_band"]
            a = float(np.max(r["m_vap_band"] / mb))
            b = float(np.min((r["m_melt_band"] - r["m_vap_band"]) / mb))
            good = (a <= 1 + tol) and (b >= -tol)
            ok &= good
            print(f"   {label:<24}{e:>6.2f}{a:>16.4f}{b:>22.4f}"
                  f"{'OK' if good else 'ПРОВАЛ':>7}")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}\n")
    assert ok, "нарушено: испарено <= расплавлено <= масса пояса"


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
        assert r["lumped_valid"], f"{label}: стенка толще глубины прогрева"
    R["diffusion_depth_cm"] = d * 100
    print("\n   Стенка тоньше 14 см участвует целиком, толще — только")
    print("   приповерхностный слой.\n")


def test_clausius_clapeyron():
    print("8. ТЕМПЕРАТУРА КИПЕНИЯ ПРИ ДАВЛЕНИИ: КЛАПЕЙРОН-КЛАУЗИУС ПРОТИВ CRC")
    print("   Таблица давления паров металлов (CRC Handbook) для Al:\n")
    crc = {1.0: 1482.0, 10.0: 1632.0, 100.0: 1817.0, 1e3: 2054.0,
           1e4: 2364.0, 1e5: 2790.0}
    mat = Aluminium()
    ok = True
    print(f"   {'p, Па':>9}{'CRC, K':>9}{'модель, K':>11}{'разница':>10}")
    for p, T in crc.items():
        Tm = float(mat.T_boil_at(p))
        d = Tm / T - 1
        ok &= abs(d) < 0.02
        R[f"crc.p{p:g}.model"] = Tm
        R["crc.max_diff_pct"] = max(abs(100 * d), R.data.get("crc", {}).get("max_diff_pct", 0.0))
        print(f"   {p:>9.0f}{T:>9.0f}{Tm:>11.0f}{100*d:>+9.1f}%")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: на давлении торможения 0.3-1 кПа Al")
    print("      кипит при ~1900-2050 K, а не при 2792 K.\n")
    assert ok, "расхождение с таблицей CRC больше 2%"


class _ConstantFluxTrajectory:
    """Синтетическая траектория: постоянные q, V, rho. Для аналитики."""

    def __init__(self, q, V=7000.0, rho=1e-6, t_end=600.0):
        self.t = np.linspace(0.0, t_end, 601)
        self.V = np.full_like(self.t, V)
        self.V_rel = self.V
        self.rho = np.full_like(self.t, rho)
        self.h = np.full_like(self.t, 80e3)
        self._q = q

    def heat_flux(self, vehicle, correlation="sutton-graves"):
        return np.full_like(self.t, self._q)


def test_two_sided_plate():
    print("9. ПЛАСТИНА ИЗЛУЧАЕТ ДВУМЯ СТОРОНАМИ")
    print("   Постоянный поток, ниже плавления. Носовой пояс должен выйти на")
    print("   sides*eps*sigma*T^4 = q*mu1*(1 - h_w/h_0), sides = 2 у пластины.\n")
    from reentry.heating import SIGMA_SB, hot_wall_factor
    from scipy.optimize import brentq
    q, eps, n = 7.0e3, 0.30, 24          # обе T_eq ниже плавления (890 K)
    mu1 = 1.0 - 0.5 / n
    tr = _ConstantFluxTrajectory(q)
    mat = Aluminium(emissivity=eps)
    ok = True
    print(f"   {'объект':<16}{'sides':>6}{'T модель, K':>13}{'T аналитика, K':>16}{'разница':>10}")
    Ts = {}
    # Число излучающих сторон задаёт ГЕОМЕТРИЯ, а не флаг модели: иначе
    # тест сверял бы модель с самой собой.
    for label, veh, tw, sides in (("пластина 1 мм", THIN, 1e-3, 2),
                                  ("оболочка", Vehicle.compact(1.0, 5.0, Cd=1.5), None, 1)):
        r = surface_thermal_model(tr, veh, mat, wall_thickness=tw, n_bands=n)
        T_an = brentq(lambda T: sides * eps * SIGMA_SB * T ** 4
                      - q * mu1 * float(hot_wall_factor(7000.0, T)), 100, 3000)
        d = r["T_nose_end"] / T_an - 1
        ok &= abs(d) < 1e-3
        Ts[sides] = r["T_nose_end"]
        R[f"two_sided.sides{sides}.diff_pct"] = 100 * d
        print(f"   {label:<16}{sides:>6}{r['T_nose_end']:>13.1f}{T_an:>16.1f}{100*d:>+9.3f}%")
    print(f"\n   отношение T(пластина)/T(оболочка) = {Ts[2]/Ts[1]:.4f}"
          f"  (2^-1/4 = {2**-0.25:.4f} при h_w << h_0)")
    print(f"   -> {'OK' if ok else 'ПРОВАЛ'}\n")
    assert ok, "равновесная T не совпадает с аналитикой"


def test_flash_regularization():
    print("10. РЕГУЛЯРИЗАЦИЯ ВСКИПАНИЯ НЕ ВЛИЯЕТ НА РЕЗУЛЬТАТ")
    print("   Когда давление падает, полка кипения опускается, и перегретый")
    print("   расплав вскипает с постоянной TAU_FLASH. Меняем её вчетверо.\n")
    frags = S.build_fragments()
    name, veh, tr, tw = frags[2]
    mat = Aluminium(surface="oxide")
    vals = {}
    saved = ablation.TAU_FLASH
    try:
        for tau in (0.2, 0.05):
            ablation.TAU_FLASH = tau
            vals[tau] = surface_thermal_model(tr, veh, mat, wall_thickness=tw)["m_vap"]
            print(f"   TAU_FLASH = {tau:4.2f} с -> испарено {vals[tau]:.4f} кг ({name})")
    finally:
        ablation.TAU_FLASH = saved
    d = abs(vals[0.05] / vals[0.2] - 1)
    R["flash.diff_pct"] = 100 * d
    ok = d < 5e-3
    print(f"\n   разница {100*d:.3f}%  -> {'OK' if ok else 'ПРОВАЛ'}\n")
    assert ok, "результат зависит от TAU_FLASH"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
