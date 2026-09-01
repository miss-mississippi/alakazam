"""Проверки шага 5. Запуск: python verify_step5.py

1. Планк -> Стефан-Больцман (точное тождество).
2. Постоянная eps возвращается в точности.
3. Смещение Вина.
4. Сохранение массы в гистограмме.
5. Независимость медианы от ширины бина.
6. Кирхгоф: отражение -> eps -> обратно.
"""

from __future__ import annotations

import numpy as np

from reentry.ablation import Aluminium, surface_thermal_model
from reentry.emissivity import (SIGMA_SB, WIEN_B, planck_spectral_radiance,
                                required_band, total_emissivity,
                                total_emissivity_from_reflectance)
import run_step5 as S


def test_stefan_boltzmann():
    print("1. ПЛАНК -> СТЕФАН-БОЛЬЦМАН")
    print("   pi * int B(lam,T) dlam == sigma T^4, тождество без свободы\n")
    lam = np.geomspace(1e-8, 1e-2, 200000)
    ok = True
    print(f"   {'T, K':>7}{'pi*int B':>15}{'sigma T^4':>15}{'отн. разница':>15}")
    for T in (1000.0, 1500.0, 2000.0, 2740.0):
        I = float(np.pi * np.trapezoid(planck_spectral_radiance(lam, T), lam))
        ref = SIGMA_SB * T ** 4
        d = I / ref - 1
        ok &= abs(d) < 1e-6
        print(f"   {T:>7.0f}{I:>15.6e}{ref:>15.6e}{d:>+15.2e}")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: интегратор и константы согласованы.\n")
    return ok


def test_flat_emissivity():
    print("2. ПОСТОЯННАЯ eps ВОЗВРАЩАЕТСЯ В ТОЧНОСТИ")
    lam = np.geomspace(1e-7, 1e-3, 50000)
    ok = True
    for T in (1500.0, 2740.0):
        for c in (0.05, 0.35, 0.9):
            got = total_emissivity(lam, np.full_like(lam, c), T)
            ok &= abs(got - c) < 1e-9
            print(f"   T={T:.0f} eps={c:.2f} -> {got:.10f}")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: взвешивание нормировано.\n")
    return ok


def test_wien():
    print("3. СМЕЩЕНИЕ ВИНА")
    lam = np.geomspace(1e-7, 1e-4, 200000)
    ok = True
    print(f"   {'T, K':>7}{'пик численно':>16}{'b/T':>12}{'разница':>12}")
    for T in (1500.0, 2000.0, 2740.0):
        pk = float(lam[int(np.argmax(planck_spectral_radiance(lam, T)))])
        ref = WIEN_B / T
        ok &= abs(pk / ref - 1) < 1e-3
        print(f"   {T:>7.0f}{pk*1e6:>14.4f} мкм{ref*1e6:>10.4f}{pk/ref-1:>+12.1e}")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}\n")
    return ok


def test_kirchhoff_roundtrip():
    print("4. КИРХГОФ: ОТРАЖЕНИЕ -> eps -> ОБРАТНО")
    print("   Ровно тот путь, которым пойдут данные с прибора.\n")
    lam = np.geomspace(0.3e-6, 20e-6, 4000)
    rng = np.random.default_rng(0)
    R = np.clip(0.6 + 0.25 * np.sin(np.log(lam * 1e6)) + 0.02 * rng.normal(size=lam.size),
                0.0, 1.0)
    e1 = total_emissivity_from_reflectance(lam, R, 2000.0)
    e2 = total_emissivity(lam, 1.0 - R, 2000.0)
    print(f"   из R:      {e1:.10f}")
    print(f"   из 1-R:    {e2:.10f}")
    ok = abs(e1 - e2) < 1e-12
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}\n")
    return ok


def test_histogram_conservation():
    print("5. СОХРАНЕНИЕ МАССЫ В ГИСТОГРАММЕ")
    print("   Сумма по слоям обязана равняться полной испарённой массе Al.\n")
    frags = S.build_fragments()
    ok = True
    print(f"   {'eps':>6}{'сумма гист., кг':>18}{'прямой счёт, кг':>18}{'разница':>12}")
    for e in (0.05, 0.15, 0.30):
        mat = Aluminium(emissivity=e)
        hist, _ = S.histogram(frags, mat)
        direct = sum(surface_thermal_model(tr, veh, mat, wall_thickness=tw)["m_vap"]
                     for _, veh, tr, tw in frags) * S.F_AL
        d = abs(hist.sum() / direct - 1) if direct > 0 else 0.0
        ok &= d < 1e-6
        print(f"   {e:>6.2f}{hist.sum():>18.6f}{direct:>18.6f}{d:>12.1e}")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: ничего не теряется на краях бинов.\n")
    return ok


def test_bin_independence():
    print("6. НЕЗАВИСИМОСТЬ ОТ ШИРИНЫ БИНА")
    frags = S.build_fragments()
    mat = Aluminium(emissivity=0.10)
    saved_bins, saved_c = S.BINS, S.CENTERS
    meds, tots = [], []
    print(f"   {'бин, км':>9}{'всего, кг':>13}{'медиана, км':>14}")
    for w in (0.5, 1.0, 2.0, 5.0):
        S.BINS = np.arange(30.0, 102.0, w)
        S.CENTERS = 0.5 * (S.BINS[:-1] + S.BINS[1:])
        h, _ = S.histogram(frags, mat)
        s = S.summarize(h, S.raw_deposition(frags, mat))
        meds.append(s["median"]); tots.append(s["total"])
        print(f"   {w:>9.1f}{s['total']:>13.3f}{s['median']:>14.2f}")
    S.BINS, S.CENTERS = saved_bins, saved_c
    ok = (max(meds) - min(meds) < 1e-9) and (max(tots) / min(tots) - 1 < 1e-6)
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: медиана не зависит от бина "
          f"(разброс {max(meds)-min(meds):.2e} км), потому что считается")
    print("      по сырому ряду, а не по гистограмме. Первая версия считала")
    print("      по бинам и гуляла на 1.5 км — гистограмма теперь только")
    print("      для показа.\n")
    return ok


def test_required_band():
    print("7. ТРЕБУЕМАЯ СПЕКТРАЛЬНАЯ ПОЛОСА")
    lo_hot, _, _ = required_band(2740.0)
    _, hi_cold, _ = required_band(1500.0)
    print(f"   объединение по 1500-2740 K: {lo_hot*1e6:.2f} - {hi_cold*1e6:.2f} мкм")
    print(f"   UV-Vis-NIR 0.25-2.5 мкм + FTIR 2.5-15 мкм покрывают с запасом")
    ok = lo_hot > 0.25e-6 and hi_cold < 15e-6
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: стандартной пары приборов хватает,")
    print("      высокотемпературный эмиссометр НЕ нужен.\n")
    return ok


if __name__ == "__main__":
    print()
    res = [test_stefan_boltzmann(), test_flat_emissivity(), test_wien(),
           test_kirchhoff_roundtrip(), test_histogram_conservation(),
           test_bin_independence(), test_required_band()]
    print("ИТОГ:", "все проверки пройдены" if all(res) else "есть провалы")
    print()
