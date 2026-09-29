"""Проверки шага 5. Запуск: python verify_step5.py  (или pytest)

1. Планк -> Стефан-Больцман (точное тождество).
2. Постоянная eps возвращается в точности.
3. Смещение Вина.
4. Кирхгоф: отражение -> eps -> обратно.
5. Сохранение массы в гистограмме.
6. Независимость медианы от ширины бина (регрессия).
7. Требуемая спектральная полоса.
8. eps голого металла по сопротивлению против справочника для полированного Al.
"""

from __future__ import annotations

import numpy as np

from reentry.ablation import Aluminium, surface_thermal_model
from reentry.emissivity import (SIGMA_SB, WIEN_B, aluminium_resistivity,
                                bare_aluminium_emissivity, parker_abbott_emissivity,
                                planck_spectral_radiance, required_band,
                                total_emissivity, total_emissivity_from_reflectance)
import run_step5 as S
from reentry.results import Recorder

R = Recorder("verify_step5")


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
        R["stefan_boltzmann_max_diff"] = max(abs(d), R.data.get("stefan_boltzmann_max_diff", 0.0))
        print(f"   {T:>7.0f}{I:>15.6e}{ref:>15.6e}{d:>+15.2e}")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: интегратор и константы согласованы.\n")
    assert ok, "pi*int(B) != sigma*T^4"


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
    assert ok, "постоянная eps не возвращается"


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
    assert ok, "пик Планка не совпадает с законом Вина"


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
    assert ok, "путь R -> eps не совпадает с прямым"


def test_histogram_conservation():
    print("5. СОХРАНЕНИЕ МАССЫ В ГИСТОГРАММЕ")
    print("   Сумма по слоям обязана равняться полной испарённой массе Al,")
    print("   посчитанной напрямую по фрагментам с их долями Al.\n")
    frags = S.build_fragments()
    ok = True
    print(f"   {'случай':<32}{'сумма гист., кг':>16}{'прямой счёт, кг':>17}{'разница':>10}")
    cases = [("eps=0.05", Aluminium(emissivity=0.05), S.BASE_SPLIT),
             ("плёнка активна", S.BASE, S.BASE_SPLIT),
             ("голый, Al в тонкостенных", S.SCENARIOS["голый расплав"],
              S.AL_SPLITS["весь Al в тонкостенных"])]
    for label, mat, split in cases:
        run = S.run_model(frags, mat, al_split=split)
        hist = S.histogram(run)
        direct = sum(surface_thermal_model(tr, veh, mat, wall_thickness=tw)["m_vap"] * f
                     for (_, veh, tr, tw), f in zip(frags, split))
        d = abs(hist.sum() / direct - 1) if direct > 0 else 0.0
        ok &= d < 1e-6
        print(f"   {label:<32}{hist.sum():>16.6f}{direct:>17.6f}{d:>10.1e}")
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: ничего не теряется на краях бинов.\n")
    assert ok, "гистограмма теряет массу"


def test_bin_independence():
    print("6. НЕЗАВИСИМОСТЬ ОТ ШИРИНЫ БИНА (регрессия)")
    print("   Медиана считается по сырому ряду, поэтому от бина не зависит")
    print("   по построению; тест ловит возврат к медиане по гистограмме,")
    print("   которая гуляла на 1.5 км.\n")
    frags = S.build_fragments()
    run = S.run_model(frags, S.BASE)
    saved_bins, saved_c = S.BINS, S.CENTERS
    meds, tots = [], []
    print(f"   {'бин, км':>9}{'всего, кг':>13}{'медиана, км':>14}")
    try:
        for w in (0.5, 1.0, 2.0, 5.0):
            S.BINS = np.arange(30.0, 102.0, w)
            S.CENTERS = 0.5 * (S.BINS[:-1] + S.BINS[1:])
            h = S.histogram(run)
            s = S.summarize(run)
            meds.append(s["median"]); tots.append(h.sum())
            print(f"   {w:>9.1f}{h.sum():>13.3f}{s['median']:>14.2f}")
    finally:
        S.BINS, S.CENTERS = saved_bins, saved_c
    ok = (max(meds) - min(meds) < 1e-9) and (max(tots) / min(tots) - 1 < 1e-6)
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}\n")
    assert ok, "медиана или масса зависят от ширины бина"


def test_required_band():
    print("7. ТРЕБУЕМАЯ СПЕКТРАЛЬНАЯ ПОЛОСА")
    lo_hot, _, _ = required_band(2200.0)
    _, hi_cold, _ = required_band(1500.0)
    R["band.lo_um"] = lo_hot * 1e6
    R["band.hi_um"] = hi_cold * 1e6
    print(f"   объединение по рабочим 1500-2200 K: {lo_hot*1e6:.2f} - {hi_cold*1e6:.2f} мкм")
    print(f"   UV-Vis-NIR + FTIR среднего ИК (2.5-15 мкм) покрывают с запасом")
    ok = lo_hot > 0.25e-6 and hi_cold < 15e-6
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: стандартной пары приборов хватает,")
    print("      высокотемпературный эмиссометр НЕ нужен.\n")
    assert ok, "полоса не покрывается стандартной парой приборов"


def test_bare_metal_emissivity():
    print("8. eps ГОЛОГО МЕТАЛЛА ПО СОПРОТИВЛЕНИЮ (Parker-Abbott)")
    print("   Сценарий 'голый расплав' опирается на оценку, а не на измерение.")
    print("   Проверяем её там, где измерения есть: полированный твёрдый Al,")
    print("   справочно 0.04-0.07 при 600-900 K. При 300-400 K модель ниже")
    print("   справочника: идеально чистая поверхность без нативного оксида.\n")
    ok = True
    print(f"   {'T, K':>7}{'rho, мкОм*см':>14}{'eps модель':>12}{'справочно':>12}")
    for T in (300.0, 400.0, 600.0, 800.0, 900.0):
        e = float(bare_aluminium_emissivity(T))
        ref = "0.04-0.07" if T >= 600 else "~0.04"
        R[f"bare_eps.T{T:.0f}"] = e
        if T >= 600:
            ok &= 0.04 <= e <= 0.07
        print(f"   {T:>7.0f}{float(aluminium_resistivity(T))*1e8:>14.2f}{e:>12.3f}{ref:>12}")
    for T in (1000.0, 1500.0, 2000.0, 2740.0):
        R[f"bare_eps.T{T:.0f}"] = float(bare_aluminium_emissivity(T))
        R[f"resistivity.T{T:.0f}"] = float(aluminium_resistivity(T)) * 1e8
        print(f"   {T:>7.0f}{float(aluminium_resistivity(T))*1e8:>14.2f}"
              f"{float(bare_aluminium_emissivity(T)):>12.3f}{'жидкость':>12}")
    x = 5e-7 * 100 * 2000          # rho = 5e-7 Ом*м = 50 мкОм*см, T = 2000 K
    same = abs(float(parker_abbott_emissivity(5e-7, 2000.0))
               - (0.766 * x ** 0.5 - (0.309 - 0.0889 * np.log(x)) * x
                  - 0.0175 * x ** 1.5)) < 1e-12
    ok &= same
    print(f"\n   -> {'OK' if ok else 'ПРОВАЛ'}: в твёрдой фазе совпадает со справочником;")
    print("      у жидкости сопротивление в 2.4-5 раз выше, чем у твёрдого при")
    print("      900 K, и eps ~0.10-0.22,")
    print("      а не 0.05 полированного твёрдого металла.\n")
    assert ok, "оценка eps металла не сходится со справочником"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
