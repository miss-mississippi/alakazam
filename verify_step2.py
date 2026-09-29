"""Проверки шага 2. Запуск: python verify_step2.py  (или pytest)

1. Точность сплайна против прямого вызова pymsis.
2. Швы производной log(rho) у NRLMSISE-00 и их отсутствие у 2.1.
3. Сходимость траектории по шагу табуляции dh.
4. Механизм: почему солнечная активность на 120 км почти не работает.
5. География и сезон против солнечной активности.
"""

from __future__ import annotations

import numpy as np
import pymsis

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.constants import G0
from reentry.results import Recorder

# Шаги 1-2 определены для невращающейся Земли; integrate() по умолчанию
# вращение включает, поэтому здесь оно выключено явно.
NO_ROT = dict(earth_rotation=False)
R = Recorder("verify_step2")


VEH = Vehicle(mass=175.0, area=1.0, Cd=1.5)
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)
DATE = np.datetime64("2026-09-01T12:00")
KW = dict(lat=-40.0, lon=-140.0, f107=140.0, f107a=140.0, ap=4.0, version=0)


def _direct(alts_m, f107=140.0, ap=4.0, lat=-40.0):
    out = pymsis.calculate(DATE, -140.0, lat, np.asarray(alts_m) / 1e3,
                           f107s=f107, f107as=f107, aps=[[ap] * 7], version=0)
    return out[..., pymsis.Variable.MASS_DENSITY].ravel()


# Внутренние границы формулировки NRLMSISE-00, на которых у самой модели
# рвётся производная log(rho). Найдены численно (см. test_model_seams).
MSIS00_SEAMS = (72.5e3, 123.4e3)


def test_spline_accuracy():
    print("1. ТОЧНОСТЬ СПЛАЙНА ПРОТИВ ПРЯМОГО ВЫЗОВА pymsis")
    atm = MSISAtmosphere(**KW)
    h = np.linspace(35e3, 130e3, 20001)   # плотная сетка, НЕ узлы табуляции
    err = np.abs(atm.density(h) / _direct(h) - 1.0)

    near_seam = np.zeros_like(h, dtype=bool)
    for s in MSIS00_SEAMS:
        near_seam |= np.abs(h - s) < 1.0e3

    print(f"   вдали от швов модели:  макс {err[~near_seam].max():.2e}, "
          f"медиана {np.median(err[~near_seam]):.2e}")
    print(f"   в пределах 1 км от шва: макс {err[near_seam].max():.2e}")
    print("   Всплески НЕ от интерполяции: у самой NRLMSISE-00 на 72.5 и 123 км")
    print("   рвётся производная (внутренние границы формулировки), а кубический")
    print("   сплайн сглаживает излом. Цена 0.1% против собственной")
    print("   неопределённости MSIS в 10-30% — пренебрежимо.")
    R["spline.max_err_far"] = err[~near_seam].max()
    R["spline.median_err_far"] = np.median(err[~near_seam])
    R["spline.max_err_seam"] = err[near_seam].max()
    print()
    assert err[~near_seam].max() < 1e-4, f"ошибка сплайна {err[~near_seam].max():.1e}"
    assert err.max() < 5e-3, f"ошибка у шва {err.max():.1e}"


def test_model_seams():
    print("1b. ГДЕ У САМОЙ МОДЕЛИ РВЁТСЯ ПРОИЗВОДНАЯ")
    hh = np.arange(40e3, 130e3, 50.0)
    found = {}
    for ver in (0, 2.1):
        out = pymsis.calculate(DATE, -140.0, -40.0, hh / 1e3, f107s=140.0,
                               f107as=140.0, aps=[[4.0] * 7], version=ver)
        lr = np.log(out[..., pymsis.Variable.MASS_DENSITY].ravel())
        d2 = np.abs(np.gradient(np.gradient(lr, hh), hh))
        med = np.median(d2)
        idx = np.where(d2 > 12 * med)[0]
        if idx.size == 0:
            print(f"   NRLMSIS {ver}: швов не найдено, профиль гладкий")
            found[ver] = []
            continue
        groups = []
        cur = [idx[0]]
        for k in idx[1:]:
            if k - cur[-1] <= 5:
                cur.append(k)
            else:
                groups.append(cur)
                cur = [k]
        groups.append(cur)
        peaks = [hh[g[int(np.argmax(d2[g]))]] / 1e3 for g in groups]
        name = "NRLMSISE-00" if ver == 0 else f"NRLMSIS {ver}"
        print(f"   {name}: изломы на " + ", ".join(f"{p:.1f} км" for p in peaks))
        found[ver] = peaks
    print("   -> 2.x гладкая. Для адаптивного интегратора это плюс, и один")
    print("      из изломов MSISE-00 (72.5 км) лежит прямо в зоне абляции.\n")
    R["seams.msis00_km"] = found[0]
    R["seams.msis21_count"] = len(found[2.1])
    for s_km in (72.5, 123.4):
        assert any(abs(p - s_km) < 1.0 for p in found[0]), f"шов {s_km} км не найден"
    assert not found[2.1], f"у NRLMSIS 2.1 найдены изломы: {found[2.1]}"


def test_grid_convergence():
    print("2. СХОДИМОСТЬ ПО ШАГУ ТАБУЛЯЦИИ")
    print(f"   {'dh, м':>8}{'узлов':>8}{'макс ошибка':>14}"
          f"{'h торм., км':>13}{'h нагрева, км':>15}")
    prev = None
    ok = True
    rng = np.random.default_rng(1)
    h = rng.uniform(35e3, 130e3, 500)
    ref = _direct(h)
    for dh in (2000.0, 1000.0, 250.0, 100.0):
        atm = MSISAtmosphere(dh=dh, **KW)
        err = np.abs(atm.density(h) / ref - 1.0).max()
        tr = integrate(VEH, ENTRY, atm, **NO_ROT)
        _, h_a, _ = tr.peak_decel()
        h_q, _, _ = tr.peak_heating()
        print(f"   {dh:>8.0f}{len(atm._h_grid):>8d}{err:>14.2e}"
              f"{h_a/1e3:>13.3f}{h_q/1e3:>15.3f}")
        if prev is not None and abs(h_q - prev) > 1.0:
            ok = False
        prev = h_q
    print("   Макс. ошибка не падает монотонно с dh — она упирается в те же")
    print("   два шва модели, а не в разрешение сетки. Высоты пиков совпадают")
    print("   до метра при любом dh, а это и есть критерий.\n")
    assert ok, "высота пика нагрева сдвигается больше чем на 1 м при смене dh"


def test_solar_mechanism():
    print("3. МЕХАНИЗМ: ПОЧЕМУ F10.7 НА 120 КМ ПОЧТИ НЕ РАБОТАЕТ")
    print("   Гипотеза: солнечный нагрев управляет ТЕМПЕРАТУРОЙ ЭКЗОСФЕРЫ,")
    print("   а она задаёт шкалу высот в верхней термосфере. На 120 км")
    print("   атмосфера ещё определяется мезосферой снизу, не сверху.")
    print("   Проверка: если гипотеза верна, чувствительность должна резко")
    print("   расти с высотой.\n")
    print(f"   {'h, км':>7}{'F10.7=70':>12}{'F10.7=220':>12}{'отношение':>12}")
    ratios = {}
    for h in (70, 100, 120, 150, 200, 300, 400):
        lo = float(_direct([h * 1e3], f107=70.0)[0])
        hi = float(_direct([h * 1e3], f107=220.0)[0])
        ratios[h] = hi / lo
        print(f"   {h:>7}{lo:>12.3e}{hi:>12.3e}{hi/lo:>12.2f}")
    print(f"\n   на 120 км x{ratios[120]:.2f}, на 400 км x{ratios[400]:.1f}:")
    print("   солнечная активность меняет плотность в разы только на 300+ км.\n")
    for h, r in ratios.items():
        R[f"solar_ratio.{h}"] = r
    assert ratios[120] < 1.3 and ratios[400] > 3.0, f"отношения {ratios}"


def test_geography_dominates():
    print("4. ЧТО ЖЕ ТОГДА ГЛАВНЫЙ ИСТОЧНИК РАЗБРОСА АТМОСФЕРЫ")
    print("   Сравниваем размах трёх факторов по высоте пика нагрева.\n")
    def peak(**kw):
        atm = MSISAtmosphere(**{**KW, **kw})
        return integrate(VEH, ENTRY, atm, **NO_ROT).peak_heating()[0]

    solar = [peak(f107=f, f107a=f) for f in (70.0, 220.0)]
    geomag = [peak(ap=a) for a in (4.0, 80.0)]
    lat = [peak(lat=x) for x in (-75.0, 0.0)]
    season = [peak(date=d) for d in (np.datetime64("2026-03-01T12:00"),
                                     np.datetime64("2026-09-01T12:00"))]
    rows = [("солнечная активность F10.7 70-220", solar),
            ("геомагнитная буря Ap 4-80", geomag),
            ("широта -75...0", lat),
            ("сезон март/сентябрь", season)]
    print(f"   {'фактор':<36}{'размах, км':>12}")
    keys = {"солнечная активность F10.7 70-220": "solar", "геомагнитная буря Ap 4-80": "geomag",
            "широта -75...0": "lat", "сезон март/сентябрь": "season"}
    spans = {}
    for label, vals in sorted(rows, key=lambda r: -abs(r[1][1] - r[1][0])):
        spans[keys[label]] = abs(vals[1] - vals[0]) / 1e3
        print(f"   {label:<36}{spans[keys[label]]:>12.1f}")
        R[f"env_span_km.{keys[label]}"] = spans[keys[label]]
    print("\n   Для сравнения: свип Cd 1.0-2.2 даёт 5.8 км,")
    print("   а фрагментация — около 20 км.\n")
    assert spans["lat"] > 10 * max(spans["solar"], 0.1), "широта должна бить F10.7"
    assert spans["season"] > 5 * max(spans["solar"], 0.1), "сезон должен бить F10.7"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
