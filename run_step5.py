"""Шаг 5: главный результат — высотное распределение вброшенного алюминия.

Запуск:  python run_step5.py

Выход: step5_main.png, step5_sensitivity.png, step5_experiment.png
       и таблицы в консоли.
"""

from __future__ import annotations

import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.ablation import Aluminium, surface_thermal_model
from reentry.results import Recorder
from reentry.emissivity import (LAM_GRID, bare_aluminium_emissivity, eps_film,
                                fit_film_edge, oxide_film_emissivity,
                                planck_weight, required_band,
                                synthetic_oxide_spectrum, total_emissivity)

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.22,
    "axes.spines.top": False, "axes.spines.right": False,
})

ATM = MSISAtmosphere()
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
M0, CD, F_AL = 175.0, 1.5, 0.30
INTACT = Vehicle(mass=M0, area=1.0, Cd=CD, nose_radius=0.5)
M_AL_TOTAL = M0 * F_AL
MW_AL_IN_OXIDE = 2 * 26.98 / (2 * 26.98 + 3 * 16.00)

# Границы атмосферных слоёв. Ключевые для задачи: выше стратопаузы частица
# оседает годами, ниже попадает в стратосферу почти сразу.
STRATOPAUSE = 50.0     # км
MESOPAUSE = 85.0       # км
OBS_LO, OBS_HI = 70.0, 80.0
T_OPER = 2000.0        # K, рабочая температура: кипение Al на ~0.5-1 кПа

FRAGMENTS = [
    # имя, доля массы, высота отделения, вид, параметр
    ("солнечные панели", 0.10, 95.0e3, "plate", 1.5e-3),
    ("силовой набор", 0.50, 78.0e3, "compact", 55.0),
    ("мелкие элементы, MLI", 0.40, 78.0e3, "plate", 1.0e-3),
]

# РАСПРЕДЕЛЕНИЕ Al ПО ФРАГМЕНТАМ при той же общей доле 30%.
# Тепловая модель считает каждый фрагмент алюминиевым; доля Al говорит,
# какая часть испарённой массы — алюминий. Реальный список материалов
# конкретного аппарата неизвестен, поэтому это ось, а не константа.
AL_SPLITS = {
    "равномерно (база)": (0.30, 0.30, 0.30),
    "весь Al в силовом наборе": (0.0, 0.60, 0.0),
    "весь Al в тонкостенных": (0.60, 0.0, 0.60),
}
BASE_SPLIT = AL_SPLITS["равномерно (база)"]

# Два сценария поверхности. База — "плёнка активна": это сценарий, который
# меряется в лаборатории, и он даёт меньшую (консервативную) массу.
SCENARIOS = {
    "плёнка активна": Aluminium(surface="oxide"),
    "голый расплав": Aluminium(surface="bare"),
}
BASE = SCENARIOS["плёнка активна"]

SCEN_KEY = {"плёнка активна": "film", "голый расплав": "bare"}
SENS_KEY = {
    "сценарий поверхности (плёнка/голый)": "surface",
    "форма кривой плёнки (48 наборов)": "film_shape",
    "рост плёнки tau 1-10000 с": "growth",
    "вдув пара, eta 0-0.6": "blowing",
    "ориентация: устойчивая / кувыркание": "orientation",
    "распределение Al по фрагментам": "al_split",
    "высота разрушения ±10 км": "breakup",
    "толщина пластин x2 / /2": "thickness",
    "доля тонкостенной массы 25-75%": "thin_fraction",
    "угол входа -1...-3°": "entry_angle",
    "наклонение орбиты 0-180°": "inclination",
}
R = Recorder("step5")

BINS = np.arange(30.0, 102.0, 2.0)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])


def build_fragments(fragments=FRAGMENTS, h_shift: float = 0.0,
                    entry: EntryState = ENTRY):
    """-> [(имя, Vehicle, траектория, толщина стенки или None)]."""
    tr0 = integrate(INTACT, entry, ATM, h_stop=55.0e3)
    out = []
    for name, f_mass, h_break, kind, param in fragments:
        m = M0 * f_mass
        veh = (Vehicle.plate(m, param, Cd=CD) if kind == "plate"
               else Vehicle.compact(m, param, Cd=CD))
        t_wall = param if kind == "plate" else None
        hb = max(h_break + h_shift, 58.0e3)
        tr = integrate(veh, tr0.state_at_altitude(hb, entry.inclination_deg), ATM)
        out.append((name, veh, tr, t_wall))
    return out


def run_model(frags, mat: Aluminium, al_split=BASE_SPLIT, **model_kw) -> dict:
    """Один прогон тепловой модели по всем фрагментам.

    Возвращает полную массу испарённого Al, расплав Al и СЫРОЙ ряд
    (высота, кг Al) — из него считаются и гистограмма, и медиана.
    """
    hs, dms, per = [], [], []
    melt_al = 0.0
    for (name, veh, tr, t_wall), f_al in zip(frags, al_split):
        r = surface_thermal_model(tr, veh, mat, wall_thickness=t_wall, **model_kw)
        melt_al += r["m_melt"] * f_al
        hs.append(r["h"][:-1] / 1e3)
        dms.append(np.diff(r["m_vap_series"]) * f_al)
        per.append((name, veh, tr, r, f_al))
    h = np.concatenate(hs)
    dm = np.concatenate(dms)
    order = np.argsort(h)
    return dict(h=h[order], dm=dm[order], total=float(dm.sum()),
                melt=melt_al, per=per)


def histogram(run: dict) -> np.ndarray:
    hist = np.zeros_like(CENTERS)
    idx = np.digitize(run["h"], BINS) - 1
    ok = (idx >= 0) & (idx < len(CENTERS))
    np.add.at(hist, idx[ok], run["dm"][ok])
    return hist


def summarize(run: dict) -> dict:
    """Медиана — по СЫРОМУ ряду (высота, масса), а не по гистограмме:
    медиана по бинам зависит от их ширины (гуляла на 1.5 км при 0.5-5 км)."""
    h, dm, tot = run["h"], run["dm"], run["total"]
    if tot <= 0:
        return dict(total=0.0, median=np.nan, above_strat=np.nan,
                    above_meso=np.nan, melt=run["melt"])
    c = np.cumsum(dm)
    return dict(total=tot, median=float(np.interp(0.5 * c[-1], c, h)),
                above_strat=float(dm[h >= STRATOPAUSE].sum() / tot),
                above_meso=float(dm[h >= MESOPAUSE].sum() / tot),
                melt=run["melt"])


def eps_oper(mat: Aluminium, T: float = T_OPER) -> float:
    return float(mat.emissivity_at(np.array([T]))[0])


def film_shape_extremes(T: float = T_OPER):
    """Формы кривой плёнки с минимальной и максимальной eps(T) из свипа
    по 48 наборам (analysis_step5b.E2)."""
    rows = []
    for es, el, w in itertools.product((0.03, 0.05, 0.07, 0.10),
                                       (0.85, 0.92, 1.00), (1.0, 2.0, 3.0, 4.0)):
        try:
            lc = fit_film_edge(es, el, w)
        except ValueError:
            continue
        rows.append(((es, el, w),
                     total_emissivity(LAM_GRID, eps_film(LAM_GRID, es, lc, el, w), T)))
    rows.sort(key=lambda r: r[1])
    return rows[0], rows[-1]


def report_main():
    print("A. ГЛАВНЫЙ РЕЗУЛЬТАТ: ВБРОС АЛЮМИНИЯ ПО ВЫСОТЕ")
    print(f"   объект {M0:.0f} кг, {100*F_AL:.0f}% Al = {M_AL_TOTAL:.1f} кг Al,"
          f" Al распределён по фрагментам равномерно")
    print(f"   вход 120 км / 7500 м/с / {ENTRY.gamma_deg:+.1f}°, "
          f"i={ENTRY.inclination_deg:.0f}°; кипение при местном давлении\n")
    frags = build_fragments()
    runs = {}
    print(f"   {'сценарий':<17}{'eps(2000K)':>11}{'испар. Al, кг':>15}{'выход':>7}"
          f"{'Al2O3/кг сп.':>14}{'медиана, км':>13}{'>50 км':>8}{'>85 км':>8}"
          f"{'расплав Al':>12}")
    for name, mat in SCENARIOS.items():
        run = run_model(frags, mat)
        s = summarize(run)
        runs[name] = (run, s)
        k = f"scenarios.{SCEN_KEY[name]}"
        R[f"{k}.eps2000"] = eps_oper(mat)
        R[f"{k}.total"] = s["total"]
        R[f"{k}.yield_pct"] = 100 * s["total"] / M_AL_TOTAL
        R[f"{k}.al2o3_per_kg"] = s["total"] / MW_AL_IN_OXIDE / M0
        R[f"{k}.median"] = s["median"]
        R[f"{k}.above_strat_pct"] = 100 * s["above_strat"]
        R[f"{k}.above_meso_pct"] = 100 * s["above_meso"]
        R[f"{k}.melt"] = s["melt"]
        R[f"{k}.melt_pct"] = 100 * s["melt"] / M_AL_TOTAL
        print(f"   {name:<17}{eps_oper(mat):>11.3f}{s['total']:>15.1f}"
              f"{100*s['total']/M_AL_TOTAL:>6.0f}%"
              f"{s['total']/MW_AL_IN_OXIDE/M0:>14.3f}{s['median']:>13.1f}"
              f"{100*s['above_strat']:>7.0f}%{100*s['above_meso']:>7.0f}%"
              f"{s['melt']:>12.1f}")
    a, b = (runs[k][1]["total"] for k in SCENARIOS)
    R["scenarios.ratio"] = max(a, b) / min(a, b)
    print(f"\n   Между сценариями {max(a, b)/min(a, b):.2f}x по массе.")
    print("   Кипение идёт при 1750-2050 K, и там eps плёнки пришпилена")
    print("   справочной точкой alpha-Al2O3 при 1800 K, а eps голого металла")
    print("   по сопротивлению уже ~0.18. Сценарии сблизились.\n")

    print("   Постоянная eps (для графика и сравнения с первой версией):")
    print(f"   {'eps':>6}{'испар. Al, кг':>15}{'выход':>7}{'медиана, км':>13}"
          f"{'расплав Al':>12}")
    sweep = {}
    for e in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        run = run_model(frags, Aluminium(emissivity=e))
        s = summarize(run)
        sweep[e] = (run, s)
        k = f"sweep.e{round(e*100):03d}"
        R[f"{k}.total"] = s["total"]
        R[f"{k}.yield_pct"] = 100 * s["total"] / M_AL_TOTAL
        R[f"{k}.median"] = s["median"]
        R[f"{k}.melt"] = s["melt"]
        print(f"   {e:>6.2f}{s['total']:>15.1f}{100*s['total']/M_AL_TOTAL:>6.0f}%"
              f"{s['median']:>13.1f}{s['melt']:>12.1f}")
    print()
    return frags, runs, sweep


def report_changes(frags):
    """Декомпозиция: что дала каждая исправленная вещь по сравнению с первой
    версией (4.4-7.7 кг при eps 0.19-0.29 и 15.4 кг при eps 0.05)."""
    print("Z. ЧТО ИЗМЕНИЛОСЬ ОТНОСИТЕЛЬНО ПЕРВОЙ ВЕРСИИ (кг испарённого Al)")
    print("   первая версия: eps 0.05 -> 15.4, eps 0.19 -> 7.7, eps 0.29 -> 4.4\n")
    rows = [
        ("+ пластина 2 стороны, пояс <= своей массы,\n"
         "     расплав по максимуму (старые свойства)",
         lambda e: Aluminium.legacy(emissivity=e)),
        ("+ новые свойства Al, кипение при 1 атм (2792 K)",
         lambda e: Aluminium(emissivity=e, boil_at_local_pressure=False)),
        ("+ кипение при местном давлении (база)",
         lambda e: Aluminium(emissivity=e)),
    ]
    print(f"   {'шаг':<50}{'0.05':>7}{'0.19':>7}{'0.29':>7}")
    for (label, make), key in zip(rows, ("plates_fixed", "new_props_1atm", "local_pressure")):
        vals = [run_model(frags, make(e))["total"] for e in (0.05, 0.19, 0.29)]
        for e, v in zip((5, 19, 29), vals):
            R[f"changes.{key}.e{e:03d}"] = v
        first, *rest = label.split("\n")
        print(f"   {first:<50}" + "".join(f"{v:>7.1f}" for v in vals))
        for line in rest:
            print(f"   {line}")
    film_1atm = run_model(frags, Aluminium(surface="oxide",
                                           boil_at_local_pressure=False))["total"]
    R["changes.film_1atm"] = film_1atm
    print(f"\n   плёнка активна при кипении на 1 атм: {film_1atm:.1f} кг Al"
          " (база при местном давлении — в разделе A)")
    print()


def report_sensitivity(frags):
    print("B. ЧУВСТВИТЕЛЬНОСТЬ (база: плёнка активна, Al равномерно)")
    base = summarize(run_model(frags, BASE))
    rows = []

    def add(label, variants):
        ss = [summarize(v) for v in variants]
        tots = [s["total"] for s in ss]
        meds = [s["median"] for s in ss if np.isfinite(s["median"])]
        rows.append((label, min(tots), max(tots),
                     (max(meds) - min(meds)) if len(meds) > 1 else 0.0))

    add("сценарий поверхности (плёнка/голый)",
        [run_model(frags, m) for m in SCENARIOS.values()])
    lo_shape, hi_shape = film_shape_extremes()
    add("форма кривой плёнки (48 наборов)",
        [run_model(frags, Aluminium(surface="oxide", film_shape=sh))
         for sh, _ in (lo_shape, hi_shape)])
    add("рост плёнки tau 1-10000 с",
        [run_model(frags, Aluminium(tau_oxide=t)) for t in (1.0, 1e4)])
    add("вдув пара, eta 0-0.6",
        [run_model(frags, BASE, blowing_eta=e) for e in (0.0, 0.6)])
    add("ориентация: устойчивая / кувыркание",
        [run_model(frags, BASE, flux_mode=m) for m in ("cos", "uniform")])
    add("распределение Al по фрагментам",
        [run_model(frags, BASE, al_split=sp) for sp in AL_SPLITS.values()])
    add("высота разрушения ±10 км",
        [run_model(build_fragments(h_shift=d), BASE) for d in (-10e3, 10e3)])
    thick = [("солнечные панели", 0.10, 95e3, "plate", 3.0e-3),
             ("силовой набор", 0.50, 78e3, "compact", 55.0),
             ("мелкие элементы, MLI", 0.40, 78e3, "plate", 2.0e-3)]
    thin = [("солнечные панели", 0.10, 95e3, "plate", 0.75e-3),
            ("силовой набор", 0.50, 78e3, "compact", 55.0),
            ("мелкие элементы, MLI", 0.40, 78e3, "plate", 0.5e-3)]
    add("толщина пластин x2 / /2",
        [run_model(build_fragments(f), BASE) for f in (thick, thin)])
    heavy = [("солнечные панели", 0.05, 95e3, "plate", 1.5e-3),
             ("силовой набор", 0.75, 78e3, "compact", 55.0),
             ("мелкие элементы, MLI", 0.20, 78e3, "plate", 1.0e-3)]
    light = [("солнечные панели", 0.20, 95e3, "plate", 1.5e-3),
             ("силовой набор", 0.25, 78e3, "compact", 55.0),
             ("мелкие элементы, MLI", 0.55, 78e3, "plate", 1.0e-3)]
    add("доля тонкостенной массы 25-75%",
        [run_model(build_fragments(f), BASE) for f in (heavy, light)])
    add("угол входа -1...-3°",
        [run_model(build_fragments(entry=EntryState(gamma_deg=g,
                                                    inclination_deg=53.0)), BASE)
         for g in (-1.0, -3.0)])
    add("наклонение орбиты 0-180°",
        [run_model(build_fragments(entry=EntryState(gamma_deg=-1.5,
                                                    inclination_deg=i)), BASE)
         for i in (0.0, 180.0)])

    print(f"   {'фактор':<38}{'Al, кг: от':>11}{'до':>7}{'раз':>7}{'медиана, км':>13}")
    for label, lo, hi, dmed in sorted(rows, key=lambda r: -(r[2] - r[1])):
        k = f"sensitivity.{SENS_KEY[label]}"
        R[f"{k}.lo"] = lo
        R[f"{k}.hi"] = hi
        R[f"{k}.ratio"] = hi / lo if lo > 1e-3 else None
        R[f"{k}.median_span"] = dmed
        ratio = f"{hi/lo:>6.1f}x" if lo > 1e-3 else "     ∞"
        print(f"   {label:<38}{lo:>11.1f}{hi:>7.1f}{ratio}{dmed:>13.1f}")
    print(f"\n   База: {base['total']:.1f} кг Al, медиана {base['median']:.1f} км, "
          f"{100*base['above_strat']:.0f}% выше стратопаузы, "
          f"{100*base['above_meso']:.0f}% выше мезопаузы.")
    print(f"   Формы плёнки: eps(2000 K) от {lo_shape[1]:.3f} до {hi_shape[1]:.3f}.\n")
    R["sensitivity.base.total"] = base["total"]
    R["sensitivity.base.median"] = base["median"]
    R["film_shape.eps2000_min"] = lo_shape[1]
    R["film_shape.eps2000_max"] = hi_shape[1]
    return rows, base


def report_ferreira(runs):
    print("D. СВЕРКА С FERREIRA et al. 2024 (GRL, 10.1029/2024GL109280)")
    print("   У них: 250 кг, 30% Al = 75 кг Al. По МД окисления: окисляется")
    print("   24.0 кг Al (32%), образуя 29.8 кг кластеров AlO; 51.0 кг Al")
    print("   остаются неокисленными кластерами. То есть 32% — ВЫХОД ИХ МОДЕЛИ,")
    print("   а не допущение, и считается он при 2200 K, 86 км.")
    f_ferr = 24.0 / 75.0
    print(f"   В нашей нормировке (52.5 кг Al): {f_ferr*M_AL_TOTAL:.1f} кг Al "
          f"в оксид, {24.0/250:.3f} кг Al на кг спутника.\n")
    print(f"   {'сценарий':<42}{'Al в пар, кг':>13}{'доля Al':>9}{'на кг сп.':>11}")
    frags = runs["frags"]
    for name, mat in SCENARIOS.items():
        for sp_name, sp in AL_SPLITS.items():
            if sp_name.startswith("весь Al в силовом"):
                continue
            s = summarize(run_model(frags, mat, al_split=sp))
            spk = "uniform" if sp_name.startswith("равномерно") else "thin"
            k = f"ferreira.{SCEN_KEY[name]}_{spk}"
            R[f"{k}.total"] = s["total"]
            R[f"{k}.pct"] = 100 * s["total"] / M_AL_TOTAL
            R[f"{k}.per_kg"] = s["total"] / M0
            lbl = f"{name}, {sp_name.split(' (')[0]}"
            print(f"   {lbl:<42}{s['total']:>13.1f}{100*s['total']/M_AL_TOTAL:>8.0f}%"
                  f"{s['total']/M0:>11.3f}")
    R["ferreira.theirs.total"] = f_ferr * M_AL_TOTAL
    R["ferreira.theirs.pct"] = 100 * f_ferr
    R["ferreira.theirs.per_kg"] = 24.0 / 250
    print(f"   {'Ferreira (окислено)':<42}{f_ferr*M_AL_TOTAL:>13.1f}"
          f"{100*f_ferr:>8.0f}%{24.0/250:>11.3f}")
    print("\n   Наши 16-23% лежат НИЖЕ их 32% при равномерном Al; 32% достижимы,")
    print("   если Al сосредоточен в тонкостенных элементах. Сравниваются разные")
    print("   величины: у нас — испарённый Al (весь пар окислится), у них —")
    print("   окисленная доля при допущении, что абляции подвергается весь Al.")
    print("   Это согласие по порядку величины, не валидация.\n")


def report_experiment():
    print("C. ЧТО НАДО ИЗМЕРИТЬ — В ТЕРМИНАХ ВОЗМОЖНОСТЕЙ ПРИБОРА")
    print("   Прямая высокотемпературная эмиссометрия не нужна. Достаточно")
    print("   спектральной ОТРАЖАТЕЛЬНОЙ способности при комнатной температуре:")
    print("     eps(lam) = 1 - R(lam)          (Кирхгоф, непрозрачный образец)")
    print("     eps(T)   = int eps(lam) B(lam,T) dlam / int B(lam,T) dlam\n")
    print(f"   {'T, K':>7}{'пик Вина, мкм':>16}{'95% энергии, мкм':>22}")
    for T in (1500.0, 1800.0, 2000.0, 2200.0):
        lo, hi, pk = required_band(T)
        R[f"band.T{T:.0f}.wien_um"] = pk * 1e6
        R[f"band.T{T:.0f}.lo_um"] = lo * 1e6
        R[f"band.T{T:.0f}.hi_um"] = hi * 1e6
        print(f"   {T:>7.0f}{pk*1e6:>16.2f}{lo*1e6:>14.2f} - {hi*1e6:<6.2f}")
    lo_all, _, _ = required_band(2200.0)
    _, hi_all, _ = required_band(1500.0)
    R["band.union_lo_um"] = lo_all * 1e6
    R["band.union_hi_um"] = hi_all * 1e6
    print(f"\n   ИТОГО рабочий диапазон 1500-2200 K требует {lo_all*1e6:.2f}-"
          f"{hi_all*1e6:.1f} мкм.")
    print("   Это UV-Vis-NIR ПЛЮС FTIR среднего ИК, оба с интегрирующей")
    print("   сферой: нужна полная (зеркальная + диффузная) R.\n")


def figure_main(runs, sweep, path="figures/step5_main.png"):
    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    colors = {"плёнка активна": "tab:blue", "голый расплав": "tab:red"}
    hmax = 0.0
    for name, (run, s) in runs["scen"].items():
        hist = histogram(run)
        hmax = max(hmax, hist.max())
        ax.step(hist, CENTERS, where="mid", lw=2.0, color=colors[name],
                label=f"{name}: {s['total']:.1f} кг")
    ax.axhspan(OBS_LO, OBS_HI, color="seagreen", alpha=0.16, zorder=0)
    ax.axhline(STRATOPAUSE, color="crimson", ls="--", lw=1.3)
    ax.axhline(MESOPAUSE, color="grey", ls=":", lw=1.1)
    ax.text(hmax * 0.98, STRATOPAUSE + 1.5, "стратопауза 50 км", fontsize=7.5,
            color="crimson", ha="right")
    ax.text(hmax * 0.98, MESOPAUSE + 1.2, "мезопауза 85 км", fontsize=7.5,
            color="grey", ha="right")
    ax.text(hmax * 0.98, 66.5, "наблюдаемое разрушение\n70–80 км",
            fontsize=7.5, color="seagreen", ha="right", va="center")
    ax.set_xlabel("вброшенный Al в слое 2 км, кг")
    ax.set_ylabel("высота, км")
    ax.set_title("Высотное распределение вброса Al\n"
                 f"объект {M0:.0f} кг, {100*F_AL:.0f}% Al", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(42, 100)

    ax = fig.add_subplot(gs[1])
    eps = sorted(sweep)
    tot = [sweep[e][1]["total"] for e in eps]
    melt = [sweep[e][1]["melt"] for e in eps]
    ax.fill_between(eps, tot, melt, color="tab:purple", alpha=0.15,
                    label="вилка расплав/пар")
    ax.plot(eps, melt, lw=2.0, marker="o", ms=3.5, label="расплав")
    ax.plot(eps, tot, lw=2.0, ls="--", marker="s", ms=3.5,
            label="испарено на месте")
    for name, (run, s) in runs["scen"].items():
        e = eps_oper(SCENARIOS[name])
        ax.plot([e], [s["total"]], "D", ms=8, color=colors[name],
                label=f"{name}, ε(2000 K)={e:.2f}")
    f_ferr = 24.0 / 75.0 * M_AL_TOTAL
    ax.axhline(f_ferr, color="crimson", ls=":", lw=1.4)
    ax.text(0.055, f_ferr + 0.7, f"Ferreira: 32% Al окислено = {f_ferr:.1f} кг",
            fontsize=7.5, color="crimson", ha="left")
    ax.set_xlabel(r"постоянная $\varepsilon$"); ax.set_ylabel("Al, кг")
    ax.set_title("Масса: сценарии поверхности\nна фоне свипа по ε", fontsize=10)
    ax.legend(frameon=False, fontsize=7.2, loc="center right",
              bbox_to_anchor=(1.0, 0.66))

    ax = fig.add_subplot(gs[2])
    med = [sweep[e][1]["median"] for e in eps]
    ax.plot(eps, med, lw=2.0, marker="o", ms=3.5, color="tab:blue",
            label="медианная высота, км")
    ax.axhspan(OBS_LO, OBS_HI, color="seagreen", alpha=0.16)
    ax.set_ylim(66, 84)
    ax.set_xlabel(r"постоянная $\varepsilon$"); ax.set_ylabel("высота, км")
    ax.set_title("Медиана вброса от ε не зависит:\nеё задаёт высота разрушения",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("Шаг 5. Вброс алюминия при входе спутника: где и сколько",
                 fontsize=12, y=1.0)
    fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


def figure_sensitivity(rows, base, path="figures/step5_sensitivity.png"):
    rows = sorted(rows, key=lambda r: (r[2] - r[1]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))
    ax = axes[0]
    for k, (_, lo, hi, _) in enumerate(rows):
        ax.plot([lo, hi], [k, k], lw=6, color="tab:orange", solid_capstyle="butt")
    ax.axvline(base["total"], color="k", lw=1.0, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("испарённый Al, кг (база — пунктир)")
    ax.set_title("Бюджет по МАССЕ", fontsize=10)
    ax = axes[1]
    ax.barh(y, [r[3] for r in rows], color="tab:blue")
    ax.set_yticks(y, [""] * len(rows))
    ax.set_xlabel("размах медианной высоты, км")
    ax.set_title("Бюджет по ВЫСОТЕ", fontsize=10)
    fig.suptitle("Масса и высота управляются разными параметрами", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


def figure_experiment(path="figures/step5_experiment.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    lam = np.geomspace(0.2e-6, 30e-6, 1200)

    ax = axes[0]
    for T, c in ((1500.0, "tab:blue"), (2000.0, "tab:orange"),
                 (2740.0, "crimson")):
        w = planck_weight(lam, T)
        ax.semilogx(lam * 1e6, w / w.max(), lw=1.9, color=c, label=f"{T:.0f} K")
        lo, hi, _ = required_band(T)
        ax.axvspan(lo * 1e6, hi * 1e6, color=c, alpha=0.06)
    ax.axvspan(0.22, 1.4, color="tab:green", alpha=0.13)
    ax.axvspan(2.5, 15.0, color="tab:purple", alpha=0.13)
    ax.text(0.5, 0.55, "UV-Vis-NIR\n(есть, до 1.4)", fontsize=8,
            color="tab:green", rotation=90)
    ax.text(5.0, 0.55, "FTIR средний ИК\n(нужна сфера)", fontsize=8,
            color="tab:purple", rotation=90)
    ax.set_xlabel("длина волны, мкм"); ax.set_ylabel("планковский вес, норм.")
    ax.set_title("Какую полосу обязан покрыть прибор", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    Ts = np.linspace(900, 2800, 60)
    ax.plot(Ts, oxide_film_emissivity()(Ts), lw=2.0, color="tab:blue",
            label="плёнка активна (α-Al₂O₃)")
    ax.plot(Ts, bare_aluminium_emissivity(Ts), lw=2.0, color="tab:red",
            label="голый металл (по сопротивлению)")
    for d, c in ((0.1e-6, "0.7"), (1.0e-6, "0.5"), (5.0e-6, "0.3")):
        sp = synthetic_oxide_spectrum(lam, d)
        ax.plot(Ts, [total_emissivity(lam, sp, T) for T in Ts], lw=1.0,
                ls=":", color=c, label=f"заглушка, плёнка {d*1e6:.1f} мкм")
    ax.axvspan(1750, 2050, color="tab:orange", alpha=0.12)
    ax.text(1900, 0.02, "кипение\nна 0.3–1 кПа", fontsize=7.5, ha="center")
    ax.set_xlabel("температура, K"); ax.set_ylabel(r"полная $\varepsilon(T)$")
    ax.set_title("Цепочка спектр → ε(T)\n(данные подставятся сюда)", fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    ax.axis("off")
    ax.text(0.0, 1.0, "ЧТО ИЗМЕРИТЬ", fontsize=11, weight="bold", va="top")
    txt = (
        "Образцы: Al 6061, серия по толщине оксида\n"
        "  нативный (~3 нм) и термический (до ~0.2 мкм);\n"
        "  толще — анодирование / ПЭО / ALD-Al2O3\n\n"
        "Толщина оксида:\n"
        "  эллипсометрия (до ~1 мкм)\n"
        "  скол в СЭМ (толще)\n\n"
        "Отражение (главное):\n"
        "  UV-Vis-NIR до 1.4 мкм, интегрирующая сфера\n"
        "  FTIR 2.5–15 мкм, золотая интегрирующая сфера\n"
        "  нужна ПОЛНАЯ R, не зеркальная и не ATR\n\n"
        "Обработка: eps(lam)=1−R(lam), затем взвешивание\n"
        "  по Планку при 1500–2200 K\n\n"
        "Что это закрывает: сценарий «плёнка активна».\n"
        "Голый расплав так не меряется."
    )
    ax.text(0.0, 0.92, txt, fontsize=8.2, va="top", family="monospace")

    fig.suptitle("Мост к эксперименту: измеряется отражение при комнатной "
                 "температуре, получается ε при рабочей", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


if __name__ == "__main__":
    print()
    frags, scen, sweep = report_main()
    report_changes(frags)
    rows, base = report_sensitivity(frags)
    report_ferreira({"frags": frags})
    report_experiment()
    figure_main({"scen": scen}, sweep)
    figure_sensitivity(rows, base)
    figure_experiment()
    print()
    R.save(__file__)
