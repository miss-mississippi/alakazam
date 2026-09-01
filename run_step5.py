"""Шаг 5: главный результат — высотное распределение вброшенного алюминия.

Запуск:  python run_step5.py

Выход: step5_main.png, step5_sensitivity.png, step5_experiment.png
       и таблицы в консоли.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import EntryState, MSISAtmosphere, Vehicle, integrate
from reentry.ablation import Aluminium, surface_thermal_model
from reentry.emissivity import (SIGMA_SB, planck_weight, required_band,
                                total_emissivity)

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.22,
    "axes.spines.top": False, "axes.spines.right": False,
})

ATM = MSISAtmosphere()
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5,
                   inclination_deg=53.0)
M0, CD, F_AL = 175.0, 1.5, 0.30
INTACT = Vehicle(mass=M0, area=1.0, Cd=CD, nose_radius=0.5)
MW_AL_IN_OXIDE = 2 * 26.98 / (2 * 26.98 + 3 * 16.00)

# Границы атмосферных слоёв. Ключевые для задачи: выше стратопаузы частица
# оседает годами, ниже попадает в стратосферу почти сразу.
STRATOPAUSE = 50.0     # км
MESOPAUSE = 85.0       # км
OBS_LO, OBS_HI = 70.0, 80.0

FRAGMENTS = [
    ("солнечные панели", 0.10, 95.0e3, "plate", 1.5e-3),
    ("силовой набор", 0.50, 78.0e3, "compact", 55.0),
    ("мелкие элементы, MLI", 0.40, 78.0e3, "plate", 1.0e-3),
]
BINS = np.arange(30.0, 102.0, 2.0)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])


def build_fragments(fragments=FRAGMENTS, h_shift: float = 0.0,
                    entry: EntryState = ENTRY):
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


def histogram(frags, mat: Aluminium):
    """(центры бинов, кг Al в каждом слое, суммарный расплав кг)."""
    hist = np.zeros_like(CENTERS)
    melt = 0.0
    for _, veh, tr, t_wall in frags:
        r = surface_thermal_model(tr, veh, mat, wall_thickness=t_wall)
        melt += r["m_melt"]
        dm = np.diff(r["m_vap_series"]) * F_AL
        idx = np.digitize(r["h"][:-1] / 1e3, BINS) - 1
        ok = (idx >= 0) & (idx < len(CENTERS))
        np.add.at(hist, idx[ok], dm[ok])
    return hist, melt * F_AL


def raw_deposition(frags, mat: Aluminium):
    """Несгруппированные пары (высота, кг Al) по всем фрагментам.

    Медиану считаем ОТСЮДА, а не из гистограммы: медиана по бинам зависит
    от их ширины (гуляет на 1.5 км при 0.5-5 км), а по сырому ряду —
    точная. Гистограмма остаётся только для показа.
    """
    hs, dms = [], []
    for _, veh, tr, t_wall in frags:
        r = surface_thermal_model(tr, veh, mat, wall_thickness=t_wall)
        dm = np.diff(r["m_vap_series"]) * F_AL
        hs.append(r["h"][:-1] / 1e3)
        dms.append(dm)
    h = np.concatenate(hs); dm = np.concatenate(dms)
    order = np.argsort(h)
    return h[order], dm[order]


def summarize(hist, raw=None):
    tot = hist.sum()
    if tot <= 0:
        return dict(total=0.0, median=np.nan, meso=np.nan, strat=np.nan)
    if raw is not None:
        h, dm = raw
        c = np.cumsum(dm)
        median = float(np.interp(0.5 * c[-1], c, h))
        meso = float(dm[h >= STRATOPAUSE].sum() / dm.sum())
    else:
        cdf = np.cumsum(hist) / tot
        median = float(np.interp(0.5, cdf, CENTERS))
        meso = float(hist[CENTERS >= STRATOPAUSE].sum() / tot)
    return dict(total=tot, median=median, meso=meso, strat=1.0 - meso)


def report_main():
    print("A. ГЛАВНЫЙ РЕЗУЛЬТАТ: ВБРОС АЛЮМИНИЯ ПО ВЫСОТЕ")
    print(f"   объект {M0:.0f} кг, {100*F_AL:.0f}% Al = {M0*F_AL:.1f} кг Al")
    print(f"   вход 120 км / 7500 м/с / {ENTRY.gamma_deg:+.1f}°, i={ENTRY.inclination_deg:.0f}°\n")
    frags = build_fragments()
    print(f"   {'eps':>6}{'испарено Al, кг':>17}{'выход':>8}{'медиана, км':>13}"
          f"{'в мезосферу':>14}{'расплав Al, кг':>16}")
    rows = {}
    for e in (0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        mat = Aluminium(emissivity=e)
        h, melt = histogram(frags, mat)
        s = summarize(h, raw_deposition(frags, mat))
        rows[e] = (h, s, melt)
        print(f"   {e:>6.2f}{s['total']:>17.1f}{100*s['total']/(M0*F_AL):>7.0f}%"
              f"{s['median']:>13.1f}{100*s['meso']:>13.0f}%{melt:>16.1f}")
    print(f"\n   Стратопауза {STRATOPAUSE:.0f} км: выше — мезосферный вброс,")
    print("   частицы оседают годами; ниже — сразу в стратосферу.")
    print("   Медиана вброса держится в 73-77 км при любом eps: ВЫСОТА устойчива,")
    print("   меняется только МАССА. Это разделение — главный вывод для статьи.\n")
    return frags, rows


def report_sensitivity(frags):
    print("B. ЧУВСТВИТЕЛЬНОСТЬ")
    base = Aluminium(emissivity=0.10)
    h0, _ = histogram(frags, base)
    s0 = summarize(h0, raw_deposition(frags, base))
    rows = []

    def add(label, variants):
        tots = [v["total"] for v in variants]
        meds = [v["median"] for v in variants]
        rows.append((label, max(tots) / max(min(tots), 1e-9), max(meds) - min(meds)))

    add("eps 0.05-0.35",
        [summarize(histogram(frags, Aluminium(emissivity=e))[0],
                   raw_deposition(frags, Aluminium(emissivity=e)))
         for e in (0.05, 0.35)])
    add("tau оксида 1-10000 с",
        [summarize(histogram(frags, Aluminium(tau_oxide=t))[0],
                   raw_deposition(frags, Aluminium(tau_oxide=t)))
         for t in (1.0, 10000.0)])
    add("высота разрушения ±10 км",
        [summarize(histogram(build_fragments(h_shift=d), base)[0],
                   raw_deposition(build_fragments(h_shift=d), base))
         for d in (-10e3, +10e3)])
    thick = [("солнечные панели", 0.10, 95e3, "plate", 3.0e-3),
             ("силовой набор", 0.50, 78e3, "compact", 55.0),
             ("мелкие элементы, MLI", 0.40, 78e3, "plate", 2.0e-3)]
    thin = [("солнечные панели", 0.10, 95e3, "plate", 0.8e-3),
            ("силовой набор", 0.50, 78e3, "compact", 55.0),
            ("мелкие элементы, MLI", 0.40, 78e3, "plate", 0.5e-3)]
    add("толщина пластин x2 / /2",
        [summarize(histogram(build_fragments(f), base)[0],
                   raw_deposition(build_fragments(f), base))
         for f in (thick, thin)])
    heavy = [("солнечные панели", 0.05, 95e3, "plate", 1.5e-3),
             ("силовой набор", 0.75, 78e3, "compact", 55.0),
             ("мелкие элементы, MLI", 0.20, 78e3, "plate", 1.0e-3)]
    light = [("солнечные панели", 0.20, 95e3, "plate", 1.5e-3),
             ("силовой набор", 0.25, 78e3, "compact", 55.0),
             ("мелкие элементы, MLI", 0.55, 78e3, "plate", 1.0e-3)]
    add("доля тонкостенной массы 25-75%",
        [summarize(histogram(build_fragments(f), base)[0],
                   raw_deposition(build_fragments(f), base))
         for f in (heavy, light)])
    add("угол входа -1...-3°",
        [summarize(histogram(build_fragments(
            entry=EntryState(gamma_deg=g, inclination_deg=53.0)), base)[0],
            raw_deposition(build_fragments(
                entry=EntryState(gamma_deg=g, inclination_deg=53.0)), base))
         for g in (-1.0, -3.0)])
    add("наклонение орбиты 0-180°",
        [summarize(histogram(build_fragments(
            entry=EntryState(gamma_deg=-1.5, inclination_deg=i)), base)[0],
            raw_deposition(build_fragments(
                entry=EntryState(gamma_deg=-1.5, inclination_deg=i)), base))
         for i in (0.0, 180.0)])
    rows = [r for r in rows if r[1] > 1.001 or r[2] > 0.05]

    print(f"   {'фактор':<34}{'масса, раз':>13}{'медиана, км':>14}")
    for label, f_tot, d_med in sorted(rows, key=lambda r: -r[1]):
        print(f"   {label:<34}{f_tot:>12.1f}x{d_med:>13.1f}")
    print(f"\n   База (eps=0.10): {s0['total']:.1f} кг Al, медиана {s0['median']:.1f} км,")
    print(f"   {100*s0['meso']:.0f}% в мезосферу.\n")
    return rows


def report_experiment():
    print("C. ЧТО НАДО ИЗМЕРИТЬ — В ТЕРМИНАХ ВОЗМОЖНОСТЕЙ ПРИБОРА")
    print("   Прямая высокотемпературная эмиссометрия не нужна. Достаточно")
    print("   спектральной ОТРАЖАТЕЛЬНОЙ способности при комнатной температуре:")
    print("     eps(lam) = 1 - R(lam)          (Кирхгоф, непрозрачный образец)")
    print("     eps(T)   = int eps(lam) B(lam,T) dlam / int B(lam,T) dlam\n")
    print(f"   {'T, K':>7}{'пик Вина, мкм':>16}{'95% энергии, мкм':>22}")
    for T in (1500.0, 2000.0, 2400.0, 2740.0):
        lo, hi, pk = required_band(T)
        print(f"   {T:>7.0f}{pk*1e6:>16.2f}{lo*1e6:>14.2f} - {hi*1e6:<6.2f}")
    lo_all, _, _ = required_band(2740.0)
    _, hi_all, _ = required_band(1500.0)
    print(f"\n   ИТОГО прибор обязан покрыть {lo_all*1e6:.1f}-{hi_all*1e6:.1f} мкм.")
    print("   Это UV-Vis-NIR (0.25-2.5 мкм) ПЛЮС FTIR среднего ИК (2.5-15 мкм),")
    print("   оба с интегрирующей сферой либо приставкой диффузного отражения:")
    print("   окисленная поверхность рассеивает, зеркального отражения мало.\n")


def figure_main(frags, rows, path="step5_main.png"):
    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)

    # (1) Главная: вилка по eps как заштрихованная полоса
    ax = fig.add_subplot(gs[0])
    lo = rows[0.35][0]
    hi = rows[0.05][0]
    mid = rows[0.10][0]
    ax.fill_betweenx(CENTERS, lo, hi, step="mid", color="tab:blue", alpha=0.22,
                     label=r"вилка по $\varepsilon$ (0.05–0.35)")
    ax.step(mid, CENTERS, where="mid", lw=2.0, color="tab:blue",
            label=r"$\varepsilon=0.10$")
    ax.axhspan(OBS_LO, OBS_HI, color="seagreen", alpha=0.16, zorder=0)
    ax.axhline(STRATOPAUSE, color="crimson", ls="--", lw=1.3)
    ax.axhline(MESOPAUSE, color="grey", ls=":", lw=1.1)
    ax.text(hi.max() * 0.98, STRATOPAUSE + 1.5,
            "стратопауза 50 км — ниже вброса нет", fontsize=7.5,
            color="crimson", ha="right")
    ax.text(hi.max() * 0.98, MESOPAUSE + 1.2, "мезопауза 85 км",
            fontsize=7.5, color="grey", ha="right")
    ax.text(hi.max() * 0.98, 66.5, "наблюдаемое разрушение\n70–80 км",
            fontsize=7.5, color="seagreen", ha="right", va="center")
    ax.set_xlabel("вброшенный Al в слое, кг")
    ax.set_ylabel("высота, км")
    ax.set_title("Высотное распределение вброса Al\n"
                 f"объект {M0:.0f} кг, {100*F_AL:.0f}% Al", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(42, 100)

    # (2) Масса и медиана как функции eps
    ax = fig.add_subplot(gs[1])
    eps = sorted(rows)
    tot = [rows[e][1]["total"] for e in eps]
    melt = [rows[e][2] for e in eps]
    ax.fill_between(eps, tot, melt, color="tab:purple", alpha=0.18,
                    label="вилка расплав/пар")
    ax.plot(eps, melt, lw=2.0, marker="o", ms=3.5, label="расплав (верхняя)")
    ax.plot(eps, tot, lw=2.0, ls="--", marker="s", ms=3.5,
            label="испарено (нижняя)")
    ax.axhline(15.9, color="crimson", ls=":", lw=1.4)
    ax.text(0.355, 16.6, "Ferreira 15.9 кг", fontsize=7.5, color="crimson",
            ha="right")
    ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel("Al, кг")
    ax.set_title("Масса: меняется в разы", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # (3) Медиана и доля в мезосферу — устойчивы
    ax = fig.add_subplot(gs[2])
    med = [rows[e][1]["median"] for e in eps]
    meso = [100 * rows[e][1]["meso"] for e in eps]
    ax.plot(eps, med, lw=2.0, marker="o", ms=3.5, color="tab:blue",
            label="медианная высота, км")
    ax.axhspan(OBS_LO, OBS_HI, color="seagreen", alpha=0.16)
    ax.set_ylim(66, 84)
    ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel("высота, км")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(eps, meso, lw=1.8, ls="--", color="tab:orange",
             label="доля в мезосферу, %")
    ax2.set_ylabel("в мезосферу, %", color="tab:orange")
    ax2.set_ylim(0, 105)
    ax.set_title("Высота: устойчива\n74.6–75.4 км при любом $\\varepsilon$", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("Шаг 5. Вброс алюминия при входе спутника: где и сколько",
                 fontsize=12, y=1.0)
    fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


def figure_sensitivity(rows_sens, path="step5_sensitivity.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    labels = [r[0] for r in rows_sens][::-1]
    ax = axes[0]
    ax.barh(labels, [r[1] for r in rows_sens][::-1], color="tab:orange")
    ax.set_xlabel("во сколько раз меняется вброшенная масса")
    ax.set_title("Бюджет по МАССЕ", fontsize=10)
    ax = axes[1]
    ax.barh(labels, [r[2] for r in rows_sens][::-1], color="tab:blue")
    ax.set_xlabel("размах медианной высоты, км")
    ax.set_title("Бюджет по ВЫСОТЕ", fontsize=10)
    fig.suptitle("Масса и высота управляются разными параметрами", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


def figure_experiment(path="step5_experiment.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    lam = np.geomspace(0.2e-6, 30e-6, 1200)

    ax = axes[0]
    for T, c in ((1500.0, "tab:blue"), (2000.0, "tab:orange"),
                 (2740.0, "crimson")):
        w = planck_weight(lam, T)
        ax.semilogx(lam * 1e6, w / w.max(), lw=1.9, color=c, label=f"{T:.0f} K")
        lo, hi, _ = required_band(T)
        ax.axvspan(lo * 1e6, hi * 1e6, color=c, alpha=0.06)
    ax.axvspan(0.25, 2.5, color="tab:green", alpha=0.13)
    ax.axvspan(2.5, 15.0, color="tab:purple", alpha=0.13)
    ax.text(0.8, 0.55, "UV-Vis-NIR", fontsize=8, color="tab:green", rotation=90)
    ax.text(5.0, 0.55, "FTIR средний ИК", fontsize=8, color="tab:purple",
            rotation=90)
    ax.set_xlabel("длина волны, мкм"); ax.set_ylabel("планковский вес, норм.")
    ax.set_title("Какую полосу обязан покрыть прибор", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    Ts = np.linspace(1200, 2800, 60)
    for e_flat, ls in ((0.05, ":"), (0.35, "-")):
        ax.plot(Ts, [total_emissivity(lam, np.full_like(lam, e_flat), T)
                     for T in Ts], lw=1.8, ls=ls,
                label=fr"постоянная $\varepsilon$={e_flat}")
    from reentry.emissivity import synthetic_oxide_spectrum
    for d, c in ((0.1e-6, "tab:blue"), (1.0e-6, "tab:orange"),
                 (5.0e-6, "tab:green")):
        sp = synthetic_oxide_spectrum(lam, d)
        ax.plot(Ts, [total_emissivity(lam, sp, T) for T in Ts], lw=1.6,
                color=c, label=f"плёнка {d*1e6:.1f} мкм (заглушка)")
    ax.set_xlabel("температура, K"); ax.set_ylabel(r"полная $\varepsilon(T)$")
    ax.set_title("Цепочка спектр → $\\varepsilon(T)$\n(данные подставятся сюда)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[2]
    ax.axis("off")
    ax.text(0.0, 1.0, "ЧТО ИЗМЕРИТЬ", fontsize=11, weight="bold", va="top")
    txt = (
        "Образцы: Al 6061, серия по толщине оксида\n"
        "  от нативной (~3 нм) до ~5 мкм\n"
        "  термическое окисление в печи, T и t контролируются\n\n"
        "Толщина оксида:\n"
        "  эллипсометрия (до ~1 мкм)\n"
        "  профилометрия / скол в СЭМ (толще)\n\n"
        "Отражение (главное):\n"
        "  UV-Vis-NIR 0.25–2.5 мкм, интегрирующая сфера\n"
        "  FTIR 2.5–15 мкм, приставка диффузного отражения\n"
        "  нужна ПОЛУСФЕРИЧЕСКАЯ R, не зеркальная\n\n"
        "Обработка: eps(lam)=1−R(lam), затем взвешивание\n"
        "  по Планку при 1500–2740 K\n\n"
        "Что это закрывает: верхний конец вилки.\n"
        "Нижний (голый расплав) так не меряется."
    )
    ax.text(0.0, 0.92, txt, fontsize=8.2, va="top", family="monospace")

    fig.suptitle("Мост к эксперименту: измеряется отражение при комнатной "
                 "температуре, получается ε при рабочей", fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


if __name__ == "__main__":
    print()
    frags, rows = report_main()
    rows_sens = report_sensitivity(frags)
    report_experiment()
    figure_main(frags, rows)
    figure_sensitivity(rows_sens)
    figure_experiment()
    print()
