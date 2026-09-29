"""Разбор возражений к шагу 5. Запуск: python analysis_step5b.py

D. Передаточная функция "высота разрушения -> высота вброса".
   Модель НЕ предсказывает высоту вброса независимо: она её наследует.
E. Валидация метода Кирхгофа на справочных данных по alpha-Al2O3 —
   ДО того, как трогать прибор; рабочее число при температуре кипения.
F. Цена обрезанной спектральной полосы на реальных приборах НУ, включая
   интерференцию в плёнке, которую гладкая модель не видит.
"""

from __future__ import annotations

import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_step5 as S
from reentry import EntryState
from reentry.ablation import Aluminium
from reentry.results import Recorder
from reentry.emissivity import (ALUMINA_REF, LAM_GRID, eps_film, fit_film_edge,
                                planck_spectral_radiance, total_emissivity)

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.22,
    "axes.spines.top": False, "axes.spines.right": False,
})

LAM = LAM_GRID
H_BODY_BASE = 78.0     # км, базовая высота разрушения корпуса
T_WORK = (1800.0, 2000.0, 2100.0)   # K, кипение Al на 0.3-2 кПа
R = Recorder("step5b")


def part_d_transfer():
    print("D. ВЫСОТА ВБРОСА НАСЛЕДУЕТСЯ, А НЕ ПРЕДСКАЗЫВАЕТСЯ")
    print("   Возражение: 'вы получили вброс на 75 км, потому что задали")
    print("   разрушение на 78'. Считаем передаточную функцию явно.")
    print("   База: плёнка активна.\n")
    print(f"   {'сдвиг, км':>10}{'h разруш.':>11}{'медиана вброса':>17}"
          f"{'смещение':>11}{'масса, кг':>12}")
    mat = S.BASE
    hb, med, mass = [], [], []
    for d in (-15e3, -10e3, -5e3, 0.0, 5e3, 10e3, 15e3):
        s = S.summarize(S.run_model(S.build_fragments(h_shift=d), mat))
        b = H_BODY_BASE + d / 1e3
        hb.append(b); med.append(s["median"]); mass.append(s["total"])
        k = f"transfer.{'m' if d < 0 else 'p'}{abs(d)/1e3:.0f}"
        R[f"{k}.h_break"] = b
        R[f"{k}.median"] = s["median"]
        R[f"{k}.offset"] = b - s["median"]
        R[f"{k}.mass"] = s["total"]
        print(f"   {d/1e3:>+10.0f}{b:>11.0f}{s['median']:>17.2f}"
              f"{b - s['median']:>+11.2f}{s['total']:>12.1f}")
    hb, med = np.array(hb), np.array(med)
    gain_all = float(np.polyfit(hb, med, 1)[0])
    m = (hb >= 68) & (hb <= 83)
    fit = np.polyfit(hb[m], med[m], 1)
    gain_mid = float(fit[0])
    med78 = float(np.polyval(fit, 78.0))
    R["transfer.gain_all"] = gain_all
    R["transfer.gain_mid"] = gain_mid
    R["transfer.intercept78"] = med78
    R["transfer.mass_ratio"] = max(mass) / min(mass)
    print(f"\n   коэффициент передачи по всему размаху   {gain_all:.2f}")
    print(f"   в правдоподобном окне 68-83 км          {gain_mid:.2f}")
    print(f"   -> в окне: h_вброс ≈ {med78:.1f} + {gain_mid:.2f}·(H − 78) км")
    print(f"   масса при этом меняется в {max(mass)/min(mass):.1f} раза —")
    print("   высота разрушения — главный рычаг и по МАССЕ тоже.\n")

    print("   Смещение (78 км − медиана вброса) при ФИКСИРОВАННОЙ h разрушения:")
    frags0 = S.build_fragments()
    cases = [("плёнка активна", S.BASE, None, {}),
             ("голый расплав", S.SCENARIOS["голый расплав"], None, {}),
             ("tau=1 с", Aluminium(tau_oxide=1.0), None, {}),
             ("tau=10000 с", Aluminium(tau_oxide=1e4), None, {}),
             ("вдув eta=0.6", S.BASE, None, dict(blowing_eta=0.6)),
             ("кувыркание", S.BASE, None, dict(flux_mode="uniform")),
             ("угол входа -1°", S.BASE, EntryState(gamma_deg=-1.0, inclination_deg=53.), {}),
             ("угол входа -3°", S.BASE, EntryState(gamma_deg=-3.0, inclination_deg=53.), {})]
    offs = []
    okeys = ["film", "bare", "tau1", "tau10000", "blowing", "tumbling", "gamma1", "gamma3"]
    for (lbl, mt, ent, kw), ok_ in zip(cases, okeys):
        f = S.build_fragments(entry=ent) if ent else frags0
        s = S.summarize(S.run_model(f, mt, **kw))
        offs.append(H_BODY_BASE - s["median"])
        R[f"offsets.{ok_}"] = H_BODY_BASE - s["median"]
        print(f"     {lbl:<20}{H_BODY_BASE - s['median']:>+8.2f} км")
    offs = np.array(offs)
    R["offsets.min"] = offs.min()
    R["offsets.max"] = offs.max()
    print(f"\n   -> Смещение {offs.mean():+.1f} км, размах {offs.min():+.1f}"
          f" ... {offs.max():+.1f} км: устойчиво")
    print("      ко всему, КРОМЕ самой высоты разрушения.")
    print("\n   ЧЕСТНАЯ ФОРМУЛИРОВКА: модель не предсказывает высоту вброса, а")
    print("   переводит высоту разрушения в высоту вброса с коэффициентом ~0.8.")
    print("   Главная неопределённость по высоте — в наблюдательных данных.\n")
    return hb, med, fit


def part_e_kirchhoff():
    print("E. ВАЛИДАЦИЯ МЕТОДА ДО ПРИБОРА")
    print(f"   Справочно для alpha-Al2O3: eps_полн = {ALUMINA_REF[300.0]} при 300 K")
    print(f"   и {ALUMINA_REF[1800.0]} при 1800 K. Гипотеза: падение — эффект "
          "ВЗВЕШИВАНИЯ,")
    print("   а не изменения eps(lam) с температурой.")
    print("   Проверка: ОДНА фиксированная eps(lam) должна дать обе цифры.\n")
    lam_c = fit_film_edge()
    e = eps_film(LAM, 0.05, lam_c)
    e300 = total_emissivity(LAM, e, 300.)
    R["kirchhoff.lam_c_um"] = lam_c * 1e6
    R["kirchhoff.e300"] = e300
    R["kirchhoff.err_pct"] = 100 * (e300 / ALUMINA_REF[300.0] - 1)
    print(f"   Край фононной области подобран ПО ОДНОЙ точке (1800 K):"
          f" {lam_c*1e6:.2f} мкм")
    print(f"   Вторая точка предсказывается: eps(300 K) = {e300:.3f}"
          f"  против справочных {ALUMINA_REF[300.0]}")
    print(f"   Ошибка {100*(e300/ALUMINA_REF[300.0]-1):+.1f}%.\n")
    print(f"   {'T, K':>7}{'eps полная':>13}{'пик Вина, мкм':>16}")
    for T in (300, 800, 1200, 1500, 1800, 2000, 2200, 2740):
        R[f"film_eps.T{T}"] = total_emissivity(LAM, e, T)
        print(f"   {T:>7}{total_emissivity(LAM, e, T):>13.3f}"
              f"{2.8977e-3/T*1e6:>16.2f}")
    print("\n   Рабочая температура — кипение Al на местном давлении, 1750-2050 K,")
    print("   то есть рядом с опорной точкой 1800 K. Экстраполяция к 2740 K")
    print("   (выше плавления самого Al2O3, 2345 K) больше не нужна.\n")
    return lam_c


def part_e2_shape_sweep():
    """Насколько валидация и рабочее число зависят от РУЧНОГО выбора формы.

    В eps_film четыре параметра. По справочной точке 1800 K подбирается
    только lam_c; eps_short, eps_long и width выбраны рукой. Свипуем все,
    каждый раз перефитируя lam_c под ту же точку 1800 K.
    """
    print("E2. НАСКОЛЬКО ВАЛИДАЦИЯ ЗАВИСИТ ОТ ВЫБОРА ФОРМЫ")
    print("   Свип по eps_short, eps_long, width; lam_c каждый раз")
    print("   перефитируется под справочную точку 1800 K.\n")
    rows = []
    for es, el, w in itertools.product((0.03, 0.05, 0.07, 0.10),
                                       (0.85, 0.92, 1.00), (1.0, 2.0, 3.0, 4.0)):
        try:
            lc = fit_film_edge(es, el, w)
        except ValueError:
            continue
        sp = eps_film(LAM, es, lc, el, w)
        rows.append([total_emissivity(LAM, sp, T) for T in (300.,) + T_WORK + (2740.,)])
    r = np.array(rows)
    e300 = r[:, 0]
    ok300 = np.abs(e300 / ALUMINA_REF[300.0] - 1) < 0.10
    print(f"   наборов формы: {len(rows)}")
    print(f"   ПРЕДСКАЗАННАЯ eps(300 K), справочное {ALUMINA_REF[300.0]}:")
    print(f"     разброс {e300.min():.3f} - {e300.max():.3f}  "
          f"({100*(e300.min()/0.83-1):+.0f}% ... {100*(e300.max()/0.83-1):+.0f}%)")
    print(f"     в пределах ±10% от справочного: {ok300.sum()} из {len(rows)}")
    R["shape.n"] = len(rows)
    R["shape.e300_min"] = e300.min()
    R["shape.e300_max"] = e300.max()
    R["shape.e300_min_pct"] = 100 * (e300.min() / 0.83 - 1)
    R["shape.e300_max_pct"] = 100 * (e300.max() / 0.83 - 1)
    R["shape.n_ok300"] = int(ok300.sum())
    print("\n   -> Формулировка: в разумном семействе кривых СУЩЕСТВУЕТ набор,")
    print("      воспроизводящий обе справочные точки. Это ПОДДЕРЖКА гипотезы,")
    print("      не подтверждение.\n")
    print(f"   {'T, K':>7}{'eps по 48 наборам':>22}{'фактор':>8}"
          f"{'только воспр. 300 K':>22}")
    for k, T in enumerate(T_WORK + (2740.,), start=1):
        c = r[:, k]
        R[f"shape.T{T:.0f}.min"] = c.min()
        R[f"shape.T{T:.0f}.max"] = c.max()
        R[f"shape.T{T:.0f}.factor"] = c.max() / c.min()
        R[f"shape.T{T:.0f}.ok_min"] = c[ok300].min()
        R[f"shape.T{T:.0f}.ok_max"] = c[ok300].max()
        print(f"   {T:>7.0f}{c.min():>12.3f} - {c.max():.3f}{c.max()/c.min():>8.2f}"
              f"{c[ok300].min():>12.3f} - {c[ok300].max():.3f}")
    print("\n   -> При рабочей температуре свобода формы почти не проходит в")
    print("      число: край пришпилен условием при 1800 K, а 2000 K от него")
    print("      недалеко. При 2740 K размах был бы 1.5x.\n")
    return r


def _al_index(lam):
    """Комплексный показатель преломления Al по Друде (Rakic 1998:
    hbar*wp = 14.98 эВ, hbar*gamma = 0.047 эВ). Без межзонного пика у
    0.8 мкм — для оценки интерференционных полос в 1-5 мкм достаточно."""
    E = 1.23984e-6 / lam
    eps = 1.0 - 14.98 ** 2 / (E ** 2 + 1j * 0.047 * E)
    return np.sqrt(eps)


def film_on_al_emissivity(lam, d, n_film=1.65):
    """eps(lam) = 1 - R прозрачной плёнки Al2O3 толщины d на Al, нормальное
    падение, с интерференцией. n = 1.65 — аморфный/анодный оксид."""
    r01 = (1.0 - n_film) / (1.0 + n_film)
    N = _al_index(lam)
    r12 = (n_film - N) / (n_film + N)
    ph = np.exp(2j * 2 * np.pi * n_film * d / lam)
    r = (r01 + r12 * ph) / (1.0 + r01 * r12 * ph)
    return 1.0 - np.abs(r) ** 2


def part_f_instruments(lam_c):
    print("F. ЦЕНА ОБРЕЗАННОЙ ПОЛОСЫ НА РЕАЛЬНЫХ ПРИБОРАХ")
    print("   По каталогу НУ:")
    print("     UV-2600i + ISR-2600Plus: 0.22-1.4 мкм")
    print("     FTIR только ATR (Bruker Alpha II, Thermo Nicolet iS12)\n")
    print(f"   {'T, K':>7}{'<1.4 мкм':>11}{'1.4-2.5':>10}{'2.5-5':>9}{'>5 мкм':>9}")
    for T in (1500., 2000., 2200.):
        B = planck_spectral_radiance(LAM, T)
        tot = np.trapezoid(B, LAM)

        def fr(a, b):
            m = (LAM >= a) & (LAM <= b)
            return np.trapezoid(B[m], LAM[m]) / tot
        R[f"fractions.T{T:.0f}.lt14"] = 100 * fr(0, 1.4e-6)
        R[f"fractions.T{T:.0f}.b14_25"] = 100 * fr(1.4e-6, 2.5e-6)
        R[f"fractions.T{T:.0f}.b25_5"] = 100 * fr(2.5e-6, 5e-6)
        R[f"fractions.T{T:.0f}.gt5"] = 100 * fr(5e-6, 1e-3)
        print(f"   {T:>7.0f}{100*fr(0,1.4e-6):>10.1f}%{100*fr(1.4e-6,2.5e-6):>9.1f}%"
              f"{100*fr(2.5e-6,5e-6):>8.1f}%{100*fr(5e-6,1e-3):>8.1f}%")

    true = eps_film(LAM, 0.05, lam_c)
    gap = (LAM > 1.4e-6) & (LAM < 2.5e-6)
    i0 = int(np.argmax(LAM > 1.4e-6)) - 1
    i1 = int(np.argmax(LAM > 2.5e-6))

    def interp_gap(spec):
        out = spec.copy()
        out[gap] = np.interp(LAM[gap], [LAM[i0], LAM[i1]], [spec[i0], spec[i1]])
        return out

    print("\n   ПРОВАЛ 1.4-2.5 мкм на ГЛАДКОЙ модели (там она плоская по построению,")
    print("   поэтому это проверка арифметики, а не физики):")
    print(f"   {'T, K':>7}{'истинная':>11}{'интерполяция':>15}{'ошибка':>10}")
    for T in (1500., 2000., 2200.):
        a = total_emissivity(LAM, true, T)
        b = total_emissivity(LAM, interp_gap(true), T)
        R[f"gap_smooth.T{T:.0f}.true"] = a
        R[f"gap_smooth.T{T:.0f}.interp"] = b
        R[f"gap_smooth.T{T:.0f}.err_pct"] = 100 * (b / a - 1)
        print(f"   {T:>7.0f}{a:>11.4f}{b:>15.4f}{100*(b/a-1):>+9.2f}%")

    print("\n   ПРОВАЛ С ИНТЕРФЕРЕНЦИЕЙ: прозрачная плёнка Al2O3 (n = 1.65) на Al.")
    print("   Полосы с периодом ~lam^2/(2nd) ложатся как раз в 1.4-2.5 мкм.")
    print("   Ошибка eps(2000 K) от интерполяции провала, абсолютная, в долях")
    print("   от рабочей eps плёнки ~0.32:")
    print(f"   {'d, мкм':>8}{'вклад провала':>15}{'интерполяция':>14}{'ошибка':>9}{'от 0.32':>9}")
    B = planck_spectral_radiance(LAM, 2000.)
    Btot = np.trapezoid(B, LAM)
    worst = 0.0
    for d in (0.1e-6, 0.3e-6, 0.5e-6, 1.0e-6, 2.0e-6, 5.0e-6):
        sp = film_on_al_emissivity(LAM, d)
        a = np.trapezoid((sp * B)[gap], LAM[gap]) / Btot
        b = np.trapezoid((interp_gap(sp) * B)[gap], LAM[gap]) / Btot
        worst = max(worst, abs(b - a))
        R[f"interference.d{d*1e7:.0f}.err_abs"] = b - a
        R[f"interference.d{d*1e7:.0f}.err_pct"] = 100 * (b - a) / 0.32
        print(f"   {d*1e6:>8.1f}{a:>15.4f}{b:>14.4f}{b-a:>+9.4f}{100*(b-a)/0.32:>+8.1f}%")
    R["interference.worst_pct"] = 100 * worst / 0.32
    print(f"   -> худший случай {100*worst/0.32:.1f}% от рабочей eps. Контраст полос")
    print("      мал: подложка Al отражает ~95%, и в прозрачной области вклад")
    print("      плёнки в eps — сотые. Провал интерполируется, расширять UV-Vis")
    print("      за 1.4 мкм не нужно. Оговорка: Друде без межзонного пика Al и")
    print("      гладкая поверхность; шероховатость это надо проверить на образце.\n")

    print("   МИД-ИК, если его НЕ мерить (экстраполяция постоянной от 1.4 мкм):")
    print(f"   {'T, K':>7}{'истинная':>11}{'без мид-ИК':>14}{'ошибка':>10}")
    for T in (1500., 2000., 2200.):
        a = total_emissivity(LAM, true, T)
        b = total_emissivity(LAM, np.full_like(LAM, 0.05), T)
        R[f"no_midir.T{T:.0f}.true"] = a
        R[f"no_midir.T{T:.0f}.err_pct"] = 100 * (b / a - 1)
        print(f"   {T:>7.0f}{a:>11.3f}{b:>14.3f}{100*(b/a-1):>+9.0f}%")
    print("\n   А если край взять из литературы, но не мерить (разброс 3-6 мкм):")
    print(f"   {'T, K':>7}{'край 3 мкм':>13}{'край 6 мкм':>13}{'разброс':>10}")
    for T in (1500., 2000., 2200.):
        a = total_emissivity(LAM, eps_film(LAM, 0.05, 3e-6), T)
        b = total_emissivity(LAM, eps_film(LAM, 0.05, 6e-6), T)
        R[f"edge.T{T:.0f}.edge3"] = a
        R[f"edge.T{T:.0f}.edge6"] = b
        R[f"edge.T{T:.0f}.spread_pct"] = 100 * (a / b - 1)
        print(f"   {T:>7.0f}{a:>13.3f}{b:>13.3f}{100*(a/b-1):>+9.0f}%")

    print("\n   ВЫВОД ПО ПРИБОРАМ:")
    print("     1. Средний ИК — ОБЯЗАТЕЛЕН: без него ошибка ~80-85%.")
    print("     2. Нужна ПОЛНАЯ направленно-полусферическая R: золотая")
    print("        интегрирующая сфера для среднего ИК. DRIFT (Praying Mantis)")
    print("        собирает не всю полусферу и не даёт абсолютной R — только")
    print("        запасной вариант с калибровкой по эталону.")
    print("     3. ATR не годится: меряет затухающей волной (глубина ~0.5-2 мкм")
    print("        в среднем ИК) через прижатый кристалл, отражения не даёт,")
    print("        оптический контакт с жёстким купоном не обеспечить.")
    print("     4. Провал 1.4-2.5 мкм интерполируется даже с интерференцией.\n")
    return true


def figure(hb, med, fit, lam_c, path="step5b_revision.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    ax = axes[0]
    ax.plot(hb, med, "o-", lw=2.0, ms=5, label="медиана вброса")
    ax.plot(hb, hb, ls=":", color="grey", lw=1.3, label="1:1")
    m = (hb >= 68) & (hb <= 83)
    ax.plot(hb[m], np.polyval(fit, hb[m]), lw=1.4, color="crimson",
            label=f"наклон {fit[0]:.2f} в окне 68–83")
    ax.axvspan(70, 80, color="seagreen", alpha=0.14)
    ax.set_xlabel("заданная высота разрушения, км")
    ax.set_ylabel("медианная высота вброса, км")
    ax.set_title("Высота вброса НАСЛЕДУЕТСЯ\nот высоты разрушения", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    e = eps_film(LAM, 0.05, lam_c)
    ax.semilogx(LAM * 1e6, e, lw=2.0, color="k", label=r"$\varepsilon(\lambda)$, одна кривая")
    ax.semilogx(LAM * 1e6, film_on_al_emissivity(LAM, 1e-6), lw=0.9,
                color="tab:purple", alpha=0.8, label="плёнка 1 мкм на Al\n(интерференция)")
    ax.axvspan(1.4, 2.5, color="grey", alpha=0.15)
    for T, c in ((300.0, "tab:blue"), (1800.0, "tab:orange"), (2000.0, "crimson")):
        B = planck_spectral_radiance(LAM, T)
        ax.semilogx(LAM * 1e6, B / B.max() * 0.9, lw=1.2, ls="--", color=c,
                    alpha=0.75, label=f"Планк {T:.0f} K")
    ax.set_xlim(0.3, 40); ax.set_ylim(0, 1.02)
    ax.set_xlabel("длина волны, мкм"); ax.set_ylabel(r"$\varepsilon$ / Планк, норм.")
    ax.set_title("Падение ε с температурой — эффект\nвзвешивания; серое — провал приборов",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="center left")

    ax = axes[2]
    Ts = np.linspace(300, 2800, 60)
    ax.plot(Ts, [total_emissivity(LAM, e, T) for T in Ts], lw=2.0, color="k",
            label="модель, одна ε(λ)")
    ax.plot([300, 1800], [ALUMINA_REF[300.0], ALUMINA_REF[1800.0]], "o", ms=8,
            mfc="none", mew=2, color="crimson", label="справочные α-Al₂O₃")
    ax.axvspan(1750, 2050, color="tab:orange", alpha=0.15)
    ax.text(1900, 0.72, "кипение Al\nна 0.3–1 кПа", fontsize=8, ha="center",
            color="tab:orange")
    e2000 = total_emissivity(LAM, e, 2000.)
    ax.annotate(f"ε(2000 K) = {e2000:.2f}", xy=(2000, e2000),
                xytext=(1300, 0.22), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.set_ylim(0.15, 0.92)
    ax.set_xlabel("температура, K"); ax.set_ylabel(r"полная $\varepsilon(T)$")
    ax.set_title("Валидация до прибора:\nвторая точка предсказана", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Ревизия шага 5: что модель на самом деле предсказывает "
                 "и что можно измерить", fontsize=11.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


if __name__ == "__main__":
    print()
    hb, med, fit = part_d_transfer()
    lam_c = part_e_kirchhoff()
    part_e2_shape_sweep()
    part_f_instruments(lam_c)
    figure(hb, med, fit, lam_c)
    print()
    R.save(__file__)
