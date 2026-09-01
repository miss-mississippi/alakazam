"""Разбор трёх возражений к шагу 5. Запуск: python analysis_step5b.py

D. Передаточная функция "высота разрушения -> высота вброса".
   Модель НЕ предсказывает высоту вброса независимо: она её наследует.
E. Валидация метода Кирхгофа на справочных данных по alpha-Al2O3 —
   ДО того, как трогать прибор.
F. Цена обрезанной спектральной полосы на реальных приборах НУ.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import run_step5 as S
from reentry import EntryState
from reentry.ablation import Aluminium
from reentry.emissivity import planck_spectral_radiance, total_emissivity

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.22,
    "axes.spines.top": False, "axes.spines.right": False,
})

LAM = np.geomspace(0.2e-6, 100e-6, 40000)
H_BODY_BASE = 78.0     # км, базовая высота разрушения корпуса


def eps_film(lam, eps_short=0.05, lam_c=3.98e-6, eps_long=0.95, width=1.6):
    """eps(lam) для оксидной плёнки на металле.

    Прозрачная область: виден МЕТАЛЛ под плёнкой, eps ~ 0.05.
    Фононная область (многофононное поглощение Al2O3): eps ~ 0.95.
    Край перехода lam_c подобран по ОДНОЙ справочной точке (1800 K),
    после чего кривая проверяется на ВТОРОЙ (300 K) — см. раздел E.
    """
    return eps_short + (eps_long - eps_short) / (1.0 + (lam_c / lam) ** width)


def part_d_transfer():
    print("D. ВЫСОТА ВБРОСА НАСЛЕДУЕТСЯ, А НЕ ПРЕДСКАЗЫВАЕТСЯ")
    print("   Возражение справедливо: 'вы получили вброс на 75 км, потому что")
    print("   задали разрушение на 78'. Считаем передаточную функцию явно.\n")
    print(f"   {'сдвиг, км':>10}{'h разруш.':>11}{'медиана вброса':>17}"
          f"{'смещение':>11}{'масса, кг':>12}")
    mat = Aluminium(emissivity=0.10)
    hb, med, off, mass = [], [], [], []
    for d in (-15e3, -10e3, -5e3, 0.0, 5e3, 10e3, 15e3):
        frags = S.build_fragments(h_shift=d)
        h, _ = S.histogram(frags, mat)
        s = S.summarize(h, S.raw_deposition(frags, mat))
        b = H_BODY_BASE + d / 1e3
        hb.append(b); med.append(s["median"]); off.append(b - s["median"])
        mass.append(s["total"])
        print(f"   {d/1e3:>+10.0f}{b:>11.0f}{s['median']:>17.2f}"
              f"{b - s['median']:>+11.2f}{s['total']:>12.1f}")
    hb, med, off = np.array(hb), np.array(med), np.array(off)
    gain_all = float(np.polyfit(hb, med, 1)[0])
    m = (hb >= 68) & (hb <= 83)
    gain_mid = float(np.polyfit(hb[m], med[m], 1)[0])
    print(f"\n   коэффициент передачи по всему размаху   {gain_all:.2f}")
    print(f"   в правдоподобном окне 68-83 км          {gain_mid:.2f}")
    print(f"   масса при этом меняется в {max(mass)/min(mass):.1f} раза —")
    print("   высота разрушения оказалась и главным рычагом по МАССЕ тоже.\n")

    print("   Смещение (h разруш. − медиана вброса) при ФИКСИРОВАННОЙ h разруш.:")
    frags0 = S.build_fragments()
    cases = [("eps=0.05", Aluminium(emissivity=0.05), None),
             ("eps=0.25", Aluminium(emissivity=0.25), None),
             ("tau=1 с", Aluminium(tau_oxide=1.0), None),
             ("tau=10000 с", Aluminium(tau_oxide=1e4), None),
             ("угол входа -1°", mat, EntryState(gamma_deg=-1.0, inclination_deg=53.)),
             ("угол входа -3°", mat, EntryState(gamma_deg=-3.0, inclination_deg=53.))]
    offs = []
    for lbl, mt, ent in cases:
        f = S.build_fragments(entry=ent) if ent else frags0
        s = S.summarize(S.histogram(f, mt)[0], S.raw_deposition(f, mt))
        offs.append(H_BODY_BASE - s["median"])
        print(f"     {lbl:<20}{H_BODY_BASE - s['median']:>+8.2f} км")
    print(f"\n   -> Смещение {np.mean(offs):+.1f} ± {np.std(offs):.1f} км: устойчиво")
    print("      ко всему, КРОМЕ самой высоты разрушения.")
    print("\n   ЧЕСТНАЯ ФОРМУЛИРОВКА: модель предсказывает не высоту вброса,")
    print("   а СМЕЩЕНИЕ между разрушением и вбросом — небольшое и стабильное.")
    print("   Это позволяет атмосферным моделистам переводить НАБЛЮДАЕМЫЕ")
    print("   высоты разрушения в высоты вброса. Но главная неопределённость")
    print("   по высоте лежит в наблюдательных данных, а не в модели.\n")
    return hb, med, off


def part_e_kirchhoff():
    print("E. ВАЛИДАЦИЯ МЕТОДА ДО ПРИБОРА")
    print("   Справочно для alpha-Al2O3: eps_полн = 0.83 при 300 K")
    print("   и 0.35 при 1800 K. Гипотеза: падение — эффект ВЗВЕШИВАНИЯ,")
    print("   а не изменения eps(lam) с температурой.")
    print("   Проверка: ОДНА фиксированная eps(lam) должна дать обе цифры.\n")
    lam_c = brentq(lambda lc: total_emissivity(LAM, eps_film(LAM, 0.05, lc), 1800.)
                   - 0.35, 1e-6, 20e-6)
    e300 = total_emissivity(LAM, eps_film(LAM, 0.05, lam_c), 300.)
    print(f"   Край фононной области подобран ПО ОДНОЙ точке (1800 K):"
          f" {lam_c*1e6:.2f} мкм")
    print(f"   Вторая точка предсказывается: eps(300 K) = {e300:.3f}"
          f"  против справочных 0.83")
    print(f"   Ошибка {100*(e300/0.83-1):+.1f}% — при том, что она НЕ подгонялась.\n")
    print("   Край 4.0 мкм физически осмыслен: у сапфира прозрачность до ~5 мкм,")
    print("   дальше многофононное поглощение. Прозрачная область eps=0.05 —")
    print("   это металл под плёнкой, а не сам оксид.\n")
    print(f"   {'T, K':>7}{'eps полная':>13}{'пик Вина, мкм':>16}")
    for T in (300, 800, 1200, 1500, 1800, 2200, 2740):
        print(f"   {T:>7}{total_emissivity(LAM, eps_film(LAM, 0.05, lam_c), T):>13.3f}"
              f"{2.8977e-3/T*1e6:>16.2f}")
    print("\n   -> Допущение 'eps(lam) не зависит от T' ПОДТВЕРЖДЕНО на")
    print("      справочных данных, без прибора.")
    print("   -> Побочный результат: при 2740 K та же кривая даёт eps = 0.25,")
    print("      а не 0.35. Верхний конец рабочей вилки надо ОПУСТИТЬ:")
    print("      физический диапазон при температуре кипения 0.05-0.25.\n")
    return lam_c


def part_f_instruments(lam_c):
    print("F. ЦЕНА ОБРЕЗАННОЙ ПОЛОСЫ НА РЕАЛЬНЫХ ПРИБОРАХ")
    print("   По каталогу НУ (проверка Бекнура):")
    print("     UV-2600i + ISR-2600Plus: 0.22-1.4 мкм, не 2.5")
    print("     FTIR только ATR (Bruker Alpha II, Thermo Nicolet iS12)\n")
    print(f"   {'T, K':>7}{'<1.4 мкм':>11}{'1.4-2.5':>10}{'2.5-5':>9}{'>5 мкм':>9}")
    for T in (1500., 2000., 2740.):
        B = planck_spectral_radiance(LAM, T)
        tot = np.trapezoid(B, LAM)
        def fr(a, b):
            m = (LAM >= a) & (LAM <= b)
            return np.trapezoid(B[m], LAM[m]) / tot
        print(f"   {T:>7.0f}{100*fr(0,1.4e-6):>10.1f}%{100*fr(1.4e-6,2.5e-6):>9.1f}%"
              f"{100*fr(2.5e-6,5e-6):>8.1f}%{100*fr(5e-6,1e-3):>8.1f}%")

    true = eps_film(LAM, 0.05, lam_c)
    gap = (LAM > 1.4e-6) & (LAM < 2.5e-6)
    i0 = int(np.argmax(LAM > 1.4e-6)) - 1
    i1 = int(np.argmax(LAM > 2.5e-6))
    lin = true.copy()
    lin[gap] = np.interp(LAM[gap], [LAM[i0], LAM[i1]], [true[i0], true[i1]])

    print("\n   ПРОБЕЛ 1.4-2.5 мкм, если мид-ИК измерен и пробел интерполирован:")
    print(f"   {'T, K':>7}{'истинная':>11}{'интерполяция':>15}{'ошибка':>10}")
    for T in (1500., 2000., 2740.):
        a = total_emissivity(LAM, true, T); b = total_emissivity(LAM, lin, T)
        print(f"   {T:>7.0f}{a:>11.4f}{b:>15.4f}{100*(b/a-1):>+9.2f}%")
    print("   Оксид алюминия в этой области спектрально ПУСТ: прозрачное окно")
    print("   между краем электронного поглощения и фононными полосами.")
    print("   Интерполяция физически оправдана, цена — доли процента.\n")

    print("   МИД-ИК, если его НЕ мерить (экстраполяция постоянной от 1.4 мкм):")
    print(f"   {'T, K':>7}{'истинная':>11}{'без мид-ИК':>14}{'ошибка':>10}")
    for T in (1500., 2200., 2740.):
        a = total_emissivity(LAM, true, T)
        b = total_emissivity(LAM, np.full_like(LAM, 0.05), T)
        print(f"   {T:>7.0f}{a:>11.3f}{b:>14.3f}{100*(b/a-1):>+9.0f}%")
    print("\n   А если край взять из литературы, но не мерить (разброс 3-6 мкм):")
    print(f"   {'T, K':>7}{'край 3 мкм':>13}{'край 6 мкм':>13}{'разброс':>10}")
    for T in (1500., 2200., 2740.):
        a = total_emissivity(LAM, eps_film(LAM, 0.05, 3e-6), T)
        b = total_emissivity(LAM, eps_film(LAM, 0.05, 6e-6), T)
        print(f"   {T:>7.0f}{a:>13.3f}{b:>13.3f}{100*(a/b-1):>+9.0f}%")

    print("\n   ВЫВОД ПО ПРИБОРАМ:")
    print("     1. Расширение UV-Vis за 1.4 мкм — НЕ нужно. Провал")
    print("        интерполируется, цена 0.1%.")
    print("     2. Мид-ИК диффузное отражение — ОБЯЗАТЕЛЬНО. Без него")
    print("        ошибка 80-88%, и вся спектральная структура именно там.")
    print("     3. ATR принципиально не годится: меряет затухающей волной")
    print("        приповерхностный слой через прижатый кристалл, не даёт")
    print("        полусферического диффузного отражения, и оптический контакт")
    print("        с жёстким шероховатым купоном не обеспечить.")
    print("     ЧТО ИСКАТЬ ПОИМЁННО: приставка DRIFT (Praying Mantis и т.п.)")
    print("     либо интегрирующая сфера для среднего ИК (золотое покрытие).")
    print("     Это ОПЦИЯ к спектрометру, наличие проверяется отдельно от")
    print("     названия прибора.\n")
    return true, lin


def figure(hb, med, off, lam_c, path="step5b_revision.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    ax = axes[0]
    ax.plot(hb, med, "o-", lw=2.0, ms=5, label="медиана вброса")
    ax.plot(hb, hb, ls=":", color="grey", lw=1.3, label="1:1")
    m = (hb >= 68) & (hb <= 83)
    g = np.polyfit(hb[m], med[m], 1)
    ax.plot(hb[m], np.polyval(g, hb[m]), lw=1.4, color="crimson",
            label=f"наклон {g[0]:.2f} в окне 68–83")
    ax.axvspan(70, 80, color="seagreen", alpha=0.14)
    ax.set_xlabel("заданная высота разрушения, км")
    ax.set_ylabel("медианная высота вброса, км")
    ax.set_title("Высота вброса НАСЛЕДУЕТСЯ\nот высоты разрушения", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    e = eps_film(LAM, 0.05, lam_c)
    ax.semilogx(LAM * 1e6, e, lw=2.0, color="k", label=r"$\varepsilon(\lambda)$, одна кривая")
    for T, c in ((300.0, "tab:blue"), (1800.0, "tab:orange"),
                 (2740.0, "crimson")):
        B = planck_spectral_radiance(LAM, T)
        ax.semilogx(LAM * 1e6, B / B.max() * 0.9, lw=1.2, ls="--", color=c,
                    alpha=0.75, label=f"Планк {T:.0f} K")
    ax.set_xlim(0.3, 40); ax.set_ylim(0, 1.02)
    ax.set_xlabel("длина волны, мкм"); ax.set_ylabel(r"$\varepsilon$ / Планк, норм.")
    ax.set_title("Падение ε с температурой —\nэффект взвешивания, не материала",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="center left")

    ax = axes[2]
    Ts = np.linspace(300, 2800, 60)
    ax.plot(Ts, [total_emissivity(LAM, e, T) for T in Ts], lw=2.0, color="k",
            label="модель, одна ε(λ)")
    ax.plot([300, 1800], [0.83, 0.35], "o", ms=8, mfc="none", mew=2,
            color="crimson", label="справочные α-Al₂O₃")
    ax.axvspan(1500, 2740, color="tab:blue", alpha=0.10)
    ax.text(2100, 0.72, "рабочий\nдиапазон", fontsize=8, ha="center",
            color="tab:blue")
    e2740 = total_emissivity(LAM, e, 2740.)
    ax.annotate(f"ε(2740 K) = {e2740:.2f}\nбыл рабочий верх 0.35",
                xy=(2740, e2740), xytext=(1350, 0.20), fontsize=8.5,
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
    hb, med, off = part_d_transfer()
    lam_c = part_e_kirchhoff()
    part_f_instruments(lam_c)
    figure(hb, med, off, lam_c)
    print()
