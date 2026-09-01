"""Диагностика по итогам ревью шага 1. Запуск: python explore_step1b.py

НЕ реализация шага 4 — это what-if прогоны на существующей траекторной
модели, чтобы проверить три вещи до того, как менять план сборки:

  A. Где пикует НАГРЕВ (а не торможение). Высота пика sqrt(rho)*V^3 не
     зависит ни от k Саттона-Грейвса, ни от радиуса затупления, поэтому
     считается уже сейчас.
  B. Разброс по Cd = 1.0 / 1.5 / 2.2.
  C. Закрывает ли фрагментация разрыв в 25-30 км до наблюдаемых 70-80 км.

Наблюдательный репер: ATV-1 разрушился на 74 км, Cygnus OA6 на 70 км,
Cluster II (SALSA) на 80 км (Ferreira, UNOOSA/IAF 2024). Настройки SESAM
по умолчанию: отрыв панелей 95 км, разрушение корпуса 78 км (Lips, SDC6).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reentry import EntryState, ExponentialAtmosphere, Vehicle, integrate
from reentry.constants import G0

ATM = ExponentialAtmosphere()
ENTRY = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)

OBSERVED_LO, OBSERVED_HI = 70.0, 80.0   # км, наблюдённые события разрушения


def part_a_heating_peak():
    print("A. ГДЕ ПИКУЕТ НАГРЕВ, А НЕ ТОРМОЖЕНИЕ")
    print("   q ~ sqrt(rho)*V^3 против a ~ rho*V^2: скорость весит сильнее,")
    print("   плотность слабее, значит пик нагрева выше и раньше.\n")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    tr = integrate(veh, ENTRY, ATM)
    a, h_a, v_a = tr.peak_decel()
    h_q, v_q, t_q = tr.peak_heating()
    print(f"   пик торможения   {h_a/1e3:6.1f} км   V = {v_a:5.0f} м/с   "
          f"({a/G0:.1f} g)")
    print(f"   пик нагрева      {h_q/1e3:6.1f} км   V = {v_q:5.0f} м/с   "
          f"t = {t_q:.0f} с")
    print(f"   разнос           {(h_q - h_a)/1e3:6.1f} км\n")
    return h_q


def part_b_cd_sweep():
    print("B. РАЗБРОС ПО Cd")
    print("   1.0 — тупое тело в континууме, нижняя граница")
    print("   1.5 — беспорядочно кувыркающееся нерегулярное тело")
    print("   2.2 — свободномолекулярный предел / очень нерегулярное тело\n")
    print(f"   {'Cd':>5}{'beta, кг/м^2':>14}{'h торм., км':>14}"
          f"{'h нагрева, км':>16}{'макс g':>9}")
    out = {}
    for Cd in (1.0, 1.5, 2.2):
        veh = Vehicle(mass=175.0, area=1.0, Cd=Cd)
        tr = integrate(veh, ENTRY, ATM)
        a, h_a, _ = tr.peak_decel()
        h_q, _, _ = tr.peak_heating()
        out[Cd] = (h_a, h_q)
        print(f"   {Cd:>5.1f}{veh.ballistic_coefficient:>14.1f}"
              f"{h_a/1e3:>14.1f}{h_q/1e3:>16.1f}{a/G0:>9.1f}")
    span_a = (out[2.2][0] - out[1.0][0]) / 1e3
    span_q = (out[2.2][1] - out[1.0][1]) / 1e3
    print(f"\n   разброс по высоте: торможение {span_a:+.1f} км, "
          f"нагрев {span_q:+.1f} км")
    print("   -> неопределённость Cd стоит меньше, чем разрыв до 70-80 км\n")
    return out


def fragment_run(beta_ratio: float, h_frag: float, Cd: float = 1.5):
    """Целый объект до h_frag, затем осколки с beta / beta_ratio.

    При геометрически подобном дроблении m ~ L^3, A ~ L^2, значит
    beta = m/(Cd A) ~ L. Куски вдесятеро меньше по линейному размеру
    дают beta вдесятеро меньше.
    """
    intact = Vehicle(mass=175.0, area=1.0, Cd=Cd)
    tr0 = integrate(intact, ENTRY, ATM, h_stop=h_frag)
    restart = tr0.state_at_altitude(h_frag)

    # Сохраняем массу, режем beta через площадь: A_эфф = A * beta_ratio
    frag = Vehicle(mass=175.0, area=1.0 * beta_ratio, Cd=Cd)
    tr1 = integrate(frag, restart, ATM)
    return tr0, tr1, frag


def part_c_fragmentation():
    print("C. ЗАКРЫВАЕТ ЛИ ФРАГМЕНТАЦИЯ РАЗРЫВ")
    print(f"   наблюдения: разрушение на {OBSERVED_LO:.0f}-{OBSERVED_HI:.0f} км "
          "(ATV-1 74, Cygnus OA6 70, SALSA 80)")
    print("   SESAM по умолчанию: панели 95 км, корпус 78 км\n")

    print("   C1. Разрушение на 78 км, разное дробление (Cd = 1.5):")
    print(f"   {'beta осколка':>14}{'отношение':>12}{'h нагрева, км':>16}"
          f"{'в окне?':>10}")
    for ratio in (1, 2, 5, 10, 20, 50):
        _, tr1, frag = fragment_run(ratio, 78e3)
        h_q, _, _ = tr1.peak_heating()
        inside = OBSERVED_LO <= h_q / 1e3 <= OBSERVED_HI
        clipped = "  <- пик НА разрушении" if h_q > 77.9e3 else ""
        print(f"   {frag.ballistic_coefficient:>14.1f}{'1/' + str(ratio):>12}"
              f"{h_q/1e3:>16.1f}{'да' if inside else '':>10}{clipped}")

    print("\n   C2. Дробление в 10 раз, разная высота разрушения:")
    print(f"   {'h разруш., км':>15}{'h нагрева, км':>16}{'сдвиг':>10}")
    base = None
    for hf in (95e3, 84e3, 78e3, 70e3):
        _, tr1, _ = fragment_run(10, hf)
        h_q, _, _ = tr1.peak_heating()
        if base is None:
            base = h_q
        print(f"   {hf/1e3:>15.0f}{h_q/1e3:>16.1f}{(h_q-base)/1e3:>+10.1f}")
    print()

    # Обратная задача: какой beta осколка кладёт пик нагрева в середину окна
    target = 0.5 * (OBSERVED_LO + OBSERVED_HI) * 1e3
    ratios = np.logspace(0, 2.3, 40)
    heights = []
    for r in ratios:
        _, tr1, _ = fragment_run(float(r), 78e3)
        heights.append(tr1.peak_heating()[0])
    heights = np.array(heights)
    i = int(np.argmin(np.abs(heights - target)))
    beta_needed = 175.0 / (1.5 * ratios[i])
    beta_intact = 175.0 / 1.5
    factor = beta_intact / beta_needed
    print(f"   C3. Обратная задача: чтобы пик нагрева лёг на {target/1e3:.0f} км,")
    print(f"       нужен beta осколка ~ {beta_needed:.1f} кг/м^2 против "
          f"{beta_intact:.1f} у целого,")
    print(f"       то есть падение beta в {factor:.1f} раза.")
    print(f"       При геометрически подобном дроблении beta ~ L, а L ~ N^(-1/3),")
    print(f"       значит нужно N ~ {factor**3:.0f} равных осколков.")
    print(f"       Это много. Реальный разброс beta даёт не равное дробление,")
    print(f"       а спектр: панели и MLI имеют beta ~1-10, силовой набор ~30-80.\n")

    print("   ВАЖНАЯ ОГОВОРКА: пик sqrt(rho)*V^3 — это ВЕРХНЯЯ ГРАНИЦА высоты")
    print("   вброса, а не сам вброс. Испарение требует НАКОПЛЕННОГО тепла:")
    print("   сначала прогрев до 930 K, потом плавление, потом кипение при 2740 K.")
    print("   Масса пойдёт ниже пика потока. Насколько — покажет шаг 4.")
    print("   Поэтому подгонять beta так, чтобы пик потока лёг ровно в 70-80 км,")
    print("   было бы ошибкой: тогда сама масса окажется слишком низко.\n")
    return ratios, heights


def figure(ratios, heights, path="step1b_fragmentation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # 1: форма нагрева vs торможения для целого объекта
    ax = axes[0]
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.5)
    tr = integrate(veh, ENTRY, ATM)
    q = tr.heat_flux_shape / tr.heat_flux_shape.max()
    a = tr.decel / tr.decel.max()
    ax.plot(q, tr.h / 1e3, lw=1.7, label=r"нагрев $\sqrt{\rho}V^3$")
    ax.plot(a, tr.h / 1e3, lw=1.7, ls="--", label=r"торможение $\rho V^2$")
    ax.axhspan(OBSERVED_LO, OBSERVED_HI, color="seagreen", alpha=0.15)
    ax.text(0.5, (OBSERVED_LO + OBSERVED_HI) / 2, "наблюдаемое\nразрушение",
            fontsize=7.5, color="seagreen", ha="center", va="center")
    ax.set_xlabel("нормировано на максимум"); ax.set_ylabel("высота, км")
    ax.set_title("целый объект: оба пика\nсильно ниже наблюдений", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(30, 120)

    # 2: траектории осколков разного дробления
    ax = axes[1]
    for ratio, color in zip((1, 5, 20, 50),
                            ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        _, tr1, frag = fragment_run(ratio, 78e3)
        qq = tr1.heat_flux_shape / tr1.heat_flux_shape.max()
        ax.plot(qq, tr1.h / 1e3, lw=1.6, color=color,
                label=fr"$\beta$={frag.ballistic_coefficient:.0f}")
    ax.axhspan(OBSERVED_LO, OBSERVED_HI, color="seagreen", alpha=0.15)
    ax.axhline(78, color="k", ls=":", lw=1)
    ax.text(0.02, 79, "разрушение 78 км", fontsize=7.5)
    ax.set_xlabel("нагрев, нормирован"); ax.set_ylabel("высота, км")
    ax.set_title("осколки после разрушения на 78 км", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_ylim(30, 100)

    # 3: высота пика нагрева как функция beta осколка
    ax = axes[2]
    betas = 175.0 / (1.5 * ratios)
    ax.semilogx(betas, heights / 1e3, lw=1.8)
    ax.axhspan(OBSERVED_LO, OBSERVED_HI, color="seagreen", alpha=0.15,
               label="наблюдения 70-80 км")
    ax.plot(175 / 1.5, heights[0] / 1e3, "o", color="crimson", ms=7,
            label="целый объект")
    ax.set_xlabel(r"$\beta$ осколка, кг/м$^2$")
    ax.set_ylabel("высота пика нагрева, км")
    ax.set_title(r"наклон $H\ln\beta$, затем полка" "\n" r"на высоте разрушения", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Диагностика: фрагментация как механизм, поднимающий высоту вброса",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"   сохранено: {path}")


if __name__ == "__main__":
    print()
    part_a_heating_peak()
    part_b_cd_sweep()
    ratios, heights = part_c_fragmentation()
    figure(ratios, heights)
    print()
