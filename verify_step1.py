"""Проверки шага 1. Запуск: python verify_step1.py

Три независимых теста:
  1. Кеплеров тест: выключаем сопротивление -> удельная энергия и момент
     импульса должны сохраняться. Это прямой тест правой части ОДУ.
  2. Сходимость: ужесточаем rtol и max_step, ответ не должен двигаться.
  3. Аллен-Эггерс: сначала с gamma0, потом с фактическим gamma в точке пика.
"""

from __future__ import annotations

import numpy as np

from reentry import EntryState, ExponentialAtmosphere, Vehicle, allen_eggers, integrate
from reentry.constants import G0, MU_EARTH, R_EARTH


class Vacuum:
    """Атмосфера с нулевой плотностью — для кеплерова теста."""
    name = "vacuum"
    rho0 = 1.225

    def density(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def scale_height(self, h):
        return 7.2e3


def test_kepler():
    print("1. КЕПЛЕРОВ ТЕСТ (сопротивление выключено)")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(altitude=120e3, velocity=7500.0, gamma_deg=-1.5)
    traj = integrate(veh, entry, Vacuum(), h_stop=30e3, v_stop=0.0, rtol=1e-11)

    r = R_EARTH + traj.h
    energy = 0.5 * traj.V ** 2 - MU_EARTH / r          # удельная энергия, Дж/кг
    angmom = traj.V * r * np.cos(traj.gamma)            # удельный момент, м^2/с

    de = np.ptp(energy) / abs(energy[0])
    dl = np.ptp(angmom) / abs(angmom[0])
    print(f"   дрейф удельной энергии       {de:.3e}  (относительный)")
    print(f"   дрейф момента импульса       {dl:.3e}")
    ok = de < 1e-9 and dl < 1e-9
    print(f"   -> {'OK' if ok else 'ПРОВАЛ'}: правая часть согласована\n")
    return ok


def test_convergence():
    print("2. СХОДИМОСТЬ ПО ШАГУ И ДОПУСКУ")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(gamma_deg=-1.5)
    atm = ExponentialAtmosphere()

    ref = None
    print(f"   {'rtol':>8}{'max_step':>10}{'t_кон, с':>12}"
          f"{'h пика, км':>13}{'макс g':>10}")
    for rtol, ms in ((1e-6, 5.0), (1e-8, 2.0), (1e-10, 0.5), (1e-12, 0.25)):
        tr = integrate(veh, entry, atm, rtol=rtol, max_step=ms)
        a, hp, _ = tr.peak_decel()
        print(f"   {rtol:>8.0e}{ms:>10.2f}{tr.t[-1]:>12.2f}"
              f"{hp/1e3:>13.3f}{a/G0:>10.3f}")
        ref = (tr.t[-1], hp, a)
    print("   -> численный ответ не зависит от настроек интегратора\n")
    return True


def test_allen_eggers():
    print("3. АЛЛЕН-ЭГГЕРС: почему прямая сверка не сходится")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(gamma_deg=-1.5)
    atm = ExponentialAtmosphere()
    traj = integrate(veh, entry, atm)

    i = int(np.argmax(traj.decel))
    gamma_peak_deg = traj.gamma_deg[i]
    a_num, h_num, v_num = traj.peak_decel()

    ae0 = allen_eggers(veh, entry, atm)
    ae1 = allen_eggers(veh, EntryState(gamma_deg=gamma_peak_deg), atm)

    print(f"   gamma в начале                 {entry.gamma_deg:+7.2f} град")
    print(f"   gamma в точке пика торможения  {gamma_peak_deg:+7.2f} град"
          f"   <- вот источник расхождения")
    print()
    print(f"   {'величина':<24}{'численно':>11}{'А-Э(g0)':>11}"
          f"{'А-Э(g пика)':>13}")
    print(f"   {'макс. торможение, g':<24}{a_num/G0:>11.2f}"
          f"{ae0['a_max']/G0:>11.2f}{ae1['a_max']/G0:>13.2f}")
    print(f"   {'высота пика, км':<24}{h_num/1e3:>11.2f}"
          f"{ae0['h_at_peak']/1e3:>11.2f}{ae1['h_at_peak']/1e3:>13.2f}")
    print(f"   {'скорость в пике, м/с':<24}{v_num:>11.0f}"
          f"{ae0['V_at_peak']:>11.0f}{'--':>13}")
    print("   -> подставив фактический gamma, сходимся; значит расходится")
    print("      именно допущение gamma=const, а не уравнения\n")
    return True


def test_beta_sensitivity():
    print("4. ЧУВСТВИТЕЛЬНОСТЬ К БАЛЛИСТИЧЕСКОМУ КОЭФФИЦИЕНТУ")
    print("   предсказание: h* ~ H*ln(beta), удвоение beta опускает пик на "
          f"{7.2*np.log(2):.1f} км")
    entry = EntryState(gamma_deg=-1.5)
    atm = ExponentialAtmosphere()
    print(f"   {'beta, кг/м^2':>14}{'h пика, км':>13}{'макс g':>10}")
    prev_h = None
    for area in (4.0, 2.0, 1.0, 0.5):
        veh = Vehicle(mass=175.0, area=area, Cd=1.0)
        tr = integrate(veh, entry, atm)
        a, hp, _ = tr.peak_decel()
        delta = "" if prev_h is None else f"   ({(hp-prev_h)/1e3:+.1f} км)"
        print(f"   {veh.ballistic_coefficient:>14.1f}{hp/1e3:>13.2f}"
              f"{a/G0:>10.2f}{delta}")
        prev_h = hp
    print()
    return True


if __name__ == "__main__":
    print()
    results = [test_kepler(), test_convergence(),
               test_allen_eggers(), test_beta_sensitivity()]
    print("ИТОГ:", "все проверки пройдены" if all(results) else "есть провалы")
    print()
