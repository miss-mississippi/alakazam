"""Проверки шага 1. Запуск: python verify_step1.py  (или pytest)

1. Кеплеров тест: без сопротивления удельная энергия и момент импульса
   сохраняются. Прямой тест правой части ОДУ.
2. Сходимость: ужесточаем rtol и max_step, ответ не должен двигаться.
3. Аллен-Эггерс: с gamma0 не сходится, с фактическим gamma в пике — сходится
   по высоте. Значит расходится допущение gamma = const, а не уравнения.
4. Чувствительность к beta: h* ~ H ln(beta), удвоение beta опускает пик на H ln2.
"""

from __future__ import annotations

import numpy as np

from reentry import EntryState, ExponentialAtmosphere, Vehicle, allen_eggers, integrate
from reentry.constants import G0, H_SCALE_FIT, MU_EARTH, R_EARTH
from reentry.results import Recorder

# Шаги 1-2 определены для невращающейся Земли; integrate() по умолчанию
# вращение включает, поэтому здесь оно выключено явно.
NO_ROT = dict(earth_rotation=False)
R = Recorder("verify_step1")


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
    traj = integrate(veh, entry, Vacuum(), h_stop=30e3, v_stop=0.0, rtol=1e-11,
                     **NO_ROT)

    r = R_EARTH + traj.h
    energy = 0.5 * traj.V ** 2 - MU_EARTH / r          # удельная энергия, Дж/кг
    angmom = traj.V * r * np.cos(traj.gamma)            # удельный момент, м^2/с

    de = np.ptp(energy) / abs(energy[0])
    dl = np.ptp(angmom) / abs(angmom[0])
    print(f"   дрейф удельной энергии       {de:.3e}  (относительный)")
    print(f"   дрейф момента импульса       {dl:.3e}\n")
    R["kepler.energy_drift"] = de
    R["kepler.angmom_drift"] = dl
    assert de < 1e-9 and dl < 1e-9, f"дрейф {de:.1e}, {dl:.1e}"


def test_convergence():
    print("2. СХОДИМОСТЬ ПО ШАГУ И ДОПУСКУ")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(gamma_deg=-1.5)
    atm = ExponentialAtmosphere()
    print(f"   {'rtol':>8}{'max_step':>10}{'t_кон, с':>12}"
          f"{'h пика, км':>13}{'макс g':>10}")
    hs, ts = [], []
    for rtol, ms in ((1e-6, 5.0), (1e-8, 2.0), (1e-10, 0.5), (1e-12, 0.25)):
        tr = integrate(veh, entry, atm, rtol=rtol, max_step=ms, **NO_ROT)
        a, hp, _ = tr.peak_decel()
        hs.append(hp); ts.append(tr.t[-1])
        print(f"   {rtol:>8.0e}{ms:>10.2f}{tr.t[-1]:>12.2f}"
              f"{hp/1e3:>13.3f}{a/G0:>10.3f}")
    print()
    R["convergence.h_peak_km"] = hs[-1] / 1e3
    R["convergence.h_spread_m"] = max(hs) - min(hs)
    assert max(hs) - min(hs) < 1.0, f"высота пика гуляет на {max(hs)-min(hs):.2f} м"
    assert max(ts) - min(ts) < 0.05, f"время гуляет на {max(ts)-min(ts):.3f} с"


def test_allen_eggers():
    print("3. АЛЛЕН-ЭГГЕРС: почему прямая сверка не сходится")
    veh = Vehicle(mass=175.0, area=1.0, Cd=1.0)
    entry = EntryState(gamma_deg=-1.5)
    atm = ExponentialAtmosphere()
    traj = integrate(veh, entry, atm, **NO_ROT)

    i = int(np.argmax(traj.decel))
    gamma_peak_deg = traj.gamma_deg[i]
    a_num, h_num, v_num = traj.peak_decel()

    ae0 = allen_eggers(veh, entry)
    ae1 = allen_eggers(veh, EntryState(gamma_deg=gamma_peak_deg))

    print(f"   gamma в начале                 {entry.gamma_deg:+7.2f} град")
    print(f"   gamma в точке пика торможения  {gamma_peak_deg:+7.2f} град\n")
    print(f"   {'величина':<24}{'численно':>11}{'А-Э(g0)':>11}{'А-Э(g пика)':>13}")
    print(f"   {'макс. торможение, g':<24}{a_num/G0:>11.2f}"
          f"{ae0['a_max']/G0:>11.2f}{ae1['a_max']/G0:>13.2f}")
    print(f"   {'высота пика, км':<24}{h_num/1e3:>11.2f}"
          f"{ae0['h_at_peak']/1e3:>11.2f}{ae1['h_at_peak']/1e3:>13.2f}")
    print(f"   {'скорость в пике, м/с':<24}{v_num:>11.0f}"
          f"{ae0['V_at_peak']:>11.0f}{'--':>13}\n")
    R["ae.gamma_peak_deg"] = gamma_peak_deg
    R["ae.num.amax_g"] = a_num / G0
    R["ae.num.h_km"] = h_num / 1e3
    R["ae.num.V"] = v_num
    R["ae.g0.amax_g"] = ae0["a_max"] / G0
    R["ae.g0.h_km"] = ae0["h_at_peak"] / 1e3
    R["ae.g0.V"] = ae0["V_at_peak"]
    R["ae.gpeak.amax_g"] = ae1["a_max"] / G0
    R["ae.gpeak.h_km"] = ae1["h_at_peak"] / 1e3
    R["ae.amax_dev_pct"] = 100 * (a_num / ae0["a_max"] - 1)
    dh = abs(h_num - ae1["h_at_peak"])
    assert dh < 0.5e3, f"с фактическим gamma высота расходится на {dh/1e3:.2f} км"
    assert abs(h_num - ae0["h_at_peak"]) > 5e3, "с gamma0 ожидалось расхождение"


def test_beta_sensitivity():
    print("4. ЧУВСТВИТЕЛЬНОСТЬ К БАЛЛИСТИЧЕСКОМУ КОЭФФИЦИЕНТУ")
    shift = H_SCALE_FIT * np.log(2) / 1e3
    print(f"   предсказание: h* ~ H*ln(beta), удвоение beta опускает пик на {shift:.1f} км")
    entry = EntryState(gamma_deg=-1.5)
    atm = ExponentialAtmosphere()
    print(f"   {'beta, кг/м^2':>14}{'h пика, км':>13}{'макс g':>10}")
    hs, gs = [], []
    for k, area in enumerate((4.0, 2.0, 1.0, 0.5)):
        veh = Vehicle(mass=175.0, area=area, Cd=1.0)
        tr = integrate(veh, entry, atm, **NO_ROT)
        a, hp, _ = tr.peak_decel()
        delta = "" if not hs else f"   ({(hp - hs[-1])/1e3:+.1f} км)"
        print(f"   {veh.ballistic_coefficient:>14.1f}{hp/1e3:>13.2f}{a/G0:>10.2f}{delta}")
        R[f"beta.{k}.beta"] = veh.ballistic_coefficient
        R[f"beta.{k}.h_km"] = hp / 1e3
        R[f"beta.{k}.amax_g"] = a / G0
        hs.append(hp); gs.append(a)
    print()
    steps = -np.diff(hs) / 1e3
    for k, s in enumerate(steps):
        R[f"beta_step.{k}"] = -s
    assert np.all(np.abs(steps - shift) < 0.3), f"шаги {steps} против {shift:.2f}"
    assert max(gs) / min(gs) - 1 < 0.10, "перегрузка не должна зависеть от beta"


if __name__ == "__main__":
    from reentry.checks import run_checks
    raise SystemExit(run_checks(globals(), R, __file__))
