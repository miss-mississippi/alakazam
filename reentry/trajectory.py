"""Траектория входа: планарные уравнения движения и интегрирование.

Система координат: связанная с планетой полярная, состояние [V, gamma, h, s].
Земля сферическая и невращающаяся. Подъёмной и боковой силы нет, поэтому
движение СТРОГО планарное — это не приближение, а следствие допущений.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import H_SCALE_FIT, MU_EARTH, R_EARTH, RHO0_SEA_LEVEL
from .vehicle import EntryState, Vehicle

# Индексы вектора состояния
I_V, I_GAMMA, I_H, I_S = 0, 1, 2, 3


def eom(t, y, vehicle: Vehicle, atmosphere, v_corot: float = 0.0):
    """Правая часть системы.

        dV/dt     = -(1/2) rho V^2 Cd A / m  -  (mu/r^2) sin(gamma)
        dgamma/dt = cos(gamma) * ( V/r  -  mu/(r^2 V) )
        dh/dt     = V sin(gamma)
        ds/dt     = V cos(gamma) * R_E/r

    Второе уравнение — ядро задачи. V/r — центробежный член,
    mu/(r^2 V) — гравитационный. При околоорбитальной скорости они почти
    сокращаются, поэтому gamma эволюционирует медленно, и траектория
    чувствительна к gamma0.
    """
    V, gamma, h, s = y

    r = R_EARTH + h
    g = MU_EARTH / (r * r)
    rho = atmosphere.density(h)

    # Сопротивление и нагрев зависят от скорости ОТНОСИТЕЛЬНО АТМОСФЕРЫ,
    # а она вращается вместе с Землёй. v_corot = omega*R*cos(i), см.
    # EntryState.corotation_speed. При i=53 это 280 м/с, то есть 3.7% в V
    # и ~11% в тепловом потоке, потому что q ~ V^3.
    V_rel = V - v_corot

    drag_decel = 0.5 * rho * V_rel * V_rel * vehicle.Cd * vehicle.area / vehicle.mass

    dV = -drag_decel - g * np.sin(gamma)
    dgamma = np.cos(gamma) * (V / r - g / V)
    dh = V * np.sin(gamma)
    ds = V * np.cos(gamma) * R_EARTH / r

    return np.array([dV, dgamma, dh, ds])


def _make_events(h_stop: float, v_stop: float):
    def hit_floor(t, y, *args):
        return y[I_H] - h_stop
    hit_floor.terminal = True
    hit_floor.direction = -1

    def too_slow(t, y, *args):
        return y[I_V] - v_stop
    too_slow.terminal = True
    too_slow.direction = -1

    return [hit_floor, too_slow]


@dataclass
class TrajectoryResult:
    t: np.ndarray
    V: np.ndarray
    gamma: np.ndarray
    h: np.ndarray
    s: np.ndarray
    rho: np.ndarray
    decel: np.ndarray          # полное аэродинамическое замедление, м/с^2
    stop_reason: str
    raw: object                # объект solve_ivp, если нужен dense_output
    V_rel: np.ndarray = None   # скорость относительно вращающейся атмосферы

    @property
    def gamma_deg(self):
        return np.rad2deg(self.gamma)

    def peak_decel(self):
        """(a_max [м/с^2], h [м], V [м/с]) в точке максимального торможения."""
        i = int(np.argmax(self.decel))
        return self.decel[i], self.h[i], self.V[i]

    @property
    def heat_flux_shape(self):
        """sqrt(rho)*V^3 — ФОРМА теплового потока Саттона-Грейвса без констант.

        q = k*sqrt(rho/Rn)*V^3, и k, Rn — константы вдоль траектории.
        Значит ВЫСОТА пика нагрева не зависит ни от k, ни от радиуса
        затупления, ни от размерности. Её можно считать уже на шаге 1;
        абсолютную величину потока — только на шаге 3.
        """
        V = self.V_rel if self.V_rel is not None else self.V
        return np.sqrt(self.rho) * V ** 3

    def heat_flux(self, vehicle, correlation="sutton-graves") -> np.ndarray:
        """Тепловой поток в точке торможения вдоль траектории, Вт/м^2."""
        from .heating import detra_kemp_riddell_shape, sutton_graves
        V = self.V_rel if self.V_rel is not None else self.V
        f = sutton_graves if correlation == "sutton-graves" else detra_kemp_riddell_shape
        return f(self.rho, V, vehicle.nose_radius)

    def heat_load(self, vehicle, correlation="sutton-graves") -> float:
        """Интегральный тепловой поток, Дж/м^2. Именно он греет материал."""
        return float(np.trapezoid(self.heat_flux(vehicle, correlation), self.t))

    def peak_heating(self):
        """(h [м], V [м/с], t [с]) в точке максимума sqrt(rho)*V^3."""
        i = int(np.argmax(self.heat_flux_shape))
        return self.h[i], self.V[i], self.t[i]

    def state_at_altitude(self, h_target: float, inclination_deg: float = 53.0):
        """EntryState на заданной высоте — точка рестарта после фрагментации.

        Наклонение надо передавать явно: оно не восстанавливается из
        состояния [V, gamma, h, s], а от него зависит соатмосферный снос.
        """
        i = int(np.argmin(np.abs(self.h - h_target)))
        return EntryState(altitude=float(self.h[i]),
                          velocity=float(self.V[i]),
                          gamma_deg=float(self.gamma_deg[i]),
                          inclination_deg=inclination_deg)


def integrate(
    vehicle: Vehicle,
    entry: EntryState,
    atmosphere,
    h_stop: float = 30.0e3,
    v_stop: float = 300.0,
    earth_rotation: bool = True,
    t_max: float = 3000.0,
    max_step: float = 2.0,
    rtol: float = 1e-8,
) -> TrajectoryResult:
    """Интегрирует вход от начальных условий до h_stop или v_stop.

    max_step=2 с — страховка, а не точность. На первых ~100 секундах
    торможение пренебрежимо, интегратор разгоняет шаг до сотен секунд и
    рискует перепрыгнуть включение сопротивления около 100 км.

    v_stop=300 м/с: ниже Mach ~1 вся гиперзвуковая физика (и Саттон-Грейвс
    на шаге 3) неприменима.

    atol задан покомпонентно: скорость в м/с, угол в рад, высоты в м —
    один скаляр на разномасштабные величины давать нельзя.
    """
    y0 = entry.to_vector()
    atol = np.array([1e-3, 1e-9, 1e-3, 1e-3])
    v_corot = entry.corotation_speed if earth_rotation else 0.0

    sol = solve_ivp(
        eom,
        t_span=(0.0, t_max),
        y0=y0,
        args=(vehicle, atmosphere, v_corot),
        method="DOP853",
        events=_make_events(h_stop, v_stop),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True,
    )

    # Пересэмплируем на равномерную мелкую сетку через dense_output.
    # Без этого argmax по узлам солвера зависит от max_step: при max_step=5
    # высота пика торможения гуляет на ~1 км чисто от разрешения сетки.
    t = np.arange(0.0, sol.t[-1], 0.1)
    if t.size == 0 or t[-1] < sol.t[-1]:
        t = np.append(t, sol.t[-1])
    V, gamma, h, s = sol.sol(t)

    rho = atmosphere.density(h)
    V_rel = V - v_corot
    decel = 0.5 * rho * V_rel ** 2 * vehicle.Cd * vehicle.area / vehicle.mass

    if sol.t_events[0].size:
        reason = f"достигнута высота {h_stop/1e3:.0f} км"
    elif sol.t_events[1].size:
        reason = f"скорость упала до {v_stop:.0f} м/с"
    else:
        reason = f"истёк лимит времени {t_max:.0f} с"

    return TrajectoryResult(t, V, gamma, h, s, rho, decel, reason, sol, V_rel)


# ---------------------------------------------------------------------------
# Аналитика Аллена-Эггерса — только для сверки, в расчёте не участвует.
# ---------------------------------------------------------------------------

def allen_eggers(vehicle: Vehicle, entry: EntryState,
                 rho0: float = RHO0_SEA_LEVEL, H: float = H_SCALE_FIT) -> dict:
    """Классическое замкнутое решение (Allen & Eggers, NACA TR-1381, 1958).

    Допущения: gamma = const, сопротивление >> гравитации, экспоненциальная
    атмосфера, постоянный баллистический коэффициент.

    rho0 и H передаются ЯВНО, а не берутся из объекта атмосферы. Причина:
    А-Э определён только для экспоненциальной атмосферы, и обёртка над
    NRLMSISE-00 никакого rho0 иметь не будет. Сверка сознательно привязана
    к заглушке — на реальной атмосфере формула всё равно неприменима.

    У нас нарушены ПЕРВЫЕ ДВА: gamma успевает измениться вдвое, а первые
    100 с полёт вообще бездрагвый. Поэтому ожидание — совпадение в пределах
    20-30%, а не 1%. Расхождение в 2 раза и больше означает баг в коде.

    Три цели:
      V*    = V0/sqrt(e) = 0.6065*V0   — не зависит НИ ОТ ЧЕГО
      a_max = V0^2 sin|gamma| / (2 e H) — не зависит от beta
      rho*  = beta sin|gamma| / H      -> h* = -H ln(rho*/rho0)
    """
    V0 = entry.velocity
    sin_g = abs(np.sin(entry.gamma_rad))
    beta = vehicle.ballistic_coefficient

    v_at_peak = V0 / np.sqrt(np.e)
    a_max = V0 ** 2 * sin_g / (2 * np.e * H)
    rho_at_peak = beta * sin_g / H
    h_at_peak = -H * np.log(rho_at_peak / rho0)

    return {
        "V_at_peak": v_at_peak,
        "a_max": a_max,
        "rho_at_peak": rho_at_peak,
        "h_at_peak": h_at_peak,
    }
