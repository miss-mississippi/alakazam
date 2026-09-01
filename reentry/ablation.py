"""Тепловой отклик и абляция: две границы вместо одной оценки.

ГЛАВНОЕ АРХИТЕКТУРНОЕ РЕШЕНИЕ. Модель НЕ выдаёт одно число испарённой массы.
Она выдаёт вилку:

  ВЕРХНЯЯ ГРАНИЦА — расплавленная масса. Критерий ORSAT/DRAMA: поглощённая
  энергия против теплоты плавления (у ORSAT для generic aluminum
  heat of ablation = 934.5 кДж/кг, что и есть нагрев до плавления плюс
  плавление). Это ВСЁ, что в принципе может стать оксидом.

  НИЖНЯЯ ГРАНИЦА — испарённая НА МЕСТЕ масса. Локальный температурный
  критерий: испаряется та часть поверхности, где местный поток превышает
  то, что излучение способно отвести при температуре кипения.

Между ними — судьба расплава, сорванного сдвигом в поток. Капля может
испариться дальше по траектории, может окислиться только по поверхности и
выпасть миллиметровой сферулой (так находят абляционные сферулы метеороидов
в глубоководных осадках), может застыть целиком. Ни одна текущая модель
этого не разрешает, и именно поэтому вилка — результат, а не компромисс.

Отношение границ — 14x по энергии (0.97 против 13.6 МДж/кг).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .heating import (SIGMA_SB, hot_wall_factor, local_flux_fraction,
                      vaporizing_area_fraction, vaporization_rate)


@dataclass
class Aluminium:
    """Свойства Al 6061. Каждое — с источником и неопределённостью.

    ЭМИССИВНОСТЬ — самая слабая константа модели, и она же решает исход.
    Тонкость: при 1900 K излучает НЕ окисленный твёрдый алюминий, а расплав
    с оксидной плёнкой. Это другой материал, и тренд у него другой.

      полированный Al       0.04-0.07  (300-900 K)
      окисленный Al         0.07-0.09  (450-900 K)
      слабо окисленный Al   0.11-0.19  (473-873 K)
      альфа-Al2O3           0.83 -> 0.35  (300 -> 1800 K)   ПАДАЕТ с ростом T

    То есть у оксида эмиссивность с температурой ПАДАЕТ, а не растёт, и при
    1900 K оксидная плёнка даёт около 0.3-0.35. Голый жидкий алюминий под ней —
    блестящий жидкий металл, ~0.05. Физическая вилка: 0.05...0.35, фактор 7.

    Порог кипения прямо пропорционален eps, поэтому eps — ГЛАВНАЯ ОСЬ СВИПА,
    а не константа с оговоркой. Результат подаётся как функция eps.
    """

    c_p: float = 900.0            # Дж/(кг*К), ~10%
    T_melt: float = 933.0         # K
    T_boil: float = 2740.0        # K
    L_fusion: float = 0.397e6     # Дж/кг, ~5%
    L_vapour: float = 10.5e6      # Дж/кг, ~5%
    T_initial: float = 300.0      # K
    emissivity: float = 0.3       # ГЛАВНАЯ ОСЬ СВИПА, диапазон 0.05-0.35

    @property
    def h_melt_complete(self) -> float:
        """Энергия на килограмм до полного расплавления, Дж/кг.

        Критерий ORSAT: 934.5 кДж/кг для generic aluminum. Наш расчёт даёт
        900*(933-300) + 397e3 = 0.97 МДж/кг — совпадение в пределах 4%,
        то есть мы воспроизводим их критерий, а не изобретаем свой.
        """
        return self.c_p * (self.T_melt - self.T_initial) + self.L_fusion

    @property
    def h_vapour_complete(self) -> float:
        """Энергия на килограмм до полного испарения, Дж/кг."""
        return (self.h_melt_complete
                + self.c_p * (self.T_boil - self.T_melt) + self.L_vapour)

    @property
    def demise_ratio(self) -> float:
        """Во сколько раз испарение дороже плавления."""
        return self.h_vapour_complete / self.h_melt_complete


def temperature_from_enthalpy(h_s, mat: "Aluminium"):
    """T(h_s) — монотонная кусочная функция с ПОЛКАМИ на фазовых переходах.

    Именно так фазовые переходы попадают в правую часть ОДУ без ветвления:
    состояние — удельная энтальпия, а температура из неё выводится.

        h1 = c_p*(T_пл - T0)               начало плавления
        h2 = h1 + L_пл                     конец плавления
        h3 = h2 + c_p*(T_кип - T_пл)       начало кипения
    """
    h1 = mat.c_p * (mat.T_melt - mat.T_initial)
    h2 = h1 + mat.L_fusion
    h3 = h2 + mat.c_p * (mat.T_boil - mat.T_melt)
    h = np.asarray(h_s, dtype=float)
    return np.where(
        h < h1, mat.T_initial + h / mat.c_p,
        np.where(h < h2, mat.T_melt,
                 np.where(h < h3, mat.T_melt + (h - h2) / mat.c_p, mat.T_boil)))


def thermal_history(traj, vehicle, material: "Aluminium",
                    correlation: str = "sutton-graves") -> dict:
    """ВЕРХНЯЯ ГРАНИЦА: сосредоточенная тепловая модель, честно проинтегрированная.

    ПОЧЕМУ НЕЛЬЗЯ ПРОСТО ПОДЕЛИТЬ ЭНЕРГИЮ НА ТЕПЛОТУ ПЛАВЛЕНИЯ. Переизлучение
    зависит от температуры, температура — от накопленной энтальпии. Отношение
    "поглощённая энергия / теплота плавления" не зависит от eps вообще, а
    физически eps решает всё: он задаёт равновесную температуру.

    Постоянная времени тут не запас: для целого объекта m*c_p = 1.6e5 Дж/К,
    приход ~9e5 Вт, то есть ~6 К/с, и до плавления надо ~110 с при полётном
    времени высокого потока ~100 с. Плавление у крупного тела МАРГИНАЛЬНО,
    и это видно только при честном интегрировании.

        dh_s/dt = (A_омыв/m) * (phi*q_stag*(1-h_w/h_0) - eps*sigma*T(h_s)^4)
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline

    from .heating import SHAPE_FACTOR_TUMBLING

    V = traj.V_rel if traj.V_rel is not None else traj.V
    q_raw = traj.heat_flux(vehicle, correlation)
    q_spline = CubicSpline(traj.t, q_raw)
    V_spline = CubicSpline(traj.t, V)

    A_over_m = vehicle.wetted / vehicle.mass
    eps_sig = material.emissivity * SIGMA_SB

    def rhs(t, y):
        T = float(temperature_from_enthalpy(y[0], material))
        q = float(q_spline(t)) * float(hot_wall_factor(float(V_spline(t)), T))
        return [A_over_m * (SHAPE_FACTOR_TUMBLING * max(q, 0.0) - eps_sig * T ** 4)]

    sol = solve_ivp(rhs, (traj.t[0], traj.t[-1]), [0.0], method="LSODA",
                    t_eval=traj.t, rtol=1e-8, atol=1e-3, max_step=2.0)
    h_s = np.maximum(sol.y[0], 0.0)
    T = temperature_from_enthalpy(h_s, material)

    h1 = material.c_p * (material.T_melt - material.T_initial)
    h2 = h1 + material.L_fusion
    # Доля массы, доведённая до расплава: полка плавления пройдена целиком
    # -> 100%; пройдена частично -> линейная доля.
    hmax = float(h_s.max())
    frac = 0.0 if hmax <= h1 else min(1.0, (hmax - h1) / material.L_fusion)
    return {"t": traj.t, "h_s": h_s, "T": T, "T_peak": float(T.max()),
            "fraction": frac, "mass": frac * vehicle.mass,
            "specific": hmax}


def melted_fraction(traj, vehicle, material: Aluminium,
                    correlation: str = "sutton-graves") -> dict:
    """Совместимая обёртка над thermal_history."""
    return thermal_history(traj, vehicle, material, correlation)


def vaporized_mass(traj, vehicle, material: Aluminium,
                   correlation: str = "sutton-graves") -> dict:
    """НИЖНЯЯ ГРАНИЦА: масса, испарённая НА МЕСТЕ.

    Локальный температурный критерий по УГЛОВОМУ РАСПРЕДЕЛЕНИЮ потока,
    а не по среднему. Среднее в пороговой задаче с T^4 систематически
    занижает: испарение идёт там, где поток выше среднего, а среднее
    этого места не видит.
    """
    V = traj.V_rel if traj.V_rel is not None else traj.V
    q_s = traj.heat_flux(vehicle, correlation) * hot_wall_factor(V, material.T_boil)

    mdot = vaporization_rate(q_s, vehicle.wetted, material.emissivity,
                             material.T_boil, material.L_vapour)
    m_vap = float(np.trapezoid(mdot, traj.t))
    frac_area = vaporizing_area_fraction(q_s, material.emissivity, material.T_boil)
    return {"mass": min(m_vap, vehicle.mass),
            "fraction": min(m_vap / vehicle.mass, 1.0),
            "peak_area_fraction": float(np.max(frac_area)),
            "mdot_peak": float(np.max(mdot))}


def bracket(traj, vehicle, material: Aluminium) -> dict:
    """Обе границы разом плюс доля алюминия."""
    up = melted_fraction(traj, vehicle, material)
    lo = vaporized_mass(traj, vehicle, material)
    f_al = vehicle.al_mass_fraction
    return {
        "melt": up, "vapour": lo,
        "al_upper_kg": up["mass"] * f_al,
        "al_lower_kg": lo["mass"] * f_al,
        "width": (up["mass"] / lo["mass"]) if lo["mass"] > 0 else np.inf,
    }


# ---------------------------------------------------------------------------
# ПОЭЛЕМЕНТНАЯ модель поверхности — одна согласованная схема вместо двух
# ---------------------------------------------------------------------------

K_ALUMINIUM = 167.0      # Вт/(м*К), теплопроводность Al 6061 (~10%)
RHO_ALUMINIUM = 2700.0   # кг/м^3


def thermal_diffusion_depth(t_flight: float, k=K_ALUMINIUM,
                            rho=RHO_ALUMINIUM, c_p=900.0) -> float:
    """Глубина прогрева за время полёта, м: sqrt(alpha*t), alpha = k/(rho*c_p).

    Критерий применимости сосредоточенной модели. Для Al alpha = 6.9e-5 м^2/с,
    за 300 с это 14 см. Стенка тоньше — прогревается насквозь, модель
    применима. Толще — участвует только приповерхностный слой.
    """
    return float(np.sqrt(k / (rho * c_p) * t_flight))


def surface_thermal_model(traj, vehicle, material: Aluminium,
                          wall_thickness: float | None = None,
                          n_bands: int = 24,
                          correlation: str = "sutton-graves") -> dict:
    """Поэлементный энергобаланс по угловому распределению потока.

    ЗАЧЕМ ЭТО ВМЕСТО ДВУХ ОТДЕЛЬНЫХ УЧЁТОВ. Сосредоточенная модель с одной
    температурой и локальный критерий испарения давали несогласованный ответ:
    у толстого тела первая говорила "расплава 0%", вторая — "испарилось 1.6%".
    Обе правы по-своему, но вилка из них не складывается, потому что это два
    разных счёта, а не границы одного.

    Здесь один счёт. Поверхность делится на пояса по углу theta от точки
    торможения. Каждый пояс:
      - получает q_stag * cos(theta)  (угловое распределение, без параметров)
      - переизлучает eps*sigma*T^4 при СВОЕЙ температуре
      - имеет свою участвующую массу rho*t_стенки на единицу площади
      - проходит свои фазовые переходы через полки T(h_s)

    Тогда расплавленная и испарённая массы получаются из ОДНОГО поля
    температуры, и вилка между ними имеет смысл: расплав — это всё, что
    в принципе может стать оксидом, испарение на месте — то, что точно им
    стало. Разница — судьба сорванного расплава.

    Толщина стенки: для пластины известна, для компактного фрагмента берётся
    эквивалентная оболочка t = m/(rho*A_омыв). Ограничена сверху глубиной
    прогрева: глубже тепло за полёт не доходит.
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline

    V = traj.V_rel if traj.V_rel is not None else traj.V
    q_spline = CubicSpline(traj.t, traj.heat_flux(vehicle, correlation))
    V_spline = CubicSpline(traj.t, V)
    h_spline = CubicSpline(traj.t, traj.h)

    if wall_thickness is None:
        wall_thickness = vehicle.mass / (RHO_ALUMINIUM * vehicle.wetted)
    depth = thermal_diffusion_depth(float(traj.t[-1] - traj.t[0]))
    t_eff = min(wall_thickness, depth)          # участвующая толщина
    lumped_valid = wall_thickness <= depth

    # Пояса по theta, равные ПО ПЛОЩАДИ (площадь ~ sin(th) dth, то есть
    # равные шаги по cos(th)). Считаем только освещённую полусферу: за
    # миделем f(theta)=0, эти пояса не греются и в систему не нужны.
    # Их площадь учитывается тем, что dA берётся от полной поверхности.
    mu_edges = np.linspace(1.0, 0.0, n_bands + 1)       # cos(theta), 0..90 град
    mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    flux_frac = mu                                      # f(theta) = cos(theta)
    dA = vehicle.wetted / (2.0 * n_bands)               # половина поверхности

    sigma_eps = material.emissivity * SIGMA_SB
    m_band = RHO_ALUMINIUM * t_eff * dA                 # участвующая масса пояса
    h3 = (material.c_p * (material.T_melt - material.T_initial)
          + material.L_fusion
          + material.c_p * (material.T_boil - material.T_melt))

    def rhs(t, y):
        h_s = y[:n_bands]
        T = temperature_from_enthalpy(h_s, material)
        q = float(q_spline(t)) * hot_wall_factor(float(V_spline(t)), T)
        q_in = np.maximum(q, 0.0) * flux_frac
        rad = sigma_eps * T ** 4
        net = q_in - rad
        # Переключение "греется / кипит" СГЛАЖЕНО по узкому окну энтальпии.
        # Жёсткий np.where даёт разрыв правой части, на котором LSODA
        # захлёбывается: тонкая пластина считалась 14 с вместо долей секунды.
        # Окно 1% от h3 физически неразличимо, численно решает всё.
        w = np.clip((h_s - h3) / (0.01 * h3), 0.0, 1.0)

        # ЗАЖИМ ТОЛЬКО НА НАГРЕВ. Пояс на кипении, у которого приход упал
        # ниже переизлучения, обязан ОСТЫВАТЬ. Симметричный зажим (1-w)*net
        # запрещал это и заставлял пояс излучать энергию, которой у него нет:
        # энергетический баланс расходился на 50%. Поймано test_energy_balance.
        pos = np.maximum(net, 0.0)
        neg = np.minimum(net, 0.0)
        dh = (neg + (1.0 - w) * pos) / (RHO_ALUMINIUM * t_eff)
        dm_vap = np.sum(w * pos * dA) / material.L_vapour

        # Аудит: приход и переизлучение по всей освещённой поверхности.
        # Разбиение точное: neg + (1-w)*pos + w*pos = net = q_in - rad,
        # поэтому баланс замыкается по построению.
        return np.concatenate([dh, [dm_vap, float(np.sum(q_in) * dA),
                                    float(np.sum(rad) * dA)]])

    y0 = np.zeros(n_bands + 3)
    # Сетка вывода реже траекторной: тепловая задача меняется медленнее,
    # а стоимость растёт линейно по числу точек вывода.
    t_out = np.linspace(traj.t[0], traj.t[-1], 400)
    sol = solve_ivp(rhs, (traj.t[0], traj.t[-1]), y0, method="LSODA",
                    t_eval=t_out, rtol=1e-6, atol=1e-1, max_step=5.0)
    h_end = sol.y[:n_bands, -1]
    m_vap_series = sol.y[n_bands]
    E_in = float(sol.y[n_bands + 1, -1])
    E_rad = float(sol.y[n_bands + 2, -1])
    E_stored = float(np.sum(h_end * RHO_ALUMINIUM * t_eff * dA))
    E_vap = float(m_vap_series[-1] * material.L_vapour)
    residual = (E_in - E_rad - E_stored - E_vap) / E_in if E_in > 0 else 0.0
    T_end = temperature_from_enthalpy(h_end, material)

    h1 = material.c_p * (material.T_melt - material.T_initial)
    melt_frac_band = np.clip((h_end - h1) / material.L_fusion, 0.0, 1.0)
    m_melt = float(np.sum(melt_frac_band * m_band))
    m_vap = float(min(m_vap_series[-1], vehicle.mass))
    # Испарённое — часть расплавленного: расплав есть необходимая стадия
    m_melt = max(m_melt, m_vap)

    return {
        "m_melt": min(m_melt, vehicle.mass), "m_vap": m_vap,
        "f_melt": min(m_melt, vehicle.mass) / vehicle.mass,
        "f_vap": m_vap / vehicle.mass,
        "T_max": float(T_end.max()), "T_nose": float(T_end[0]),
        "wall_thickness": wall_thickness, "participating": t_eff,
        "lumped_valid": lumped_valid, "diffusion_depth": depth,
        "m_vap_series": m_vap_series, "t": sol.t,
        "h": h_spline(sol.t),
        "E_in": E_in, "E_rad": E_rad, "E_stored": E_stored, "E_vap": E_vap,
        "energy_residual": residual,
    }
