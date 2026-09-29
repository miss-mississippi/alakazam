"""Тепловой отклик и абляция: две границы вместо одной оценки.

ГЛАВНОЕ АРХИТЕКТУРНОЕ РЕШЕНИЕ. Модель НЕ выдаёт одно число испарённой массы.
Она выдаёт вилку:

  ВЕРХНЯЯ ГРАНИЦА — расплавленная масса (по максимуму энтальпии за полёт).
  Критерий ORSAT/DRAMA: поглощённая энергия против нагрева до плавления плюс
  теплоты плавления (у ORSAT для generic aluminum heat of ablation =
  934.5 кДж/кг). Это всё, что в принципе может стать оксидом.

  ИСПАРЕНИЕ НА МЕСТЕ — масса, испарённая при допущении, что расплав
  УДЕРЖИВАЕТСЯ на фрагменте, пока не закипит (оксидная корка это допускает).
  Это не строгая нижняя граница: если расплав срывает потоком, на месте
  испарится меньше, а судьба капель не моделируется.

Между ними — судьба расплава, сорванного сдвигом в поток. Капля может
испариться дальше по траектории, может окислиться только по поверхности и
выпасть миллиметровой сферулой (так находят абляционные сферулы метеороидов
в глубоководных осадках), может застыть целиком. Ни одна текущая модель
этого не разрешает, и именно поэтому вилка — результат, а не компромисс.

ТЕМПЕРАТУРА КИПЕНИЯ ЗАВИСИТ ОТ ДАВЛЕНИЯ. 2792 K — это кипение Al при 1 атм.
На поверхности фрагмента давление — это давление торможения,
~0.4-1 кПа на 70-80 км, и Al там кипит при ~1900-2050 K
(Клапейрон-Клаузиус). Кипение при 1 атм завышало бы порог испарения
eps*sigma*T^4 примерно втрое.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .heating import CP_AIR_HOT, SIGMA_SB, hot_wall_factor, stagnation_pressure

R_GAS = 8.314462618      # Дж/(моль*К)
M_AL = 0.026982          # кг/моль
P_ATM = 101325.0         # Па

# Kirchhoff: dL/dT = c_p(пар) - c_p(жидкость) = 20.79 - 31.75 Дж/(моль*К)
# (одноатомный газ 5/2 R; жидкость — NIST Shomate). На кг: -406 Дж/(кг*К).
DCP_VAPOUR = (2.5 * R_GAS - 31.751) / M_AL

K_ALUMINIUM = 167.0      # Вт/(м*К), теплопроводность Al 6061 (~10%)
RHO_ALUMINIUM = 2700.0   # кг/м^3

# Релаксация перегретого расплава. Когда давление падает, падает и T_кип,
# и пояс оказывается выше новой полки кипения. Избыток энтальпии уходит в
# испарение (вскипание) с этой постоянной времени. Физически это мгновенно;
# 0.2 с — численная регуляризация, на результат не влияет (verify_step4).
TAU_FLASH = 0.2          # с


@dataclass
class Aluminium:
    """Свойства Al 6061. Каждое — с источником и неопределённостью.

    ТЕПЛОФИЗИКА
      c_p            эффективная теплоёмкость твёрдой фазы 300 K -> T_melt:
                     среднее по NIST Shomate для Al(s) (c_p растёт от 899
                     при 300 K до 1190 при 890 K), сохраняет энтальпию точно.
      c_p_liquid     31.75 Дж/(моль*К) = 1177 Дж/(кг*К), NIST Shomate для
                     Al(l).
      T_melt         6061: солидус 855 K, ликвидус 925 K (ASM); одна полка
                     посередине интервала. 933 K — это чистый Al.
      L_fusion       397 кДж/кг (чистый Al), ~5%.

    ИСПАРЕНИЕ
      T_boil_1atm    2792 K (CRC Handbook; в части справочников 2740-2743 K).
      L_vapour       294 кДж/моль = 10.90 МДж/кг при T_boil_1atm (CRC).
      T_boil_at(p)   Клапейрон-Клаузиус с опорой на (2792 K, 1 атм).
                     Против таблицы давления паров CRC: 100 Па -> 1817 K,
                     1 кПа -> 2054 K, 10 кПа -> 2364 K, расхождение < 2%
                     (verify_step4).
      L_vapour_at(T) поправка Кирхгофа, +0.3 МДж/кг при 2000 K.

    ИЗЛУЧЕНИЕ — главная ось для тонкостенных фрагментов.
      surface = "constant"  eps = emissivity при любой T (свипы).
      surface = "oxide"     плёнка оптически активна: eps(T) одной
                            спектральной кривой, пришпиленной к alpha-Al2O3
                            (~0.32 при 2000 K).
      surface = "bare"      плёнки нет: eps(T) голого металла по
                            сопротивлению (~0.18 при 2000 K).
      tau_oxide             рост плёнки: eps(t, T) переходит от "bare" к
                            "oxide" как 1 - exp(-sqrt(t/tau)), t — время
                            с момента образования поверхности (разрушения).
    """

    c_p: float = 1038.0
    c_p_liquid: float = 1177.0
    T_melt: float = 890.0
    L_fusion: float = 0.397e6
    T_initial: float = 300.0

    T_boil_1atm: float = 2792.0
    L_vapour: float = 10.90e6
    boil_at_local_pressure: bool = True
    p_nominal: float = 1.0e3      # Па, для сводных оценок (энергия, Pi)

    emissivity: float = 0.10
    surface: str = "constant"
    tau_oxide: float | None = None
    film_shape: tuple = (0.05, 0.95, 1.6)   # eps_short, eps_long, width

    @classmethod
    def legacy(cls, **kw):
        """Свойства первой версии: c_p = 900 везде, T_melt = 933 K,
        кипение 2740 K при любом давлении, L = 10.5 МДж/кг. Только для
        сравнения "было -> стало"."""
        base = dict(c_p=900.0, c_p_liquid=900.0, T_melt=933.0,
                    T_boil_1atm=2740.0, L_vapour=10.5e6,
                    boil_at_local_pressure=False)
        base.update(kw)
        return cls(**base)

    # --- кипение ---------------------------------------------------------

    def T_boil_at(self, p):
        """Температура кипения при давлении p, K."""
        if not self.boil_at_local_pressure:
            return np.full_like(np.asarray(p, dtype=float), self.T_boil_1atm)
        p = np.maximum(np.asarray(p, dtype=float), 1e-6)
        inv = 1.0 / self.T_boil_1atm - R_GAS / (M_AL * self.L_vapour) * np.log(p / P_ATM)
        return 1.0 / inv

    def L_vapour_at(self, T_boil):
        """Теплота испарения при температуре кипения T_boil, Дж/кг."""
        if not self.boil_at_local_pressure:
            return np.full_like(np.asarray(T_boil, dtype=float), self.L_vapour)
        return self.L_vapour + DCP_VAPOUR * (np.asarray(T_boil) - self.T_boil_1atm)

    @property
    def T_boil_nominal(self) -> float:
        """T кипения при p_nominal (1 кПа) — для сводных оценок."""
        return float(self.T_boil_at(self.p_nominal))

    # --- энтальпия -------------------------------------------------------

    @property
    def h1(self) -> float:
        """Начало плавления, Дж/кг."""
        return self.c_p * (self.T_melt - self.T_initial)

    @property
    def h2(self) -> float:
        """Конец плавления, Дж/кг."""
        return self.h1 + self.L_fusion

    def h3(self, T_boil):
        """Начало кипения при данной T_кип, Дж/кг."""
        return self.h2 + self.c_p_liquid * (np.asarray(T_boil) - self.T_melt)

    @property
    def h_melt_complete(self) -> float:
        """Энергия на килограмм до полного расплавления, Дж/кг.

        Критерий ORSAT: 934.5 кДж/кг для generic aluminum (NTRS 20140016958).
        """
        return self.h2

    @property
    def h_vapour_complete(self) -> float:
        """Энергия на килограмм до полного испарения при p_nominal, Дж/кг."""
        Tb = self.T_boil_nominal
        return float(self.h3(Tb) + self.L_vapour_at(Tb))

    @property
    def demise_ratio(self) -> float:
        """Во сколько раз испарение дороже плавления."""
        return self.h_vapour_complete / self.h_melt_complete

    # --- излучение -------------------------------------------------------

    def emissivity_at(self, T, t_since_surface: float = 0.0):
        """Полная полусферическая eps при температуре T (массив)."""
        from .emissivity import bare_aluminium_emissivity, oxide_film_emissivity
        T = np.asarray(T, dtype=float)
        if self.tau_oxide is not None:
            g = 1.0 - np.exp(-np.sqrt(max(t_since_surface, 0.0) / self.tau_oxide))
            e_b = bare_aluminium_emissivity(T)
            return e_b + (oxide_film_emissivity(*self.film_shape)(T) - e_b) * g
        if self.surface == "oxide":
            return oxide_film_emissivity(*self.film_shape)(T)
        if self.surface == "bare":
            return bare_aluminium_emissivity(T)
        return np.full_like(T, self.emissivity)

    def label(self) -> str:
        if self.tau_oxide is not None:
            return f"рост плёнки tau={self.tau_oxide:g} с"
        return {"oxide": "плёнка активна", "bare": "голый расплав"}.get(
            self.surface, f"eps={self.emissivity:.2f}")


def temperature_from_enthalpy(h_s, mat: Aluminium, T_boil=None):
    """T(h_s) — монотонная кусочная функция с ПОЛКАМИ на фазовых переходах.

    Именно так фазовые переходы попадают в правую часть ОДУ без ветвления:
    состояние — удельная энтальпия, а температура из неё выводится.

        h1 = c_p(тв)*(T_пл - T0)                начало плавления
        h2 = h1 + L_пл                          конец плавления
        h3 = h2 + c_p(ж)*(T_кип - T_пл)         начало кипения

    T_boil — температура кипения при текущем местном давлении.
    """
    if T_boil is None:
        T_boil = mat.T_boil_nominal
    h1, h2 = mat.h1, mat.h2
    h3 = mat.h3(T_boil)
    h = np.asarray(h_s, dtype=float)
    return np.where(
        h < h1, mat.T_initial + h / mat.c_p,
        np.where(h < h2, mat.T_melt,
                 np.where(h < h3, mat.T_melt + (h - h2) / mat.c_p_liquid, T_boil)))


# ---------------------------------------------------------------------------
# Критерии режимов
# ---------------------------------------------------------------------------

def thermal_diffusion_depth(t_flight: float, k=K_ALUMINIUM,
                            rho=RHO_ALUMINIUM, c_p=900.0) -> float:
    """Глубина прогрева за время полёта, м: sqrt(alpha*t), alpha = k/(rho*c_p).

    Что этот критерий ГОВОРИТ: температура по толщине успевает выровняться,
    то есть сосредоточенная по толщине модель законна. Для Al alpha = 6.9e-5,
    за 300 с это 14 см.

    Что он НЕ говорит: он НЕ разделяет энергетический и радиационный режимы.
    И 1 мм, и 16 мм много тоньше 14 см, то есть сосредоточенная модель законна
    для обеих, а ведут они себя противоположно. Разделяет их поверхностная
    теплоёмкость — см. regime_number().
    """
    return float(np.sqrt(k / (rho * c_p) * t_flight))


def areal_mass(vehicle, t_flight: float) -> float:
    """Масса на единицу ОМЫВАЕМОЙ площади, участвующая в прогреве, кг/м^2.

    m / A_омыв, но не больше rho * глубина прогрева. Для пластины толщины t
    это rho*t/2: омываемая площадь — обе грани.
    """
    return min(vehicle.mass / vehicle.wetted,
               RHO_ALUMINIUM * thermal_diffusion_depth(t_flight))


def regime_number(traj, vehicle, material: Aluminium) -> dict:
    """Pi = доступная энергия / энергия, нужная чтобы дойти до кипения.

    ЭТО и есть разделитель режимов, а не глубина прогрева.

        Pi = phi * int(q_stag dt)  /  (m'' * h_кип)

    m'' — масса на единицу омываемой площади (для пластины rho*t/2, для
    оболочки m/A_омыв), h_кип — энтальпия от T0 до начала кипения при
    номинальном давлении 1 кПа.

    Pi >> 1 — тело выходит на радиационное равновесие, дальше всё решает eps.
    Pi << 1 — энергии не хватает даже на разогрев, задача ЭНЕРГЕТИЧЕСКАЯ,
              eps почти не влияет, потому что переизлучение мало.
    """
    from .heating import SHAPE_FACTOR_TUMBLING
    t_flight = float(traj.t[-1] - traj.t[0])
    m_area = areal_mass(vehicle, t_flight)
    h_boil = float(material.h3(material.T_boil_nominal))
    need = m_area * h_boil
    avail = SHAPE_FACTOR_TUMBLING * traj.heat_load(vehicle)
    return {"Pi": avail / need, "areal_mass": m_area,
            "equiv_thickness": m_area / RHO_ALUMINIUM,
            "need": need, "available": avail,
            "regime": "радиационный" if avail > need else "энергетический"}


# ---------------------------------------------------------------------------
# ПОЭЛЕМЕНТНАЯ модель поверхности — одна согласованная схема
# ---------------------------------------------------------------------------

def surface_thermal_model(traj, vehicle, material: Aluminium,
                          wall_thickness: float | None = None,
                          n_bands: int = 24,
                          correlation: str = "sutton-graves",
                          flux_mode: str = "cos",
                          blowing_eta: float = 0.0,
                          n_out: int = 1200) -> dict:
    """Поэлементный энергобаланс по угловому распределению потока.

    Поверхность делится на пояса по углу theta от точки торможения. Каждый пояс:
      - получает q_stag * cos(theta) * (1 - h_w/h_0)
      - переизлучает sides * eps(T) * sigma * T^4 при СВОЕЙ температуре
        (sides = 2 для пластины: тыльная сторона излучает тоже)
      - имеет свою участвующую массу rho*t_стенки*dA
      - проходит фазовые переходы через полки T(h_s); полка кипения стоит
        на T_кип(p_торм(t)) и сдвигается вниз по мере падения давления
      - испаряет не больше своей массы: выкипевший пояс исчезает (прогар)
        и перестаёт получать и излучать

    Расплавленная и испарённая массы получаются из ОДНОГО поля температуры.
    Расплав считается по МАКСИМУМУ энтальпии пояса за полёт: пояс, который
    расплавился и потом остыл, расплавленным быть не перестал. Испарение
    идёт только выше полки кипения, поэтому в каждом поясе испарено <=
    расплавлено <= масса пояса — по построению, без принудительного max().

    flux_mode:
      "cos"      устойчивая ориентация: греется освещённая половина
                 поверхности, распределение cos(theta), без свободных
                 параметров. Для оболочки подветренная половина массы в
                 нагреве не участвует — расплав оболочки не больше 50%.
      "uniform"  быстрое кувыркание: каждый элемент всей омываемой
                 поверхности получает средний поток 0.25*q_stag. Предел,
                 когда период кувыркания много меньше тепловой постоянной
                 времени стенки (~5 с для пластины 1 мм).

    blowing_eta: блокировка потока вдувом пара в кипящих поясах,
      q_исп = (q_in - rad) / (1 + eta*(h_0 - h_w)/L). 0 — выключено (база).

    Толщина стенки: для пластины — её толщина, для компактного фрагмента —
    эквивалентная оболочка m/(rho*A_омыв). Ограничена сверху глубиной
    прогрева.
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline

    t0, t1 = float(traj.t[0]), float(traj.t[-1])
    V = traj.V_rel if traj.V_rel is not None else traj.V
    q_spline = CubicSpline(traj.t, traj.heat_flux(vehicle, correlation))
    V_spline = CubicSpline(traj.t, V)
    h_spline = CubicSpline(traj.t, traj.h)
    lnp_spline = CubicSpline(
        traj.t, np.log(np.maximum(stagnation_pressure(traj.rho, V), 1e-9)))

    depth = thermal_diffusion_depth(t1 - t0)
    if flux_mode == "cos":
        if wall_thickness is None:
            wall_thickness = vehicle.mass / (RHO_ALUMINIUM * vehicle.wetted)
        heated_area = vehicle.wetted / 2.0
        sides = float(vehicle.radiating_sides)
        # Пояса по theta, равные ПО ПЛОЩАДИ (равные шаги по cos(theta)).
        mu_edges = np.linspace(1.0, 0.0, n_bands + 1)
        flux_frac = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    elif flux_mode == "uniform":
        # Вся омываемая поверхность, средний поток; обе грани пластины уже
        # входят в омываемую площадь, поэтому каждый элемент излучает одной
        # стороной, а масса на единицу площади — m/A_омыв.
        wall_thickness = vehicle.mass / (RHO_ALUMINIUM * vehicle.wetted)
        heated_area = vehicle.wetted
        sides = 1.0
        flux_frac = np.full(n_bands, 0.25)
    else:
        raise ValueError(f"flux_mode: {flux_mode!r}")

    t_eff = min(wall_thickness, depth)          # участвующая толщина
    lumped_valid = wall_thickness <= depth
    dA = heated_area / n_bands
    m_band = RHO_ALUMINIUM * t_eff * dA
    n = n_bands
    mat = material
    h1 = mat.h1

    def rhs(t, y):
        h_s = y[:n]
        m_v = y[n:2 * n]
        T_b = float(mat.T_boil_at(np.exp(float(lnp_spline(t)))))
        L_v = float(mat.L_vapour_at(T_b))
        h3 = float(mat.h3(T_b))
        T = temperature_from_enthalpy(h_s, mat, T_b)
        V_now = float(V_spline(t))
        q = max(float(q_spline(t)), 0.0) * hot_wall_factor(V_now, T)
        eps = mat.emissivity_at(T, t - t0)

        # Выкипевший пояс исчезает. Сглажено по 1% массы пояса.
        alive = np.clip((m_band - m_v) / (0.01 * m_band), 0.0, 1.0)
        q_in = q * flux_frac * alive
        rad = sides * eps * SIGMA_SB * T ** 4 * alive
        net = q_in - rad

        # Переключение "греется / кипит" СГЛАЖЕНО по окну 1% от h3: жёсткий
        # разрыв правой части LSODA проходит в 200 раз медленнее.
        # ЗАЖИМ ТОЛЬКО НА НАГРЕВ: пояс на кипении, у которого приход упал
        # ниже переизлучения, обязан остывать, иначе он излучал бы энергию,
        # которой у него нет.
        w = np.clip((h_s - h3) / (0.01 * h3), 0.0, 1.0)
        pos = np.maximum(net, 0.0)
        neg = np.minimum(net, 0.0)

        # Вдув: часть избытка потока в кипящем поясе блокируется паром.
        dh_gas = max(0.5 * V_now ** 2 - CP_AIR_HOT * T_b, 0.0)
        evap = w * pos / (1.0 + blowing_eta * dh_gas / L_v)
        block = w * pos - evap

        # Вскипание перегретого расплава, когда T_кип падает с давлением.
        flash = np.maximum(h_s - 1.01 * h3, 0.0) / TAU_FLASH * alive

        dh = (neg + (1.0 - w) * pos) / (RHO_ALUMINIUM * t_eff) - flash
        dm = evap * dA / L_v + m_band * flash / L_v

        # Аудит внутри той же системы ОДУ. Разбиение точное:
        # q_in - rad = [neg + (1-w)pos] + evap + block, и m_band*flash
        # переходит из запаса в испарение.
        return np.concatenate([dh, dm, [
            float(np.sum(q_in) * dA),
            float(np.sum(rad) * dA),
            float(np.sum(evap) * dA + np.sum(m_band * flash)),
            float(np.sum(block) * dA)]])

    y0 = np.zeros(2 * n + 4)
    atol = np.concatenate([np.full(n, 1e-1), np.full(n, 1e-7 * max(m_band, 1e-9)),
                           np.full(4, 1e-1)])
    t_out = np.linspace(t0, t1, n_out)
    sol = solve_ivp(rhs, (t0, t1), y0, method="LSODA", t_eval=t_out,
                    rtol=1e-6, atol=atol, max_step=5.0)

    H = sol.y[:n]
    MV = sol.y[n:2 * n]
    h_end = H[:, -1]
    m_vap_band = MV[:, -1]
    E_in, E_rad, E_vap, E_block = (float(sol.y[2 * n + k, -1]) for k in range(4))
    E_stored = float(np.sum(h_end * m_band))
    residual = (E_in - E_rad - E_stored - E_vap - E_block) / E_in if E_in > 0 else 0.0

    melt_frac_band = np.clip((H.max(axis=1) - h1) / mat.L_fusion, 0.0, 1.0)
    m_melt_band = melt_frac_band * m_band
    m_vap_series = MV.sum(axis=0)
    p_series = np.exp(lnp_spline(sol.t))
    Tb_series = mat.T_boil_at(p_series)
    T_end = temperature_from_enthalpy(h_end, mat, float(Tb_series[-1]))

    m_melt = float(m_melt_band.sum())
    m_vap = float(m_vap_band.sum())
    return {
        "m_melt": m_melt, "m_vap": m_vap,
        "f_melt": m_melt / vehicle.mass, "f_vap": m_vap / vehicle.mass,
        "T_max": float(temperature_from_enthalpy(H.max(), mat,
                                                 float(Tb_series.max()))),
        "T_nose_end": float(T_end[0]),
        "wall_thickness": wall_thickness, "participating": t_eff,
        "lumped_valid": lumped_valid, "diffusion_depth": depth,
        "m_band": m_band, "m_vap_band": m_vap_band, "m_melt_band": m_melt_band,
        "m_vap_series": m_vap_series, "t": sol.t, "h": h_spline(sol.t),
        "p_stag": p_series, "T_boil": Tb_series,
        "E_in": E_in, "E_rad": E_rad, "E_stored": E_stored, "E_vap": E_vap,
        "E_block": E_block, "energy_residual": residual,
    }
