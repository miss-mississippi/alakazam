"""Модели атмосферы.

Шаг 1: экспоненциальная заглушка (ExponentialAtmosphere).
Шаг 2: NRLMSISE-00 / NRLMSIS 2.x через pymsis (MSISAtmosphere).

Обе модели дают одинаковый интерфейс: .density(h) и .scale_height(h),
поэтому траекторный код их не различает.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import H_SCALE_FIT, RHO0_SEA_LEVEL


class ExponentialAtmosphere:
    """rho(h) = rho0 * exp(-h/H).

    Заглушка. Никаких свойств кроме плотности не даёт — на шаге 1 больше
    ничего и не нужно (Саттон-Грейвс тоже требует только rho).

    Ожидаемая точность против U.S. Standard Atmosphere 1976 при
    rho0=1.225, H=7.2 км:

        120 км : завышает в ~3.2 раза
        100 км : завышает в ~2 раза
         80 км : ~1.0
         60 км : ~1.0
         40 км : завышает в ~1.2 раза

    То есть модель точна ровно там, где идёт абляция, и врёт там, где
    торможение пренебрежимо. Но подгонять под неё ничего нельзя.
    """

    name = "exponential (placeholder)"

    def __init__(self, rho0: float = RHO0_SEA_LEVEL, H: float = H_SCALE_FIT):
        self.rho0 = rho0
        self.H = H

    def density(self, h):
        """Плотность, кг/м^3. h — геометрическая высота, м."""
        # Клампим снизу: без этого exp(-h/H) при h < 0 уходит в overflow,
        # если интегратор пробует шаг ниже поверхности.
        h_clipped = np.maximum(h, 0.0)
        return self.rho0 * np.exp(-h_clipped / self.H)

    def scale_height(self, h):
        """Локальная шкала высот, м. Для экспоненты — константа по определению."""
        return self.H


class MSISAtmosphere:
    """NRLMSISE-00 / NRLMSIS 2.x через pymsis, затабулированная и сплайненная.

    ПОЧЕМУ НЕ ЗОВЁМ pymsis НАПРЯМУЮ ИЗ ПРАВОЙ ЧАСТИ ОДУ.
    solve_ivp вызывает density() десятки тысяч раз за прогон. Прямой вызов
    MSIS на каждом шаге — это секунды на траекторию и часы на свип.
    Табулируем один раз на сетке по высоте и интерполируем.

    ПОЧЕМУ СПЛАЙН ПО log(rho), А НЕ ПО rho.
    Плотность меняется на 6 порядков на нашем диапазоне; линейная
    интерполяция по rho в разреженной части даёт чудовищную ошибку.
    log(rho) почти линеен по высоте (это и есть смысл шкалы высот),
    поэтому интерполируется отлично.

    ПОЧЕМУ КУБИЧЕСКИЙ, А НЕ ЛИНЕЙНЫЙ.
    Линейная интерполяция по log(rho) даёт кусочно-экспоненциальную
    плотность: непрерывную, но с изломами производной. Адаптивный
    интегратор на каждом изломе режет шаг. CubicSpline даёт C2 и
    правая часть ОДУ остаётся гладкой.

    Параметры среды — физические, а не косметические:
      f107, f107a : поток на 10.7 см, индекс солнечной активности.
                    ~70 в минимуме цикла, ~140 умеренно, ~220 в максимуме.
                    Влияет в основном на термосферу (выше 100 км), где
                    торможение всё равно пренебрежимо. Проверяется свипом.
      ap          : геомагнитный индекс. 4 — спокойно, 50+ — буря.
      lat, lon    : контролируемые сходы обычно целят в южную часть Тихого
                    океана (SPOUA), примерно -40 град широты.
      version     : 2.1 = NRLMSIS 2.1 (БАЗОВАЯ), 0 = NRLMSISE-00 (для сверки
                    с инструментами демиза вроде DRAMA и для бюджета
                    неопределённостей).

                    Почему базовая 2.1. В NRLMSISE-00 термосферные плотности
                    считались независимо от нижних слоёв, а профили сшивались
                    апостериорно — отсюда излом производной ln(rho) на 72.5 км,
                    прямо в зоне абляции (см. verify_step2.test_model_seams).
                    В NRLMSIS 2.0 сшивку убрали: гидростатический профиль
                    непрерывен от земли до экзосферы, переход от перемешанной
                    области к диффузионному разделению идёт непрерывно начиная
                    примерно с 70 км. Emmert et al. 2021 (Earth and Space
                    Science): "In the mesosphere and below, residual biases and
                    standard deviations are considerably lower than NRLMSISE-00";
                    туда же ассимилированы новые данные по температуре мезосферы
                    и стратосферы, атомарный кислород продлён вниз до 50 км.
                    Расхождение с 00 составляет 9-16% именно на 50-90 км.
    """

    def __init__(
        self,
        date: "datetime | np.datetime64" = np.datetime64("2026-09-01T12:00"),
        lat: float = -40.0,
        lon: float = -140.0,
        f107: float = 140.0,
        f107a: float = 140.0,
        ap: float = 4.0,
        version: float = 2.1,
        h_max: float = 200.0e3,
        dh: float = 250.0,
    ):
        import pymsis  # локальный импорт: шаг 1 работает без pymsis

        self.date = date
        self.lat, self.lon = lat, lon
        self.f107, self.f107a, self.ap = f107, f107a, ap
        self.version = version
        self.name = f"NRLMSISE-00" if version == 0 else f"NRLMSIS {version}"
        self.name += (f" (F10.7={f107:.0f}, Ap={ap:.0f}, "
                      f"lat={lat:+.0f}, {str(date)[:10]})")

        self._h_grid = np.arange(0.0, h_max + dh, dh)
        out = pymsis.calculate(
            date, lon, lat, self._h_grid / 1e3,
            f107s=f107, f107as=f107a, aps=[[ap] * 7],
            version=version,
        )
        rho = out[..., pymsis.Variable.MASS_DENSITY].ravel()

        # MSIS может вернуть NaN у самой земли для некоторых версий —
        # обрезаем сетку по валидным значениям, а не подставляем заглушку.
        good = np.isfinite(rho) & (rho > 0)
        if not good.all():
            self._h_grid = self._h_grid[good]
            rho = rho[good]

        self._log_rho = CubicSpline(self._h_grid, np.log(rho))
        self._h_lo, self._h_hi = self._h_grid[0], self._h_grid[-1]

        # Шкала высот на верхней границе — для экспоненциальной экстраполяции
        self._H_top = -1.0 / self._log_rho(self._h_hi, 1)
        self._log_rho_top = float(self._log_rho(self._h_hi))

        # Совместимость с ExponentialAtmosphere: rho у поверхности
        self.rho0 = float(np.exp(self._log_rho(self._h_lo)))

    def density(self, h):
        """Плотность, кг/м^3. h — геометрическая высота, м."""
        h = np.asarray(h, dtype=float)
        h_clipped = np.clip(h, self._h_lo, self._h_hi)
        log_rho = self._log_rho(h_clipped)
        # Выше сетки — экспоненциальная экстраполяция с верхней шкалой высот.
        above = h > self._h_hi
        if np.any(above):
            log_rho = np.where(
                above,
                self._log_rho_top - (h - self._h_hi) / self._H_top,
                log_rho,
            )
        return np.exp(log_rho)

    def scale_height(self, h):
        """ЛОКАЛЬНАЯ шкала высот H = -1/(d ln rho / dh), м.

        В отличие от экспоненциальной модели это не константа. Именно
        разброс этой величины по высоте и есть мера того, насколько
        однопараметрическая заглушка была неправа.
        """
        h_clipped = np.clip(np.asarray(h, dtype=float), self._h_lo, self._h_hi)
        return -1.0 / self._log_rho(h_clipped, 1)
