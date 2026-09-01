"""Модели атмосферы.

Шаг 1: экспоненциальная заглушка.
Шаг 2: сюда придёт NRLMSISE-00 с тем же интерфейсом .density(h).
"""

from __future__ import annotations

import numpy as np

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
