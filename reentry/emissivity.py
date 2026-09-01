"""Мост между моделью и лабораторным измерением.

ЗАЧЕМ ЭТОТ МОДУЛЬ. Модель показала, что весь ответ упирается в полную
излучательную способность окисленного алюминия при 1500-2740 K. Померить её
НАПРЯМУЮ при таких температурах почти невозможно: нужен высокотемпературный
эмиссометр, которого в обычной материаловедческой лаборатории нет.

Обходной путь стандартный и полностью реализуемый на комнатной оптике:

  1. Меряем СПЕКТРАЛЬНУЮ ОТРАЖАТЕЛЬНУЮ СПОСОБНОСТЬ R(lambda) образца при
     комнатной температуре, с интегрирующей сферой (нужна полусферическая,
     а не зеркальная: окисленная поверхность рассеивает).
  2. Для НЕПРОЗРАЧНОГО образца по закону Кирхгофа eps(lambda) = 1 - R(lambda).
  3. Полная излучательная способность при температуре T получается
     интегрированием по планковскому распределению:

         eps(T) = int eps(lam) B(lam,T) dlam / int B(lam,T) dlam

Это даёт eps ПРИ РАБОЧЕЙ ТЕМПЕРАТУРЕ из измерения ПРИ КОМНАТНОЙ.

ЧЕСТНЫЕ ОГРАНИЧЕНИЯ, которые надо назвать до, а не после:
  - Оптические константы сами зависят от температуры. Метод их не ловит.
  - Реальная поверхность на входе — РАСПЛАВ с плёнкой, а меряем твёрдый
    окисленный образец. То есть измерение прижимает ВЕРХНИЙ конец вилки
    (сплошной оксид) и не трогает нижний (голый жидкий металл).
  - Кирхгоф требует непрозрачности. Плёнка тоньше ~1 мкм в ИК полупрозрачна,
    и тогда меряется система "плёнка на металле", что как раз и нужно, но
    результат зависит от подложки — её надо фиксировать.

Даже с этими оговорками измерение превращает диапазон 0.05-0.35 (фактор 7)
в 0.05-[измеренное], то есть снимает половину неопределённости.
"""

from __future__ import annotations

import numpy as np

H_PLANCK = 6.62607015e-34    # Дж*с (СИ, точное определение)
C_LIGHT = 2.99792458e8       # м/с (точное)
K_BOLTZ = 1.380649e-23       # Дж/К (точное)
SIGMA_SB = 5.670374419e-8    # Вт/(м^2*К^4)
WIEN_B = 2.897771955e-3      # м*К, постоянная смещения Вина


def planck_spectral_radiance(wavelength_m, T: float):
    """Спектральная яркость чёрного тела B(lambda,T), Вт/(м^2*ср*м)."""
    lam = np.asarray(wavelength_m, dtype=float)
    a = 2.0 * H_PLANCK * C_LIGHT ** 2 / lam ** 5
    x = H_PLANCK * C_LIGHT / (lam * K_BOLTZ * T)
    # Клампим показатель: на очень коротких волнах exp переполняется,
    # а вклад там всё равно нулевой.
    return np.where(x < 700.0, a / np.expm1(np.minimum(x, 700.0)), 0.0)


def planck_weight(wavelength_m, T: float):
    """Нормированный планковский вес: int w dlam = 1 на заданной сетке."""
    lam = np.asarray(wavelength_m, dtype=float)
    B = planck_spectral_radiance(lam, T)
    return B / np.trapezoid(B, lam)


def total_emissivity(wavelength_m, eps_lambda, T: float) -> float:
    """Полная излучательная способность при температуре T.

        eps(T) = int eps(lam) B(lam,T) dlam / int B(lam,T) dlam

    wavelength_m : м, возрастающая сетка
    eps_lambda   : спектральная излучательная способность, 0..1
                   (из измерения: eps = 1 - R для непрозрачного образца)
    """
    lam = np.asarray(wavelength_m, dtype=float)
    e = np.asarray(eps_lambda, dtype=float)
    B = planck_spectral_radiance(lam, T)
    return float(np.trapezoid(e * B, lam) / np.trapezoid(B, lam))


def total_emissivity_from_reflectance(wavelength_m, reflectance, T: float) -> float:
    """То же, но на вход подаётся ИЗМЕРЕННАЯ отражательная способность.

    Ровно та функция, в которую лягут данные с прибора: массив длин волн
    и массив R, снятые интегрирующей сферой.
    """
    return total_emissivity(wavelength_m, 1.0 - np.asarray(reflectance), T)


def required_band(T: float, coverage: float = 0.95,
                  lam_lo: float = 1e-7, lam_hi: float = 1e-3):
    """Диапазон длин волн, несущий заданную долю планковской энергии.

    Отвечает на прямой экспериментальный вопрос: какой спектральный диапазон
    обязан покрыть прибор, чтобы интеграл не поехал.

    Возвращает (lam_min, lam_max, lam_peak) в метрах.
    """
    lam = np.geomspace(lam_lo, lam_hi, 20000)
    B = planck_spectral_radiance(lam, T)
    cdf = np.concatenate([[0.0], np.cumsum(np.diff(lam) * (B[:-1] + B[1:]) / 2)])
    cdf /= cdf[-1]
    tail = (1.0 - coverage) / 2.0
    lo = float(np.interp(tail, cdf, lam))
    hi = float(np.interp(1.0 - tail, cdf, lam))
    return lo, hi, WIEN_B / T


def synthetic_oxide_spectrum(wavelength_m, thickness_m: float,
                             eps_metal: float = 0.05,
                             eps_oxide: float = 0.60,
                             lam_transition: float = 3.0e-6):
    """ЗАГЛУШКА до появления реальных данных. Не выдавать за измерение.

    Простейшая модель "плёнка на металле": плёнка непрозрачна там, где её
    оптическая толщина велика, то есть на коротких волнах относительно
    lam_transition ~ толщины; на длинных волнах просвечивает металл.

        eps(lam) = eps_мет + (eps_окс - eps_мет) / (1 + (lam/(delta*k))^2)

    Нужна ровно для одного: проверить, что цепочка
    "спектр -> планковское взвешивание -> eps(T) -> модель" работает
    целиком, ДО того как появятся образцы. Численные значения смысла
    не имеют.
    """
    lam = np.asarray(wavelength_m, dtype=float)
    lam_c = max(thickness_m, 1e-9) * (lam_transition / 1e-6) * 1e6 * 1e-6
    return eps_metal + (eps_oxide - eps_metal) / (1.0 + (lam / lam_c) ** 2)
