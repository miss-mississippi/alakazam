"""Излучательная способность: модели поверхности и мост к лабораторному измерению.

ЗАЧЕМ ЭТОТ МОДУЛЬ. Для тонкостенных фрагментов испарённая масса зависит от
полной излучательной способности поверхности при рабочей температуре. Рабочая
температура — это температура кипения Al при МЕСТНОМ давлении торможения,
~1900-2050 K (см. ablation.Aluminium.T_boil_at), а не 2740 K при 1 атм.

Две модели поверхности, обе как функции температуры:

  oxide_film_emissivity(T)   плёнка Al2O3 на металле: одна спектральная кривая
                             eps(lam), подобранная по справочной точке
                             alpha-Al2O3 (0.35 при 1800 K), и взвешенная по Планку.
  bare_aluminium_emissivity(T)  голый металл без плёнки: оценка по
                             электросопротивлению (Parker-Abbott).

Обходной путь к измерению стандартный и реализуемый на комнатной оптике:

  1. Меряем СПЕКТРАЛЬНУЮ ОТРАЖАТЕЛЬНУЮ СПОСОБНОСТЬ R(lambda) образца при
     комнатной температуре. Нужна полная (зеркальная + диффузная)
     направленно-полусферическая R: интегрирующая сфера, в среднем ИК — с
     золотым покрытием.
  2. Для НЕПРОЗРАЧНОГО образца по закону Кирхгофа eps(lambda) = 1 - R(lambda).
  3. Полная излучательная способность при температуре T получается
     интегрированием по планковскому распределению:

         eps(T) = int eps(lam) B(lam,T) dlam / int B(lam,T) dlam

Это даёт eps ПРИ РАБОЧЕЙ ТЕМПЕРАТУРЕ из измерения ПРИ КОМНАТНОЙ.

ЧЕСТНЫЕ ОГРАНИЧЕНИЯ, которые надо назвать до, а не после:
  - Оптические константы сами зависят от температуры. Метод их не ловит.
  - Реальная поверхность на входе — РАСПЛАВ с плёнкой, а меряем твёрдый
    окисленный образец. Измерение прижимает сценарий "плёнка активна" и
    порог по толщине; сценарий "голый расплав" так не меряется.
  - Кирхгоф требует непрозрачности. Плёнка тоньше ~1 мкм в ИК полупрозрачна,
    и тогда меряется система "плёнка на металле", что как раз и нужно, но
    результат зависит от подложки — её надо фиксировать.
  - Сфера даёт eps около нормали, модели нужна ПОЛУСФЕРИЧЕСКАЯ. Для металлов
    полусферическая выше нормальной на 10-30%, для диэлектриков чуть ниже.
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


# ---------------------------------------------------------------------------
# Сценарий "плёнка активна": спектральная кривая плёнки на металле
# ---------------------------------------------------------------------------

LAM_GRID = np.geomspace(0.2e-6, 100e-6, 40000)   # м, сетка для взвешивания

# Справочные полные eps alpha-Al2O3 (сводка излучательных способностей
# оксидов, White Rose eprints 133266): 0.83 при 300 K, 0.35 при 1800 K.
ALUMINA_REF = {300.0: 0.83, 1800.0: 0.35}

# Форма по умолчанию. eps_short, eps_long, width выбраны рукой; lam_c
# подбирается по ОДНОЙ справочной точке (1800 K). Свобода формы оценена
# свипом в analysis_step5b (48 наборов).
FILM_SHAPE_DEFAULT = dict(eps_short=0.05, eps_long=0.95, width=1.6)


def eps_film(lam, eps_short=0.05, lam_c=3.98e-6, eps_long=0.95, width=1.6):
    """eps(lam) для оксидной плёнки на металле.

    Прозрачная область: виден МЕТАЛЛ под плёнкой, eps ~ eps_short.
    Фононная область (многофононное поглощение Al2O3): eps ~ eps_long.
    Край lam_c подбирается по справочной точке, см. fit_film_edge().
    """
    return eps_short + (eps_long - eps_short) / (1.0 + (lam_c / lam) ** width)


def fit_film_edge(eps_short=0.05, eps_long=0.95, width=1.6,
                  T_ref: float = 1800.0, eps_ref: float = ALUMINA_REF[1800.0]):
    """lam_c, при котором кривая даёт справочную eps(T_ref). Метры."""
    from scipy.optimize import brentq
    return float(brentq(lambda lc: total_emissivity(
        LAM_GRID, eps_film(LAM_GRID, eps_short, lc, eps_long, width), T_ref)
        - eps_ref, 0.3e-6, 40e-6))


class TabulatedEmissivity:
    """eps(T) по таблице с линейной интерполяцией.

    Правая часть тепловой ОДУ вызывает eps(T) сотни тысяч раз, а честное
    планковское взвешивание стоит интеграла на 40 000 точек. Поэтому кривая
    табулируется один раз на сетке 300-3500 K (шаг 25 K; eps(T) гладкая,
    ошибка интерполяции < 3e-5 по всему диапазону).
    """

    def __init__(self, func_T, T_lo=300.0, T_hi=3500.0, dT=25.0, label=""):
        self.T = np.arange(T_lo, T_hi + dT, dT)
        self.eps = np.array([func_T(T) for T in self.T])
        self.label = label

    def __call__(self, T):
        return np.interp(np.asarray(T, dtype=float), self.T, self.eps)


_OXIDE_CACHE: dict = {}


def oxide_film_emissivity(eps_short=0.05, eps_long=0.95, width=1.6):
    """Сценарий "плёнка оптически активна": eps(T) одной спектральной кривой.

    Кривая пришпилена к справочной точке alpha-Al2O3 при 1800 K; вторая
    справочная точка (300 K) предсказывается, см. analysis_step5b.E.
    Выше 2345 K (плавление Al2O3) это экстраполяция, но при кипении
    Al на местном давлении (~2000 K) плёнка ещё твёрдая.
    """
    key = (eps_short, eps_long, width)
    if key not in _OXIDE_CACHE:
        lc = fit_film_edge(eps_short, eps_long, width)
        spec = eps_film(LAM_GRID, eps_short, lc, eps_long, width)
        _OXIDE_CACHE[key] = TabulatedEmissivity(
            lambda T: total_emissivity(LAM_GRID, spec, T), label="oxide")
    return _OXIDE_CACHE[key]


# ---------------------------------------------------------------------------
# Сценарий "голый расплав": металл без оптически активной плёнки
# ---------------------------------------------------------------------------

def aluminium_resistivity(T, T_melt: float = 933.5):
    """Удельное электросопротивление чистого Al, Ом*м.

    Твёрдая фаза: 2.65 мкОм*см при 293 K -> ~10.9 при плавлении, линейно.
    Жидкость: ~24.2 мкОм*см при плавлении, наклон ~0.0145 мкОм*см/К.
    Порядок величин — рекомендованные данные для Al (Desai et al., J. Phys.
    Chem. Ref. Data 13, 1131, 1984); выше ~1500 K жидкая ветвь —
    экстраполяция. Сплав 6061 в твёрдой фазе имеет сопротивление выше
    (~4 мкОм*см при 293 K), то есть для него eps здесь занижена.
    """
    T = np.asarray(T, dtype=float)
    solid = 2.65 + (10.9 - 2.65) * (T - 293.0) / (T_melt - 293.0)
    liquid = 24.2 + 0.0145 * (T - T_melt)
    return np.where(T < T_melt, solid, liquid) * 1e-8


def parker_abbott_emissivity(rho_e_ohm_m, T):
    """Полная полусферическая eps металла по сопротивлению (Parker & Abbott,
    NASA SP-55, 1965), обобщение соотношения Хагена-Рубенса:

        eps = 0.766 x^0.5 - (0.309 - 0.0889 ln x) x - 0.0175 x^1.5,
        x = rho_e[Ом*см] * T[K]

    Проверка на справочных данных: для полированного твёрдого Al при
    600-900 K даёт 0.045-0.068 (справочник: 0.04-0.07), см. verify_step5.
    """
    x = np.asarray(rho_e_ohm_m, dtype=float) * 100.0 * np.asarray(T, dtype=float)
    return 0.766 * np.sqrt(x) - (0.309 - 0.0889 * np.log(x)) * x - 0.0175 * x ** 1.5


def bare_aluminium_emissivity(T):
    """Сценарий "плёнки нет": eps(T) голого Al по сопротивлению.

    ОЦЕНКА, а не измерение: прямых данных по полной eps жидкого Al выше
    ~1500 K найти не удалось. Даёт ~0.10 у плавления и ~0.18 при 2000 K.
    Справочное 0.05 — это полированный ТВЁРДЫЙ Al при 300-900 K; у жидкого
    металла сопротивление в 2.4-5 раз выше, чем у твёрдого при 900 K, и eps
    растёт примерно как корень из него.
    """
    T = np.asarray(T, dtype=float)
    return parker_abbott_emissivity(aluminium_resistivity(T), T)


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
