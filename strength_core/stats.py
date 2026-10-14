"""
Глава 1.1 — Статистические характеристики случайных величин и законы распределения.

Формулы (1.1.1) – (1.1.27). Все функции векторизованы через NumPy/scipy:
принимают массивы (например, выборки напряжений по тысячам КЭ-узлов) и
обрабатывают их за одну проходку без явных циклов в Python.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid, quad
from scipy.special import erf

__all__ = [
    "relative_frequency",            # (1.1.1)
    "probability_from_pdf",          # (1.1.2)
    "mean_discrete",                 # (1.1.3)
    "mean_continuous",               # (1.1.4)
    "variance_discrete",             # (1.1.5)
    "variance_continuous",           # (1.1.6)
    "std_dev",                       # (1.1.7)
    "coeff_variation",               # (1.1.8)
    "cdf_from_pdf",                  # (1.1.9)
    "pdf_of_sum_convolution",        # (1.1.10)
    "mean_of_sum_double_integral",   # (1.1.11)
    "normalization_and_means",       # (1.1.12)
    "mean_sum",                      # (1.1.13)
    "mean_difference",               # (1.1.14)
    "mean_total",                    # (1.1.15)
    "pdf_exponential",               # (1.1.16)
    "pdf_uniform",                   # (1.1.17)
    "pdf_triangular",                # (1.1.18)
    "pdf_normal",                    # (1.1.19)
    "prob_normal_interval",          # (1.1.20)
    "standardize",                   # (1.1.21)
    "laplace_function",              # (1.1.22)
    "prob_normal_laplace",           # (1.1.23)
    "tolerance_limits_normal",       # (1.1.24)
    "three_sigma_rule",              # (1.1.25)
    "normal_params_from_range",      # (1.1.26)
    "pdf_weibull",                   # (1.1.27)
]

# ---------------------------------------------------------------------------
# 1.1.1  p_i = n_i / N
# ---------------------------------------------------------------------------
def relative_frequency(n_i, N):
    """(1.1.1): относительная частота p_i = n_i / N.

    Parameters
    ----------
    n_i : array_like — частоты попадания в i-й интервал;
    N   : float — общее число наблюдений.
    """
    n_i = np.asarray(n_i, dtype=float)
    return n_i / N


# ---------------------------------------------------------------------------
# 1.1.2  P(x1 < x < x2) = ∫_{x1}^{x2} f(x) dx
# ---------------------------------------------------------------------------
def probability_from_pdf(f, x1, x2, grid=None):
    """(1.1.2): вероятность попадания в интервал (x1, x2) по Плотности f(x).

    Два режима:
      * f — callable: точное интегрирование scipy.integrate.quad;
      * f — ndarray значений на сетке `grid`: численное интегрирование
        методом трапеций (быстрый путь для больших массивов из МКЭ).
    """
    if callable(f):
        val, _err = quad(f, x1, x2, limit=200)
        return val
    f = np.asarray(f, dtype=float)
    if grid is None:
        raise ValueError("Для табличной ПП требуется сетка `grid`.")
    grid = np.asarray(grid, dtype=float)
    mask = (grid >= min(x1, x2)) & (grid <= max(x1, x2))
    if mask.sum() < 2:
        return 0.0
    return float(trapezoid(f[mask], grid[mask]))


# ---------------------------------------------------------------------------
# 1.1.3  m_x = Σ x_i n_i / N = Σ x_i p_i
# ---------------------------------------------------------------------------
def mean_discrete(x, weights=None, probabilities=None):
    """(1.1.3): математическое ожидание дискретной СВ.

    m_x = Σ x_i·p_i, где p_i = n_i/N (weights — частоты n_i,
    probabilities — уже нормированные p_i).
    """
    x = np.asarray(x, dtype=float)
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)
        return float(np.dot(x, p))
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        return float(np.dot(x, w) / w.sum())
    return float(x.mean())


# ---------------------------------------------------------------------------
# 1.1.4  m_x = ∫ x f(x) dx
# ---------------------------------------------------------------------------
def mean_continuous(f, grid=None, bounds=(-np.inf, np.inf)):
    """(1.1.4): мат. ожидание непрерывной СВ, m_x = ∫ x·f(x) dx.

    Быстрый путь: табличные f на равномерной/заданной сетке `grid`
    (trapezoid). Точный путь: callable f + quad.
    """
    if callable(f):
        val, _err = quad(lambda t: t * f(t), *bounds, limit=300)
        return val
    f = np.asarray(f, dtype=float)
    grid = np.asarray(grid, dtype=float)
    return float(trapezoid(grid * f, grid))


# ---------------------------------------------------------------------------
# 1.1.5  D_x = Σ (x_i - m_x)^2 p_i
# ---------------------------------------------------------------------------
def variance_discrete(x, m_x=None, weights=None, probabilities=None):
    """(1.1.5): дисперсия дискретной СВ, D_x = Σ (x_i - m_x)^2 p_i."""
    x = np.asarray(x, dtype=float)
    if m_x is None:
        m_x = mean_discrete(x, weights=weights, probabilities=probabilities)
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)
    elif weights is not None:
        p = relative_frequency(weights, np.sum(weights))
    else:
        # выборка без весов: несмещённая оценка невозможна в формуле (1.1.5),
        # используем p_i = 1/n
        p = np.full_like(x, 1.0 / x.size)
    d = x - m_x
    return float(np.dot(d * d, p))


# ---------------------------------------------------------------------------
# 1.1.6  D_x = ∫ (x - m_x)^2 f(x) dx
# ---------------------------------------------------------------------------
def variance_continuous(f, m_x, grid=None, bounds=(-np.inf, np.inf)):
    """(1.1.6): дисперсия непрерывной СВ, D_x = ∫ (x-m_x)^2 f(x) dx."""
    if callable(f):
        val, _err = quad(lambda t: (t - m_x) ** 2 * f(t), *bounds, limit=300)
        return val
    f = np.asarray(f, dtype=float)
    grid = np.asarray(grid, dtype=float)
    return float(trapezoid((grid - m_x) ** 2 * f, grid))


# ---------------------------------------------------------------------------
# 1.1.7  sigma_x = sqrt(D_x)
# ---------------------------------------------------------------------------
def std_dev(D_x):
    """(1.1.7): среднее квадратическое отклонение σ_x = √D_x."""
    D_x = np.asarray(D_x, dtype=float)
    return np.sqrt(D_x)


# ---------------------------------------------------------------------------
# 1.1.8  nu = sigma_x / m_x
# ---------------------------------------------------------------------------
def coeff_variation(sigma_x, m_x):
    """(1.1.8): коэффициент вариации ν = σ_x / m_x (векторно по узлам МКЭ)."""
    sigma_x = np.asarray(sigma_x, dtype=float)
    m_x = np.asarray(m_x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return sigma_x / m_x


# ---------------------------------------------------------------------------
# 1.1.9  F(x) = P(X < x) = ∫_{-inf}^{x} f(t) dt
# ---------------------------------------------------------------------------
def cdf_from_pdf(f, x, grid=None):
    """(1.1.9): функция распределения F(x) по ПП f(x).

    Табличный режим: кумулятивные трапеции по сетке `grid` с последующей
    линейной интерполяцией в точках `x` — O(N) для всего массива.
    """
    x = np.asarray(x, dtype=float)
    if callable(f):
        out = np.empty(x.shape, dtype=float)
        flat_x, flat_out = x.ravel(), out.ravel()
        for i, xi in enumerate(flat_x):
            flat_out[i] = quad(f, -np.inf, xi, limit=200)[0]
        return out
    f = np.asarray(f, dtype=float)
    grid = np.asarray(grid, dtype=float)
    F_grid = cumulative_trapezoid(f, grid, initial=0.0)
    return np.interp(x, grid, F_grid)


# ---------------------------------------------------------------------------
# 1.1.10  f(x) = ∫ f1(x1) f2(x - x1) dx1  (свертка)
# ---------------------------------------------------------------------------
def pdf_of_sum_convolution(f1, f2, grid1, grid2=None):
    """(1.1.10): ПП суммы двух независимых СВ как свертка плотностей.

    Векторизованный быстрый путь: дискретная свертка `scipy.signal.fftconvolve`
    эквивалентна интегралу (1.1.10); здесь — реализация через NumPy-свертку
    с шагом сетки. Возвращает (сетка x, значения f(x)).
    """
    from scipy.signal import fftconvolve

    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    grid1 = np.asarray(grid1, dtype=float)
    grid2 = grid1 if grid2 is None else np.asarray(grid2, dtype=float)
    dx1 = grid1[1] - grid1[0]
    dx2 = grid2[1] - grid2[0]
    if not np.isclose(dx1, dx2):
        raise ValueError("Сетки должны иметь одинаковый шаг.")
    fx = fftconvolve(f1, f2, mode="full") * dx1
    x = np.arange(fx.size) * dx1 + grid1[0] + grid2[0]
    return x, fx


# ---------------------------------------------------------------------------
# 1.1.11 / 1.1.12  m_x = ∬ (x1+x2) f1 f2 dx1 dx2 ; свойства нормировки
# ---------------------------------------------------------------------------
def mean_of_sum_double_integral(f1, f2, grid1, grid2):
    """(1.1.11): мат. ожидание суммы через двойной интеграл.

    Считается как ∬ (x1+x2)·f1(x1)·f2(x2) dx1 dx2 внешним произведением —
    одна векторизованная операция над сетками.
    """
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    grid1 = np.asarray(grid1, dtype=float)
    grid2 = np.asarray(grid2, dtype=float)
    X1, X2 = np.meshgrid(grid1, grid2, indexing="ij")
    dens = f1[:, None] * f2[None, :]
    inner = trapezoid((X1 + X2) * dens, grid2, axis=1)
    return float(trapezoid(inner, grid1))


def normalization_and_means(f1, f2, grid1, grid2):
    """(1.1.12): проверки ∫f1=1, ∫f2=1 и значения m_x1, m_x2."""
    f1, f2 = np.asarray(f1, float), np.asarray(f2, float)
    g1, g2 = np.asarray(grid1, float), np.asarray(grid2, float)
    return {
        "norm_f1": float(trapezoid(f1, g1)),
        "norm_f2": float(trapezoid(f2, g2)),
        "m_x1": float(trapezoid(g1 * f1, g1)),
        "m_x2": float(trapezoid(g2 * f2, g2)),
    }


# ---------------------------------------------------------------------------
# 1.1.13 – 1.1.15  линейность мат. ожидания
# ---------------------------------------------------------------------------
def mean_sum(m_x1, m_x2):
    """(1.1.13): m_x = m_x1 + m_x2."""
    return np.asarray(m_x1, float) + np.asarray(m_x2, float)


def mean_difference(m_x1, m_x2):
    """(1.1.14): m_x = m_x1 − m_x2."""
    return np.asarray(m_x1, float) - np.asarray(m_x2, float)


def mean_total(*m_list):
    """(1.1.15): m_x = Σ_i m_{x_i}."""
    arrs = [np.asarray(m, float) for m in (m_list[0] if len(m_list) == 1 and np.ndim(m_list[0]) == 1 and not np.isscalar(m_list[0]) else m_list)]
    return np.sum(arrs, axis=0)


# ---------------------------------------------------------------------------
# 1.1.16 – 1.1.19  законы распределения (векторизованные ПП)
# ---------------------------------------------------------------------------
def pdf_exponential(x, lam):
    """(1.1.16): показательное распределение f(x)=λe^{−λx}, x≥0."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0.0, lam * np.exp(-lam * np.clip(x, 0.0, None)), 0.0)


def pdf_uniform(x, a, b):
    """(1.1.17): равномерное распределение f(x)=1/(b−a) на [a,b]."""
    x = np.asarray(x, dtype=float)
    return np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)


def pdf_triangular(x, a, b):
    """(1.1.18): треугольное распределение с модой в середине [a,b]."""
    x = np.asarray(x, dtype=float)
    mid = 0.5 * (a + b)
    denom = (b - a) ** 2
    left = 4.0 * (x - a) / denom
    right = 4.0 * (b - x) / denom
    return np.where((x >= a) & (x <= mid), left,
                    np.where((x > mid) & (x <= b), right, 0.0))


def pdf_normal(x, m_x, sigma_x):
    """(1.1.19): нормальное распределение.

    f(x) = 1/(σ√(2π)) · exp[−(x−m)²/(2σ²)]
    Массивы любой формы; σ может быть массивом (поле по узлам МКЭ).
    """
    x = np.asarray(x, dtype=float)
    m_x = np.asarray(m_x, dtype=float)
    sigma_x = np.asarray(sigma_x, dtype=float)
    inv_2s2 = 1.0 / (2.0 * sigma_x ** 2)
    norm = 1.0 / (sigma_x * np.sqrt(2.0 * np.pi))
    return norm * np.exp(-((x - m_x) ** 2) * inv_2s2)


# ---------------------------------------------------------------------------
# 1.1.20 – 1.1.23  вероятности для нормального закона
# ---------------------------------------------------------------------------
def prob_normal_interval(x1, x2, m_x, sigma_x):
    """(1.1.20): P(x1<x<x2) для нормального закона (через erf — точно и быстро)."""
    z1 = (np.asarray(x1, float) - m_x) / sigma_x
    z2 = (np.asarray(x2, float) - m_x) / sigma_x
    return 0.5 * (erf(z2 / np.sqrt(2.0)) - erf(z1 / np.sqrt(2.0)))


def standardize(x, m_x, sigma_x):
    """(1.1.21): стандартизация z = (x − m_x)/σ_x (массив или кортеж границ)."""
    x = np.asarray(x, dtype=float)
    return (x - np.asarray(m_x, float)) / np.asarray(sigma_x, float)


def laplace_function(z):
    """(1.1.22): функция Лапласа Φ(z) = (1/√(2π))∫_0^z e^{−t²/2} dt.

    Непрерывная функция ошибок: Φ(z) = ½·erf(z/√2). Векторизовано.
    """
    z = np.asarray(z, dtype=float)
    return 0.5 * erf(z / np.sqrt(2.0))


def prob_normal_laplace(x1, x2, m_x, sigma_x):
    """(1.1.23): P(x1<x<x2) = Φ((x2−m)/σ) − Φ((x1−m)/σ)."""
    return laplace_function((x2 - m_x) / sigma_x) - laplace_function((x1 - m_x) / sigma_x)


# ---------------------------------------------------------------------------
# 1.1.24 – 1.1.26  интервальные оценки
# ---------------------------------------------------------------------------
def tolerance_limits_normal(m_x, sigma_x, z_p):
    """(1.1.24): x1 = m − z_p·σ, x2 = m + z_p·σ."""
    dz = np.asarray(z_p, float) * np.asarray(sigma_x, float)
    m_x = np.asarray(m_x, float)
    return m_x - dz, m_x + dz


def three_sigma_rule():
    """(1.1.25): P(|x − m| < 3σ) ≈ 0.9973 (точное значение через erf)."""
    return float(2.0 * laplace_function(3.0))  # ≈ 0.997300203...


def normal_params_from_range(x_min, x_max):
    """(1.1.26): m = (xmax+xmin)/2, σ = (xmax−xmin)/6 (векторно по узлам)."""
    x_min = np.asarray(x_min, float)
    x_max = np.asarray(x_max, float)
    m = 0.5 * (x_max + x_min)
    s = (x_max - x_min) / 6.0
    return m, s


# ---------------------------------------------------------------------------
# 1.1.27  распределение Вейбулла
# ---------------------------------------------------------------------------
def pdf_weibull(x, alpha, beta):
    """(1.1.27): ПП Вейбулла f(x)=(α/β)(x/β)^{α−1} exp[−(x/β)^α], x≥0."""
    x = np.asarray(x, dtype=float)
    xs = np.clip(x, 0.0, None)
    xb = xs / beta
    # при α<1 в нуле ПП → ∞; обнуляем отрицательные аргументы
    val = (alpha / beta) * np.power(xb, alpha - 1.0) * np.exp(-np.power(xb, alpha))
    return np.where(x >= 0.0, val, 0.0)
