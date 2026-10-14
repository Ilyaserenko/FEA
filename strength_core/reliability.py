"""
Глава 1.3 — Надежность: безотказность, долговечность, вероятность безотказной работы.

Формулы (1.3.1) – (1.3.21). Векторизованный расчет по массивам времени/узлов;
структурные схемы надежности (1.3.14–1.3.16) принимают матрицы P_i(t) вида
(элементы × моменты времени).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid, quad

# переиспользуем функцию Лапласа из главы 1.1
from .stats import laplace_function

__all__ = [
    "prob_lifetime_interval",        # (1.3.1)
    "cdf_lifetime",                  # (1.3.2)
    "reliability_func",              # (1.3.3)
    "prob_interval_via_P",           # (1.3.4)
    "hazard_rate_estimate",          # (1.3.5)
    "hazard_rate_limit",             # (1.3.6)
    "pdf_from_reliability",          # (1.3.7)
    "hazard_rate_from_P",            # (1.3.8)
    "reliability_general_exponential",   # (1.3.9)
    "reliability_exponential",       # (1.3.10)
    "conditional_reliability",       # (1.3.10, условная P(t|t1))
    "hazard_constant",               # (1.3.11)
    "prob_lifespan_normal",          # (1.3.12)
    "reliability_normal",            # (1.3.13)
    "weibull_alpha1",                # (1.3.13, Вейбулл при α=1)
    "reliability_series",            # (1.3.14)
    "reliability_parallel",          # (1.3.15) и (1.3.16)
    "pdf_strength_limit",            # (1.3.17)
    "pdf_stress",                    # (1.3.18)
    "safety_margin_params",          # (1.3.19)
    "pdf_margin",                    # (1.3.20)
    "safety_index_zp",               # (1.3.21)
    "coefficient_n_p",               # (1.3.21, n_p через ν, ν_lim, z_p)
]


# ---------------------------------------------------------------------------
# 1.3.1  P(t1 < T < t2) = ∫_{t1}^{t2} f(t) dt
# ---------------------------------------------------------------------------
def prob_lifetime_interval(f, t1, t2, grid=None):
    """(1.3.1): вероятность безотказной работы в интервале (t1, t2)."""
    if callable(f):
        return float(quad(f, t1, t2, limit=200)[0])
    f = np.asarray(f, float)
    grid = np.asarray(grid, float)
    mask = (grid >= t1) & (grid <= t2)
    if mask.sum() < 2:
        return 0.0
    return float(trapezoid(f[mask], grid[mask]))


# ---------------------------------------------------------------------------
# 1.3.2  F(t) = P(T ≤ t) = ∫_0^t f(τ) dτ
# ---------------------------------------------------------------------------
def cdf_lifetime(f, t, grid=None):
    """(1.3.2): F(t) — распределение ресурса; табличный режим O(N)."""
    t = np.asarray(t, float)
    if callable(f):
        flat = t.ravel()
        out = np.array([quad(f, 0.0, ti, limit=200)[0] for ti in flat])
        return out.reshape(t.shape)
    f = np.asarray(f, float)
    grid = np.asarray(grid, float)
    F_grid = cumulative_trapezoid(f, grid, initial=0.0)
    return np.interp(t, grid, F_grid)


# ---------------------------------------------------------------------------
# 1.3.3  P(t) = P(T > t) = 1 − F(t) = ∫_t^∞ f(τ) dτ
# ---------------------------------------------------------------------------
def reliability_func(f_or_F, t=None, grid=None):
    """(1.3.3): вероятность безотказной работы P(t) = 1 − F(t).

    Принимает либо значения F(t) (массив), либо ПП f на сетке `grid` и точки t.
    """
    arr = np.asarray(f_or_F, float)
    if t is None:                      # arr — уже F(t)
        return 1.0 - arr
    F = cdf_lifetime(arr, t, grid=grid)
    return 1.0 - F


# ---------------------------------------------------------------------------
# 1.3.4  P(t1<T<t2) = P(t1) − P(t2) = F(t2) − F(t1)
# ---------------------------------------------------------------------------
def prob_interval_via_P(P_t1, P_t2, F_t1=None, F_t2=None):
    """(1.3.4): вероятность интервала через P или F."""
    p = np.asarray(P_t1, float) - np.asarray(P_t2, float)
    if F_t1 is not None and F_t2 is not None:
        q = np.asarray(F_t2, float) - np.asarray(F_t1, float)
        return p, q
    return p


# ---------------------------------------------------------------------------
# 1.3.5  λ = Δn / (N(t)·Δt)
# ---------------------------------------------------------------------------
def hazard_rate_estimate(delta_n, N_t, delta_t):
    """(1.3.5): оценка интенсивности отказов по статистике партий."""
    dn = np.asarray(delta_n, float)
    N = np.asarray(N_t, float)
    dt = np.asarray(delta_t, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dn / (N * dt)


# ---------------------------------------------------------------------------
# 1.3.6  λ(t) = lim_{Δt→0} Δn/(N(t)Δt)
# ---------------------------------------------------------------------------
def hazard_rate_limit(delta_n, N_t, delta_t, order=2):
    """(1.3.6): λ(t) как предел — экстраполяция оценок (1.3.5) к Δt→0.

    Считает λ_i по окнам и выполняет полиномиальную extrapolation (Richardson)
    к нулевому шагу; для больших рядов данных — векторизованная скользящая
    оценка с шагом `delta_t`.
    """
    lam = hazard_rate_estimate(delta_n, N_t, delta_t)
    lam = np.asarray(lam, float)
    if lam.size == 1:
        return float(lam[0])
    # Richardson-экстраполяция по последним двум оценкам разного шага
    dt = np.asarray(delta_t, float)
    if dt.size >= 2 and not np.isclose(dt[-1], dt[-2]):
        r = dt[-2] / dt[-1]
        rich = (lam[-1] * r ** order - lam[-2]) / (r ** order - 1.0)
        return float(rich)
    return float(np.mean(lam))


# ---------------------------------------------------------------------------
# 1.3.7  f(t) = −dP(t)/dt
# ---------------------------------------------------------------------------
def pdf_from_reliability(P_t, t):
    """(1.3.7): ПП ресурса из численного дифференцирования P(t).

    Центрально-разностная схема `np.gradient` — O(N), годится для массивов
    миллионов точек.
    """
    P_t = np.asarray(P_t, float)
    t = np.asarray(t, float)
    return -np.gradient(P_t, t)


# ---------------------------------------------------------------------------
# 1.3.8  λ(t) = f(t)/P(t) = −(1/P) dP/dt
# ---------------------------------------------------------------------------
def hazard_rate_from_P(P_t, t, f_t=None):
    """(1.3.8): λ(t) = f(t)/P(t); если f не задан — из (1.3.7)."""
    P_t = np.asarray(P_t, float)
    f_t = pdf_from_reliability(P_t, t) if f_t is None else np.asarray(f_t, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = f_t / P_t
    return lam


# ---------------------------------------------------------------------------
# 1.3.9  P(t) = exp[−∫_0^t λ(τ) dτ]
# ---------------------------------------------------------------------------
def reliability_general_exponential(lambda_t, t, grid=None):
    """(1.3.9): общая экспоненциальная форма P(t) при произвольном λ(t).

    Быстрый путь: табличный λ на `grid`, кумулятивные трапеции + exp.
    Точный: callable λ и скалярные t через quad.
    """
    t = np.asarray(t, float)
    if callable(lambda_t):
        flat = t.ravel()
        out = np.exp([-quad(lambda_t, 0.0, ti, limit=200)[0] for ti in flat])
        return out.reshape(t.shape)
    lam = np.asarray(lambda_t, float)
    grid = np.asarray(grid, float)
    Lam = cumulative_trapezoid(lam, grid, initial=0.0)
    return np.exp(-np.interp(t, grid, Lam))


# ---------------------------------------------------------------------------
# 1.3.10  P(t) = e^{−λt};  P(t|t1) = e^{−λ(t−t1)}
# ---------------------------------------------------------------------------
def reliability_exponential(t, lam):
    """(1.3.10): экспоненциальная надежность P(t)=e^{−λt}.

    t и lam могут быть массивами одинаковой формы (поле по элементам МКЭ).
    """
    t = np.asarray(t, float)
    lam = np.asarray(lam, float)
    return np.exp(-lam * t)


def conditional_reliability(t, t1, lam):
    """(1.3.10'): условная вероятность P(t|t1)=e^{−λ(t−t1)}, t>t1."""
    t = np.asarray(t, float)
    t1 = np.asarray(t1, float)
    lam = np.asarray(lam, float)
    dt = np.clip(t - t1, 0.0, None)
    return np.exp(-lam * dt)


# ---------------------------------------------------------------------------
# 1.3.11  λ(t) = const
# ---------------------------------------------------------------------------
def hazard_constant(lam):
    """(1.3.11): постоянная интенсивность отказов — возвращает константу/массив."""
    return np.asarray(lam, float)


# ---------------------------------------------------------------------------
# 1.3.12 / 1.3.13  логарифмически-нормальный (нормальный) закон ресурса
# ---------------------------------------------------------------------------
def prob_lifespan_normal(t1, t2, m_t, sigma_t):
    """(1.3.12): P(t1<T<t2) = Φ((t2−m)/σ) − Φ((t1−m)/σ)."""
    return (laplace_function((np.asarray(t2, float) - m_t) / sigma_t)
            - laplace_function((np.asarray(t1, float) - m_t) / sigma_t))


def reliability_normal(t, m_t, sigma_t):
    """(1.3.13): P(t) = 0.5 − Φ((t − m_t)/σ_t)."""
    return 0.5 - laplace_function((np.asarray(t, float) - m_t) / sigma_t)


def weibull_alpha1(t, beta):
    """(1.3.13, примечание): Вейбулл при α=1 → F(t)=1−e^{−t/β}, P(t)=e^{−t/β}, λ=1/β."""
    t = np.asarray(t, float)
    P = np.exp(-t / beta)
    return {"F": 1.0 - P, "P": P, "lambda": np.full_like(t, 1.0 / beta)}


# ---------------------------------------------------------------------------
# 1.3.14  Pc(t) = Π Pi(t)  — последовательная структура
# ---------------------------------------------------------------------------
def reliability_series(P_components, axis=-1):
    """(1.3.14): надежность последовательной системы Pc(t) = Π Pi(t).

    P_components — массив (n_элементов, n_моментов) или список массивов;
    произведение выполняется векторно вдоль оси элементов.
    """
    arrs = np.asarray(P_components, float)
    return np.prod(arrs, axis=axis)


# ---------------------------------------------------------------------------
# 1.3.15  Qc(t) = Π Qi(t) = Π (1 − Pi(t))  — параллельная структура (отказ)
# ---------------------------------------------------------------------------
def reliability_parallel(P_components, axis=-1):
    """(1.3.15)+(1.3.16): Qc = Π(1−Pi); надежность параллельной Pc = 1 − Qc.

    Возвращает кортеж (Qc, Pc).
    """
    arrs = np.asarray(P_components, float)
    Qc = np.prod(1.0 - arrs, axis=axis)
    return Qc, 1.0 - Qc


# ---------------------------------------------------------------------------
# 1.3.17 – 1.3.21  метод "напряжение – прочность"
# ---------------------------------------------------------------------------
def pdf_strength_limit(sigma_lim, mean_sigma_lim, S_lim):
    """(1.3.17): ПП предела прочности f_lim(σ_lim) — нормальный закон."""
    from .stats import pdf_normal
    return pdf_normal(sigma_lim, mean_sigma_lim, S_lim)


def pdf_stress(sigma, mean_sigma, S):
    """(1.3.18): ПП действующих напряжений f(σ) — нормальный закон."""
    from .stats import pdf_normal
    return pdf_normal(sigma, mean_sigma, S)


def safety_margin_params(mean_sigma_lim, mean_sigma, S_lim, S):
    """(1.3.19): m = σ̄_lim − σ̄, S_0 = √(S_lim² + S²)."""
    m = np.asarray(mean_sigma_lim, float) - np.asarray(mean_sigma, float)
    S0 = np.sqrt(np.asarray(S_lim, float) ** 2 + np.asarray(S, float) ** 2)
    return m, S0


def pdf_margin(z, m, S0):
    """(1.3.20): ПП запаса z = σ_lim − σ есть нормальный закон N(m, S0)."""
    from .stats import pdf_normal
    return pdf_normal(z, m, S0)


def safety_index_zp(mean_sigma_lim, mean_sigma, S_lim, S):
    """(1.3.21): z_p = −(σ̄_lim − σ̄)/√(S_lim² + S²)  (квантиль/индекс надежности).

    Соглашение о знаке: при положительном среднем запасе прочности
    (σ̄_lim > σ̄) возвращаемое значение ПОЛОЖИТЕЛЬНО и совпадает с классическим
    индексом надежности β = m/S_0; вероятность безотказной работы P = Φ(z_p),
    а в терминах функции Лапласа из (1.1.22): P = 0.5 + Φ(z_p).
    Это соответствует формуле n_p(z_p) из (1.3.21), где z_p > 0 ⟹ n_p > 1.
    (Знак «−» в записи учебника обусловлен тем, что там z_p — квантиль области
    отказа z_p = (0 − m)/S_0.)
    """
    m, S0 = safety_margin_params(mean_sigma_lim, mean_sigma, S_lim, S)
    return m / S0


def coefficient_n_p(z_p, v, v_lim):
    """(1.3.21): коэффициент надежности

        n_p = [1 + z_p·√(v² + v_lim² − z_p²·v²·v_lim²)] / (1 − z_p²·v_lim²)

    где n_p = σ̄_lim/σ̄, v = S/σ̄, v_lim = S_lim/σ̄_lim. Векторно по массивам z_p.
    """
    z_p = np.asarray(z_p, float)
    v = np.asarray(v, float)
    vl = np.asarray(v_lim, float)
    disc = np.maximum(v ** 2 + vl ** 2 - (z_p ** 2) * (v ** 2) * (vl ** 2), 0.0)
    num = 1.0 + z_p * np.sqrt(disc)
    den = 1.0 - (z_p ** 2) * (vl ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den
