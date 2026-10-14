"""
Глава 1.2 — Теория допусков и посадок (статистический метод).

Формулы (1.2.1) – (1.2.15). Функции векторизованы: допускают расчет
полей допусков для массивов соединений (тысясы пар "вал-отверстие").
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

__all__ = [
    "roughness_ra",                  # (1.2.1)
    "mean_diameters",                # (1.2.2)
    "clearance_nominal",             # (1.2.3)
    "std_clearance_components",      # (1.2.4)
    "std_from_extremes",             # (1.2.5)
    "std_clearance_extremes",        # (1.2.6)
    "clearance_limits_3sigma",       # (1.2.7)
    "interference_nominal",          # (1.2.8)
    "interference_limits",           # (1.2.9)
    "dim_mid_and_tol",               # (1.2.10)
    "dim_extremes_from_mid",         # (1.2.11)
    "chain_mean",                    # (1.2.12)
    "chain_dispersion",              # (1.2.13)
    "component_std_from_tol",        # (1.2.14)
    "chain_tolerance",               # (1.2.15)
]


# ---------------------------------------------------------------------------
# 1.2.1  R_a = (1/l) ∫_0^l |y(x)| dx
# ---------------------------------------------------------------------------
def roughness_ra(y, x=None):
    """(1.2.1): параметр шероховатости Ra по профилю y(x).

    y — массив отклонений профиля (например, данные профилометра или
    результат МКЭ-моделирования поверхности); интегрирование трапециями.
    """
    y = np.asarray(y, dtype=float)
    if x is None:
        l = float(y.size - 1)   # единичный шаг по индексу
        integral = float(trapezoid(np.abs(y)))
    else:
        x = np.asarray(x, dtype=float)
        l = float(x[-1] - x[0])
        integral = float(trapezoid(np.abs(y), x))
    return integral / l


# ---------------------------------------------------------------------------
# 1.2.2  d = (d_max+d_min)/2, D = (D_max+D_min)/2
# ---------------------------------------------------------------------------
def mean_diameters(d_max, d_min, D_max, D_min):
    """(1.2.2): средние диаметры вала d и отверстия D (векторно по соединениям)."""
    d = 0.5 * (np.asarray(d_max, float) + np.asarray(d_min, float))
    D = 0.5 * (np.asarray(D_max, float) + np.asarray(D_min, float))
    return d, D


# ---------------------------------------------------------------------------
# 1.2.3  N = d − D   (зазор/натяк по средним)
# ---------------------------------------------------------------------------
def clearance_nominal(d, D):
    """(1.2.3): N = d − D."""
    return np.asarray(d, float) - np.asarray(D, float)


# ---------------------------------------------------------------------------
# 1.2.4  σ_N = sqrt(σ_d² + σ_D²)
# ---------------------------------------------------------------------------
def std_clearance_components(sigma_d, sigma_D):
    """(1.2.4): σ_N = √(σ_d² + σ_D²)."""
    sd = np.asarray(sigma_d, float)
    sD = np.asarray(sigma_D, float)
    return np.sqrt(sd ** 2 + sD ** 2)


# ---------------------------------------------------------------------------
# 1.2.5  σ_d = (d_max−d_min)/6, σ_D = (D_max−D_min)/6
# ---------------------------------------------------------------------------
def std_from_extremes(x_max, x_min):
    """(1.2.5): СКО размера из крайних значений, σ = (xmax − xmin)/6."""
    return (np.asarray(x_max, float) - np.asarray(x_min, float)) / 6.0


# ---------------------------------------------------------------------------
# 1.2.6  σ_N = (1/6)·sqrt((d_max−d_min)² + (D_max−D_min)²)
# ---------------------------------------------------------------------------
def std_clearance_extremes(d_max, d_min, D_max, D_min):
    """(1.2.6): СКО зазора непосредственно из крайних значений размеров."""
    dd = np.asarray(d_max, float) - np.asarray(d_min, float)
    DD = np.asarray(D_max, float) - np.asarray(D_min, float)
    return np.sqrt(dd ** 2 + DD ** 2) / 6.0


# ---------------------------------------------------------------------------
# 1.2.7  N_max = N + 3σ_N, N_min = N − 3σ_N
# ---------------------------------------------------------------------------
def clearance_limits_3sigma(N, sigma_N):
    """(1.2.7): предельные значения зазора по правилу трех сигм."""
    N = np.asarray(N, float)
    sN = np.asarray(sigma_N, float)
    return N + 3.0 * sN, N - 3.0 * sN


# ---------------------------------------------------------------------------
# 1.2.8  S = D − d   (зазор посадки)
# ---------------------------------------------------------------------------
def interference_nominal(D, d):
    """(1.2.8): S = D − d."""
    return np.asarray(D, float) - np.asarray(d, float)


# ---------------------------------------------------------------------------
# 1.2.9  S_min = S − z_p σ_S, S_max = S + z_p σ_S, σ_S = √(σ_D² + σ_d²)
# ---------------------------------------------------------------------------
def interference_limits(S, sigma_D, sigma_d, z_p):
    """(1.2.9): предельные зазоры посадки при вероятности P (через z_p)."""
    sigma_S = std_clearance_components(sigma_d, sigma_D)
    S = np.asarray(S, float)
    dz = np.asarray(z_p, float) * sigma_S
    return S - dz, S + dz


# ---------------------------------------------------------------------------
# 1.2.10  m_Ai = (Aimax + Aimin)/2, Ti = Aimax − Aimin
# ---------------------------------------------------------------------------
def dim_mid_and_tol(A_max, A_min):
    """(1.2.10): середина поля допуска m_A и допуск T (векторно по звеньям)."""
    A_max = np.asarray(A_max, float)
    A_min = np.asarray(A_min, float)
    m = 0.5 * (A_max + A_min)
    T = A_max - A_min
    return m, T


# ---------------------------------------------------------------------------
# 1.2.11  Aimax = m_A + T/2, Aimin = m_A − T/2
# ---------------------------------------------------------------------------
def dim_extremes_from_mid(m_A, T):
    """(1.2.11): крайние значения размера из середины поля и допуска."""
    m_A = np.asarray(m_A, float)
    halfT = np.asarray(T, float) / 2.0
    return m_A + halfT, m_A - halfT


# ---------------------------------------------------------------------------
# 1.2.12  m_Δ = m_A1 − m_A2 − m_A3
# ---------------------------------------------------------------------------
def chain_mean(*m_components, increasing=None):
    """(1.2.12): средний размер замыкающего звена размерной цепи.

    chain_mean(m1, m2, m3) → m1 − m2 − m3 (первое звено увеличивающее,
    остальные уменьшающие — как в формуле). Либо явный список знаков
    `increasing=[True, False, False]`.
    """
    ms = [np.asarray(m, float) for m in m_components]
    if increasing is None:
        signs = [1.0] + [-1.0] * (len(ms) - 1)
    else:
        signs = [1.0 if s else -1.0 for s in increasing]
    out = np.zeros_like(ms[0])
    for s, m in zip(signs, ms):
        out = out + s * m
    return out


# ---------------------------------------------------------------------------
# 1.2.13  D_Δ = D_A1 + D_A2 + D_A3
# ---------------------------------------------------------------------------
def chain_dispersion(*D_components):
    """(1.2.13): дисперсия замыкающего звена = сумма дисперсий звеньев."""
    arrs = [np.asarray(D, float) for D in D_components]
    return np.sum(arrs, axis=0)


# ---------------------------------------------------------------------------
# 1.2.14  D_Ai = (Ti/6)²  либо  σ_Ai = Ti/6
# ---------------------------------------------------------------------------
def component_std_from_tol(T, return_variance=False):
    """(1.2.14): σ_Ai = Ti/6; при return_variance=True — D_Ai = (Ti/6)²."""
    sigma = np.asarray(T, float) / 6.0
    return sigma ** 2 if return_variance else sigma


# ---------------------------------------------------------------------------
# 1.2.15  T_Δ = sqrt(T1² + T2² + T3²)
# ---------------------------------------------------------------------------
def chain_tolerance(*T_components):
    """(1.2.15): допуск замыкающего звена по статистической формуле."""
    arrs = [np.asarray(T, float) for T in T_components]
    return np.sqrt(np.sum(np.square(arrs), axis=0))
