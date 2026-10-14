"""
Общие примитивы расчетного ядра (Этап 0 плана модернизации).

Векторизованные сервисы, переиспользуемые модулями глав 2.x / 3.x:
    - приведение входов к общим формам (поля МКЭ произвольной формы);
    - собственные значения симметричного тензора напряжений (массово);
    - инварианты и главные характеристики напряженного состояния;
    - von Mises / Треска-Сен-Венана из компонент (для полей из МКЭ);
    - кусочно-линейные таблицы (S-N кривые, коэффициенты материалов);
    - решение скалярных уравнений векторно (brentq, массовый прогон).

Единицы: конвенция СИ+МПа (Н, мм, МПа) — см. README.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "as_float_array",
    "broadcast_fields",
    "principal_stresses",
    "stress_invariants",
    "von_mises_from_components",
    "tresca_from_components",
    "interp_table",
    "solve_scalar_vectorized",
]


def as_float_array(x):
    """Привести вход к float-массиву NumPy (скаляры тоже работают)."""
    return np.asarray(x, dtype=float)


def broadcast_fields(*fields):
    """Привести набор полей (узлы/элементы МКЭ) к общей форме вещания."""
    arrs = [as_float_array(f) for f in fields]
    return np.broadcast_arrays(*arrs)


# ---------------------------------------------------------------------------
# Напряженное состояние: главный вектор для symmetric eigvalsh над полем
# ---------------------------------------------------------------------------
def principal_stresses(sxx, syy, szz, sxy=0.0, syz=0.0, sxz=0.0, sort_desc=True):
    """Главные напряжения тензора (σ1 ≥ σ2 ≥ σ3 при sort_desc=True).

    Вход — поля компонент любой одинаковой/вещаемой формы, например
    (n_nodes,) или (n_nodes, 6) с инженерной записью [xx, yy, zz, xy, yz, xz].
    Реализация: аналитический метод Кардано (векторно, O(N)), без циклов —
    быстрее и устойчивее, чем np.linalg.eigvalsh на больших полях.

    Формула: корни характеристического уравнения
        λ³ − I1 λ² + I2 λ − I3 = 0
    через тригонометрическое решение (случай 3 вещественных корней).
    """
    sxx, syy, szz, sxy, syz, sxz = broadcast_fields(sxx, syy, szz, sxy, syz, sxz)

    I1 = sxx + syy + szz
    I2 = sxx * syy + syy * szz + szz * sxx - sxy**2 - syz**2 - sxz**2
    I3 = (sxx * syy * szz + 2.0 * sxy * syz * sxz
          - sxx * syz**2 - syy * sxz**2 - szz * sxy**2)

    p2 = I1 * I1 / 3.0 - I2                     # >0 для симм. тензора
    p2 = np.maximum(p2, 0.0)
    # det(C) тензора девиатора C = S − (I1/3) E
    detC = (2.0 * I1**3 / 27.0 - I1 * I2 / 3.0 + I3)
    # Девиатор: p = −½ tr(CᵀC) ≤ 0, q = det(C). Тригонометрический корень:
    #   cos(3θ) = 3√3·q / (2·(−p)^{3/2}),  λ_k = I1/3 + 2√(−p/3)·cos(θ − 2πk/3)
    with np.errstate(divide="ignore", invalid="ignore"):
        neg_p = np.maximum(p2, 0.0)             # −p ≥ 0
        cos3t = np.where(neg_p > 0.0,
                         3.0 * np.sqrt(3.0) * detC /
                         (2.0 * np.power(np.maximum(neg_p, 1e-300), 1.5)),
                         1.0)
    cos3t = np.clip(cos3t, -1.0, 1.0)
    theta = np.arccos(cos3t) / 3.0
    amp = 2.0 * np.sqrt(p2 / 3.0)
    mean = I1 / 3.0

    # Собственные значения (λ1 ≥ λ2 ≥ λ3):
    #   λ_k = mean + amp·cos(θ − 2πk/3), θ ∈ [0, π/3]
    l1 = mean + amp * np.cos(theta)
    l2 = mean + amp * np.cos(theta - 2.0 * np.pi / 3.0)
    l3 = mean + amp * np.cos(theta + 2.0 * np.pi / 3.0)

    if sort_desc:
        # l1 >= l2 >= l3 по построению (cos θ ≥ cos(θ−2π/3) ≥ cos(θ−4π/3), θ∈[0,π/3])
        return l1, l2, l3
    return l3, l2, l1


def stress_invariants(sxx, syy, szz, sxy=0.0, syz=0.0, sxz=0.0):
    """Инварианты тензора напряжений I1, I2, I3 (поле любой формы)."""
    sxx, syy, szz, sxy, syz, sxz = broadcast_fields(sxx, syy, szz, sxy, syz, sxz)
    I1 = sxx + syy + szz
    I2 = sxx * syy + syy * szz + szz * sxx - sxy**2 - syz**2 - sxz**2
    I3 = (sxx * syy * szz + 2.0 * sxy * syz * sxz
          - sxx * syz**2 - syy * sxz**2 - szz * sxy**2)
    return I1, I2, I3


def von_mises_from_components(sxx, syy, szz, sxy=0.0, syz=0.0, sxz=0.0):
    """Эквивалентное напряжение Мизеса из компонент (3.1.31-контекст, 2.7.31).

    σ_v = √( ½[(σx−σy)²+(σy−σz)²+(σz−σx)²] + 3(τxy²+τyz²+τxz²) )
    """
    sxx, syy, szz, sxy, syz, sxz = broadcast_fields(sxx, syy, szz, sxy, syz, sxz)
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy**2 + syz**2 + sxz**2))


def tresca_from_components(sxx, syy, szz, sxy=0.0, syz=0.0, sxz=0.0):
    """Эквивалентное напряжение по теории макс. касательных (Треска)."""
    a, b, c = principal_stresses(sxx, syy, szz, sxy, syz, sxz)
    return 0.5 * (a - c)


def interp_table(x, xp, fp, left=None, right=None):
    """Кусочно-линейная интерполяция справочных таблиц (S-N, k_t, ψ...).

    Обёртка над np.interp с поддержкой массивов x любой формы и явными
    экстраполяционными значениями краёв.
    """
    x = as_float_array(x)
    shape = x.shape
    out = np.interp(x.ravel(), np.asarray(xp, float), np.asarray(fp, float),
                    left=-np.inf if left is None else left,
                    right=-np.inf if right is None else right)
    if left is not None:
        out[x.ravel() < xp[0]] = left
    if right is not None:
        out[x.ravel() > xp[-1]] = right
    return out.reshape(shape)


def solve_scalar_vectorized(func_factory, x0_bounds, args_fields, tol=1e-10):
    """Массовое решение скалярного уравнения func(x)=0 по элементам поля.

    func_factory(args_tuple) -> callable f(x);
    x0_bounds — кортеж (a, b) гарантированного интервала корня;
    args_fields — broadcasting-набор параметров (например поля σm, σa);
    Возвращает массив решений той же формы. Используется для неявных
    формул (например обратные S-N зависимости, k_σ(ψ)).
    """
    arrs = broadcast_fields(*args_fields)
    flat_shape = arrs[0].shape
    it = zip(*[a.ravel() for a in arrs])
    out = np.empty(flat_shape, dtype=float)
    o = out.ravel()
    a, b = x0_bounds
    for i, vals in enumerate(it):
        f = func_factory(vals)
        o[i] = brentq(f, a, b, xtol=tol)
    return out.reshape(flat_shape)
