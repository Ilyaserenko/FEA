"""
Глава 2.7 — Напряженное состояние в точке. Теории предельных напряженных
состояний (теории прочности). Формулы (2.7.1) – (2.7.42).

Все функции векторизованы: принимают поля главных/компонентных напряжений
МКЭ-модели произвольной формы ((n_nodes,), (n_elem, m), скаляры) и возвращают
поля того же размера. LaTeX-записи восстановлены из контекста учебника
(картинки формул в исходном HTML утрачены) — см. docs/formula_registry.csv.

Обозначения: s1, s2, s3 — главные напряжения; sx..sz, txy.. — компоненты;
[σ] — допускаемые напряжения; σ_B — предел прочности, σ_T — предел текучести.
"""

from __future__ import annotations

import numpy as np

from .core import broadcast_fields, principal_stresses

__all__ = [
    "stresses_on_incline",            # (2.7.3)/(2.7.4)
    "max_shear_in_plane",             # (2.7.5)
    "mohr_circle_params",             # (2.7.6)
    "hooke_plane_stress_strain",      # (2.7.7)
    "hooke_general_strain",           # (2.7.8)+(2.7.9)
    "dilatation_from_stress",         # (2.7.10)/(2.7.11)
    "volumetric_strain_from_volume",  # (2.7.12)
    "strain_energy_density_components",   # (2.7.13)/(2.7.14)
    "strain_energy_density_principal",    # (2.7.15)
    "strain_energy_total",            # (2.7.16)
    "deviatoric_stress",              # (2.7.25)
    "energy_dilatation_principal",    # (2.7.27)
    "energy_distortion_principal",    # (2.7.28)/(2.7.29)
    "equivalent_t1_max_normal",       # (2.7.20) теория 1
    "equivalent_t2_max_strain",       # (2.7.22) теория 2
    "equivalent_t3_max_shear",        # (2.7.23) теория 3 (Треска)
    "equivalent_t4_mises",            # (2.7.30) теория 4 (Губер-Мизес)
    "shear_yield_mises",              # (2.7.34)
    "shear_yield_tresca",             # (2.7.35)
    "mohr_limit_stresses",            # (2.7.36)
    "equivalent_mohr",                # (2.7.39) теория Мора
    "safety_factor_mohr",             # (2.7.40)/(2.7.42)
    "allowable_stress",               # (2.7.18)/(2.7.19)
    "check_strength",                 # (2.7.18)/(2.7.19) условия прочности
]


# ---------------------------------------------------------------------------
# (2.7.1)-(2.7.5) Площадка под углом α в плоскости (s1, s2)
# ---------------------------------------------------------------------------
def stresses_on_incline(s1, s2, alpha):
    """(2.7.4): нормальное и касательное напряжения на площадке под углом α:
        σα = (s1+s2)/2 + (s1−s2)/2·cos2α ;  τα = (s1−s2)/2·sin2α.
    alpha — радианы, поле любой формы. Возвращает (σα, τα).
    """
    s1, s2, alpha = broadcast_fields(s1, s2, alpha)
    sa = 0.5 * (s1 + s2) + 0.5 * (s1 - s2) * np.cos(2.0 * alpha)
    ta = 0.5 * (s1 - s2) * np.sin(2.0 * alpha)
    return sa, ta


def max_shear_in_plane(s1, s2):
    """(2.7.5): τ_max = |s1 − s2| / 2 (при α = π/4)."""
    s1, s2 = broadcast_fields(s1, s2)
    return np.abs(s1 - s2) / 2.0


def mohr_circle_params(s1, s2):
    """(2.7.6): круг Мора для пары главных напряжений: центр c, радиус R."""
    s1, s2 = broadcast_fields(s1, s2)
    c = 0.5 * (s1 + s2)
    r = 0.5 * np.abs(s1 - s2)
    return c, r


# ---------------------------------------------------------------------------
# (2.7.7)-(2.7.12) Обобщенный закон Гука
# ---------------------------------------------------------------------------
def hooke_plane_stress_strain(E, ex, ey, nu):
    """(2.7.7): плоское напряженное состояние — обратная форма закона Гука:
        sx = E/(1−ν²)·(εx + ν·εy),  sy = E/(1−ν²)·(εy + ν·εx).
    Возвращает (sx, sy)."""
    E, ex, ey, nu = broadcast_fields(E, ex, ey, nu)
    k = E / (1.0 - nu**2)
    return k * (ex + nu * ey), k * (ey + nu * ex)


def hooke_general_strain(E, nu, s1, s2, s3):
    """(2.7.8)/(2.7.9): обобщенный закон Гука в главных осях:
        e1 = 1/E·[s1 − ν(s2+s3)] и т.д. Возвращает (e1, e2, e3)."""
    s1, s2, s3, E, nu = broadcast_fields(s1, s2, s3, E, nu)
    e1 = (s1 - nu * (s2 + s3)) / E
    e2 = (s2 - nu * (s1 + s3)) / E
    e3 = (s3 - nu * (s1 + s2)) / E
    return e1, e2, e3


def dilatation_from_stress(E, nu, s1, s2, s3):
    """(2.7.10)/(2.7.11): относительная объемная деформация
        θ = (1−2ν)/E · (s1+s2+s3)."""
    e1, e2, e3 = hooke_general_strain(E, nu, s1, s2, s3)
    return e1 + e2 + e3


def volumetric_strain_from_volume(V0, V1):
    """(2.7.12): ΔV/V0 = (V1 − V0)/V0 ≈ θ (с точностью до первого порядка)."""
    V0, V1 = broadcast_fields(V0, V1)
    return (V1 - V0) / V0


# ---------------------------------------------------------------------------
# (2.7.13)-(2.7.16) Потенциальная энергия деформации
# ---------------------------------------------------------------------------
def strain_energy_density_components(E, nu, sx, sy, sz, txy=0.0, tyz=0.0, txz=0.0):
    """(2.7.14): удельная потенциальная энергия через компоненты напряжений:
        A = 1/(2E)[σx²+σy²+σz² − 2ν(σxσy+σyσz+σzσx)]
          + (1+ν)/E·(τxy²+τyz²+τxz²).
    (использован G = E/(2(1+ν)), закон Гука при сдвиге 2.4.12)."""
    sx, sy, sz, txy, tyz, txz, E, nu = broadcast_fields(
        sx, sy, sz, txy, tyz, txz, E, nu)
    A_norm = (sx**2 + sy**2 + sz**2 - 2.0 * nu * (sx * sy + sy * sz + sz * sx)) / (2.0 * E)
    A_shear = (1.0 + nu) / E * (txy**2 + tyz**2 + txz**2)
    return A_norm + A_shear


def strain_energy_density_principal(E, nu, s1, s2, s3):
    """(2.7.15): удельная энергия в главных напряжениях."""
    return strain_energy_density_components(E, nu, s1, s2, s3)


def strain_energy_total(A_density, dV=None):
    """(2.7.16): полная энергия — сумма по элементам объема.
    A_density — поле удельных энергий; dV — поле объемов КЭ (или единый шаг);
    если dV=None — простое суммирование (равные объемы с весом 1)."""
    A_density = np.asarray(A_density, dtype=float)
    if dV is None:
        return float(np.sum(A_density))
    dV = np.asarray(dV, dtype=float)
    return float(np.sum(A_density.ravel() * dV.ravel()))


# ---------------------------------------------------------------------------
# (2.7.24)-(2.7.29) Разделение на энергию объема и формоизменения
# ---------------------------------------------------------------------------
def deviatoric_stress(s1, s2, s3):
    """(2.7.25): s'_i = s_i − (s1+s2+s3)/3. Возвращает (s1', s2', s3')."""
    s1, s2, s3 = broadcast_fields(s1, s2, s3)
    m = (s1 + s2 + s3) / 3.0
    return s1 - m, s2 - m, s3 - m


def energy_dilatation_principal(E, nu, s1, s2, s3):
    """(2.7.27): A_v = (1−2ν)/(6E)·(s1+s2+s3)² — энергия изменения объема."""
    s1, s2, s3, E, nu = broadcast_fields(s1, s2, s3, E, nu)
    return (1.0 - 2.0 * nu) / (6.0 * E) * (s1 + s2 + s3) ** 2


def energy_distortion_principal(E, nu, s1, s2, s3):
    """(2.7.28): A_f = (1+ν)/(3E)·[s1²+s2²+s3² − (s1s2+s2s3+s3s1)]
                = (1+ν)/(6E)·[(s1−s2)²+(s2−s3)²+(s3−s1)²]."""
    s1, s2, s3, E, nu = broadcast_fields(s1, s2, s3, E, nu)
    return (1.0 + nu) / (6.0 * E) * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)


# ---------------------------------------------------------------------------
# (2.7.17)-(2.7.42) Эквивалентные напряжения и теории прочности
# ---------------------------------------------------------------------------
def equivalent_t1_max_normal(s1, s2, s3):
    """(2.7.20) Теория 1 (наибольшие нормальные напряжения): σэ = max(s1,s2,s3)."""
    s1, s2, s3 = broadcast_fields(s1, s2, s3)
    return np.maximum(np.maximum(s1, s2), s3)


def equivalent_t2_max_strain(nu, s1, s2, s3):
    """(2.7.22) Теория 2 (наибольшие линейные деформации):
        σэ = max_i [ s_i − ν(s_j + s_k) ]."""
    s1, s2, s3, nu = broadcast_fields(s1, s2, s3, nu)
    e1 = s1 - nu * (s2 + s3)
    e2 = s2 - nu * (s1 + s3)
    e3 = s3 - nu * (s1 + s2)
    return np.maximum(np.maximum(e1, e2), e3)


def equivalent_t3_max_shear(s1, s2, s3):
    """(2.7.23) Теория 3 (Треска, наибольшие касательные):
        σэ = 2τ_max = max|si − sj| = s_max − s_min."""
    s1, s2, s3 = broadcast_fields(s1, s2, s3)
    return np.max(np.stack([np.abs(s1 - s2), np.abs(s2 - s3), np.abs(s3 - s1)], 0), 0)


def equivalent_t4_mises(s1, s2, s3):
    """(2.7.30) Теория 4 (Губер–Мизес, энергия формоизменения):
        σэ = √(½[(s1−s2)²+(s2−s3)²+(s3−s1)²])."""
    s1, s2, s3 = broadcast_fields(s1, s2, s3)
    return np.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))


def shear_yield_mises(sigma_T):
    """(2.7.34): предел текучести при сдвиге по Мизесу: τ_T = σ_T/√3 ≈ 0.577σ_T."""
    return np.asarray(sigma_T, dtype=float) / np.sqrt(3.0)


def shear_yield_tresca(sigma_T):
    """(2.7.35): по Треске: τ_T = σ_T/2 = 0.5σ_T."""
    return np.asarray(sigma_T, dtype=float) / 2.0


def mohr_limit_stresses(sigma_lim_t, sigma_lim_c, n):
    """(2.7.36): предельные значения главных напряжений при запасе n:
        [s1] = σ_T/p... — возвращает (s1_pred, s3_pred) = (σt/n, −σc/n).
    sigma_lim_t — предел (текучести/прочности) при растяжении (>0),
    sigma_lim_c — то же при сжатии (передается положительным числом)."""
    a, b, n = broadcast_fields(sigma_lim_t, sigma_lim_c, n)
    return a / n, -b / n


def equivalent_mohr(s1, s3, sigma_lim_t, sigma_lim_c):
    """(2.7.39)/(2.7.42): эквивалентное напряжение по теории Мора при разных
    пределах σт (растяжение) и σс (сжатие), оба > 0:
        σэ = (σс·s1 − σт·s3) / (σс − σт)   (при σс ≠ σт; с точностью до
        постоянного множителя, не влияющего на условие прочности)
    Условие прочности: n = σт/σэ_pred... Фактический запас — safety_factor_mohr.
    При σс → σт вырождается к комбинации теорий 1 и 3: σэ ~ s1 − s3."""
    s1, s3, st, sc = broadcast_fields(s1, s3, sigma_lim_t, sigma_lim_c)
    st = np.abs(st)
    sc = np.abs(sc)
    with np.errstate(divide="ignore", invalid="ignore"):
        seq = np.where(np.abs(sc - st) > 1e-12 * sc,
                       (sc * s1 - st * s3) / (sc - st),
                       s1 - s3)
    return seq


def safety_factor_mohr(s1, s3, sigma_lim_t, sigma_lim_c):
    """(2.7.40)/(2.7.42): коэффициент запаса по теории Мора:
        n = σт·σс / (σс·s1 − σт·s3),  s1 ≥ 0 ≥ s3, σт, σс > 0.
    Вне четверти растяжение-сжатие используется консервативная редукция
    к теориям 1/3 (n = min(σт/s1+, σс/|s3−|))."""
    s1, s3, st, sc = broadcast_fields(s1, s3, sigma_lim_t, sigma_lim_c)
    st = np.abs(st)
    sc = np.abs(sc)
    # общий канонический вид (справедлив и при σс = σт): n = σтσс/(σс·s1 − σт·s3)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_mohr = st * sc / (sc * s1 - st * s3)
    n1 = np.where(s1 > 0, st / np.where(s1 > 0, s1, 1.0), np.inf)
    n3 = np.where(s3 < 0, sc / np.where(s3 < 0, -s3, 1.0), np.inf)
    return np.minimum(n_mohr, np.minimum(n1, n3))


def allowable_stress(sigma_lim, n):
    """(2.7.18)/(2.7.19): допускаемое напряжение [σ] = σпред / n."""
    sigma_lim, n = broadcast_fields(sigma_lim, n)
    return sigma_lim / n


def check_strength(sigma_eq, sigma_lim, n_required):
    """Условия прочности (2.7.18)/(2.7.19): σэ ≤ [σ] = σпред/n.
    Возвращает булево поле «прочность обеспечена» и поле фактических запасов."""
    sigma_eq, sigma_lim, n_required = broadcast_fields(sigma_eq, sigma_lim, n_required)
    n_fact = sigma_lim / sigma_eq
    return n_fact >= n_required, n_fact
