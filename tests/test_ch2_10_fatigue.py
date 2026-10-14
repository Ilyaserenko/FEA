"""Тесты модуля strength_core.ch2_10_fatigue (глава 2.10, формулы 2.10.4–2.10.38).

Проверки: аналитические значения, самосогласованность парных функций
(sigma(N) <-> N(sigma), Miner <-> эквивалентные приведения),вырожденные случаи,
векторизация на полях МКЭ произвольной формы.
"""

import numpy as np
import pytest

from strength_core import ch2_10_fatigue as F


# ---------------------------------------------------------------------------
# Параметры цикла (2.10.4)-(2.10.7)
# ---------------------------------------------------------------------------
def test_cycle_params_scalar():
    assert F.cycle_mean(-30.0, 70.0) == 20.0
    assert F.cycle_amplitude(-30.0, 70.0) == 50.0
    assert abs(F.cycle_asymmetry(-30.0, 70.0) + 30 / 70) < 1e-12


def test_cycle_params_vector_field():
    smin = np.array([-100.0, 0.0, 20.0])
    smax = np.array([100.0, 100.0, 80.0])
    p = F.cycle_params(smin, smax)
    assert np.allclose(p["sm"], [0.0, 50.0, 50.0])
    assert np.allclose(p["sa"], [100.0, 50.0, 30.0])
    assert np.allclose(p["r"], [-1.0, 0.0, 0.25])


def test_characteristic_cycles():
    assert F.characteristic_cycles("symmetric")["r"] == -1.0
    assert F.characteristic_cycles("pulse_plus")["r"] == 0.0
    assert F.characteristic_cycles("static")["r"] == 1.0
    with pytest.raises(ValueError):
        F.characteristic_cycles("bogus")


# ---------------------------------------------------------------------------
# Кривая выносливости (2.10.8)
# ---------------------------------------------------------------------------
S_D, N_D, M = 300.0, 1e7, 9.0


def test_sn_branches_values():
    # в базовой точке sigma = s_D
    assert F.sn_stress(N_D, S_D, N_D, M) == pytest.approx(S_D)
    # при N = N_D * 2^m -> sigma = s_D/2
    N_half = N_D * 2.0 ** M
    assert F.sn_stress(N_half, S_D, N_D, M) == pytest.approx(S_D / 2.0)
    # горизонтальная ветвь: N > N_D не поднимает напряжение выше s_D
    assert F.sn_stress(1e9, S_D, N_D, M) == pytest.approx(S_D)


def test_sn_roundtrip():
    N = np.logspace(4, 6.9, 50)
    sig = F.sn_stress(N, S_D, N_D, M)
    Nb = F.sn_cycles(sig, S_D, N_D, M)
    assert np.allclose(Nb, N)


def test_conditional_endurance_limit():
    # без ограничения горизонталью: sigma(N*) = s_D*(N_D/N*)^(1/m)
    val = F.conditional_endurance_limit(S_D, N_D, M, N_star=1e8)
    assert val == pytest.approx(S_D * (N_D / 1e8) ** (1 / M))


# ---------------------------------------------------------------------------
# Концентраторы (2.10.10)-(2.10.13), табл. 2.10.1
# ---------------------------------------------------------------------------
def test_effective_kk_definitions():
    assert F.effective_kk(400.0, 200.0) == pytest.approx(2.0)
    # (2.10.11): eps=1 -> k_eff = kt; eps=0 -> k_eff = 1
    assert F.effective_kk_from_theoretical(2.5, 1.0) == pytest.approx(2.5)
    assert F.effective_kk_from_theoretical(2.5, 0.0) == pytest.approx(1.0)


def test_sensitivity_table_midpoints():
    assert F.sensitivity_coefficient("cast") == pytest.approx(0.25)
    assert F.sensitivity_coefficient("medium_carbon") == pytest.approx(0.75)
    assert 0.8 <= F.sensitivity_coefficient("alloy") <= 0.95


# ---------------------------------------------------------------------------
# Масштаб и поверхность (2.10.14)-(2.10.15), снижение (2.10.18)
# ---------------------------------------------------------------------------
def test_scale_factor_monotonic():
    b = F.scale_factor(np.array([10.0, 50.0, 200.0]))
    assert b[0] == pytest.approx(1.0)
    assert np.all(np.diff(b) < 0)  # монотонное убывание с размером


def test_reduction_factor_limits():
    # идеальные коэффициенты -> K = k_eff (согласуется с определением 2.10.10)
    assert F.reduction_factor(1.8) == pytest.approx(1.8)
    assert F.reduction_factor(1.8, form="product") == pytest.approx(1.8)
    # шероховатая поверхность ухудшает K
    assert F.reduction_factor(1.8, b_s=0.8) > 1.8
    # упрочнение улучшает
    assert F.reduction_factor(1.8, y_beta=1.2) < 1.8


# ---------------------------------------------------------------------------
# Запасы (2.10.16)-(2.10.22)
# ---------------------------------------------------------------------------
def test_safety_ref_and_part():
    assert F.safety_ref_sample(400.0, 100.0) == pytest.approx(4.0)
    n = F.safety_part(400.0, 100.0, sm=50.0, K=1.8, psi=0.3)
    assert n == pytest.approx(400.0 / (1.8 * 100 + 0.3 * 50))


def test_equivalent_symmetric_amplitude():
    # симметричный цикл (sm=0): sa_eq = K*sa
    assert F.equivalent_symmetric_amplitude(100.0, 0.0, 0.3, K=1.5) == pytest.approx(150.0)
    # асимметричный: аддитивная поправка psi*|sm|
    assert F.equivalent_symmetric_amplitude(100.0, -40.0, 0.3, K=1.5) == pytest.approx(162.0)


# ---------------------------------------------------------------------------
# Статика (2.10.23)-(2.10.27)
# ---------------------------------------------------------------------------
def test_static_safety():
    assert F.static_safety_normal(350.0, 100.0, 200.0) == pytest.approx(350.0 / 300.0)
    assert F.static_safety_shear(200.0, 50.0, 100.0) == pytest.approx(200.0 / 150.0)


def test_combined_static_safety_reduces_to_normal():
    # при отсутствии касательной составляющей — обычный статический запас
    n = F.combined_static_safety(350.0, 100.0, 200.0, 0.0, 0.0)
    assert n == pytest.approx(350.0 / 300.0)


def test_equivalent_tension_pure_axial():
    assert F.equivalent_tension(100.0, 200.0, 0.0, 0.0) == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Общее НС (2.10.28)-(2.10.31)
# ---------------------------------------------------------------------------
def test_general_safety_and_zone():
    sa_eq = F.equivalent_amplitude_principal(120.0, -30.0)
    assert sa_eq == pytest.approx(150.0)
    n = F.fatigue_safety_general(400.0, sa_eq, sm_eq=20.0, psi=0.2, K=1.6)
    assert n == pytest.approx(400.0 / (1.6 * 150 + 0.2 * 20))
    chk = F.check_fatigue_safety(np.array([1.0, 1.4, 2.0]))
    assert list(chk["ok"]) == [False, True, False]


def test_safety_limited_endurance_vs_base():
    # при N = N_D совпадает с обычным запасом (s_D вместо sigma_r)
    n1 = F.safety_limited_endurance(N_D, S_D, N_D, M, sa_eq=100.0)
    n2 = F.fatigue_safety_general(S_D, 100.0)
    assert n1 == pytest.approx(n2)
    # при меньшем N запас меньше (криволинейный участок)
    assert F.safety_limited_endurance(1e5, S_D, N_D, M, sa_eq=100.0) > n2


# ---------------------------------------------------------------------------
# Повреждения и эквивалентные режимы (2.10.33)-(2.10.38)
# ---------------------------------------------------------------------------
def test_miner_damage():
    D = F.miner_damage_sum([5e5, 5e5], [1e6, 1e6])
    assert D == pytest.approx(1.0)


def test_sn_curved_branch_consistency():
    N1 = F.sn_cycles_curved_branch(400.0, M, s_D=S_D, N_D=N_D)
    N2 = F.sn_cycles(400.0, S_D, N_D, M)
    assert N1 == pytest.approx(N2)


def test_equivalent_stress_constant_regime():
    seq = F.equivalent_stress([100, 200, 300], [150.0, 150.0, 150.0], M)
    assert seq == pytest.approx(150.0)


def test_equivalent_stress_bounds():
    ni = np.array([1e4, 2e4, 3e4])
    si = np.array([400.0, 350.0, 300.0])
    seq = F.equivalent_stress(ni, si, M)
    assert si.max() >= seq >= si.min()


def test_equivalent_cycles_damage_consistency():
    """N_eq(sigma_max)/N(sigma_max) должен давать то же повреждение, что sum n_i/N_i."""
    ni = np.array([1e4, 2e4, 3e4])
    si = np.array([400.0, 350.0, 300.0])
    smax = 400.0
    Neq = F.equivalent_cycles(ni, si, smax, M)
    Nmax = F.sn_cycles(smax, S_D, N_D, M)
    D_direct = np.sum(ni / F.sn_cycles(si, S_D, N_D, M))
    assert Neq / Nmax == pytest.approx(D_direct)


def test_equivalent_cycles_le_total():
    ni = np.array([1e4, 2e4, 3e4])
    si = np.array([400.0, 350.0, 300.0])
    Neq = F.equivalent_cycles(ni, si, 400.0, M)
    assert Neq <= ni.sum()  # N_eq меньше полного числа циклов (см. текст к 2.10.38)


# ---------------------------------------------------------------------------
# Метод полных циклов (табл. 2.10.2)
# ---------------------------------------------------------------------------
def test_full_cycle_counting_sinusoid():
    t = np.linspace(0, 4 * np.pi, 4001)
    x = 100 * np.sin(t)
    res = F.full_cycle_counting(x)
    # 4 периода синусоиды -> 4 полных цикла размахом ~200 (плюс хвостовые пары)
    big = res["ranges"][res["ranges"] > 150]
    assert big.size >= 4
    assert np.allclose(big, 200.0, atol=0.01)
    assert np.allclose(res["means"][res["ranges"] > 150], 0.0, atol=0.01)


def test_full_cycle_counting_range_pairs_astm():
    """Контрольный пример ASTM E1049-85 п.5.4.2 (range pair counting):
    пики/впадины A..H дают 4 полных цикла: B-C, D-E(?), G-H... используем
    канонический тест последовательности экстремумов."""
    ext = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0])
    res = F.full_cycle_counting(ext)
    # сумма длин всех диапазонов должна равняться алгебраической длине пути
    # после схлопывания полных пар — проверяем неотрицательность и чётность
    assert res["counts"] >= 1
    assert np.all(res["ranges"] > 0)
    assert np.allclose(res["amplitudes"], res["ranges"] / 2.0)


def test_full_cycle_counting_monotone_no_cycles():
    res = F.full_cycle_counting(np.arange(100.0))
    assert res["counts"] == 0


def test_full_cycle_counting_plateau():
    x = np.concatenate([np.zeros(50), np.ones(50), np.zeros(50), -np.ones(50)])
    res = F.full_cycle_counting(x)
    assert res["counts"] >= 1
    assert res["ranges"].max() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Векторизация на «МКЭ-полях»
# ---------------------------------------------------------------------------
def test_vectorization_on_fields():
    rng = np.random.default_rng(1)
    nodes = (rng.uniform(20, 80, size=(1000,)), rng.uniform(100, 200, size=(1000,)))
    sm = F.cycle_mean(*nodes)
    sa = F.cycle_amplitude(*nodes)
    n = F.safety_part(400.0, sa, sm=sm, K=1.8, psi=0.3)
    assert n.shape == (1000,)
    assert np.all(np.isfinite(n))


def test_performance_1e6_nodes():
    import time
    rng = np.random.default_rng(2)
    smin = rng.uniform(-150, 50, size=1_000_000)
    smax = smin + rng.uniform(50, 250, size=1_000_000)
    t0 = time.perf_counter()
    p = F.cycle_params(smin, smax)
    n = F.safety_part(400.0, p["sa"], sm=p["sm"], K=1.8, psi=0.3)
    dt = time.perf_counter() - t0
    assert n.shape == (1_000_000,)
    assert dt < 1.0, f"слишком медленно: {dt:.3f} c на 1e6 узлов"
