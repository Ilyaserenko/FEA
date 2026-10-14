"""Тесты всех формул глав 1.1–1.3 против аналитических/эталонных значений."""
import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import norm, expon, uniform, weibull_min

from strength_core import stats, tolerance, reliability


# ============================ Глава 1.1 =====================================

def test_1_1_1_relative_frequency():
    p = stats.relative_frequency([10, 20, 70], 100)
    assert np.allclose(p, [0.1, 0.2, 0.7])


def test_1_1_2_probability_from_pdf():
    grid = np.linspace(0, 2, 20001)
    f = stats.pdf_uniform(grid, 0, 2)
    assert abs(stats.probability_from_pdf(f, 0.5, 1.5, grid=grid) - 0.5) < 1e-9
    fn = lambda x: float(stats.pdf_normal(x, 0.0, 1.0))
    assert abs(stats.probability_from_pdf(fn, -1, 1) - 0.682689492) < 1e-6


def test_1_1_3_mean_discrete():
    m = stats.mean_discrete([1, 2, 3], weights=[1, 1, 2])   # (1+2+6)/4
    assert abs(m - 2.25) < 1e-12
    m2 = stats.mean_discrete([1, 2, 3], probabilities=[0.5, 0.3, 0.2])
    assert abs(m2 - 1.7) < 1e-12


def test_1_1_4_mean_continuous():
    grid = np.linspace(0, 1, 100001)
    f = stats.pdf_uniform(grid, 0, 1)
    assert abs(stats.mean_continuous(f, grid=grid) - 0.5) < 1e-9
    g = np.linspace(0, 50, 200001)
    fe = stats.pdf_exponential(g, 2.0)
    assert abs(stats.mean_continuous(fe, grid=g) - 0.5) < 1e-6


def test_1_1_5_and_1_1_7_variance_discrete():
    D = stats.variance_discrete([0, 10], probabilities=[0.5, 0.5])  # m=5, D=25
    assert abs(D - 25.0) < 1e-12
    assert abs(stats.std_dev(D) - 5.0) < 1e-12                      # 1.1.7


def test_1_1_6_variance_continuous():
    grid = np.linspace(-10, 10, 400001)
    f = stats.pdf_normal(grid, 0.0, 2.0)
    m = stats.mean_continuous(f, grid=grid)
    D = stats.variance_continuous(f, m, grid=grid)
    assert abs(D - 4.0) < 1e-4


def test_1_1_8_coeff_variation():
    nu = stats.coeff_variation(np.array([2.0, 0.5]), np.array([10.0, 5.0]))
    assert np.allclose(nu, [0.2, 0.1])


def test_1_1_9_cdf_from_pdf():
    grid = np.linspace(-6, 6, 120001)
    f = stats.pdf_normal(grid, 0.0, 1.0)
    F = stats.cdf_from_pdf(f, [0.0, 1.96], grid=grid)
    assert np.allclose(F, [0.5, 0.975], atol=1e-5)


def test_1_1_10_convolution_sum_of_normals():
    # свертка N(0,1)*N(0,1) → N(0,√2)
    grid = np.linspace(-12, 12, 200001)
    f1 = stats.pdf_normal(grid, 0, 1)
    x, fx = stats.pdf_of_sum_convolution(f1, f1, grid)
    m = stats.mean_continuous(fx, grid=x)
    D = stats.variance_continuous(fx, m, grid=x)
    assert abs(m) < 1e-3 and abs(np.sqrt(D) - np.sqrt(2)) < 5e-3
    assert abs(trapezoid(fx, x) - 1.0) < 1e-3                        # нормировка


def test_1_1_11_double_integral_mean():
    grid = np.linspace(-8, 8, 2001)          # сетка умеренная: метод O(N²) по памяти
    f1 = stats.pdf_normal(grid, 3.0, 1.0)
    f2 = stats.pdf_normal(grid, -1.0, 2.0)
    m = stats.mean_of_sum_double_integral(f1, f2, grid, grid)
    assert abs(m - 2.0) < 5e-3


def test_1_1_12_normalization():
    # равномерная ПП задана таблично на широкой сетке — границы [0,4]
    # попадают между узлами, поэтому учтём погрешность дискретизации ~dx/2
    grid = np.linspace(-10, 10, 200001)      # широкий диапазон для точной нормировки N(2,1)
    f1 = stats.pdf_normal(grid, 2.0, 1.0)
    f2 = stats.pdf_uniform(grid, 0, 4)
    res = stats.normalization_and_means(f1, f2, grid, grid)
    assert abs(res["norm_f1"] - 1) < 1e-6 and abs(res["norm_f2"] - 1) < 1e-4
    assert abs(res["m_x1"] - 2.0) < 1e-6 and abs(res["m_x2"] - 2.0) < 1e-4


def test_1_1_13_14_15_linearity():
    assert stats.mean_sum(3.0, 4.0) == 7.0                           # 1.1.13
    assert stats.mean_difference(3.0, 4.0) == -1.0                  # 1.1.14
    assert stats.mean_total([1.0, 2.0], [3.0, 5.0], [6.0, 5.0]) == pytest.approx([10.0, 12.0])  # 1.1.15


@pytest.mark.parametrize("pdf_fn,args,ref", [
    (stats.pdf_exponential, (1.0, 2.0), expon.pdf(1.0, scale=0.5)),                       # 1.1.16
    (stats.pdf_uniform, (0.3, 0.0, 1.0), uniform.pdf(0.3)),                               # 1.1.17
    (stats.pdf_triangular, (0.5, 0.0, 1.0), 2.0),                                         # 1.1.18 пик
    (stats.pdf_normal, (1.0, 0.0, 2.0), norm.pdf(1.0, 0, 2)),                             # 1.1.19
    (stats.pdf_weibull, (1.5, 2.0, 3.0), weibull_min.pdf(1.5, c=2.0, scale=3.0)),         # 1.1.27
])
def test_pdfs_vs_scipy(pdf_fn, args, ref):
    assert pdf_fn(*args) == pytest.approx(ref, rel=1e-9)


def test_1_1_18_triangular_shape():
    x = np.linspace(-1, 2, 3001)
    f = stats.pdf_triangular(x, 0, 1)
    assert np.all(f >= 0)
    assert abs(trapezoid(f, x) - 1.0) < 1e-4
    assert f[np.argmin(abs(x - 0.25))] == pytest.approx(1.0, rel=1e-3)
    assert f[np.argmin(abs(x - 0.75))] == pytest.approx(1.0, rel=1e-3)


def test_1_1_20_21_22_23_normal_probabilities():
    m, s = 10.0, 2.0
    p = stats.prob_normal_interval(8, 12, m, s)                     # 1.1.20, |z|<=1
    assert p == pytest.approx(0.682689492, rel=1e-8)
    z = stats.standardize(np.array([12.0, 8.0]), m, s)              # 1.1.21
    assert np.allclose(z, [1.0, -1.0])
    assert stats.laplace_function(0.0) == pytest.approx(0.0)        # 1.1.22
    assert stats.laplace_function(1.0) == pytest.approx(0.341344746, rel=1e-7)
    assert stats.laplace_function(-1.0) == pytest.approx(-0.341344746, rel=1e-7)  # нечетность
    p2 = stats.prob_normal_laplace(8, 12, m, s)                     # 1.1.23
    assert p2 == pytest.approx(p, rel=1e-12)


def test_1_1_24_25_26_limits():
    x1, x2 = stats.tolerance_limits_normal(10.0, 2.0, 1.96)         # 1.1.24
    assert (x1, x2) == pytest.approx((6.08, 13.92))
    assert stats.three_sigma_rule() == pytest.approx(0.9973, abs=1e-4)   # 1.1.25
    m, s = stats.normal_params_from_range(np.array([4.0, 8.0]), np.array([10.0, 20.0]))  # 1.1.26
    assert np.allclose(m, [7.0, 14.0]) and np.allclose(s, [1.0, 2.0])


# ============================ Глава 1.2 =====================================

def test_1_2_1_ra():
    x = np.linspace(0, 4 * np.pi, 200001)
    y = np.sin(x)
    assert tolerance.roughness_ra(y, x) == pytest.approx(2 / np.pi, rel=1e-6)  # Ra синуса


def test_1_2_2_to_1_2_7_clearance():
    d_max, d_min = 20.021, 20.000
    D_max, D_min = 20.048, 20.028
    d, D = tolerance.mean_diameters(d_max, d_min, D_max, D_min)                # 1.2.2
    assert (d, D) == pytest.approx((20.0105, 20.038))
    N = tolerance.clearance_nominal(d, D)                                       # 1.2.3
    assert N == pytest.approx(-0.0275)
    sd = tolerance.std_from_extremes(d_max, d_min)                              # 1.2.5
    sD = tolerance.std_from_extremes(D_max, D_min)
    sN = tolerance.std_clearance_components(sd, sD)                             # 1.2.4
    sN2 = tolerance.std_clearance_extremes(d_max, d_min, D_max, D_min)          # 1.2.6
    assert sN == pytest.approx(sN2, rel=1e-12)
    Nmax, Nmin = tolerance.clearance_limits_3sigma(N, sN)                       # 1.2.7
    assert Nmax > N > Nmin
    assert Nmax - N == pytest.approx(3 * sN)


def test_1_2_8_9_interference():
    D, d = 50.05, 50.02
    S = tolerance.interference_nominal(D, d)                                    # 1.2.8
    assert S == pytest.approx(0.03)
    Smin, Smax = tolerance.interference_limits(S, 0.005, 0.004, 3.0)            # 1.2.9
    sS = np.hypot(0.005, 0.004)
    assert Smin == pytest.approx(S - 3 * sS) and Smax == pytest.approx(S + 3 * sS)


def test_1_2_10_11_roundtrip():
    Amax, Amin = 10.2, 9.8
    m, T = tolerance.dim_mid_and_tol(Amax, Amin)                                # 1.2.10
    assert (m, T) == pytest.approx((10.0, 0.4))
    rmax, rmin = tolerance.dim_extremes_from_mid(m, T)                          # 1.2.11
    assert rmax == pytest.approx(Amax) and rmin == pytest.approx(Amin)


def test_1_2_12_to_1_2_15_chain():
    m1, T1 = 60.0, 0.06
    m2, T2 = 20.0, 0.03
    m3, T3 = 15.0, 0.03
    md = tolerance.chain_mean(m1, m2, m3)                                       # 1.2.12
    assert md == pytest.approx(25.0)
    Ds = [tolerance.component_std_from_tol(T, return_variance=True) for T in (T1, T2, T3)]  # 1.2.14
    Dsum = tolerance.chain_dispersion(*Ds)                                      # 1.2.13
    assert Dsum == pytest.approx((0.06 ** 2 + 0.03 ** 2 + 0.03 ** 2) / 36.0)
    Tdelta = tolerance.chain_tolerance(T1, T2, T3)                              # 1.2.15
    assert Tdelta == pytest.approx(np.sqrt(0.06 ** 2 + 0.03 ** 2 + 0.03 ** 2))
    assert Tdelta == pytest.approx(6 * np.sqrt(Dsum))                           # согласование 1.2.13/1.2.15


# ============================ Глава 1.3 =====================================

def test_1_3_1_to_1_3_4_consistency():
    lam = 0.001
    t = np.linspace(0, 5000, 500001)
    f = stats.pdf_exponential(t, lam)
    F = reliability.cdf_lifetime(f, t, grid=t)                                  # 1.3.2
    P = reliability.reliability_func(f, t, grid=t)                              # 1.3.3
    assert np.allclose(P, 1 - F, atol=1e-9)
    i1, i2 = np.searchsorted(t, 1000), np.searchsorted(t, 2000)
    p_int = reliability.prob_lifetime_interval(f, 1000, 2000, grid=t)           # 1.3.1
    via_P = reliability.prob_interval_via_P(P[i1], P[i2])                       # 1.3.4
    assert p_int == pytest.approx(via_P, rel=1e-4)
    assert p_int == pytest.approx(np.exp(-1.0) - np.exp(-2.0), rel=1e-4)


def test_1_3_5_hazard_estimate():
    lam = reliability.hazard_rate_estimate([5, 8], [1000, 990], [100, 100])     # 1.3.5
    assert np.allclose(lam, [5 / 1e5, 8 / 99000])


def test_1_3_6_hazard_limit():
    lam_true = 2e-4
    dt = np.array([200.0, 100.0, 50.0])
    dn = lam_true * 1000 * dt                                                   # точные Δn при const λ
    est = reliability.hazard_rate_limit(dn, 1000, dt)                           # 1.3.6
    assert est == pytest.approx(lam_true, rel=1e-6)


def test_1_3_7_8_from_exponential():
    t = np.linspace(0, 5000, 500001)
    lam = 0.001
    P = reliability.reliability_exponential(t, lam)                             # 1.3.10
    f = reliability.pdf_from_reliability(P, t)                                  # 1.3.7
    assert f[100] == pytest.approx(lam * np.exp(-lam * t[100]), rel=1e-3)
    lh = reliability.hazard_rate_from_P(P, t, f_t=f)                            # 1.3.8
    assert lh[100] == pytest.approx(lam, rel=1e-3)


def test_1_3_9_general_exponential():
    t = np.linspace(0, 3000, 300001)
    lam_tab = np.full_like(t, 5e-4)
    P = reliability.reliability_general_exponential(lam_tab, t, grid=t)         # 1.3.9
    assert P[-1] == pytest.approx(np.exp(-5e-4 * 3000), rel=1e-6)
    Pc = reliability.reliability_general_exponential(lambda tt: 1e-6 * tt, np.array([1000.0]))
    assert Pc[0] == pytest.approx(np.exp(-1e-6 * 1000 ** 2 / 2), rel=1e-8)      # λ(t)=a·t


def test_1_3_10_conditional():
    P = reliability.conditional_reliability(np.array([3000.0]), 1000.0, 1e-4)   # 1.3.10'
    assert P[0] == pytest.approx(np.exp(-1e-4 * 2000), rel=1e-12)


def test_1_3_11_constant_hazard():
    lam = reliability.hazard_constant(1e-5)                                     # 1.3.11
    assert float(lam) == 1e-5


def test_1_3_12_13_normal_life():
    m_t, s_t = 1000.0, 200.0
    p = reliability.prob_lifespan_normal(800, 1200, m_t, s_t)                   # 1.3.12
    assert p == pytest.approx(norm.cdf(1) - norm.cdf(-1), rel=1e-9)
    assert reliability.reliability_normal(1000.0, m_t, s_t) == pytest.approx(0.5, abs=1e-12)  # 1.3.13
    assert reliability.reliability_normal(1200.0, m_t, s_t) == pytest.approx(1 - norm.cdf(1), rel=1e-9)


def test_1_3_13_weibull_alpha1():
    res = reliability.weibull_alpha1(np.array([100.0]), 250.0)                  # Вейбулл α=1
    assert res["P"][0] == pytest.approx(np.exp(-0.4))
    assert res["lambda"][0] == pytest.approx(1 / 250.0)
    assert res["F"][0] == pytest.approx(1 - np.exp(-0.4))


def test_1_3_14_15_16_structures():
    Pi = np.array([[0.99, 0.98], [0.995, 0.97], [0.985, 0.99]])                 # 3 элемента × 2 момента
    Pc = reliability.reliability_series(Pi, axis=0)                              # 1.3.14
    assert np.allclose(Pc, np.prod(Pi, axis=0))
    Qc, Ppar = reliability.reliability_parallel(Pi, axis=0)                      # 1.3.15/1.3.16
    assert np.allclose(Qc, np.prod(1 - Pi, axis=0))
    assert np.allclose(Ppar, 1 - Qc)
    assert np.all(Ppar >= Pc)                                                    # параллель ≥ последовательно


def test_1_3_17_to_1_3_21_stress_strength():
    s_lim_bar, s_bar = 500.0, 300.0
    S_lim, S = 40.0, 30.0
    x_lim = np.linspace(350, 650, 300001)
    flim = reliability.pdf_strength_limit(x_lim, s_lim_bar, S_lim)               # 1.3.17
    assert flim[np.argmin(abs(x_lim - 500))] == pytest.approx(norm.pdf(0, scale=40), rel=1e-6)
    x_s = np.linspace(150, 450, 300001)     # своя сетка: среднее напряжения 300 внутри диапазона
    fs = reliability.pdf_stress(x_s, s_bar, S)                                   # 1.3.18
    assert fs[np.argmin(abs(x_s - 300))] == pytest.approx(norm.pdf(0, scale=30), rel=1e-6)
    m, S0 = reliability.safety_margin_params(s_lim_bar, s_bar, S_lim, S)         # 1.3.19
    assert m == 200.0 and S0 == pytest.approx(np.hypot(40, 30))
    fz = reliability.pdf_margin(np.zeros(5), m, S0)                              # 1.3.20
    assert fz[0] == pytest.approx(norm.pdf(0, loc=200, scale=S0), rel=1e-12)
    zp = reliability.safety_index_zp(s_lim_bar, s_bar, S_lim, S)                 # 1.3.21
    assert zp == pytest.approx(200 / 50)                                         # индекс надежности β=m/S0
    assert norm.cdf(zp) == pytest.approx(norm.cdf(m / S0), rel=1e-12)            # P = Φ(z_p)


def test_1_3_21_n_p_formula():
    # обратная проверка тождества: n_p из (1.3.21) должен давать тот же z_p
    v, v_lim = 0.10, 0.08
    z_p = 4.0
    n_p = float(reliability.coefficient_n_p(z_p, v, v_lim))
    assert n_p > 1.0                                                             # запас прочности есть
    # по определениям: σ̄lim = n_p σ̄, S_lim = v_lim σ̄lim, S = v σ̄
    sig = 100.0
    slimb, slimS, sS = n_p * sig, v_lim * n_p * sig, v * sig
    z_back = reliability.safety_index_zp(slimb, sig, slimS, sS)
    assert z_back == pytest.approx(z_p, rel=1e-9)


def test_big_data_vectorized():
    """Демонстрация работы на больших данных (миллионы узлов МКЭ)."""
    rng = np.random.default_rng(1)
    sigma = rng.normal(300, 30, size=2_000_000)
    s_lim = rng.normal(500, 40, size=sigma.shape)
    zp = reliability.safety_index_zp(s_lim.mean(), sigma.mean(), s_lim.std(), sigma.std())
    assert zp > 3                                                                # большой запас
    nu = stats.coeff_variation(sigma.std(axis=0), sigma.mean(axis=0))
    assert 0.09 < nu < 0.11
