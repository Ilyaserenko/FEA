# strength_core — расчетное ядро программы прочности (МКЭ-постобработка)

Библиотека реализует формулы глав 1.1–1.3 в виде функций, оптимизированных
для работы с большими массивами данных (результаты МКЭ: поля напряжений по
сотням тысяч/миллионам узлов, статистика партий, временные ряды ресурса).

## Структура

```
strength_core/
    stats.py         # Глава 1.1, формулы (1.1.1) – (1.1.27)
    tolerance.py     # Глава 1.2, формулы (1.2.1) – (1.2.15)
    reliability.py   # Глава 1.3, формулы (1.3.1) – (1.3.21)
tests/
    test_formulas.py # 38 тестов против аналитики и scipy.stats
```

## Соответствие формул функциям

### Глава 1.1 (`stats`)
| Формула | Функция |
|---|---|
| (1.1.1) p_i = n_i/N | `relative_frequency` |
| (1.1.2) P(x1<x<x2)=∫f dx | `probability_from_pdf` |
| (1.1.3) m_x = Σ x_i p_i | `mean_discrete` |
| (1.1.4) m_x = ∫ x f dx | `mean_continuous` |
| (1.1.5) D_x = Σ(x−m)²p | `variance_discrete` |
| (1.1.6) D_x = ∫(x−m)²f dx | `variance_continuous` |
| (1.1.7) σ = √D | `std_dev` |
| (1.1.8) ν = σ/m | `coeff_variation` |
| (1.1.9) F(x) = ∫_{−∞}^x f dt | `cdf_from_pdf` |
| (1.1.10) свертка плотностей суммы | `pdf_of_sum_convolution` (FFT) |
| (1.1.11) двойной интеграл мат.ожидания | `mean_of_sum_double_integral` |
| (1.1.12) нормировки и частные мат.ожидания | `normalization_and_means` |
| (1.1.13)/(1.1.14)/(1.1.15) линейность m | `mean_sum` / `mean_difference` / `mean_total` |
| (1.1.16) показательное | `pdf_exponential` |
| (1.1.17) равномерное | `pdf_uniform` |
| (1.1.18) треугольное | `pdf_triangular` |
| (1.1.19) нормальное | `pdf_normal` |
| (1.1.20) вероятность интервала (норма) | `prob_normal_interval` |
| (1.1.21) стандартизация z | `standardize` |
| (1.1.22) функция Лапласа Φ(z) | `laplace_function` (через erf) |
| (1.1.23) P = Φ(z2) − Φ(z1) | `prob_normal_laplace` |
| (1.1.24) x = m ± z_p σ | `tolerance_limits_normal` |
| (1.1.25) правило 3σ ≈ 0.9973 | `three_sigma_rule` |
| (1.1.26) m, σ из диапазона | `normal_params_from_range` |
| (1.1.27) Вейбулл | `pdf_weibull` |

### Глава 1.2 (`tolerance`)
| Формула | Функция |
|---|---|
| (1.2.1) Ra = (1/l)∫|y|dx | `roughness_ra` |
| (1.2.2) средние d, D | `mean_diameters` |
| (1.2.3) N = d − D | `clearance_nominal` |
| (1.2.4) σ_N = √(σ_d²+σ_D²) | `std_clearance_components` |
| (1.2.5) σ = (xmax−xmin)/6 | `std_from_extremes` |
| (1.2.6) σ_N из крайних | `std_clearance_extremes` |
| (1.2.7) N ± 3σ_N | `clearance_limits_3sigma` |
| (1.2.8) S = D − d | `interference_nominal` |
| (1.2.9) S_min/S_max, σ_S | `interference_limits` |
| (1.2.10) m_A, T | `dim_mid_and_tol` |
| (1.2.11) A_max/A_min из m,T | `dim_extremes_from_mid` |
| (1.2.12) m_Δ цепи | `chain_mean` |
| (1.2.13) D_Δ = ΣD_i | `chain_dispersion` |
| (1.2.14) σ=T/6, D=(T/6)² | `component_std_from_tol` |
| (1.2.15) T_Δ = √ΣT_i² | `chain_tolerance` |

### Глава 1.3 (`reliability`)
| Формула | Функция |
|---|---|
| (1.3.1) P(t1<T<t2)=∫f dt | `prob_lifetime_interval` |
| (1.3.2) F(t)=∫_0^t f dτ | `cdf_lifetime` |
| (1.3.3) P(t)=1−F(t) | `reliability_func` |
| (1.3.4) P(t1)−P(t2)=F(t2)−F(t1) | `prob_interval_via_P` |
| (1.3.5) λ = Δn/(N Δt) | `hazard_rate_estimate` |
| (1.3.6) λ(t) = lim (Ричардсон) | `hazard_rate_limit` |
| (1.3.7) f = −dP/dt | `pdf_from_reliability` |
| (1.3.8) λ = f/P | `hazard_rate_from_P` |
| (1.3.9) P = exp(−∫λ dτ) | `reliability_general_exponential` |
| (1.3.10) P = e^{−λt}; P(t|t1) | `reliability_exponential`, `conditional_reliability` |
| (1.3.11) λ = const | `hazard_constant` |
| (1.3.12) P через Φ для ресурса | `prob_lifespan_normal` |
| (1.3.13) P = 0.5 − Φ; Вейбулл α=1 | `reliability_normal`, `weibull_alpha1` |
| (1.3.14) последовательная Π P_i | `reliability_series` |
| (1.3.15)/(1.3.16) параллельная | `reliability_parallel` |
| (1.3.17) f_lim(σ_lim) | `pdf_strength_limit` |
| (1.3.18) f(σ) | `pdf_stress` |
| (1.3.19) m, S_0 | `safety_margin_params` |
| (1.3.20) f(z) | `pdf_margin` |
| (1.3.21) z_p и n_p(ν, ν_lim, z_p) | `safety_index_zp`, `coefficient_n_p` |

## Соглашение о знаке z_p (формула 1.3.21)

В учебной записи z_p = (0 − m)/S_0 — квантиль **области отказа** (отрицателен
при запасе прочности). В ядре `safety_index_zp` возвращает классический
**индекс надежности** β = m/S_0 = (σ̄_lim − σ̄)/√(S_lim² + S²) > 0, что делает
функцию `coefficient_n_p(z_p, v, v_lim)` самосогласованной: при β > 0
коэффициент n_p > 1, а вероятность безотказной работы P = Φ(z_p).

## Оптимизация для больших данных

* Все ПП, вероятности, моменты — векторизованы NumPy (без циклов Python);
  тест `test_big_data_vectorized` считает 2 млн «узлов» за ~мсек.
* Интегрирование табличных функций — `scipy.integrate.trapezoid` /
  `cumulative_trapezoid` (O(N)); точные callable-режимы — `quad`.
* Свертка плотностей (1.1.10) — БПФ (`fftconvolve`), O(N log N).
* Функция Лапласа — через `scipy.special.erf` (векторно, машинная точность).
* Численное дифференцирование (1.3.7) — `np.gradient` (центральные разности, O(N)).
* Структурные схемы надежности (1.3.14–1.3.16) — матрицы `(элементы × времена)`,
  произведение вдоль нужной оси одним вызовом `np.prod`.

## Запуск тестов

```bash
python -m pytest tests/ -q
```

## Пример использования

```python
import numpy as np
from strength_core import stats, reliability

sigma = np.load("nodal_stress_von_mises.npy")      # поле напряжений МКЭ
m, s = sigma.mean(), sigma.std()
nu = stats.coeff_variation(s, m)                   # (1.1.8)
zp = reliability.safety_index_zp(500.0, m, 40.0, s)  # (1.3.21)
n_p = reliability.coefficient_n_p(zp, s / m, 40.0 / 500.0)
```
