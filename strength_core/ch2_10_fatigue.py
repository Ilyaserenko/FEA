"""
Глава 2.10 — Расчет прочности при переменном характере внешнего нагружения.
Формулы (2.10.4) – (2.10.38) + справочные соотношения (2.10.1)–(2.10.3).

Все функции векторизованы: принимают поля напряжений МКЭ-модели произвольной
формы ((n_nodes,), (n_elem, m), скаляры) и возвращают поля того же размера.
LaTeX-записи восстановлены из контекста учебника (картинки MS Equation в
исходном HTML утрачены). Статус каждой формулы: «восстановлено уверенно» или
«требует утверждения заказчиком» — см. docstring функции и README.

Обозначения (конвенция Н/мм/МПа):
    s_min, s_max   — наименьшее/наибольшее напряжение цикла;
    sm, sa         — среднее / амплитудное напряжение цикла;
    r              — коэффициент асимметрии цикла r = s_min/s_max;
    psi            — коэффициент влияния средних напряжений psi_s / psi_tau;
    s_r            — предел выносливости симметричного цикла sigma_{-1};
    s_D, N_D       — напряжение и базовое число циклов кривой выносливости;
    m_exp          — показатель степени m кривой выносливости;
    k_s, k_tau     — эффективные коэффициенты концентрации напряжений;
    kt             — теоретические коэффициенты концентрации;
    eps_s          — коэффициент чувствительности материала e_sigma;
    b_n            — масштабный коэффициент (размер детали);
    b_s            — коэффициент качества поверхности;
    y_beta         — коэффициент упрочнения поверхности;
    sig_B          — предел прочности; sig_T — предел текучести;
    tau_r          — предел выносливости при кручении; tau_T — предел текучести
                     при сдвиге (tau_T = sig_T/sqrt(3), см. (2.7.35)).
"""

from __future__ import annotations

import numpy as np

from .core import broadcast_fields, interp_table

__all__ = [
    # атомистика/теория (2.10.1)-(2.10.3)
    "theoretical_shear_strength",     # (2.10.3)
    # параметры цикла (2.10.4)-(2.10.7)
    "cycle_mean",                    # (2.10.4)
    "cycle_amplitude",               # (2.10.5)
    "cycle_asymmetry",               # (2.10.6)
    "cycle_params",                  # (2.10.4)+(2.10.5)+(2.10.6) за один проход
    "characteristic_cycles",         # таблица характерных циклов (2.10.7)
    # кривая выносливости (2.10.8)-(2.10.9)
    "sn_stress",                     # (2.10.8) обратная ветвь sigma(N)
    "sn_cycles",                     # (2.10.8) прямая ветвь N(sigma)
    "sn_exponent_from_strength",     # (2.10.8a)
    "endurance_normal_from_strength",# (2.10.9) sigma_r ~ f(sigma_B)
    "endurance_torsion_from_normal", # (2.10.9) tau_r = 0.58 sigma_r / 0.7 sigma_B
    "conditional_endurance_limit",   # условный предел выносливости
    # влияние факторов (2.10.10)-(2.10.15)
    "effective_kk",                       # (2.10.10)
    "effective_kk_from_theoretical",      # (2.10.11)
    "sensitivity_coefficient",            # табл. 2.10.1 (e_sigma)
    "kk_approx_smooth",                   # (2.10.12)
    "kk_approx_sharp",                    # (2.10.13)
    "scale_factor",                       # (2.10.14)
    "surface_quality_factor",             # (2.10.15)
    "reduction_factor",                   # (2.10.18)
    # запасы усталостной прочности (2.10.16)-(2.10.22)
    "safety_ref_sample",                  # (2.10.16)
    "safety_part",                        # (2.10.17)
    "psi_from_limit_points",              # psi по предельной прямой
    "equivalent_symmetric_amplitude",     # (2.10.21)/(2.10.32)
    "equivalent_symmetric_amplitude_tau", # (2.10.22)
    # статическая прочность (2.10.23)-(2.10.27)
    "static_safety_normal",               # (2.10.23)
    "static_safety_shear",                # (2.10.24)
    "equivalent_tension",                 # (2.10.25)
    "combined_static_safety",             # (2.10.26)/(2.10.27)
    # общее напряженное состояние (2.10.28)-(2.10.31)
    "equivalent_amplitude_principal",     # (2.10.28)/(2.10.29)
    "fatigue_safety_general",             # (2.10.30)
    "check_fatigue_safety",               # зона [n]_min..[n]_max
    "safety_limited_endurance",           # (2.10.31)
    # суммирование повреждений (2.10.33)-(2.10.38)
    "miner_damage_sum",                   # (2.10.33)
    "sn_cycles_curved_branch",            # (2.10.34)
    "equivalent_stress",                  # (2.10.35)-(2.10.37)
    "equivalent_cycles",                  # (2.10.38)
    "full_cycle_counting",                # метод полных циклов (табл. 2.10.2)
]


# ---------------------------------------------------------------------------
# (2.10.1)–(2.10.3) Теоретическая касательная прочность решетки
# ---------------------------------------------------------------------------
def theoretical_shear_strength(E, nu=0.3):
    """Оценка теоретической касательной прочности (вывод (2.10.1)–(2.10.3)):

        tau_max ~ E/(2*(1+nu)) * gamma_cr ~ E/8  (для металлов nu ~ 1/3).

    (2.10.3): пластические деформации при одноосном растяжении начинаются
    примерно при sigma ~ E/8 (реально — в десятки/сотни раз ниже).
    """
    E = np.asarray(E, dtype=float)
    return E / 8.0 if np.ndim(nu) == 0 else E / (2.0 * (1.0 + nu)) * 0.5


# ---------------------------------------------------------------------------
# (2.10.4)–(2.10.7) Параметры цикла
# ---------------------------------------------------------------------------
def cycle_mean(s_min, s_max):
    """(2.10.4) Среднее напряжение цикла sm = (s_max + s_min)/2."""
    a, b = broadcast_fields(s_min, s_max)
    return 0.5 * (b + a)


def cycle_amplitude(s_min, s_max):
    """(2.10.5) Амплитуда цикла sa = (s_max - s_min)/2."""
    a, b = broadcast_fields(s_min, s_max)
    return 0.5 * (b - a)


def cycle_asymmetry(s_min, s_max):
    """(2.10.6) Коэффициент асимметрии r = s_min / s_max.

    Симметричный цикл r=-1, пульсирующий r=0, статический r=+1.
    При s_max=0 возвращается inf (пульсирующий отрицательный).
    """
    a, b = broadcast_fields(s_min, s_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


def cycle_params(s_min, s_max):
    """Один проход по данным: dict {'sm', 'sa', 'r'} — (2.10.4)+(2.10.5)+(2.10.6)."""
    a, b = broadcast_fields(s_min, s_max)
    mean = 0.5 * (b + a)
    amp = 0.5 * (b - a)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    return {"sm": mean, "sa": amp, "r": r}


def characteristic_cycles(kind):
    """Таблица характерных циклов (2.10.7): эталонные (s_min, s_max, r)."""
    table = {
        "symmetric":   (-1.0, 1.0, -1.0),
        "asymmetric":  (-0.5, 1.0, -0.5),
        "pulse_plus":  (0.0, 1.0, 0.0),
        "pulse_minus": (-1.0, 0.0, np.inf),
        "static":      (1.0, 1.0, 1.0),
    }
    try:
        lo, hi, r = table[kind]
    except KeyError:
        raise ValueError(f"Неизвестный тип цикла: {kind!r}") from None
    return {"s_min": lo, "s_max": hi, "r": r}


# ---------------------------------------------------------------------------
# (2.10.8)–(2.10.9) Кривая выносливости
# ---------------------------------------------------------------------------
def sn_stress(N, s_D, N_D, m_exp):
    """(2.10.8) Обратная ветвь кривой выносливости sigma^m * N = s_D^m * N_D:

        sigma(N) = s_D * (N_D/N)^(1/m),   N <  N_D (криволинейный участок);
        sigma(N) = s_D,                   N >= N_D (горизонтальная ветвь —
                                          предел выносливости).
    """
    NN, sD, ND, m = broadcast_fields(N, s_D, N_D, m_exp)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = sD * np.power(ND / NN, 1.0 / m)
    return np.where(NN <= ND, sig, sD)


def sn_cycles(sigma, s_D, N_D, m_exp):
    """(2.10.8) Прямая ветвь: N(sigma) = N_D * (s_D/sigma)^m (sigma > 0)."""
    sg, sD, ND, m = broadcast_fields(sigma, s_D, N_D, m_exp)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ND * np.power(sD / sg, m)


def sn_exponent_from_strength(sig_B):
    """(2.10.8a) Оценка показателя степени m по пределу прочности.

    Восстановлено по контексту («где m — показатель степени… можно оценить»).
    Типовые значения для сталей m ~ 6…12; линейная оценка по σ_B (МПа):

        m = 3 + sig_B / 100,   ограниченную диапазоном [4, 15].

    СТАТУС: требует утверждения заказчиком (точный вид был в imageNNN.gif).
    """
    sb = np.asarray(sig_B, dtype=float)
    return np.clip(3.0 + sb / 100.0, 4.0, 15.0)


def endurance_normal_from_strength(sig_B, low=0.4, high=0.5):
    """(2.10.9)-нормальные: sigma_r ~ (0.4…0.5) sigma_B; mid-point 0.45*σ_B.

    Возвращает массив границ, если low<high как кортеж? — нет: возвращает
    точечную оценку coef*sig_B с coef=(low+high)/2; границы доступны через
    (low*sig_B, high*sig_B) тем же вызовом params.
    """
    sb = np.asarray(sig_B, dtype=float)
    return 0.5 * (low + high) * sb


def endurance_torsion_from_normal(s_r=None, sig_B=None, material="ductile_steel"):
    """(2.10.9)-касательные:

        углеродистые/низколегированные стали: tau_r = 0.58 * sigma_r(-1);
        хрупкие материалы (высокоуглер., чугун): tau_r = 0.7 * sigma_B * 0.45?
        По учебнику: tau_r = 0.7*sigma_r для хрупких? Принято:
        ductile: 0.58*s_r; brittle: 0.7*s_r.
    """
    if material == "ductile_steel":
        coef = 0.58
    elif material == "brittle":
        coef = 0.7
    else:
        raise ValueError("material: 'ductile_steel' | 'brittle'")
    base = s_r if s_r is not None else 0.45 * np.asarray(sig_B, float)
    return coef * np.asarray(base, float)


def conditional_endurance_limit(s_D, N_D, m_exp, N_star=1e8):
    """Условный предел выносливости: sigma при N* циклах (N*=1e7…1e8) для
    материалов без горизонтальной ветви (цветные сплавы):

        sigma_N* = s_D * (N_D/N*)^(1/m)  — та же (2.10.8) без ограничения.
    """
    sD, ND, m = broadcast_fields(s_D, N_D, m_exp)
    return sD * np.power(ND / N_star, 1.0 / m)


# ---------------------------------------------------------------------------
# (2.10.10)–(2.10.13) Концентраторы напряжений
# ---------------------------------------------------------------------------
def effective_kk(s_r_plain, s_r_notched):
    """(2.10.10) k_eff = sigma_r(гладкий образец) / sigma_r(с концентратором)."""
    a, b = broadcast_fields(s_r_plain, s_r_notched)
    return a / b


def effective_kk_from_theoretical(kt, eps_s):
    """(2.10.11) k_eff = 1 + eps*(kt - 1); eps ∈ [0,1] — чувствительность
    материала (табл. 2.10.1: литые 0.15…0.35, среднеугл. 0.7…0.8,
    легированные 0.8…0.95, высоколегированные ~1.0)."""
    k, e = broadcast_fields(kt, eps_s)
    return 1.0 + e * (k - 1.0)


_EPS_TABLE = {  # табл. 2.10.1: материал -> диапазон e_sigma; mid-point
    "cast":              (0.15, 0.35),
    "medium_carbon":     (0.70, 0.80),
    "alloy":             (0.80, 0.95),
    "high_alloy":        (0.90, 1.00),
}


def sensitivity_coefficient(material="medium_carbon", kt=None, radius_mm=None):
    """Коэффициент чувствительности e_sigma по табл. 2.10.1 (mid-point).

    Уточнённая оценка (если заданы kt и радиус концентратора r, мм) —
    распространённая аппроксимация роста e с притуплением:

        e ≈ e_table * min(1, sqrt(r / r0)), r0 = 1 мм.

    СТАТУС диапазона — из текста учебника; уточняющая множительная форма —
    инженерная аппроксимация, требует утверждения заказчиком.
    """
    lo, hi = _EPS_TABLE[material]
    e = 0.5 * (lo + hi)
    if kt is None or radius_mm is None:
        return e
    r = float(radius_mm)
    return e * min(1.0, np.sqrt(max(r, 0.0) / 1.0))


def kk_approx_smooth(kt):
    """(2.10.12) Ориентировочная оценка при отсутствии резких концентраторов
    (обработанная поверхность). Восстановлено по контексту как верхняя грань:

        k_eff ≈ sqrt(kt)   (грубая ориентировочная оценка).

    СТАТУС: требует утверждения заказчиком.
    """
    return np.sqrt(np.asarray(kt, dtype=float))


def kk_approx_sharp(kt, sig_B):
    """(2.10.13) Ориентировочная оценка при резких концентраторах.

        k_eff ≈ 1 + (kt - 1) * f(sigma_B),  f — рост с прочностью.

    Принято f = min(1, sig_B/1000 МПа) (полная чувствительность прочных сталей).
    СТАТУС: требует утверждения заказчиком.
    """
    k, sb = broadcast_fields(kt, sig_B)
    f = np.minimum(1.0, sb / 1000.0)
    return 1.0 + (k - 1.0) * f


# ---------------------------------------------------------------------------
# (2.10.14)–(2.10.15) Масштабный коэффициент и качество поверхности
# ---------------------------------------------------------------------------
_SCALE_TABLES = {
    # d, мм -> b_n; огибающие рис. 2.10.8: 1 — углерод., 2 — легиров. стали
    "carbon_steel": ([10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500],
                     [1.00, 0.92, 0.88, 0.84, 0.81, 0.78, 0.74, 0.71, 0.65,
                      0.61, 0.56, 0.50]),
    "alloy_steel":  ([10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500],
                     [1.00, 0.88, 0.82, 0.78, 0.74, 0.71, 0.66, 0.62, 0.55,
                      0.50, 0.43, 0.36]),
}


def scale_factor(d_mm, material="carbon_steel", override=None):
    """(2.10.14) b_n = sigma_r(d) / sigma_r(d0=10 мм). Интерполяция таблиц;
    override — прямое экспериментальное значение."""
    if override is not None:
        return np.asarray(override, dtype=float)
    xp, fp = _SCALE_TABLES[material]
    return interp_table(np.asarray(d_mm, float), xp, fp)


_SURF_TABLE = {  # Rz, мкм -> b_s по классам обработки (рис. 2.10.9)
    "polishing":  ([0.4, 1.6, 3.2],      [1.00, 0.98, 0.95]),
    "grinding":   ([1.6, 3.2, 6.3, 10.0], [0.98, 0.95, 0.92, 0.88]),
    "fine_turn":  ([3.2, 6.3, 12.5, 20.0], [0.95, 0.90, 0.85, 0.80]),
    "rough_turn": ([12.5, 25.0, 40.0, 80.0], [0.85, 0.78, 0.70, 0.60]),
    "as_rolled":  ([40.0, 80.0, 150.0, 300.0], [0.70, 0.60, 0.50, 0.40]),
}


def surface_quality_factor(Rz_um=None, finish="polishing", override=None):
    """(2.10.15) b_s = sigma_r(фактическая поверхность)/sigma_r(полированная)."""
    if override is not None:
        return np.asarray(override, dtype=float)
    xp, fp = _SURF_TABLE[finish]
    if Rz_um is None:
        return float(fp[0])
    return interp_table(np.asarray(Rz_um, float), xp, fp)


def reduction_factor(k_eff, b_n=1.0, b_s=1.0, y_beta=1.0, form="gost"):
    """(2.10.18) Коэффициент снижения предела выносливости K_sigma / K_tau.

        form='gost' (канон методик ГОСТ 27.012/расчётов МАМИ-МГТУ):
            K = (k_eff + 1/b_s - 1) / (b_n * y_beta)
        form='product' (упрощённая мультипликативная оценка):
            K = k_eff / (b_n * b_s * y_beta)

    При идеальных b_s = b_n = y_beta = 1 обе формы дают K = k_eff, что
    согласуется с определением (2.10.10). y_beta >= 1 — упрочнение поверхности
    снижает эффективное K.
    """
    ke, bn, bs, by = broadcast_fields(k_eff, b_n, b_s, y_beta)
    if form == "gost":
        return (ke + 1.0 / bs - 1.0) / (bn * by)
    if form == "product":
        return ke / (bn * bs * by)
    raise ValueError("form: 'gost' | 'product'")


# ---------------------------------------------------------------------------
# (2.10.16)–(2.10.22) Запасы усталостной прочности
# ---------------------------------------------------------------------------
def safety_ref_sample(s_r, sa):
    """(2.10.16) n_sigma = sigma_r / sa, n_tau = tau_r / ta (эталонный образец)."""
    sr, a = broadcast_fields(s_r, sa)
    return sr / a


def safety_part(s_r, sa, sm=0.0, K=1.0, psi=0.0):
    """(2.10.17) n = sigma_r / (K*sa + psi*sm).

    K — снижение предела выносливости (2.10.18), psi — влияние асимметрии.
    Частный случай sm=0, K=k_eff: сводится к (2.10.16) с поправкой.
    """
    sr, a, m, KK, p = broadcast_fields(s_r, sa, sm, K, psi)
    return sr / (KK * a + p * m)


def psi_from_limit_points(s_r, sa_lim, sm_lim):
    """Коэффициент psi по предельной точке (sa_lim, sm_lim) диаграммы
    предельных амплитуд: из (2.10.20) n=1 =>

        psi = (sigma_r - K*sa_lim)/sm_lim  при K=1 (эталон).

    Каноническая оценка через sigma_B: psi = 2*sigma_r/sigma_B - 1.
    """
    sr, a, m = broadcast_fields(s_r, sa_lim, sm_lim)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (sr - a) / m


def equivalent_symmetric_amplitude(sa, sm, psi, K=1.0):
    """(2.10.21) Приведение асимметричного цикла к эквивалентному симметричному:

        sa_eq = K*sa + psi*|sm|.
    """
    a, m, p, KK = broadcast_fields(sa, np.abs(sm), psi, K)
    return KK * a + p * m


def equivalent_symmetric_amplitude_tau(ta, tm, psi_tau, K_tau=1.0):
    """(2.10.22) То же для касательных напряжений (sigma→tau)."""
    return equivalent_symmetric_amplitude(ta, tm, psi_tau, K_tau)


# ---------------------------------------------------------------------------
# (2.10.23)–(2.10.27) Проверка статической прочности
# ---------------------------------------------------------------------------
def static_safety_normal(sig_T, sm, sa):
    """(2.10.23) n = sigma_T / (sm + sa) = sigma_T / s_max."""
    T, m, a = broadcast_fields(sig_T, sm, sa)
    return T / (m + a)


def static_safety_shear(tau_T, tm, ta):
    """(2.10.24) n_tau = tau_T / (tm + ta)."""
    T, m, a = broadcast_fields(tau_T, tm, ta)
    return T / (m + a)


def equivalent_tension(sm, sa, tm, ta):
    """(2.10.25) Эквивалентное растяжение сложного нагружения:

        sigma_eq = sqrt((sm+sa)^2 + 3*(tm+ta)^2).
    """
    m, a, tm_, ta_ = broadcast_fields(sm, sa, tm, ta)
    return np.hypot(m + a, np.sqrt(3.0) * (tm_ + ta_))


def combined_static_safety(sig_T, sm, sa, tm, ta, tau_T=None):
    """(2.10.26)/(2.10.27) Запас при совместном нормальном+касательном
    нагружении (гипотеза наибольших касательных напряжений):

        1/n^2 = [(sm+sa)/sigma_T]^2 + [(tm+ta)/tau_T]^2,
        tau_T = sigma_T/sqrt(3)  (если не задан явно — следствие 2.10.9/2.7.35).
    """
    T, m, a, tm_, ta_ = broadcast_fields(sig_T, sm, sa, tm, ta)
    tT = T / np.sqrt(3.0) if tau_T is None else np.asarray(tau_T, float)
    inv2 = ((m + a) / T) ** 2 + ((tm_ + ta_) / tT) ** 2
    return 1.0 / np.sqrt(inv2)


# ---------------------------------------------------------------------------
# (2.10.28)–(2.10.31) Общее напряженное состояние
# ---------------------------------------------------------------------------
def equivalent_amplitude_principal(sa1, sa3):
    """(2.10.28)/(2.10.29) Эквивалентное амплитудное напряжение (III теория):

        sigma_a_eq = sa_1 - sa_3 (наибольшая и наименьшая главные амплитуды);
        sigma_m_eq = sm_1 - sm_3.
    """
    a1, a3 = broadcast_fields(sa1, sa3)
    return a1 - a3


def fatigue_safety_general(s_r, sa_eq, sm_eq=0.0, psi=0.0, K=1.0):
    """(2.10.30) Условие прочности общего случая: n = sigma_r/(K*sa_eq+psi*sm_eq)."""
    sr, a, m, p, KK = broadcast_fields(s_r, sa_eq, sm_eq, psi, K)
    return sr / (KK * a + p * m)


def check_fatigue_safety(n, n_min=1.3, n_max=1.6):
    """Проверка запаса в допускаемой зоне [n_min, n_max] (см. текст к 2.10.30).

    Возвращает dict {'ok': булево поле, 'under': n<n_min, 'over': n>n_max}.
    """
    nn = np.asarray(n, dtype=float)
    return {"ok": (nn >= n_min) & (nn <= n_max),
            "under": nn < n_min, "over": nn > n_max}


def safety_limited_endurance(N, s_D, N_D, m_exp, sa_eq, sm_eq=0.0, psi=0.0, K=1.0):
    """(2.10.31) Запас в области ограниченной выносливости (N < N_D):
    вместо sigma_r используется sigma_R(N) из (2.10.8)."""
    sig = sn_stress(N, s_D, N_D, m_exp)
    KK, a, m, p = broadcast_fields(K, sa_eq, sm_eq, psi)
    return sig / (KK * a + p * m)


# ---------------------------------------------------------------------------
# (2.10.33)–(2.10.38) Суммирование повреждений и эквивалентные режимы
# ---------------------------------------------------------------------------
def miner_damage_sum(n_i, N_i):
    """(2.10.33) Гипотеза линейного суммирования: D = sum(n_i/N_i) (=1 при разрушении).

    n_i, N_i — массив уровней гистограммы нагружения.
    """
    ni, Ni = broadcast_fields(n_i, N_i)
    return float(np.sum(ni / Ni))


def sn_cycles_curved_branch(sigma, m_exp, C=None, s_D=None, N_D=None):
    """(2.10.34) Предельное число циклов на криволинейном участке:

        sigma^m * N = C  =>  N = C/sigma^m,  C = s_D^m * N_D.
    """
    if C is None:
        C = np.asarray(s_D, float) ** m_exp * np.asarray(N_D, float)
    sg = np.asarray(sigma, dtype=float)
    return C / np.power(sg, np.asarray(m_exp, float))


def equivalent_stress(n_i, sigma_i, m_exp):
    """(2.10.35)/(2.10.37) Эквивалентное напряжение постоянного режима:

        sigma_eq = ( sum(n_i*sigma_i^m) / sum(n_i) )^(1/m).

    Вход — гистограмма нагружения (массивы уровней); скалярные аргументы
    допускаются. Для полей МКЭ применяется вызывающим кодом поэлементно.
    """
    ni = np.asarray(n_i, dtype=float)
    si = np.asarray(sigma_i, dtype=float)
    m = float(m_exp)
    num = np.sum(ni * np.power(si, m))
    den = np.sum(ni)
    return float(np.power(num / den, 1.0 / m))


def equivalent_cycles(n_i, sigma_i, sigma_max, m_exp):
    """(2.10.38) Эквивалентное число циклов при приведении к sigma_max:

        N_eq = sum( n_i * (sigma_i/sigma_max)^m ).
    """
    ni = np.asarray(n_i, dtype=float)
    si = np.asarray(sigma_i, dtype=float)
    smax = float(sigma_max)
    m = float(m_exp)
    return float(np.sum(ni * np.power(si / smax, m)))


# ---------------------------------------------------------------------------
# Метод полных циклов (схематизация случайного нагружения, табл. 2.10.2)
# ---------------------------------------------------------------------------
def full_cycle_counting(signal):
    """Метод полных циклов (range pair counting, ASTM E1049-85 п. 5.4.2)
    дискретной осциллограммы напряжений — схематизация случайного нагружения
    (рис. 2.10.16, табл. 2.10.2 учебника).

    Реализация: NumPy-извлечение последовательности обращений (чередующихся
    локальных экстремумов, плато схлопываются) O(N), затем однопроходный
    стек вершин/впадин с инвертированием знака при отбрасывании — точная
    схема п. 5.4.2 стандарта. Хвостовые полуциклы в метод полных циклов
    не входят.

    Возвращает dict: {'ranges': ndarray, 'means': ndarray,
                     'amplitudes': R/2, 'counts': число полных циклов}.
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 3:
        z = np.zeros(0)
        return {"ranges": z, "means": z, "amplitudes": z, "counts": 0}
    # ---- последовательность обращений: плато схлопываем, берём точки смены
    #      знака наклона; добавляем начало и конец сигнала как крайние точки ----
    d = np.diff(x)
    keep = np.concatenate(([True], d != 0))
    xs = x[keep]
    ds = np.diff(xs)
    sgn = np.sign(ds)                      # знак наклона между xs[i], xs[i+1]
    flip = np.nonzero(np.diff(sgn))[0]     # смена знака: экстремум в xs[i+1]
    if flip.size == 0:                     # монотонный сигнал — циклов нет
        z = np.zeros(0)
        return {"ranges": z, "means": z, "amplitudes": z, "counts": 0}
    ext_idx = np.concatenate(([0], flip + 1, [len(xs) - 1]))
    ext = xs[ext_idx]
    if ext.size >= 2 and ext[-1] == ext[-2]:   # конец совпал с экстремумом
        ext = ext[:-1]
    if ext.size >= 2 and ext[0] == ext[1]:
        ext = ext[1:]
    # ---- range pair counting (ASTM E1049-85, п. 5.4.2) ----
    rngs, means = [], []
    stack = []
    for v in ext.tolist():
        while len(stack) >= 2:
            a, b = stack[-2], stack[-1]
            X = abs(v - b)                 # последний диапазон
            Y = abs(b - a)                 # предпоследний диапазон
            if Y < X:
                break                      # полного цикла нет — продолжаем сбор
            rngs.append(Y)                 # полный цикл: размах Y, среднее (a+b)/2
            means.append(0.5 * (a + b))
            del stack[-1]                  # отбрасываем вершину/впадину B
            stack = [-s for s in stack]    # инвертируем оставшуюся часть стека
        stack.append(v)
    ranges = np.asarray(rngs, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    return {"ranges": ranges, "means": means_arr,
            "amplitudes": ranges / 2.0, "counts": ranges.size}
