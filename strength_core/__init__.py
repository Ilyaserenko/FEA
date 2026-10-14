"""
strength_core — расчетное ядро программы прочности (МКЭ-постобработка).

Модули:
    stats       — Глава 1.1: статистика СВ и законы распределения (1.1.1–1.1.27)
    tolerance   — Глава 1.2: теория допусков и посадок (1.2.1–1.2.15)
    reliability — Глава 1.3: надёжность и вероятность разрушения (1.3.1–1.3.21)

Все функции векторизованы (NumPy/scipy): принимают массивы значений узлов/
элементов/партий и возвращают массивы результатов; скалярные входы тоже
поддерживаются.
"""

from . import stats, tolerance, reliability
from .stats import *        # noqa: F401,F403
from .tolerance import *    # noqa: F401,F403
from .reliability import *  # noqa: F401,F403

__all__ = ["stats", "tolerance", "reliability"] + stats.__all__ + tolerance.__all__ + reliability.__all__
