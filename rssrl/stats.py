"""Правило статистического решения и перекрестный бутстреп (письмо 29.10).

Данные одной ячейки (система × условие) — матрица исходов
[политики × сцены] (1 — столкновение, 0 — нет). Сцены общие для всех систем
и условий (одни и те же seed сцен и расписания), поэтому столбец j во всех
матрицах — одна и та же сцена. Политики (seed обучения PPO) — строки; у
IDM/MOBIL + RSS строка одна (обучения нет).

Перекрестный бутстреп (crossed_bootstrap): в каждом повторе
- индексы сцен выбираются с возвращением ОДИН раз и применяются ко всем
  сравниваемым матрицам (общие индексы сцен);
- индексы политик выбираются с возвращением; если обе матрицы — одни и те же
  обученные политики в разных условиях (например, с фильтром и без при
  исполнении, схема 2×2), индексы политик тоже общие (paired_policies=True),
  иначе — свои для каждой системы.
Так учитываются обе случайности: какие сцены попались и какие политики
получились при обучении. Политики и сцены — перекрестная структура, а не
независимые наборы эпизодов в каждой ячейке.

Решение — две отдельные характеристики (decide):
1. Статистический знак разности Δ = P(столкновение | A) − P(... | B):
   «хуже» — весь ДИ > 0, «лучше» — весь ДИ < 0, иначе «не различимо».
2. Практическая величина относительно допуска ±margin (2 п.п.):
   «в пределах допуска» — весь ДИ внутри [−margin, margin];
   «за пределами допуска» — весь ДИ вне его по одну сторону;
   «не определена» — ДИ пересекает границу допуска.
Например, ДИ [0,5; 1,5] п.п. — «хуже, в пределах допуска»: увеличение
риска статистически обнаружено, но меньше выбранного практического порога.

Нулевые исходы: если в обеих матрицах нет ни одного столкновения, бутстреп
дает вырожденный интервал [0; 0] — это не доказательство отсутствия
риска. Тогда для каждой системы отдельно — точная односторонняя верхняя
граница Клоппера — Пирсона zero_upper_bound (для 0 из n независимых сцен:
1 − α^(1/n), при n = 100 и α = 0,05 около 3 %). Это граница для одного
варианта при независимых сценах, а не общая гарантия всех вариантов.
"""

import math

import numpy as np

MARGIN = 0.02  # практический допуск, доля (±2 п.п.)
ALPHA = 0.05

STAT_WORSE, STAT_BETTER, STAT_NONE = "хуже", "лучше", "не различимо"
MAG_WITHIN, MAG_OUTSIDE, MAG_UNCLEAR = "в пределах допуска", "за пределами допуска", "не определена"


def _matrix(x):
    x = np.asarray(x, dtype=float)
    return x[None, :] if x.ndim == 1 else x


def crossed_bootstrap(a, b, n_boot=10_000, seed=0, paired_policies=False):
    """Повторы бутстрепа разности средних Δ = mean(a) − mean(b).

    a, b — матрицы [политики × сцены] с общими сценами (одинаковое число
    столбцов). paired_policies=True — строки a и b — одни и те же политики
    (тогда и число строк одинаковое). Возвращает массив из n_boot значений Δ.
    """
    a, b = _matrix(a), _matrix(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("нужны общие сцены: одинаковое число столбцов")
    if paired_policies and a.shape[0] != b.shape[0]:
        raise ValueError("paired_policies: нужны одни и те же политики")
    rng = np.random.default_rng(seed)
    n_s = a.shape[1]
    out = np.empty(n_boot)
    for k in range(n_boot):
        sc = rng.integers(n_s, size=n_s)
        pa = rng.integers(a.shape[0], size=a.shape[0])
        pb = pa if paired_policies else rng.integers(b.shape[0], size=b.shape[0])
        out[k] = a[np.ix_(pa, sc)].mean() - b[np.ix_(pb, sc)].mean()
    return out


def percentile_ci(boot, alpha=ALPHA):
    """Двусторонний процентильный ДИ уровня 1 − alpha."""
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def bootstrap_p(boot):
    """Двусторонний бутстреп-p для гипотезы Δ = 0: удвоенная доля повторов
    по другую сторону от нуля (не больше 1)."""
    boot = np.asarray(boot)
    return float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))


def decide(ci, margin=MARGIN):
    """Две характеристики по ДИ разности: (статистический знак, величина)."""
    lo, hi = ci
    if lo > 0:
        sign = STAT_WORSE
    elif hi < 0:
        sign = STAT_BETTER
    else:
        sign = STAT_NONE
    if -margin <= lo and hi <= margin:
        mag = MAG_WITHIN
    elif lo > margin or hi < -margin:
        mag = MAG_OUTSIDE
    else:
        mag = MAG_UNCLEAR
    return sign, mag


def zero_upper_bound(n, alpha=ALPHA):
    """Точная односторонняя верхняя граница Клоппера — Пирсона для 0
    событий из n: p такое, что (1 − p)^n = alpha."""
    if n <= 0:
        return 1.0
    return 1.0 - alpha ** (1.0 / n)


def holm(pvalues, alpha=ALPHA):
    """Поправка Холма. Возвращает (отвергнута ли гипотеза, уровень alpha_i,
    с которым сравнивалось p_i) в исходном порядке. ДИ для решения по
    сравнению i строится на уровне 1 − alpha_i."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    reject = np.zeros(m, bool)
    level = np.empty(m)
    stopped = False
    for rank, i in enumerate(order):
        level[i] = alpha / (m - rank)
        if not stopped and p[i] <= level[i]:
            reject[i] = True
        else:
            stopped = True
    return reject, level


def compare(a, b, paired_policies=False, alpha=ALPHA, margin=MARGIN, n_boot=10_000, seed=0):
    """Одно сравнение: разность частот, ДИ, p и две характеристики.
    При полностью нулевых исходах — без бутстрепа, с точными границами."""
    a, b = _matrix(a), _matrix(b)
    out = {"diff": float(a.mean() - b.mean()), "n_scenes": a.shape[1],
           "n_policies_a": a.shape[0], "n_policies_b": b.shape[0]}
    if a.sum() == 0 and b.sum() == 0:
        out.update(ci=(math.nan, math.nan), p=math.nan, sign=STAT_NONE, magnitude=MAG_UNCLEAR,
                   zero_bound_a=zero_upper_bound(a.shape[1], alpha),
                   zero_bound_b=zero_upper_bound(b.shape[1], alpha),
                   note="нет столкновений ни в одной ячейке: бутстреп не применяем, "
                        "точные границы для каждой системы отдельно")
        return out
    boot = crossed_bootstrap(a, b, n_boot, seed, paired_policies)
    ci = percentile_ci(boot, alpha)
    sign, mag = decide(ci, margin)
    out.update(ci=ci, p=bootstrap_p(boot), sign=sign, magnitude=mag, note="")
    return out


def beta_star(levels, signs):
    """β* — наименьший уровень сетки со знаком «хуже», подтвержденный «хуже»
    на следующем уровне. Возвращает (β* или None, пояснение).

    Верхняя точка сетки: если «хуже» только на последнем уровне (или на
    предпоследнем и последнем нет подтверждения), подтвердить нечем —
    исход «β* не подтвержден: хуже только на верхней точке сетки»
    (β* > предпоследнего уровня, если он существует, неизвестен).
    """
    for i, (lvl, s) in enumerate(zip(levels, signs)):
        if s != STAT_WORSE:
            continue
        if i + 1 < len(levels):
            if signs[i + 1] == STAT_WORSE:
                return lvl, f"хуже при β = {lvl:g}, подтверждено при β = {levels[i + 1]:g}"
        else:
            return None, f"β* не подтвержден: хуже только на верхней точке сетки β = {lvl:g}"
    return None, "β* не найден: ни на одном уровне сетки нет подтвержденного «хуже»"


# Основные сравнения и семьи поправки Холма (план, письмо 29.10, п. 6–7).
# Системы: PPO_S — политика, обученная с фильтром; PPO_N — без фильтра;
# «+R» — исполнение с фильтром (базовая надлежащая реакция), «−R» — без.
# IDM_R — IDM/MOBIL + RSS без RL (одна строка, без seed обучения).
COMPARISONS = {
    "A (основная, Холм по 6)": [
        f"β = {b}: PPO_S+R − PPO_N−R" for b in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
    ],
    "B (деградация, Холм по 5)": [
        f"PPO_S+R: β = {b} − β = 1" for b in (1.2, 1.4, 1.6, 1.8, 2.0)
    ],
    "C (дополнительный γ, Холм по 3)": [
        f"PPO_S+R: γ = {g} − γ = 1" for g in (0.75, 0.5, 0.25)
    ],
}
