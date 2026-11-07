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

Единая процедура для семьи сравнений (decide_family, письмо 07.11, п. 2):
1. Для каждого сравнения считаем бутстреп и двусторонний p (bootstrap_p).
2. Поправка Холма по всем p семьи: идем от меньшего p к большему и
   останавливаемся на первом неотклоненном; все дальше тоже не отклонены.
3. Статистический знак разности Δ = P(столкновение | A) − P(... | B) ставится
   ТОЛЬКО отклоненным сравнениям: «хуже» (Δ > 0) или «лучше» (Δ < 0).
   Остальные — «не различимо», какими бы ни были их отдельные интервалы.
4. Интервал для подписи (ci): у отклоненного — на уровне его шага Холма, у
   остальных — на уровне шага, где Холм остановился. p и интервал построены из
   одних повторов так, что отклоненный интервал всегда не содержит 0, а
   неотклоненный содержит (percentile_ci, проверено тестом). Подпись и решение
   не могут разойтись. Совместного покрытия у этих интервалов нет.
5. Практическая величина относительно допуска ±margin считается по интервалу
   ci_simult на уровне 1 − α/m (Бонферрони): такие интервалы накрывают все m
   разностей одновременно с вероятностью не меньше 1 − α:
   «в пределах допуска» — весь интервал внутри [−margin, margin];
   «за пределами допуска» — весь интервал вне его по одну сторону;
   «не определена» — интервал пересекает границу допуска.
6. β* семьи — по этим скорректированным знакам (beta_star).

Допуск ±2 п.п. — заранее заданная конвенция, выбранная до результатов; это
не вывод из данных. Поэтому рядом анализ чувствительности: те же интервалы при
1, 2 и 5 п.п. (margin_sensitivity).

Два порога с собственными критериями (письмо 07.11, п. 1):
β_A — семья A, гипотеза «защищенная система хуже незащищенной при том же β»;
β_B — семья B, деградация защищенной системы относительно β = 1.
Результат семьи B не подтверждает гипотезу семьи A. Перекрестная схема 2×2 —
только описание (describe_2x2): без p и знаков.

Нулевые исходы: если в обеих матрицах нет ни одного столкновения, бутстреп
дает вырожденный интервал [0; 0] — это не доказательство отсутствия риска.
Такое сравнение остается в семье (m не меняется) с p = 1 для Холма и получает
точную одностороннюю верхнюю границу Клоппера — Пирсона для каждой системы
(zero_upper_bound; для 0 из n независимых сцен: 1 − α^(1/n), при n = 100 и
α = 0,05 около 3 %). Область действия: фиксированная политика на независимых
сценах. Это не граница риска процедуры обучения PPO на новых seed и не общая
гарантия всех вариантов.
"""

import math

import numpy as np
import pandas as pd

MARGIN = 0.02  # практический допуск, доля (±2 п.п.): заранее заданная конвенция
SENSITIVITY_MARGINS = (0.01, 0.02, 0.05)  # анализ чувствительности к допуску
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
    """Двусторонний процентильный ДИ уровня 1 − alpha по порядковым статистикам.

    Границы — k-е значения с каждого края, k = floor(alpha·n/2). Так ДИ точно
    согласован с bootstrap_p: p <= alpha тогда и только тогда, когда ДИ не
    содержит 0. Интерполяция квантилей дала бы расхождение на границе."""
    b = np.sort(np.asarray(boot, dtype=float))
    k = math.floor(alpha * len(b) / 2 + 1e-9)
    return float(b[k]), float(b[len(b) - 1 - k])


def bootstrap_p(boot):
    """Двусторонний бутстреп-p для гипотезы Δ = 0: удвоенная доля повторов
    по другую сторону от нуля (не больше 1)."""
    boot = np.asarray(boot)
    return float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))


def magnitude(ci, margin=MARGIN):
    """Практическая величина по интервалу относительно допуска ±margin."""
    lo, hi = ci
    if -margin <= lo and hi <= margin:
        return MAG_WITHIN
    if lo > margin or hi < -margin:
        return MAG_OUTSIDE
    return MAG_UNCLEAR


def zero_upper_bound(n, alpha=ALPHA):
    """Точная односторонняя верхняя граница Клоппера — Пирсона для 0
    событий из n: p такое, что (1 − p)^n = alpha."""
    if n <= 0:
        return 1.0
    return 1.0 - alpha ** (1.0 / n)


def holm(pvalues, alpha=ALPHA):
    """Поправка Холма. Возвращает (отвергнута ли гипотеза, уровень alpha_i,
    с которым сравнивалось p_i) в исходном порядке. Решение со знаком и
    подпись интервала дает только decide_family: один уровень alpha_i не
    передает правило остановки."""
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


def holm_adjusted(pvalues):
    """Скорректированные по Холму p: отвергнута при adj <= alpha (те же решения,
    что у holm, но при любом alpha)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def _family_from_boots(boots, alpha, margin):
    """Решения семьи по повторам бутстрепа. boots[i] — массив Δ сравнения i или
    None, если исходы нулевые (бутстреп не применяем, p = 1 для Холма)."""
    m = len(boots)
    p = [math.nan if b is None else bootstrap_p(b) for b in boots]
    p_holm = np.array([1.0 if b is None else p[i] for i, b in enumerate(boots)])
    reject, level = holm(p_holm, alpha)
    # уровень шага, на котором Холм остановился: у всех неотклоненных он один
    stop_level = min((level[i] for i in range(m) if not reject[i]), default=math.nan)
    rows = []
    for i, b in enumerate(boots):
        ci_level = level[i] if reject[i] else stop_level
        if b is None:
            rows.append({"p": math.nan, "reject": False, "sign": STAT_NONE, "ci": (math.nan, math.nan),
                         "ci_level": ci_level, "ci_simult": (math.nan, math.nan), "magnitude": MAG_UNCLEAR})
            continue
        sign = STAT_NONE
        if reject[i]:
            sign = STAT_WORSE if (b <= 0).mean() < (b >= 0).mean() else STAT_BETTER
        ci_simult = percentile_ci(b, alpha / m)
        rows.append({"p": p[i], "reject": bool(reject[i]), "sign": sign, "ci": percentile_ci(b, ci_level),
                     "ci_level": ci_level, "ci_simult": ci_simult, "magnitude": magnitude(ci_simult, margin)})
    adj = holm_adjusted(p_holm)
    for r, x in zip(rows, adj):
        r["p_holm"] = float(x)
    return rows


def decide_family(items, alpha=ALPHA, margin=MARGIN, n_boot=10_000, seed=0):
    """Единая процедура для семьи сравнений. Других способов получить знак нет.

    items — список словарей {"level": уровень β или γ, "a": матрица, "b":
    матрица, "paired_policies": bool}, все сравнения заранее заданной семьи.
    Возвращает DataFrame в порядке items: level, diff, p, p_holm, reject, sign,
    ci и ci_level (подпись, согласована с решением), ci_simult и magnitude,
    zero_bound_a/zero_bound_b и note при нулевых исходах.
    Один и тот же seed дает общие индексы сцен для всех сравнений семьи.
    """
    boots, base = [], []
    for it in items:
        a, b = _matrix(it["a"]), _matrix(it["b"])
        row = {"level": it.get("level"), "diff": float(a.mean() - b.mean()), "n_scenes": a.shape[1],
               "n_policies_a": a.shape[0], "n_policies_b": b.shape[0], "note": "",
               "zero_bound_a": math.nan, "zero_bound_b": math.nan}
        if a.sum() == 0 and b.sum() == 0:
            boots.append(None)
            row.update(zero_bound_a=zero_upper_bound(a.shape[1], alpha),
                       zero_bound_b=zero_upper_bound(b.shape[1], alpha),
                       note="нет столкновений ни в одной ячейке: бутстреп не применяем, точные границы "
                            "для каждой системы отдельно (фиксированная политика, независимые сцены)")
        else:
            boots.append(crossed_bootstrap(a, b, n_boot, seed, it.get("paired_policies", False)))
        base.append(row)
    rows = _family_from_boots(boots, alpha, margin)
    return pd.DataFrame([{**x, **r} for x, r in zip(base, rows)])


def compare(a, b, paired_policies=False, alpha=ALPHA, margin=MARGIN, n_boot=10_000, seed=0):
    """Одно отдельное сравнение: семья из одного элемента (без поправки)."""
    item = {"a": a, "b": b, "paired_policies": paired_policies}
    return decide_family([item], alpha, margin, n_boot, seed).iloc[0].to_dict()


def family_threshold(table, levels=None):
    """Порог семьи: β* по скорректированным знакам decide_family.
    levels — вся сетка в порядке возрастания (по умолчанию уровни таблицы)."""
    levels = list(table["level"]) if levels is None else list(levels)
    signs = dict(zip(table["level"], table["sign"]))
    return beta_star(levels, [signs.get(x, STAT_NONE) for x in levels])


def margin_sensitivity(table, margins=SENSITIVITY_MARGINS):
    """Практическая величина того же интервала ci_simult при разных допусках.
    Знак от допуска не зависит, меняются только эти подписи."""
    out = pd.DataFrame({"level": table["level"], "sign": table["sign"]})
    for mg in margins:
        out[f"±{100 * mg:g} п.п."] = [magnitude(ci, mg) if np.isfinite(ci[0]) else MAG_UNCLEAR
                                      for ci in table["ci_simult"]]
    return out


def describe_2x2(cells):
    """Перекрестная схема 2×2 — только описание, диагностика возможного
    привыкания, а не доказательство механизма: без p и знаков.

    cells — {(обучена, исполняется): матрица [политики × сцены]}, например
    {("с фильтром", "с фильтром"): ...}. Частота столкновений ячейки и
    диапазон частот отдельных политик."""
    rows = []
    for (trained, executed), m in cells.items():
        per_policy = _matrix(m).mean(axis=1)
        rows.append({"обучена": trained, "исполняется": executed, "частота": float(per_policy.mean()),
                     "мин. по политикам": float(per_policy.min()), "макс. по политикам": float(per_policy.max()),
                     "политик": len(per_policy)})
    return pd.DataFrame(rows)


def beta_star(levels, signs):
    """β* — наименьший уровень сетки со знаком «хуже», подтвержденный «хуже»
    на следующем уровне. Возвращает (β* или None, пояснение). signs —
    скорректированные знаки decide_family, а не подписи отдельных интервалов.

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


# Основные сравнения и семьи поправки Холма (план, письма 29.10 и 07.11).
# Системы: PPO_S — политика, обученная с фильтром; PPO_N — без фильтра;
# «+R» — исполнение с фильтром (базовая надлежащая реакция), «−R» — без.
# IDM_R — IDM/MOBIL + RSS без RL (одна строка, без seed обучения).
# У каждой семьи свой вопрос и свой порог; 2×2 в семьи не входит.
COMPARISONS = {
    "A (основная гипотеза, порог β_A, Холм по 6)": [
        f"β = {b}: PPO_S+R − PPO_N−R" for b in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
    ],
    "B (деградация, порог β_B, Холм по 5)": [
        f"PPO_S+R: β = {b} − β = 1" for b in (1.2, 1.4, 1.6, 1.8, 2.0)
    ],
    "C (дополнительный γ, Холм по 3)": [
        f"PPO_S+R: γ = {g} − γ = 1" for g in (0.75, 0.5, 0.25)
    ],
}
