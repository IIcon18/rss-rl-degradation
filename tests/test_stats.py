"""Правило решения для семьи сравнений (письма 29.10 и 07.11), перекрестный
бутстреп, нулевые исходы."""

import numpy as np
import pytest

from rssrl.stats import (MAG_OUTSIDE, MAG_UNCLEAR, MAG_WITHIN, STAT_BETTER, STAT_NONE, STAT_WORSE, _family_from_boots,
                         beta_star, bootstrap_p, compare, crossed_bootstrap, decide_family, describe_2x2,
                         family_threshold, holm, holm_adjusted, magnitude, margin_sensitivity, percentile_ci,
                         zero_upper_bound)


def boot_with_p(p, n=100_000, center=0.05, spread=0.01):
    """Синтетические повторы бутстрепа: разность положительна, а доля повторов
    <= 0 ровно p/2, то есть bootstrap_p = p."""
    k = round(p * n / 2)
    return np.concatenate([np.full(k, -spread), np.full(n - k, center)])


def test_magnitude_by_interval():
    assert magnitude((0.005, 0.015)) == MAG_WITHIN
    assert magnitude((0.03, 0.06)) == MAG_OUTSIDE
    assert magnitude((0.01, 0.03)) == MAG_UNCLEAR
    assert magnitude((-0.01, 0.01)) == MAG_WITHIN
    assert magnitude((-0.05, 0.05)) == MAG_UNCLEAR
    assert magnitude((-0.06, -0.03)) == MAG_OUTSIDE


def test_sign_and_magnitude_are_separate():
    """Пример письма 29.10: разность около 1 п.п. с узким интервалом — «хуже,
    в пределах допуска»: риск вырос, но меньше практического порога."""
    r = _family_from_boots([np.linspace(0.005, 0.015, 10_000)], 0.05, 0.02)[0]
    assert r["sign"] == STAT_WORSE and r["magnitude"] == MAG_WITHIN


def test_holm_stop_rule_example_from_letter():
    """Письмо 07.11, п. 2: p = [0,03; 0,04], α = 0,05. Холм не отклоняет ни
    одной гипотезы. Отдельный интервал второго сравнения на его ранговом уровне
    0,05 не содержит 0, но знака «хуже» быть не должно."""
    boots = [boot_with_p(0.03), boot_with_p(0.04)]
    assert bootstrap_p(boots[0]) == pytest.approx(0.03) and bootstrap_p(boots[1]) == pytest.approx(0.04)
    assert percentile_ci(boots[1], 0.05)[0] > 0  # одиночный интервал: «вся разность > 0»
    rows = _family_from_boots(boots, 0.05, 0.02)
    assert [r["reject"] for r in rows] == [False, False]
    assert [r["sign"] for r in rows] == [STAT_NONE, STAT_NONE]
    # подпись согласована: оба интервала на уровне остановки 0,025 содержат 0
    for r in rows:
        assert r["ci_level"] == pytest.approx(0.025)
        assert r["ci"][0] <= 0 <= r["ci"][1]


def test_holm_signs_only_rejected_and_keeps_direction():
    down = -boot_with_p(0.001)
    rows = _family_from_boots([boot_with_p(0.001), down, boot_with_p(0.5)], 0.05, 0.02)
    assert [r["sign"] for r in rows] == [STAT_WORSE, STAT_BETTER, STAT_NONE]
    assert rows[0]["ci"][0] > 0 and rows[1]["ci"][1] < 0 and rows[2]["ci"][0] <= 0 <= rows[2]["ci"][1]


def test_interval_labels_always_agree_with_holm():
    """На случайных семьях: отклонено — интервал не содержит 0 и знак есть;
    не отклонено — интервал содержит 0 и знака нет. p_holm дает те же решения."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        m = rng.integers(1, 7)
        boots = [rng.normal(rng.uniform(-0.03, 0.03), 0.01, 2000) for _ in range(m)]
        rows = _family_from_boots(boots, 0.05, 0.02)
        for r in rows:
            contains_zero = r["ci"][0] <= 0 <= r["ci"][1]
            assert r["reject"] == (not contains_zero)
            assert r["reject"] == (r["sign"] != STAT_NONE)
            assert r["reject"] == (r["p_holm"] <= 0.05 + 1e-12)


def test_ci_is_exactly_dual_to_bootstrap_p():
    rng = np.random.default_rng(3)
    for _ in range(300):
        boot = rng.normal(rng.uniform(-0.02, 0.02), 0.01, 1000).round(3)  # с повторами значений
        for alpha in (0.05, 0.025, 0.05 / 6):
            lo, hi = percentile_ci(boot, alpha)
            assert (bootstrap_p(boot) <= alpha) == (lo > 0 or hi < 0)


def test_holm_adjusted_hand_example():
    # 0,01·4 = 0,04; 0,03·3 = 0,09; 0,04·2 = 0,08 -> 0,09 (монотонность); 0,2
    assert holm_adjusted([0.01, 0.04, 0.03, 0.2]).tolist() == pytest.approx([0.04, 0.09, 0.09, 0.2])


def _item(level, a, b, paired=True):
    return {"level": level, "a": a, "b": b, "paired_policies": paired}


def test_family_all_zero_outcomes():
    """Полностью нулевые исходы: без бутстрепа, нет отклонений и знаков, порог
    не найден, точная граница у каждой системы; величина не определена."""
    z = np.zeros((2, 100))
    t = decide_family([_item(b, z, z, paired=False) for b in (1.2, 1.4, 1.6)], n_boot=100)
    assert not t["reject"].any() and (t["sign"] == STAT_NONE).all()
    assert t["p"].isna().all() and (t["magnitude"] == MAG_UNCLEAR).all()
    assert t["zero_bound_a"].tolist() == pytest.approx([zero_upper_bound(100)] * 3)
    assert family_threshold(t)[0] is None


def test_family_zero_comparison_still_counts_in_m():
    """Нулевое сравнение остается в семье: отклонение второго проверяется на
    α/2, а не на α."""
    n = 40
    a = np.zeros((1, n))
    a[0, :4] = 1  # 10 % против 0: p около 0,06 — не проходит и α
    t = decide_family([_item(1.2, np.zeros((1, n)), np.zeros((1, n))), _item(1.4, a, np.zeros((1, n)))], n_boot=2000)
    assert t["sign"].tolist() == [STAT_NONE, STAT_NONE]


def test_family_no_real_differences():
    """Различий нет (одинаковые исходы на общих сценах): никаких знаков, порога нет."""
    rng = np.random.default_rng(5)
    a = (rng.random((2, 300)) < 0.1).astype(float)
    t = decide_family([_item(b, a, a.copy()) for b in (1.2, 1.4, 1.6, 1.8, 2.0)], n_boot=500)
    assert not t["reject"].any()
    b, text = family_threshold(t)
    assert b is None and "не найден" in text


def test_family_threshold_uses_adjusted_signs():
    """Порог считается по решениям Холма: явная разность начиная с β = 1,4."""
    n = 400
    rng = np.random.default_rng(6)
    base = (rng.random((1, n)) < 0.02).astype(float)
    worse = np.maximum(base, (rng.random((1, n)) < 0.3).astype(float))
    items = [_item(b, worse if b >= 1.4 else base.copy(), base) for b in (1.2, 1.4, 1.6, 1.8, 2.0)]
    t = decide_family(items, n_boot=2000)
    assert t["sign"].tolist() == [STAT_NONE, STAT_WORSE, STAT_WORSE, STAT_WORSE, STAT_WORSE]
    assert family_threshold(t)[0] == 1.4


def test_each_row_has_one_label_pair():
    rng = np.random.default_rng(0)
    a = (rng.random((3, 100)) < 0.2).astype(float)
    b = (rng.random((3, 100)) < 0.1).astype(float)
    t = decide_family([_item(1.0, a, b, paired=False), _item(1.2, b, a, paired=False)], n_boot=300)
    assert set(t["sign"]) <= {STAT_WORSE, STAT_BETTER, STAT_NONE}
    assert set(t["magnitude"]) <= {MAG_WITHIN, MAG_OUTSIDE, MAG_UNCLEAR}


def test_margin_sensitivity_changes_only_magnitude():
    r = _family_from_boots([np.linspace(0.015, 0.03, 10_000)], 0.05, 0.02)[0]
    t = margin_sensitivity(__import__("pandas").DataFrame([{"level": 1.2, **r}]))
    row = t.iloc[0]
    assert row["sign"] == STAT_WORSE
    assert row["±1 п.п."] == MAG_OUTSIDE and row["±2 п.п."] == MAG_UNCLEAR and row["±5 п.п."] == MAG_WITHIN


def test_2x2_is_descriptive_only():
    cells = {("с фильтром", "с фильтром"): np.zeros((2, 10)), ("с фильтром", "без"): np.ones((2, 10))}
    t = describe_2x2(cells)
    assert t["частота"].tolist() == [0.0, 1.0]
    assert not {"p", "sign", "reject"} & set(t.columns)


def test_zero_upper_bound_hand_example():
    # 0 из 100: 1 - 0,05^(1/100) ≈ 2,95 %
    assert zero_upper_bound(100) == pytest.approx(0.0295, abs=1e-4)


def test_zero_outcomes_do_not_use_degenerate_bootstrap():
    r = compare(np.zeros((3, 100)), np.zeros((3, 100)))
    assert np.isnan(r["p"]) and r["sign"] == STAT_NONE and not r["reject"]
    assert r["zero_bound_a"] == pytest.approx(zero_upper_bound(100))


def test_crossed_bootstrap_uses_common_scene_indices():
    """Если a и b совпадают по сценам, общие индексы сцен дают разность 0 в
    каждом повторе (при общих политиках); независимые наборы дали бы разброс."""
    rng = np.random.default_rng(1)
    a = (rng.random((3, 200)) < 0.2).astype(float)
    boot = crossed_bootstrap(a, a.copy(), n_boot=200, paired_policies=True)
    assert np.all(boot == 0)


def test_crossed_bootstrap_reflects_policy_variation():
    """Разброс между политиками расширяет интервал: при одной политике
    разброс меньше, чем при трех сильно разных."""
    rng = np.random.default_rng(2)
    n = 300
    same = np.tile((rng.random(n) < 0.1).astype(float), (3, 1))
    varied = np.stack([(rng.random(n) < p).astype(float) for p in (0.02, 0.1, 0.3)])
    b = np.zeros((1, n))
    s1 = crossed_bootstrap(same, b, n_boot=2000).std()
    s2 = crossed_bootstrap(varied, b, n_boot=2000).std()
    assert s2 > 1.5 * s1


def test_holm_hand_example():
    reject, level = holm([0.01, 0.04, 0.03, 0.2])
    # порядок 0.01 (α/4), 0.03 (α/3), 0.04 (α/2), 0.2 (α/1)
    assert level.tolist() == pytest.approx([0.05 / 4, 0.05 / 2, 0.05 / 3, 0.05])
    assert reject.tolist() == [True, False, False, False]


def test_beta_star_rules_including_top_of_grid():
    levels = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    w, n = STAT_WORSE, STAT_NONE
    assert beta_star(levels, [n, n, w, w, w, w])[0] == 1.4
    # «хуже» без подтверждения на следующем уровне не считается
    assert beta_star(levels, [n, w, n, w, w, n])[0] == 1.6
    # только на верхней точке — подтвердить нечем
    b, text = beta_star(levels, [n, n, n, n, n, w])
    assert b is None and "верхней точке" in text
    assert beta_star(levels, [n] * 6)[0] is None
