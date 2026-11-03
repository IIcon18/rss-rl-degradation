"""Шаг 6 (письмо 29.10): правило решения, перекрестный бутстреп, нулевые исходы."""

import numpy as np
import pytest

from rssrl.stats import (MAG_OUTSIDE, MAG_UNCLEAR, MAG_WITHIN, STAT_BETTER, STAT_NONE, STAT_WORSE, beta_star,
                         compare, crossed_bootstrap, decide, holm, zero_upper_bound)


def test_decide_two_separate_characteristics():
    # пример из письма: [0,5; 1,5] п.п. — хуже, но в пределах допуска ±2 п.п.
    assert decide((0.005, 0.015)) == (STAT_WORSE, MAG_WITHIN)
    assert decide((0.03, 0.06)) == (STAT_WORSE, MAG_OUTSIDE)
    assert decide((0.01, 0.03)) == (STAT_WORSE, MAG_UNCLEAR)
    assert decide((-0.01, 0.01)) == (STAT_NONE, MAG_WITHIN)
    assert decide((-0.05, 0.05)) == (STAT_NONE, MAG_UNCLEAR)
    assert decide((-0.06, -0.03)) == (STAT_BETTER, MAG_OUTSIDE)


def test_each_interval_gets_exactly_one_label_pair():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        lo, hi = sorted(rng.uniform(-0.1, 0.1, 2))
        sign, mag = decide((lo, hi))
        assert sign in {STAT_WORSE, STAT_BETTER, STAT_NONE}
        assert mag in {MAG_WITHIN, MAG_OUTSIDE, MAG_UNCLEAR}


def test_zero_upper_bound_hand_example():
    # 0 из 100: 1 - 0,05^(1/100) ≈ 2,95 %
    assert zero_upper_bound(100) == pytest.approx(0.0295, abs=1e-4)


def test_zero_outcomes_do_not_use_degenerate_bootstrap():
    r = compare(np.zeros((3, 100)), np.zeros((3, 100)))
    assert np.isnan(r["p"]) and r["sign"] == STAT_NONE
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
