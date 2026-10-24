"""Этап 7: формулы объема прогонов."""

import numpy as np
import pytest

from rssrl.volume import discordance, episodes_needed, evaluation_plan, holm_z, min_detectable


def test_holm_z_single_comparison_is_196():
    assert holm_z(1) == pytest.approx(1.959964, abs=1e-5)


def test_holm_z_grows_with_number_of_comparisons():
    assert holm_z(10) > holm_z(5) > holm_z(1)
    assert holm_z(10) == pytest.approx(2.807034, abs=1e-5)  # уровень 0,005 на хвост 0,0025


def test_discordance_counts_differing_pairs():
    a = [True, True, False, False]
    b = [True, False, True, False]
    assert discordance(a, b) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        discordance([True], [True, False])


def test_episodes_needed_hand_example():
    # d = 0,1, w = 2 п.п., z = 1,96: n = 1,96² · 0,1 / 0,02² = 960,4 -> 961
    assert episodes_needed(0.1, 0.02) == 961
    assert episodes_needed(0.0, 0.02) == 0


def test_episodes_needed_gives_claimed_ci_width():
    """Проверка формулы симуляцией: при n из формулы реальная половина
    95 % ДИ парной разности (по разбросу выборочных разностей) не шире w."""
    rng = np.random.default_rng(0)
    d, w = 0.2, 0.05
    n = episodes_needed(d, w)
    diffs = []
    for _ in range(4000):
        # исходы пар: несовпадение с вероятностью d, знак — поровну
        u = rng.random(n)
        diff = np.where(u < d / 2, 1, np.where(u < d, -1, 0))
        diffs.append(diff.mean())
    half = 1.96 * np.std(diffs)
    assert half <= w * 1.03


def test_evaluation_plan_counts():
    plan = evaluation_plan(n_scenes=100, n_policies=5)
    # (2·5 + 1)·2 с фильтром + 2·5 без = 32 эпизода на сцену и уровень
    assert plan["эпизодов на сцену и уровень"] == 32
    assert plan["эпизодов всего"] == 32 * 10 * 100


def test_min_detectable_is_the_boundary():
    """При δ = min_detectable нижняя граница ДИ ровно 0, при большем δ — выше 0."""
    n, z = 500, 3.0
    delta = min_detectable(n, z)
    lower = lambda d: d - z * np.sqrt((d - d * d) / n)
    assert lower(delta) == pytest.approx(0.0, abs=1e-12)
    assert lower(delta * 1.1) > 0 > lower(delta * 0.9)
