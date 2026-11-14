"""Объем прогонов и целевая среда обучения (rssrl/volume.py, rssrl/training.py)."""

import numpy as np
import pytest

from rssrl.training import TRAIN_SEED_OFFSET, TrainingEnv
from rssrl.volume import (budget, discordance, episodes_needed, family_a_power, family_a_precision,
                          projected_halfwidth, step_effect)


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
        u = rng.random(n)
        diff = np.where(u < d / 2, 1, np.where(u < d, -1, 0))
        diffs.append(diff.mean())
    half = 1.96 * np.std(diffs)
    assert half <= w * 1.03


def test_budget_parts_are_separate():
    b = budget(n_scenes=100, n_policies=3, delays=1, cf_scenes=50, cf_per_episode=2)
    # 2·3 PPO с фильтром + IDM/MOBIL+RSS = 7 с фильтром, 6 без: 13 на сцену и уровень
    assert b["эпизодов на сцену и уровень"] == 13
    assert b["основной β"] == 13 * 6 * 100
    assert b["дополнительный γ"] == 13 * 4 * 100
    assert b["контрфакт"] == 7 * 6 * 50 * 2


def test_projected_halfwidth_shrinks_with_scenes_and_grows_with_policy_spread():
    rng = np.random.default_rng(0)
    n = 400
    same = np.tile((rng.random(n) < 0.1).astype(float), (3, 1))
    varied = np.stack([(rng.random(n) < p).astype(float) for p in (0.02, 0.1, 0.3)])
    base = np.zeros((1, n))
    w_small = projected_halfwidth(same, base, 100, 3, n_rep=30, n_boot=200)
    w_big = projected_halfwidth(same, base, 1000, 3, n_rep=30, n_boot=200)
    assert w_big < w_small
    w_varied = projected_halfwidth(varied, base, 1000, 3, n_rep=30, n_boot=200)
    assert w_varied > 1.5 * w_big


def test_training_env_is_target_system_and_seeds_do_not_overlap_evaluation():
    env = TrainingEnv(shield=True)
    obs, _ = env.reset(seed=0)
    assert env.scene.ego.shield_enabled and not env.scene.ego.emergency_brake
    assert env.scene.traffic == "following" and env.scene.kind == "nominal"
    assert env.scene.seed >= TRAIN_SEED_OFFSET
    seeds = {env.scene.seed}
    for _ in range(3):
        obs, r, term, trunc, _ = env.step(1)
        assert env.observation_space.contains(obs.astype(env.observation_space.dtype))
    env.reset()
    seeds.add(env.scene.seed)
    assert len(seeds) == 2
    env.close()
    env = TrainingEnv(shield=False)
    env.reset(seed=0)
    assert not env.scene.ego.shield_enabled
    env.close()


def test_step_effect_is_flat_before_onset_and_shifted_after():
    effect = step_effect(1.4, 0.05, baseline=0.03)
    assert effect(1.0) == (0.03, 0.03)
    assert effect(1.2) == (0.03, 0.03)
    assert effect(1.4) == pytest.approx((0.08, 0.03))
    assert effect(2.0) == pytest.approx((0.08, 0.03))


def test_family_a_power_is_high_for_large_effect_and_many_scenes():
    """Явно отклоняемый эффект (большая дельта, много сцен, политики
    одинаковы) — процедура почти всегда находит именно этот порог."""
    effect = step_effect(1.4, 0.1, baseline=0.02)
    df = family_a_power(effect, 1.4, [3], [800], policy_spread=0.0, n_rep=25, n_boot=400, seed=1)
    assert df["мощность"].iloc[0] > 0.8


def test_family_a_power_is_near_zero_for_tiny_effect_and_few_scenes():
    effect = step_effect(1.4, 0.005, baseline=0.02)
    df = family_a_power(effect, 1.4, [2], [50], policy_spread=0.0, n_rep=25, n_boot=400, seed=1)
    assert df["мощность"].iloc[0] < 0.3


def test_family_a_power_null_effect_mostly_finds_no_threshold():
    """Под нулевым эффектом (эффект включается только выше сетки) правило
    почти всегда корректно возвращает «не установлено» — это допустимый и
    здесь даже ожидаемый исход, а не брак симуляции."""
    null = step_effect(10.0, 0.0, baseline=0.02)
    df = family_a_power(null, None, [3], [300], policy_spread=0.0, n_rep=25, n_boot=400, seed=2)
    assert df["мощность"].iloc[0] > 0.8


def test_family_a_power_decreases_with_policy_spread():
    effect = step_effect(1.4, 0.06, baseline=0.02)
    low = family_a_power(effect, 1.4, [3], [500], policy_spread=0.0, n_rep=25, n_boot=400, seed=3)
    high = family_a_power(effect, 1.4, [3], [500], policy_spread=0.03, n_rep=25, n_boot=400, seed=3)
    assert high["мощность"].iloc[0] <= low["мощность"].iloc[0]


def test_family_a_precision_shrinks_with_more_scenes():
    small = family_a_precision(3, 100, n_rep=25, n_boot=400, seed=0)
    big = family_a_precision(3, 800, n_rep=25, n_boot=400, seed=0)
    assert big < small
