"""Объем прогонов и целевая среда обучения (rssrl/volume.py, rssrl/training.py)."""

import numpy as np
import pytest

from rssrl.training import TRAIN_SEED_OFFSET, TrainingEnv
from rssrl.volume import budget, discordance, episodes_needed, projected_halfwidth


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
