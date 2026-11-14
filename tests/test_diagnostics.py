"""Диагностика замороженных политик (rssrl/diagnostics.py): с крошечной,
но настоящей моделью на целевой среде (TrainingEnv), чтобы пространства
наблюдений и действий точно совпадали с тем, что увидит make_scene."""

from pathlib import Path

import pytest

from rssrl.diagnostics import (DIAG_SEED_OFFSET, episode_stats, export_diagnostics, run_diagnostics, summarize)
from rssrl.training import TRAIN_SEED_OFFSET, limit_threads, make_training_env


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    limit_threads()
    env = DummyVecEnv([lambda: make_training_env(shield=True, worker=0)])
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, seed=0, verbose=0, device="cpu")
    model.learn(32)
    path = tmp_path_factory.mktemp("models") / "final_model"
    model.save(str(path))
    env.close()
    return model, str(path) + ".zip"


def test_diag_seed_offset_is_disjoint_from_training_seeds():
    # TrainingEnv.reset: scene_seed = TRAIN_SEED_OFFSET + base*100_000 + worker*10_000 + episode
    # даже с большим запасом по episode (десятки тысяч) не дотянется до DIAG_SEED_OFFSET
    assert DIAG_SEED_OFFSET > TRAIN_SEED_OFFSET + 1 * 100_000 + 11 * 10_000 + 50_000


def test_episode_stats_has_expected_fields_and_ranges(tiny_model):
    model, _ = tiny_model
    row = episode_stats(model, seed=DIAG_SEED_OFFSET, shield_exec=True)
    assert row["outcome"] in {"collision", "offroad", "stopped", "safe"}
    assert 0.0 <= row["shield_frac"] <= 1.0
    assert sum(row[f"action_frac_{n}"] for n in
               ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER")) == pytest.approx(1.0)
    assert row["t_end"] > 0


def test_episode_stats_without_shield_can_still_run(tiny_model):
    model, _ = tiny_model
    row = episode_stats(model, seed=DIAG_SEED_OFFSET, shield_exec=False)
    assert row["outcome"] in {"collision", "offroad", "stopped", "safe"}


def test_run_diagnostics_covers_all_runs_scenes_and_exec_modes(tiny_model):
    _, model_path = tiny_model
    runs = [("test_run", True, 0, model_path)]
    df = run_diagnostics(runs, n_scenes=3, processes=1)
    assert len(df) == 3 * 2  # 3 сцены x 2 режима исполнения
    assert set(df["exec_shield"]) == {True, False}
    assert set(df["scene_seed"]) == set(range(DIAG_SEED_OFFSET, DIAG_SEED_OFFSET + 3))


def test_summarize_has_one_row_per_run_and_exec_mode(tiny_model):
    _, model_path = tiny_model
    runs = [("test_run", True, 0, model_path)]
    df = run_diagnostics(runs, n_scenes=3, processes=1)
    s = summarize(df)
    assert len(s) == 2  # exec_shield True/False
    assert set(s["episodes"]) == {3}


def test_export_diagnostics_writes_primary_data(tmp_path, tiny_model):
    _, model_path = tiny_model
    runs = [("test_run", True, 0, model_path)]
    df = run_diagnostics(runs, n_scenes=2, processes=1)
    export_diagnostics(df, tmp_path)
    assert (tmp_path / "episodes.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "params.json").exists()
