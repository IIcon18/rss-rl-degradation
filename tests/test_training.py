"""TrainingEnv: причина окончания эпизода и вмешательства фильтра в info
(письмо 14.11, п. 2) — инфраструктура для основной серии, не для уже
прошедшего пилота."""

import pytest

from rssrl.scenarios import classify_outcome
from rssrl.training import TrainingEnv


def _run_episode(env, seed, action=1):
    """IDLE (действие 1) до конца эпизода; номинальный режим, нарушителя
    нет, эпизод либо доезжает до лимита времени, либо агент как-то
    останавливается/съезжает — с фильтром в номинале столкновений не
    бывает (см. tests/test_nominal.py)."""
    env.reset(seed=seed)
    info = {}
    terminated = truncated = False
    n = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
        n += 1
        if n > 200:  # страховка от зависания теста
            raise RuntimeError("эпизод не закончился за 200 шагов политики")
    return info


def test_non_terminal_steps_have_no_outcome_fields():
    env = TrainingEnv(shield=True)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(1)
    assert not (terminated or truncated)  # первый шаг эпизода точно не последний
    assert "outcome" not in info


def test_terminal_info_has_outcome_matching_classify_outcome():
    env = TrainingEnv(shield=True)
    info = _run_episode(env, seed=0)
    assert info["outcome"] in {"collision", "offroad", "stopped", "safe"}
    # согласовано с той же классификацией, что использует run_episode/пилот
    expected, _ = classify_outcome(env.scene.logger.frame())
    assert info["outcome"] == expected
    assert info["collision"] == (expected == "collision")
    assert info["offroad"] == (expected == "offroad")
    env.close()


def test_terminal_info_has_shield_intervention_stats():
    env = TrainingEnv(shield=True)
    info = _run_episode(env, seed=0)
    assert isinstance(info["shield_steps"], int)
    assert info["shield_steps"] >= 0
    assert 0.0 <= info["shield_frac"] <= 1.0
    env.close()


def test_nominal_episode_with_shield_is_safe_and_not_a_collision():
    """В номинальном режиме с фильтром столкновений не бывает (как в
    tests/test_nominal.py) — тут то же самое, но через TrainingEnv.info."""
    env = TrainingEnv(shield=True)
    info = _run_episode(env, seed=1)
    assert not info["collision"]
    env.close()


def test_without_shield_info_fields_are_still_populated():
    env = TrainingEnv(shield=False)
    info = _run_episode(env, seed=0)
    assert info["outcome"] in {"collision", "offroad", "stopped", "safe"}
    assert info["shield_steps"] == 0  # фильтр выключен: вмешательств нет
    assert info["shield_frac"] == 0.0
    env.close()
