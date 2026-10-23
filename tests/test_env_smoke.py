"""Этап 0: проверка, что окружение собрано и highway-v0 запускается.

Тесты не проверяют физику, только то, что:
- среда создается и делает шаги;
- при одинаковом seed сцена воспроизводится один в один (основа для
  парных сравнений на общих сценах);
- при разных seed сцены различаются.
"""

import gymnasium as gym
import highway_env  # noqa: F401  регистрирует highway-v0 в gymnasium
import numpy as np
import pytest

IDLE = 1  # DiscreteMetaAction: 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT, 3 FASTER, 4 SLOWER


def run_episode(seed, n_steps=10):
    """Проехать n_steps шагов действием IDLE, вернуть наблюдения и положения машин."""
    env = gym.make("highway-v0")
    obs, _ = env.reset(seed=seed)
    observations = [obs]
    for _ in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(IDLE)
        observations.append(obs)
        if terminated or truncated:
            break
    positions = np.array([v.position for v in env.unwrapped.road.vehicles])
    env.close()
    return np.array(observations), positions


def test_env_steps():
    env = gym.make("highway-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs, reward, terminated, truncated, info = env.step(IDLE)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    env.close()


def test_same_seed_same_scene():
    obs_a, pos_a = run_episode(seed=42)
    obs_b, pos_b = run_episode(seed=42)
    assert obs_a.shape == obs_b.shape
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(pos_a, pos_b)


def test_different_seed_different_scene():
    _, pos_a = run_episode(seed=1, n_steps=0)
    _, pos_b = run_episode(seed=2, n_steps=0)
    assert pos_a.shape != pos_b.shape or not np.allclose(pos_a, pos_b)


def test_defaults_we_rely_on():
    """Значения highway-env по умолчанию, на которые опираются следующие этапы.

    Если после обновления библиотеки тест упадет, значит изменилась
    физика среды и выводы этапов 2-3 нужно перепроверить.
    """
    from highway_env.vehicle.behavior import IDMVehicle
    from highway_env.vehicle.kinematics import Vehicle

    config = gym.make("highway-v0").unwrapped.config
    assert config["simulation_frequency"] == 15  # шаг симуляции 1/15 с
    assert config["policy_frequency"] == 1  # агент решает раз в секунду
    assert config["duration"] == 40  # лимит эпизода, с
    assert config["lanes_count"] == 4
    assert config["other_vehicles_type"] == "highway_env.vehicle.behavior.IDMVehicle"

    # Фоновые машины: IDM хочет тормозить до 5, но обрезка разрешает до 6 м/с2.
    # Это нарушает допущение RSS a_max,brake = 5, исправим на этапе 2.
    assert IDMVehicle.COMFORT_ACC_MIN == -5.0
    assert IDMVehicle.ACC_MAX == 6.0
    assert Vehicle.MIN_SPEED < 0  # симулятор не запрещает ехать задом


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
