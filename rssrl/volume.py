"""Замер времени эпизодов и расчет объема прогонов.

time_episodes      — время эпизодов на одном ядре (последовательно);
discordance        — доля сцен, где исход на двух уровнях разный (парное сравнение);
episodes_needed    — сколько сцен нужно, чтобы ДИ парной разности был не шире ±w;
min_detectable     — наименьшая разность, которую при n сценах можно назвать «хуже»;
holm_z             — критическое значение с поправкой Холма в худшем случае;
evaluation_plan    — сколько эпизодов в основной и дополнительной части.
"""

import math
import time
from statistics import NormalDist

import pandas as pd

from rssrl.pilot import _policy, _run_one


def time_episodes(jobs, kind, policy="idle"):
    """Прогнать эпизоды последовательно в этом процессе и замерить время.

    Возвращает DataFrame: seed, уровень, shield, delay, wall_s (секунды
    счета), sim_s (секунды симуляции до конца или до столкновения),
    collision.
    """
    rows = []
    for job in jobs:
        t0 = time.perf_counter()
        outcome, _ = _run_one(job, kind, policy)
        rows.append({
            "seed": job[0], "level": job[1], "shield": job[2], "reaction_delay": job[3],
            "wall_s": time.perf_counter() - t0,
            "sim_s": outcome["t_end"],
            "collision": outcome["collision"],
        })
    return pd.DataFrame(rows)


def discordance(a, b):
    """Доля пар (одна и та же сцена), где исходы a и b различаются.
    a, b — булевы массивы одинаковой длины (например, столкновение
    на уровне β и на уровне β = 1)."""
    a, b = pd.Series(a).astype(bool).to_numpy(), pd.Series(b).astype(bool).to_numpy()
    if len(a) != len(b):
        raise ValueError("нужны пары: одинаковые сцены на обоих уровнях")
    return float((a != b).mean())


def holm_z(m, alpha=0.05):
    """Двустороннее критическое значение z для самого строгого шага
    поправки Холма из m сравнений (уровень alpha/m). Для объема берем
    худший случай: интервал должен быть узким и при этом уровне."""
    return NormalDist().inv_cdf(1 - alpha / (2 * m))


def episodes_needed(d, half_width, z=1.96):
    """Сколько парных сцен нужно, чтобы половина ДИ разности частот
    столкновений была не больше half_width.

    Для парных бинарных исходов дисперсия разности на одну сцену
    d − δ², где d — доля несовпадающих пар, δ — сама разность.
    Берем верхнюю оценку d (δ = 0): n = z² · d / w².
    """
    if d <= 0:
        return 0
    return math.ceil(z * z * d / (half_width * half_width))


def min_detectable(n, z=1.96):
    """Наименьшая разность частот δ, при которой весь ДИ разности > 0,
    если все несовпадающие пары в одну сторону (d = δ, типично для
    «нарушитель только добавляет столкновения»): z·sqrt((δ − δ²)/n) < δ,
    откуда δ > z² / (n + z²)."""
    return z * z / (n + z * z)


def evaluation_plan(n_scenes, n_policies, beta_levels=6, gamma_levels=4, delays=2):
    """Сколько эпизодов в оценке после обучения PPO.

    На каждую сцену и каждый уровень (β или γ):
    - 2 политики PPO (обучена с фильтром / без) × n_policies seed обучения,
      каждая проверяется с фильтром и без (перекрестная схема 2×2);
    - IDM/MOBIL + RSS без RL — один вариант, seed обучения нет;
    - каждый вариант с фильтром — при обеих задержках реакции.
    Возвращает словарь с числом эпизодов по частям.
    """
    ppo = 2 * n_policies
    with_shield = (ppo + 1) * delays  # PPO с фильтром + IDM/MOBIL+RSS, обе задержки
    without_shield = ppo
    per_scene_level = with_shield + without_shield
    levels = beta_levels + gamma_levels
    return {
        "эпизодов на сцену и уровень": per_scene_level,
        "уровней": levels,
        "эпизодов всего": per_scene_level * levels * n_scenes,
    }


def _training_env():
    """Среда для замера скорости обучения: highway-v0 с нашим фоном.
    Функция уровня модуля, чтобы ее можно было запустить в дочернем
    процессе (там highway_env тоже нужно импортировать)."""
    import gymnasium as gym
    import highway_env  # noqa: F401  регистрирует highway-v0

    from rssrl.scenarios import BACKGROUND

    env = gym.make("highway-v0")
    env.unwrapped.configure({"other_vehicles_type": BACKGROUND})
    return env


def ppo_steps_per_second(n_envs, steps=2048, seed=0):
    """Скорость обучения PPO (stable-baselines3), шагов среды в секунду,
    при n_envs параллельных средах. Шаг среды — одно решение политики."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [_training_env] * n_envs
    env = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    env.seed(seed)
    model = PPO("MlpPolicy", env, n_steps=max(64, steps // n_envs), batch_size=64,
                seed=seed, verbose=0, device="cpu")
    t0 = time.perf_counter()
    model.learn(steps)
    elapsed = time.perf_counter() - t0
    env.close()
    return steps / elapsed
