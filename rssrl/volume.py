"""Замер времени эпизодов и план объема прогонов (письмо 29.10, п. 6, 9).

time_episodes        — время эпизодов на одном ядре (последовательно);
discordance          — доля сцен, где исходы двух условий различаются;
episodes_needed      — сколько сцен нужно, чтобы ДИ парной разности был не
                       шире ±w ПРИ ИЗВЕСТНОЙ доле несовпадающих пар d;
budget               — число эпизодов по частям: основной β, дополнительный γ,
                       контрфактические прогоны — отдельно;
projected_halfwidth  — какой будет полуширина ДИ при n сценах и k политиках,
                       оценка повторной выборкой из исходов пробных политик;
ppo_steps_per_second — скорость обучения PPO на целевой среде (с фильтром).

Чего здесь больше нет и почему. Прежняя функция min_detectable считала
границу, при которой приближенный ДИ становится положительным, в
предположении, что все несовпадающие пары в одну сторону. Это не мощность
обнаружения истинного эффекта, и вывод «500 сцен хватит для 2 п.п. после
Холма» из нее не следует — он убран. Несовпадения между β и β = 1 у IDLE
тоже не оценивают разброс будущего сравнения PPO с фильтром и без:
это разные сравнения. Окончательный объем — после 2–3 пробных политик
(projected_halfwidth на их исходах), сейчас только план.
"""

import math
import time

import numpy as np
import pandas as pd

from rssrl.pilot import _run_one
from rssrl.stats import crossed_bootstrap, percentile_ci


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
    a, b — булевы массивы одинаковой длины."""
    a, b = pd.Series(a).astype(bool).to_numpy(), pd.Series(b).astype(bool).to_numpy()
    if len(a) != len(b):
        raise ValueError("нужны пары: одинаковые сцены на обоих уровнях")
    return float((a != b).mean())


def episodes_needed(d, half_width, z=1.96):
    """Сколько парных сцен нужно, чтобы половина ДИ разности частот
    столкновений была не больше half_width, если доля несовпадающих пар
    равна d (одна политика, без разброса между политиками).

    Дисперсия парной разности на одну сцену d − δ² ≤ d, отсюда
    n = z² · d / w². d берется из пробных политик для того же сравнения,
    для которого считается объем, а не из другого сравнения.
    """
    if d <= 0:
        return 0
    return math.ceil(z * z * d / (half_width * half_width))


def budget(n_scenes, n_policies, beta_levels=6, gamma_levels=4, delays=1, cf_scenes=100,
           cf_per_episode=5.0, idm=True):
    """Число эпизодов оценки по частям (каждая часть — свой бюджет).

    На каждую сцену и уровень (β или γ) — схема 2×2: политики «обучена с
    фильтром» и «обучена без фильтра» по n_policies seed обучения, каждая
    исполняется с фильтром и без; плюс IDM/MOBIL + RSS без RL (idm=True).
    Варианты с фильтром — при delays задержках (1 — только основная ρ).
    Контрфакт: cf_scenes сцен основного эксперимента × уровни β × варианты с
    фильтром, по cf_per_episode контрфактических прогонов на эпизод (число
    вмешательств — оценка из пилота).
    """
    ppo = 2 * n_policies
    with_shield = (ppo + (1 if idm else 0)) * delays
    per_scene_level = with_shield + ppo
    main = per_scene_level * beta_levels * n_scenes
    extra = per_scene_level * gamma_levels * n_scenes
    cf = with_shield * beta_levels * cf_scenes * cf_per_episode
    return {
        "эпизодов на сцену и уровень": per_scene_level,
        "основной β": main,
        "дополнительный γ": extra,
        "контрфакт": int(round(cf)),
    }


def projected_halfwidth(a, b, n_scenes, n_policies, paired_policies=False, n_rep=200, n_boot=400, seed=0,
                        alpha=0.05):
    """Оценка полуширины ДИ разности при n_scenes сценах и n_policies
    политиках по исходам пробных политик.

    a, b — матрицы исходов пробных политик [политики × сцены] на общих сценах.
    Из них n_rep раз собираем выборку нужного размера (сцены и политики с
    возвращением; столбцы — общие) и считаем на ней перекрестный бутстреп
    ДИ. Возвращает медиану полуширины. Чем больше разброс между пробными
    политиками, тем меньше помогает увеличение числа сцен.
    """
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    rng = np.random.default_rng(seed)
    widths = []
    for r in range(n_rep):
        sc = rng.integers(a.shape[1], size=n_scenes)
        pa = rng.integers(a.shape[0], size=n_policies)
        pb = pa if paired_policies else rng.integers(b.shape[0], size=n_policies if b.shape[0] > 1 else 1)
        boot = crossed_bootstrap(a[np.ix_(pa, sc)], b[np.ix_(pb, sc)], n_boot=n_boot, seed=seed + r,
                                 paired_policies=paired_policies)
        lo, hi = percentile_ci(boot, alpha)
        widths.append((hi - lo) / 2)
    return float(np.median(widths))


def ppo_steps_per_second(n_envs, steps=20_000, seed=0, shield=True):
    """Скорость обучения PPO (stable-baselines3), шагов среды в секунду, на
    целевой среде TrainingEnv (ShieldedVehicle с фильтром или без, фон
    контролируемого следования) при n_envs параллельных средах. Шаг среды —
    одно решение политики (1 с симуляции)."""
    from functools import partial

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from rssrl.training import make_training_env

    fns = [partial(make_training_env, shield=shield, worker=i) for i in range(n_envs)]
    env = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    env.seed(seed)
    model = PPO("MlpPolicy", env, n_steps=max(64, 2048 // n_envs), batch_size=64,
                seed=seed, verbose=0, device="cpu")
    t0 = time.perf_counter()
    model.learn(steps)
    elapsed = time.perf_counter() - t0
    env.close()
    return model.num_timesteps / elapsed


def benchmark(out_csv, steps=4096, n_envs_list=(1, 8)):
    """Бенчмарк пропускной способности PPO на целевой среде: шагов в секунду и
    оценка часов на 10^5 шагов на политику, для политики с базовым фильтром и
    без фильтра. Это замер скорости, а не обучение: модели не сохраняются.

    Запускать отдельным процессом (python -m rssrl.volume <csv>): параллельные
    среды SubprocVecEnv из ядра Jupyter на macOS зависают при запуске."""
    rows = []
    for shield in (True, False):
        for n_envs in n_envs_list:
            sps = ppo_steps_per_second(n_envs, steps=steps, shield=shield)
            rows.append({"политика": "с базовым фильтром" if shield else "без фильтра", "сред": n_envs,
                         "шагов в секунду": sps, "часов на 1e5 шагов на политику": 1e5 / sps / 3600})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    import sys

    benchmark(sys.argv[1], steps=int(sys.argv[2]) if len(sys.argv) > 2 else 4096)
