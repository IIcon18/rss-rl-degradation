"""Пробное обучение PPO на целевой среде (письмо 07.11, п. 9, шаг 5).

2 типа политик («с базовым фильтром» / «без фильтра») × 2 seed обучения —
4 прогона. Конфигурация, seed и правило выбора модели зафиксированы здесь
заранее и не меняются по ходу обучения.

Фиксированная конфигурация:
- N_ENVS = 12 параллельных сред (SubprocVecEnv) — по замеру пропускной
  способности на этой машине (results/bench_parallel), лучший вариант из
  проверенных (4/6/8/10/12 сред, 2×5 одновременно).
- TOTAL_TIMESTEPS = 100_000 шагов среды на политику — по тому же замеру это
  4 политики примерно за 9.3 часа суммарно на этой машине (AMD Ryzen 5 3600,
  6 ядер / 12 потоков, без GPU, device="cpu").
- SEEDS = (0, 1) — seed обучения; каждый seed задает и начальное состояние
  политики PPO, и поток сцен TrainingEnv (TRAIN_SEED_OFFSET + seed·100000 +
  worker·10000 + номер эпизода). Не пересекается с seed сцен оценки.
- Оценка во время обучения — на отдельном потоке сцен (EVAL_SEED_OFFSET,
  непересекающемся с seed обучения), 4 среды, по N_EVAL_EPISODES эпизодов,
  раз в EVAL_EVERY_STEPS шагов среды.
- Правило выбора модели: лучшая по среднему эпизодному вознаграждению на
  этих сценах оценки за все время обучения (EvalCallback,
  best_model_save_path) — не последняя по времени модель.

Прогоны идут по очереди (не параллельно друг другу): по замеру это дает
то же суммарное время, что и параллельный запуск нескольких обучений
одновременно, но проще — не нужно разбирать перепутанные логи двух
процессов.

Запуск (отдельным процессом, Windows-ПК):
    python -m rssrl.train_pilot

Кривые обучения и конфиг каждого прогона — в results/pilot_training/
(коммитим). Сами модели — в models/pilot_training/ (вне git, см.
.gitignore).
"""

import json
import os
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

N_ENVS = 12
TOTAL_TIMESTEPS = 100_000
SEEDS = (0, 1)
SHIELD_VARIANTS = (True, False)
EVAL_SEED_OFFSET = 10_000  # не пересекается с seed обучения (0, 1, ...)
EVAL_N_ENVS = 4
N_EVAL_EPISODES = 10
EVAL_EVERY_STEPS = 10_000  # в шагах среды (не в вызовах callback)
N_STEPS = 2048 // N_ENVS  # 170: размер буфера на среду за один проход PPO
BATCH_SIZE = 120  # N_STEPS * N_ENVS = 2040, делится без остатка (2040 / 120 = 17)

RESULTS_DIR = Path("results/pilot_training")
MODELS_DIR = Path("models/pilot_training")


def run_name(shield, seed):
    return f"shield{int(shield)}_seed{seed}"


def run_one(shield, seed, total_timesteps=TOTAL_TIMESTEPS, n_envs=N_ENVS, verbose=1):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from rssrl.training import limit_threads, make_training_env

    limit_threads()  # начало главного процесса

    name = run_name(shield, seed)
    out_dir = RESULTS_DIR / name
    model_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    def make_train(i):
        def _init():
            env = make_training_env(shield=shield, worker=i)
            return Monitor(env, filename=str(out_dir / f"train_worker{i}"))
        return _init

    def make_eval(i):
        def _init():
            env = make_training_env(shield=shield, worker=i)
            return Monitor(env, filename=str(out_dir / f"eval_worker{i}"))
        return _init

    train_env = SubprocVecEnv([make_train(i) for i in range(n_envs)])
    train_env.seed(seed)

    eval_env = SubprocVecEnv([make_eval(i) for i in range(EVAL_N_ENVS)])
    eval_env.seed(EVAL_SEED_OFFSET + seed)

    config = {
        "shield": shield, "seed": seed, "n_envs": n_envs,
        "total_timesteps": total_timesteps, "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "eval_seed": EVAL_SEED_OFFSET + seed, "eval_n_envs": EVAL_N_ENVS,
        "n_eval_episodes": N_EVAL_EPISODES, "eval_every_steps": EVAL_EVERY_STEPS,
        "device": "cpu", "model_selection": "best mean reward on eval scenes during training",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False))

    model = PPO("MlpPolicy", train_env, n_steps=N_STEPS, batch_size=BATCH_SIZE,
                seed=seed, verbose=verbose, device="cpu")
    eval_cb = EvalCallback(eval_env, n_eval_episodes=N_EVAL_EPISODES,
                           eval_freq=max(1, EVAL_EVERY_STEPS // n_envs),
                           log_path=str(out_dir), best_model_save_path=str(model_dir),
                           deterministic=True, verbose=verbose)

    t0 = time.perf_counter()
    model.learn(total_timesteps, callback=eval_cb)
    elapsed = time.perf_counter() - t0

    model.save(str(model_dir / "final_model"))
    train_env.close()
    eval_env.close()

    summary = {"name": name, "shield": shield, "seed": seed, "elapsed_s": elapsed,
               "steps_per_second": model.num_timesteps / elapsed}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def run_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    t0 = time.perf_counter()
    for shield in SHIELD_VARIANTS:
        for seed in SEEDS:
            print(f"=== {run_name(shield, seed)} ===", flush=True)
            summaries.append(run_one(shield, seed))
            print(summaries[-1], flush=True)
    total_elapsed = time.perf_counter() - t0
    (RESULTS_DIR / "all_summary.json").write_text(
        json.dumps({"runs": summaries, "total_elapsed_s": total_elapsed}, indent=1, ensure_ascii=False))
    print(f"Готово: {total_elapsed / 3600:.2f} ч всего", flush=True)
    return summaries


if __name__ == "__main__":
    run_all()
