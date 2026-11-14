"""Диагностика замороженных финальных политик (письмо 14.11, п. 3).

Отдельный общий набор номинальных сцен (нарушителя нет, фон
контролируемого следования — то же, на чем обучались и оценивались
политики), не пересекающийся ни со сценами обучения (TRAIN_SEED_OFFSET,
rssrl/training.py), ни со сценами оценки во время обучения
(EVAL_SEED_OFFSET + seed, там база уже сама в районе 10^9 после формулы
TrainingEnv.reset), ни с seed основного эксперимента (небольшие целые).
DIAG_SEED_OFFSET = 2_000_000 с запасом выше обучения (~1,21*10^6) и
отдельно от обоих.

4 финальные модели (rssrl/manifest.py) исполняются в двух режимах —
с фильтром и без — независимо от того, как они обучались: перекрестное
исполнение отделяет поведение самой политики от непосредственной работы
защиты. Это ОПИСАТЕЛЬНАЯ диагностика (как describe_2x2 в rssrl/stats.py):
выводов о механизме привыкания не делает. Частота столкновений в
номинальном режиме с фильтром ожидаемо около нуля (tests/test_nominal.py)
— диагностика НЕ оценивает вероятность столкновения при нарушении и не
заменяет основной эксперимент; непрерывные показатели (скорость, зазор/d_min,
доля времени вмешательства фильтра, частоты действий) — вот что здесь
действительно различает политики.
"""

import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from rssrl.scenarios import make_scene, run_episode

DIAG_SEED_OFFSET = 2_000_000
N_DIAG_SCENES = 100
ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

_WORKER_MODEL = None


def _init_worker(model_path):
    """Загрузить модель один раз на процесс пула (а не на каждый эпизод)."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    global _WORKER_MODEL
    from stable_baselines3 import PPO

    _WORKER_MODEL = PPO.load(str(model_path), device="cpu")


def _policy_with_action_log(model, counts):
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
        counts[a] = counts.get(a, 0) + 1
        return a

    return policy


def episode_stats(model, seed, shield_exec, traffic="following"):
    """Один диагностический эпизод: сцена seed, номинальный режим,
    исполнение с фильтром (shield_exec=True) или без. Возвращает словарь
    показателей по письму 14.11, п. 3."""
    counts = {}
    scene = make_scene(seed, shield=shield_exec, kind="nominal", traffic=traffic)
    outcome, log = run_episode(scene, policy=_policy_with_action_log(model, counts))
    finite = log["gap"] < 1e6  # есть критичная машина впереди (см. rssrl/shield.py)
    ratio = (log["gap"] / log["d_min"])[finite]
    n_actions = sum(counts.values()) or 1
    row = {
        "scene_seed": seed,
        "outcome": outcome["outcome"],
        "collision": outcome["collision"],
        "offroad": outcome["outcome"] == "offroad",
        "t_end": outcome["t_end"],
        "distance_km": outcome["distance_km"],
        "mean_speed": float(log["ego_v"].mean()),
        "min_speed": float(log["ego_v"].min()),
        "gap_ratio_median": float(ratio.median()) if len(ratio) else math.nan,
        "gap_ratio_p10": float(ratio.quantile(0.1)) if len(ratio) else math.nan,
        "gap_ratio_frac_below_1": float((ratio < 1).mean()) if len(ratio) else math.nan,
        "shield_frac": float(log["shield_active"].mean()),
        "shield_steps": int(log["shield_active"].sum()),
    }
    for a, name in ACTION_NAMES.items():
        row[f"action_frac_{name}"] = counts.get(a, 0) / n_actions
    return row


def _run_one(args):
    seed, shield_exec = args
    return episode_stats(_WORKER_MODEL, seed, shield_exec)


def run_diagnostics(runs, n_scenes=N_DIAG_SCENES, seed_offset=DIAG_SEED_OFFSET, processes=None):
    """runs — [(имя, trained_shield, trained_seed, путь к модели), ...]
    (см. results/pilot_training/models_manifest.csv). Для каждой модели —
    отдельный пул процессов (модель грузится по разу на воркер), сцены и
    оба режима исполнения — внутри пула. Общие сцены для всех прогонов и
    режимов исполнения (seed_offset + 0..n_scenes-1): сравнение на общих
    сценах, как везде в проекте.
    Возвращает DataFrame: по строке на (прогон, режим исполнения, сцена)."""
    seeds = [seed_offset + i for i in range(n_scenes)]
    tasks = [(s, exec_shield) for exec_shield in (True, False) for s in seeds]
    frames = []
    for name, trained_shield, trained_seed, model_path in runs:
        with ProcessPoolExecutor(processes, initializer=_init_worker, initargs=(model_path,)) as pool:
            rows = list(pool.map(_run_one, tasks))
        df = pd.DataFrame(rows)
        df["run"] = name
        df["trained_shield"] = trained_shield
        df["trained_seed"] = trained_seed
        df["exec_shield"] = [t[1] for t in tasks]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(df):
    """Частоты исходов и средние непрерывные показатели по (прогон, режим
    исполнения) — для таблиц ноутбука 09. Описание, не статистическое
    решение (ни p, ни знаков здесь нет, как и в describe_2x2)."""
    g = df.groupby(["run", "trained_shield", "trained_seed", "exec_shield"], as_index=False)
    return g.agg(
        episodes=("scene_seed", "count"),
        collision_rate=("collision", "mean"),
        offroad_rate=("offroad", "mean"),
        mean_speed=("mean_speed", "mean"),
        min_speed=("min_speed", "min"),
        shield_frac=("shield_frac", "mean"),
        gap_ratio_median=("gap_ratio_median", "median"),
        gap_ratio_frac_below_1=("gap_ratio_frac_below_1", "mean"),
    )


def export_diagnostics(df, folder):
    """Первичные данные диагностики: episodes.csv (все эпизоды), summary.csv."""
    import json

    from rssrl.schedules import code_version

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "episodes.csv", index=False)
    summarize(df).to_csv(folder / "summary.csv", index=False)
    meta = {
        "n_scenes": int(df["scene_seed"].nunique()),
        "seed_offset": DIAG_SEED_OFFSET,
        "runs": sorted(df["run"].unique().tolist()),
        "code_version": code_version(),
    }
    (folder / "params.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    import pandas as pd

    from rssrl.train_pilot import MODELS_DIR, SEEDS, SHIELD_VARIANTS, run_name

    runs = [
        (run_name(sh, s), sh, s, str(MODELS_DIR / run_name(sh, s) / "final_model.zip"))
        for sh in SHIELD_VARIANTS for s in SEEDS
    ]
    data = run_diagnostics(runs)
    export_diagnostics(data, Path("results/pilot_diag"))
    print(summarize(data).round(4).to_string(index=False))
