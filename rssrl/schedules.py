"""Расписания нарушений: строятся один раз эталонной политикой и хранятся в файле.

Порядок работы (письмо 29.10):
1. До основной оценки build_schedules прогоняет эталон (scenarios.REFERENCE:
   IDM/MOBIL + RSS-фильтр, базовая реакция, задержка ρ) на сценах и
   записывает для каждой сцены момент нарушения и машину-нарушителя.
   Тормозящий лидер: момент — scenarios.violation_time(seed), равномерно на
   5–30 с; нарушитель — машина, за которой эталон едет в этот момент.
2. save_schedules пишет CSV в results/schedules/ вместе с параметрами
   эталона, условия запуска и версией кода (git). Файл после этого не
   меняется.
3. Оценка всех вариантов читает расписания только из файла (load_schedules),
   одинаково для всех вариантов агента и всех β. Сцены, где эталон не дошел
   до условия, остаются в выборке с пометкой «не назначено».

Расписание не зависит от β: до момента нарушения сцена от β не зависит.
Для подрезания расписание свое на каждое γ (условие запуска зависит от γ).
"""

import json
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import lru_cache, partial
from pathlib import Path

import pandas as pd

from rssrl.scenarios import (LC_DURATION, REFERENCE, TRIGGER_RATIO, TRIGGER_T_MIN, VIOLATION_T_RANGE,
                             violation_schedule)

ROOT = Path(__file__).resolve().parent.parent
SCHEDULE_DIR = ROOT / "results" / "schedules"
DEFAULT_FILE = SCHEDULE_DIR  # по умолчанию читаются все CSV этой папки


def code_version():
    """Хеш коммита git; «-dirty», если в rssrl/ есть незакоммиченные изменения."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "rssrl"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _one(task, traffic):
    seed, kind, gamma = task
    sched = violation_schedule(seed, kind=kind, gamma=gamma, traffic=traffic)
    row = {"seed": seed, "kind": kind, "gamma": gamma, "traffic": traffic,
           "t": math.nan, "uid": None, "lane_from": None, "lane_to": None, "lane_id": None}
    if sched is not None:
        lane = sched["lane"]
        row.update(t=sched["t"], uid=sched["uid"], lane_from=lane[0], lane_to=lane[1], lane_id=lane[2])
    return row


def build_schedules(seeds, kind="brake", gammas=(1.0,), traffic="following", processes=None):
    """Прогнать эталон на сценах seeds и вернуть таблицу расписаний.
    kind="brake" — одно расписание на сцену (gamma = 1); kind="cutin" — по
    одному на каждое γ из gammas."""
    gammas = (1.0,) if kind == "brake" else tuple(gammas)
    tasks = [(s, kind, g) for s in seeds for g in gammas]
    with ProcessPoolExecutor(processes) as pool:
        rows = list(pool.map(partial(_one, traffic=traffic), tasks))
    df = pd.DataFrame(rows)
    df["uid"] = df["uid"].astype("Int64")
    df["lane_id"] = df["lane_id"].astype("Int64")
    df["reference"] = json.dumps(REFERENCE, ensure_ascii=False)
    # правило выбора момента: brake — время из seed на VIOLATION_T_RANGE,
    # лидер эталона; cutin — прогноз зазора γ·d_min (TRIGGER_RATIO не используется)
    df["rule"] = "time_from_seed_leader" if kind == "brake" else "cutin_gamma_dmin"
    df["violation_t_range"] = json.dumps(VIOLATION_T_RANGE)
    df["trigger_ratio"] = TRIGGER_RATIO
    df["t_min"] = TRIGGER_T_MIN
    df["lc_duration"] = LC_DURATION
    df["code_version"] = code_version()
    df["created"] = date.today().isoformat()
    return df


def save_schedules(df, path=DEFAULT_FILE, overwrite=False):
    """Записать расписания. Существующий файл не перезаписывается без
    overwrite=True: расписания фиксируются до оценки и не меняются."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} уже есть: расписания не перезаписываем")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


@lru_cache(maxsize=None)
def load_schedules(path=DEFAULT_FILE):
    """Словарь (seed, kind, gamma, traffic) -> расписание {"t", "uid", "lane"}
    или None (эталон не дошел до условия — нарушение не назначено).
    path — CSV или папка: тогда читаются все ее CSV (новые сцены добавляются
    новым файлом, старые файлы не меняются). Одна сцена в двух файлах — ошибка."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    # узлы полосы — строки ("0", "1"); без dtype pandas прочтет их числами, а
    # при пропусках в другом файле — дробями ("0.0"), и полоса не совпадет
    df = pd.concat([pd.read_csv(f, dtype={"lane_from": str, "lane_to": str}) for f in files],
                   ignore_index=True)
    keys = df[["seed", "kind", "gamma", "traffic"]]
    if keys.duplicated().any():
        raise ValueError(f"повтор сцен в расписаниях {path}: {keys[keys.duplicated()].head().to_dict('records')}")
    out = {}
    for r in df.to_dict("records"):
        key = (int(r["seed"]), r["kind"], float(r["gamma"]), r["traffic"])
        if pd.isna(r["t"]):
            out[key] = None
        else:
            lane = (str(r["lane_from"]), str(r["lane_to"]), int(r["lane_id"]))
            out[key] = {"t": float(r["t"]), "uid": int(r["uid"]), "lane": lane}
    return out


def get_schedule(seed, kind, gamma, traffic, path=DEFAULT_FILE):
    """Расписание одной сцены из файла. Если сцены в файле нет — ошибка:
    расписания не достраиваются на ходу во время оценки. Если эталон не
    дошел до условия — расписание с t = inf (нарушение не назначено)."""
    gamma = 1.0 if kind == "brake" else float(gamma)
    key = (int(seed), kind, gamma, traffic)
    table = load_schedules(str(path))
    if key not in table:
        raise KeyError(f"нет расписания для {key} в {path}")
    sched = table[key]
    return dict(sched) if sched is not None else {"t": math.inf, "uid": -1, "lane": None}
