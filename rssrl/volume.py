"""Замер времени эпизодов и план объема прогонов (письмо 29.10, п. 6, 9).

time_episodes        — время эпизодов на одном ядре (последовательно);
discordance          — доля сцен, где исходы двух условий различаются;
episodes_needed      — сколько сцен нужно, чтобы ДИ парной разности был не
                       шире ±w ПРИ ИЗВЕСТНОЙ доле несовпадающих пар d;
budget               — число эпизодов по частям: основной β, дополнительный γ,
                       контрфактические прогоны — отдельно;
projected_halfwidth  — какой будет полуширина ДИ при n сценах и k политиках,
                       оценка повторной выборкой из исходов пробных политик;
ppo_steps_per_second — скорость обучения PPO на целевой среде (с фильтром);
family_a_power       — письмо 14.11, п. 4: симуляцией — вероятность, что
                       ВСЕ правило семьи A (поправка Холма по 6 сравнениям +
                       подтверждение «хуже» на следующем β, rssrl.stats.
                       decide_family + family_threshold) сработает при
                       заданном эффекте. projected_halfwidth считает
                       полуширину ОДНОГО интервала и такого расчета не
                       заменяет (замечание преподавателя, письмо 14.11);
family_a_precision   — альтернатива по точности: медианная полуширина
                       СОВМЕСТНОГО интервала семьи (уровень 1 − alpha/m,
                       как ci_simult в decide_family), а не обычного 95%
                       интервала по умолчанию.

Чего здесь больше нет и почему. Прежняя функция min_detectable считала
границу, при которой приближенный ДИ становится положительным, в
предположении, что все несовпадающие пары в одну сторону. Это не мощность
обнаружения истинного эффекта, и вывод «500 сцен хватит для 2 п.п. после
Холма» из нее не следует — он убран. Несовпадения между β и β = 1 у IDLE
тоже не оценивают разброс будущего сравнения PPO с фильтром и без:
это разные сравнения. Окончательный объем — после 2–3 пробных политик
(projected_halfwidth на их исходах), сейчас только план.
"""

import os

# до импорта numpy: иначе BLAS уже выберет число потоков по числу ядер, и
# параллельные процессы (SubprocVecEnv, несколько одновременных обучений)
# будут драться за потоки (настройка параллельности, письмо 07.11,
# Windows-ПК — AMD Ryzen 5 3600, 6 ядер / 12 потоков).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from rssrl.pilot import _run_one
from rssrl.stats import ALPHA, MARGIN, crossed_bootstrap, decide_family, family_threshold, percentile_ci

FAMILY_A_LEVELS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)  # как в rssrl.stats.COMPARISONS, семья A


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


def ppo_steps_per_second(n_envs, steps=20_000, seed=0, shield=True, seconds=None):
    """Скорость обучения PPO (stable-baselines3), шагов среды в секунду, на
    целевой среде TrainingEnv (ShieldedVehicle с фильтром или без, фон
    контролируемого следования) при n_envs параллельных средах. Шаг среды —
    одно решение политики (1 с симуляции). GPU не используется (device="cpu").

    Если seconds задан — вместо фиксированного числа шагов обучение идет,
    пока не истечет это время (замер пропускной способности на вариант
    фиксированной длительности, а не фиксированного объема)."""
    from functools import partial

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from rssrl.training import limit_threads, make_training_env

    limit_threads()  # начало главного процесса (torch.set_num_threads(1) + OMP/MKL)

    fns = [partial(make_training_env, shield=shield, worker=i) for i in range(n_envs)]
    env = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    env.seed(seed)
    model = PPO("MlpPolicy", env, n_steps=max(64, 2048 // n_envs), batch_size=64,
                seed=seed, verbose=0, device="cpu")

    class _TimeLimit(BaseCallback):
        def __init__(self, deadline):
            super().__init__()
            self.deadline = deadline

        def _on_step(self):
            return time.perf_counter() < self.deadline

    t0 = time.perf_counter()
    if seconds is None:
        model.learn(steps)
    else:
        model.learn(10**9, callback=_TimeLimit(t0 + seconds))
    elapsed = time.perf_counter() - t0
    env.close()
    return model.num_timesteps / elapsed


def benchmark(out_csv, steps=4096, n_envs_list=(1, 8), seconds=None):
    """Бенчмарк пропускной способности PPO на целевой среде: шагов в секунду и
    оценка часов на 10^5 шагов на политику, для политики с базовым фильтром и
    без фильтра. Это замер скорости, а не обучение: модели не сохраняются.

    Запускать отдельным процессом (python -m rssrl.volume <csv>): параллельные
    среды SubprocVecEnv из ядра Jupyter на macOS зависают при запуске."""
    rows = []
    for shield in (True, False):
        for n_envs in n_envs_list:
            sps = ppo_steps_per_second(n_envs, steps=steps, shield=shield, seconds=seconds)
            rows.append({"политика": "с базовым фильтром" if shield else "без фильтра", "сред": n_envs,
                         "шагов в секунду": sps, "часов на 1e5 шагов на политику": 1e5 / sps / 3600})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def _single_run_cli(out_json, shield, n_envs, seconds, seed=0):
    """Один замер шагов/с с заданной длительностью, результат — в JSON.
    Используется как дочерний процесс для измерения одновременных обучений
    (двух процессов PPO.learn в параллель)."""
    sps = ppo_steps_per_second(n_envs, shield=shield, seconds=seconds, seed=seed)
    with open(out_json, "w") as f:
        json.dump({"shield": shield, "n_envs": n_envs, "seconds": seconds, "steps_per_second": sps}, f)


def concurrent_benchmark(n_envs_each=5, seconds=150, shield=True, out_dir="."):
    """Пропускная способность при двух одновременных обучениях (по n_envs_each
    сред каждое) — отдельные процессы python, а не потоки/подпроцессы одного
    SubprocVecEnv, чтобы увидеть реальную конкуренцию за 12 потоков. Возвращает
    суммарные шагов/с по двум процессам."""
    paths = [os.path.join(out_dir, f"_concurrent_{i}.json") for i in range(2)]
    procs = [
        subprocess.Popen([sys.executable, "-m", "rssrl.volume", "--single", paths[i],
                          str(shield), str(n_envs_each), str(seconds), str(i)])
        for i in range(2)
    ]
    for p in procs:
        p.wait()
    results = []
    for path in paths:
        with open(path) as f:
            results.append(json.load(f))
        os.remove(path)
    total_sps = sum(r["steps_per_second"] for r in results)
    return {"политика": "с базовым фильтром" if shield else "без фильтра",
            "схема": f"2×{n_envs_each} сред одновременно", "сред всего": 2 * n_envs_each,
            "шагов в секунду": total_sps, "часов на 1e5 шагов на политику": 1e5 / total_sps / 3600}


# ---------------------------------------------------------------------------
# Объем для семьи A (письмо 14.11, п. 4): мощность всего правила семьи
# (Холм по 6 сравнениям + подтверждение «хуже» на следующем β) симуляцией,
# и отдельно — точность на уровне совместного интервала семьи.
#
# Диагностика (rssrl/diagnostics.py) — номинальный режим, нарушителя нет, и
# там частота столкновений с фильтром около нуля (tests/test_nominal.py). Она
# НЕ может дать оценку разброса между обучениями именно для вероятности
# столкновения при нарушении — такой величины в проекте пока ни у кого нет
# (основной β-эксперимент еще не проводился). Поэтому здесь разброс между
# политиками одного типа — ПРЕДПОЛАГАЕМЫЙ параметр модели (policy_spread,
# бета-биномиальная надстройка), не оценка из данных; задается сеткой
# значений для анализа чувствительности, как и сам эффект. Диагностика дает
# «полезную первичную информацию» (письмо 14.11) в другом смысле: по двум
# seed на тип видно, что политики одного типа уже по-разному ведут себя в
# номинале (скорость, зазор/d_min, доля вмешательств фильтра) — качественное
# основание, что между обучениями вообще есть разброс, но не его величину
# для столкновений при нарушении.
# ---------------------------------------------------------------------------

def _draw_policy_probs(mean, std, n_policies, rng):
    """n_policies вероятностей столкновения политик одного типа: бета-
    распределение со средним mean и стандартным отклонением std (между
    обучениями), std=0 — все политики одного типа имеют одну и ту же
    вероятность (нижняя граница разброса)."""
    mean = min(max(mean, 0.0), 1.0)
    if std <= 0 or mean <= 0 or mean >= 1:
        return np.full(n_policies, mean)
    max_var = mean * (1 - mean)
    var = min(std * std, max_var * 0.98)
    common = mean * (1 - mean) / var - 1
    a, b = max(mean * common, 1e-6), max((1 - mean) * common, 1e-6)
    return rng.beta(a, b, size=n_policies)


def step_effect(beta_onset, delta, baseline=0.03, levels=FAMILY_A_LEVELS):
    """Предполагаемый эффект: PPO_N−R (без фильтра) держит постоянную
    частоту столкновений baseline на всех уровнях сетки; PPO_S+R (с
    фильтром) — baseline при beta < beta_onset и baseline + delta начиная с
    beta_onset (ступенька, не нарастание: при бесконечных данных порог
    процедуры для такого эффекта — ровно beta_onset, если beta_onset не
    верхняя точка сетки, см. rssrl.stats.beta_star). Возвращает функцию
    beta -> (p_s, p_n)."""
    def effect(beta):
        p_n = baseline
        p_s = baseline + (delta if beta >= beta_onset - 1e-9 else 0.0)
        return min(p_s, 1.0), min(p_n, 1.0)
    return effect


def _simulate_family_a_once(effect, n_policies, n_scenes, policy_spread, rng, levels=FAMILY_A_LEVELS,
                            alpha=ALPHA, margin=MARGIN, n_boot=2000):
    """Один повтор: для каждого beta сетки разыгрываем n_policies политик
    (свои вероятности из _draw_policy_probs) x n_scenes сцен для PPO_S+R и
    PPO_N−R, считаем decide_family по всей семье сразу (поправка Холма по
    всем 6 уровням за один вызов, как и положено). Возвращает (β*, таблица)."""
    items = []
    for beta in levels:
        p_s, p_n = effect(beta)
        probs_s = _draw_policy_probs(p_s, policy_spread, n_policies, rng)
        probs_a = (rng.random((n_policies, n_scenes)) < probs_s[:, None]).astype(float)
        probs_n = _draw_policy_probs(p_n, policy_spread, n_policies, rng)
        probs_b = (rng.random((n_policies, n_scenes)) < probs_n[:, None]).astype(float)
        items.append({"level": beta, "a": probs_a, "b": probs_b, "paired_policies": False})
    table = decide_family(items, alpha=alpha, margin=margin, n_boot=n_boot, seed=int(rng.integers(1 << 31)))
    beta_star, _ = family_threshold(table, levels=levels)
    return beta_star, table


def family_a_power(effect, true_beta_star, n_policies_list, n_scenes_list, policy_spread=0.0, n_rep=200,
                   levels=FAMILY_A_LEVELS, alpha=ALPHA, margin=MARGIN, n_boot=2000, seed=0):
    """Мощность всего правила семьи A: доля повторов, где процедура
    (Холм по 6 сравнениям + подтверждение «хуже» на следующем β,
    rssrl.stats.decide_family + family_threshold) находит ИМЕННО
    true_beta_star — не просто «хоть что-то отклонила». true_beta_star=None
    проверяет долю повторов, где правило корректно не находит порога (исход
    «не установлено» допустим и может быть самим желаемым результатом —
    например под нулевым эффектом или если истинное ухудшение только на
    верхней точке сетки, см. rssrl.stats.beta_star).
    Возвращает DataFrame: политик на тип, сцен, разброс политик, мощность."""
    rng_master = np.random.default_rng(seed)
    rows = []
    for n_pol in n_policies_list:
        for n_sc in n_scenes_list:
            hits = 0
            for _ in range(n_rep):
                rng = np.random.default_rng(rng_master.integers(1 << 31))
                beta_star, _ = _simulate_family_a_once(effect, n_pol, n_sc, policy_spread, rng, levels, alpha,
                                                       margin, n_boot)
                hits += beta_star == true_beta_star
            rows.append({"политик на тип": n_pol, "сцен": n_sc, "разброс политик (std)": policy_spread,
                         "истинный β*": true_beta_star, "повторов": n_rep, "мощность": hits / n_rep})
    return pd.DataFrame(rows)


def family_a_precision(n_policies, n_scenes, baseline=0.03, policy_spread=0.0, n_rep=200, levels=FAMILY_A_LEVELS,
                       alpha=ALPHA, n_boot=2000, seed=0):
    """Точность семьи A под нулевым эффектом (PPO_S+R и PPO_N−R с одной и той
    же вероятностью baseline на каждом уровне): медианная полуширина
    СОВМЕСТНОГО интервала семьи (ci_simult в decide_family, уровень
    1 − alpha/m, m = len(levels)) по всем сравнениям и повторам. Это не
    обычный 95% интервал по умолчанию и не замена projected_halfwidth,
    который считает полуширину одного отдельного сравнения."""
    no_effect = step_effect(math.inf, 0.0, baseline, levels)  # delta никогда не включается
    rng_master = np.random.default_rng(seed)
    widths = []
    for _ in range(n_rep):
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        _, table = _simulate_family_a_once(no_effect, n_policies, n_scenes, policy_spread, rng, levels, alpha,
                                           MARGIN, n_boot)
        widths.extend((hi - lo) / 2 for lo, hi in table["ci_simult"] if np.isfinite(lo))
    return float(np.median(widths)) if widths else float("nan")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # дочерний процесс concurrent_benchmark: --single out.json shield n_envs seconds seed
        _single_run_cli(sys.argv[2], sys.argv[3] == "True", int(sys.argv[4]), float(sys.argv[5]),
                        seed=int(sys.argv[6]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--grid":
        # --grid out.csv seconds env1,env2,...
        benchmark(sys.argv[2], seconds=float(sys.argv[3]),
                  n_envs_list=tuple(int(x) for x in sys.argv[4].split(",")))
    else:
        benchmark(sys.argv[1], steps=int(sys.argv[2]) if len(sys.argv) > 2 else 4096)
