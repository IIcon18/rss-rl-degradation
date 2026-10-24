"""Прогон сетки эпизодов параллельно и график временных рядов эпизода."""

import math
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rssrl.scenarios import make_scene, run_episode

RANDOM_POLICY_SEED_OFFSET = 1000  # seed случайной политики = seed сцены + 1000


def _make(job, kind, **config):
    seed, level, shield, delay = job
    if kind == "brake":
        return make_scene(seed, beta=level, shield=shield, reaction_delay=delay, **config)
    if kind == "cutin":  # уровень — γ, β = 1
        return make_scene(seed, shield=shield, reaction_delay=delay, kind="cutin", gamma=level, **config)
    # nominal: нарушителя нет, уровень не используется (передавайте [1.0])
    return make_scene(seed, shield=shield, reaction_delay=delay, kind="nominal", **config)


def _policy(seed, policy):
    if policy == "random":
        # случайные действия; свой генератор, не связанный с генератором сцены.
        # Действия не зависят от наблюдения, поэтому при повторном прогоне
        # той же сцены последовательность действий та же.
        rng = np.random.default_rng(seed + RANDOM_POLICY_SEED_OFFSET)
        return lambda obs: int(rng.integers(5))
    return None  # IDLE


def _run_one(job, kind, policy, shield_off_at=None, t_stop=None, **config):
    scene = _make(job, kind, **config)
    if shield_off_at is not None:
        scene.ego.shield_off_at = shield_off_at
    outcome, log = run_episode(scene, policy=_policy(job[0], policy), t_stop=t_stop)
    outcome["policy"] = policy
    outcome["lc_requested"] = int(log["lc_requested"].sum())
    outcome["lc_blocked"] = int((log["lc_decision"] == "blocked").sum())
    outcome["lc_cancelled"] = int((log["lc_decision"] == "cancelled").sum())
    return outcome, log


def run_grid(seeds, levels, variants, kind="brake", policy="idle", processes=None):
    """Прогнать все сочетания seed × уровень × вариант.

    levels — значения β (kind="brake") или γ (kind="cutin", β = 1).
    variants — список пар (shield, reaction_delay).
    policy — "idle" (постоянная скорость, пилот без PPO) или "random"
    (случайные действия, проверка запрета перестроений).
    Возвращает (таблица исходов, словарь логов по ключу (seed, уровень, shield, delay)).
    Результат не зависит от числа процессов: каждый эпизод определяется
    только своим seed сцены (и seed политики).
    """
    jobs = [(s, b, sh, d) for s, b, (sh, d) in product(seeds, levels, variants)]
    n = len(jobs)
    with ProcessPoolExecutor(processes) as pool:
        results = list(pool.map(_run_one, jobs, [kind] * n, [policy] * n))
    outcomes = pd.DataFrame([o for o, _ in results])
    logs = {job: log for job, (_, log) in zip(jobs, results)}
    return outcomes, logs


# Контрфактическая проверка вмешательств фильтра
MERGE_GAP = 1.0  # с: вмешательства с перерывом меньше этого считаем одним
CF_HORIZON = 5.0  # с: сколько еще ехать без фильтра после конца вмешательства
DURATION = 40  # с: длительность эпизода highway-v0 по умолчанию


def interventions(log, merge_gap=MERGE_GAP, dt=1 / 15):
    """Вмешательства фильтра в эпизоде: торможение (shield_active) и
    запрет/отмена перестроения. Шаги вмешательства, разделенные перерывом
    меньше merge_gap, объединяются в одно вмешательство (иначе «дребезг»
    +3/−4 м/с² на границе d_min дал бы сотни вмешательств).

    Время t_start — момент решения фильтра (начало шага, на котором он
    вмешался), t_end — конец последнего шага вмешательства.
    Возвращает DataFrame: t_start, t_end, brake_steps, lc_steps.
    """
    if not len(log):
        return pd.DataFrame(columns=["t_start", "t_end", "brake_steps", "lc_steps"])
    brake = log["shield_active"].astype(bool)
    lc = log["lc_decision"].isin(["blocked", "cancelled"])
    rows = log[brake | lc]
    out = []
    for t, b, l in zip(rows["t"], brake[rows.index], lc[rows.index]):
        start = t - dt  # строка лога пишется после шага
        if out and start - out[-1]["t_end"] < merge_gap - 1e-9:
            cur = out[-1]
        else:
            cur = {"t_start": start, "t_end": t, "brake_steps": 0, "lc_steps": 0}
            out.append(cur)
        cur["t_end"] = t
        cur["brake_steps"] += int(b)
        cur["lc_steps"] += int(l)
    return pd.DataFrame(out, columns=["t_start", "t_end", "brake_steps", "lc_steps"])


def counterfactual(job, kind, policy, t_start, t_end, horizon=CF_HORIZON):
    """Тот же эпизод (та же сцена, та же политика), но с момента t_start
    фильтр не вмешивается. Едем до t_end + horizon и смотрим, будет ли
    столкновение. До t_start прогон совпадает с исходным: сцена
    детерминирована своим seed. Если окно выходит за конец эпизода (40 с),
    лимит времени среды продлевается до конца окна, иначе вмешательства в
    конце эпизода выглядели бы «лишними» только из-за обрезанного окна
    (расстановка машин от длительности эпизода не зависит).
    Возвращает словарь: collision (в пределах окна), t_crash, crash_partners, min_gap.
    """
    t_window = t_end + horizon
    duration = max(DURATION, math.ceil(t_window + 1e-9))
    outcome, log = _run_one(job, kind, policy, shield_off_at=t_start, t_stop=t_window, duration=duration)
    after = log[log["t"] > t_start]
    hit = outcome["collision"] and outcome["t_crash"] <= t_window + 1e-9
    return {
        "collision": bool(hit),
        "t_crash": outcome["t_crash"] if hit else None,
        "crash_partners": outcome["crash_partners"] if hit else [],
        "min_gap": float(after["gap"].min()) if len(after) else float("nan"),
    }


def _cf_one(task):
    job, kind, policy, iv = task
    return {**iv, **counterfactual(job, kind, policy, iv["t_start"], iv["t_end"])}


def counterfactual_grid(logs, kind="brake", policy="idle", processes=None):
    """Для каждого вмешательства фильтра во всех эпизодах с фильтром —
    контрфактический прогон без вмешательства (параллельно).

    logs — словарь логов из run_grid (ключ (seed, уровень, shield, delay)).
    Вмешательство «нужное», если без него в окне было бы столкновение,
    иначе «лишнее». Возвращает DataFrame: одна строка — одно вмешательство.
    """
    tasks = []
    for job, log in logs.items():
        if not job[2]:  # без фильтра вмешательств нет
            continue
        for iv in interventions(log).to_dict("records"):
            iv.update(seed=job[0], level=job[1], shield=job[2], reaction_delay=job[3])
            tasks.append((job, kind, policy, iv))
    with ProcessPoolExecutor(processes) as pool:
        rows = list(pool.map(_cf_one, tasks))
    df = pd.DataFrame(rows)
    if len(df):
        df["needed"] = df["collision"]
        df["duration"] = df["t_end"] - df["t_start"]
    return df


def unsafe_entries(log, settle=3.0, dt=1 / 15):
    """Моменты, когда впереди агента появляется новая опасная машина
    (продольно ближе d_min и поперечно ближе поперечного d_min), хотя сам
    агент последние settle секунд не менял полосу.

    Обычно это фоновая машина, которая быстро сдвигается к агенту: встает
    в его полосу или перестраивается в соседнюю (поперечная скорость фона
    в highway-env до ~5 м/с). Фон перестраивается по MOBIL, а не по
    поперечной RSS, и в этом смысле нарушает ее допущения даже в
    номинальном режиме. Возвращает строки лога этих моментов с колонкой
    ratio = продольный зазор / d_min.
    Первые settle секунд эпизода не учитываются: highway-env расставляет
    машины на старте случайно, без учета d_min, и смена машины впереди в
    это время — следствие стартовой расстановки, а не перестроения.

    Так выглядит перестроение фоновой машины в полосу агента в небезопасный
    промежуток: фоновые машины перестраиваются по MOBIL, а не по поперечной
    RSS, и в этом смысле нарушают ее допущения даже в номинальном режиме.
    Возвращает строки лога этих моментов с колонкой ratio = зазор / d_min.
    """
    front = log["front_id"]
    new_front = front.notna() & front.shift().notna() & (front != front.shift())
    window = max(1, round(settle / dt))
    lane_changed = (log["ego_lane"] != log["ego_lane"].shift()).iloc[1:].reindex(log.index, fill_value=False)
    recent_lane_change = lane_changed.astype(int).rolling(window, min_periods=1).max().astype(bool)
    ratio = log["gap"] / log["d_min"]
    after_start = log["t"] > settle
    events = log[new_front & ~recent_lane_change & after_start & (ratio < 1)].copy()
    events["ratio"] = ratio[events.index]
    return events


def plot_episode(log, title="", axes=None, lateral=False):
    """Панели: зазор и d_min; скорости; ускорения; при lateral=True еще
    поперечное положение нарушителя. Серым — фильтр тормозит."""
    if axes is None:
        _, axes = plt.subplots(4 if lateral else 3, 1, figsize=(8, 9 if lateral else 7), sharex=True)
    t = log["t"]
    active = log["shield_active"].astype(bool)
    t_vio = log.loc[log["vio_v"].notna(), "t"].min() if "vio_v" in log else None

    for ax in axes:
        ax.fill_between(t, 0, 1, where=active, transform=ax.get_xaxis_transform(),
                        color="0.85", step="post", linewidth=0)
        if t_vio is not None and pd.notna(t_vio):
            ax.axvline(t_vio, color="#c0392b", linewidth=1, linestyle=":")
        if log["crashed"].any():
            ax.axvline(log.loc[log["crashed"], "t"].min(), color="black", linewidth=1.5)

    ax = axes[0]
    ax.plot(t, log["gap"].where(log["gap"] < 1e6), color="#2c6fbb", label="зазор до машины впереди")
    # когда впереди никого нет, d_min не определен — не рисуем
    ax.plot(t, log["d_min"].where(log["gap"] < 1e6), color="#e67e22", label="$d_{min}$")
    ax.set_ylabel("м")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(title, fontsize=10)

    ax = axes[1]
    ax.plot(t, log["ego_v"], color="#2c6fbb", label="агент")
    if "vio_v" in log:
        ax.plot(t, log["vio_v"], color="#c0392b", label="нарушитель")
    ax.set_ylabel("скорость, м/с")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax = axes[2]
    ax.plot(t, log["a_nominal"], color="0.6", linewidth=1, label="команда политики")
    ax.plot(t, log["ego_a"], color="#2c6fbb", label="агент, фактически")
    if "vio_a" in log:
        ax.plot(t, log["vio_a"], color="#c0392b", label="нарушитель, фактически")
    ax.set_ylabel("ускорение, м/с²")
    ax.set_xlabel("время, с (серое — фильтр тормозит, пунктир — начало нарушения, "
                  "черная линия — столкновение)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    if lateral:
        ax.set_xlabel("")
        ax = axes[3]
        ax.plot(t, log["vio_y"], color="#c0392b", label="нарушитель: смещение от центра полосы агента")
        for y in (-2, 2):
            ax.axhline(y, color="0.5", linewidth=1, linestyle="--")
        ax.set_ylabel("поперек, м")
        ax.set_xlabel("время, с (серое — фильтр тормозит, пунктир — начало нарушения, "
                      "черная линия — столкновение; штрихи — разметка)")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    for a in axes:
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    return axes
