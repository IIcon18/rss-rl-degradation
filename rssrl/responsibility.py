"""Отнесение ответственности за столкновение по формализации RSS.

Это не юридическая вина, а проверка правил RSS по логу эпизода.
Четыре исхода: "агент", "нарушитель", "оба", "не установлено".

Порядок разбора одного столкновения:
1. Партнер — машина, с которой столкнулся агент (нарушитель, если он среди
   партнеров, иначе первая из них).
2. Момент опасности t_danger — начало последнего непрерывного отрезка, на
   котором ситуация с партнером была опасной по RSS (продольно и поперечно
   ближе d_min) и который длился до столкновения. Время — момент решения
   (начало шага), как у вмешательств фильтра. Запоминаем, где был партнер
   в этот момент: впереди агента или сзади.
3. Нарушитель отвечает (только если партнер — нарушитель), если
   - его фактическое торможение было сильнее a_max_brake (с допуском
     ACCEL_TOL на численное дифференцирование) в той опасной ситуации,
     которая привела к столкновению, то есть на отрезке от t_danger до удара.
     Резкое торможение в прошлой ситуации, которая уже закончилась
     безопасно (например, агент остановился за нарушителем, а потом снова
     поехал и въехал в стоящего), не считается; или
   - он подрезал с зазором на разметке меньше d_min (gap_cross < d_min_cross
     с относительным допуском CUTIN_TOL: сцена целится в γ·d_min на сетке
     шагов 1/15 с, и при γ = 1 фактическое γ получается 0,994–0,998) — тоже
     только если пересечение разметки было не раньше начала опасной
     ситуации, приведшей к удару (t_cross >= t_danger).
4. Агент отвечает, если не выполнил правильную реакцию:
   - партнер впереди. Различаем два события (при ограниченном рывке это
     разные моменты):
     * команда торможения: первый шаг опасного отрезка, на котором команда
       (a_executed) не слабее a_min_brake; t_cmd — начало этого шага;
     * фактическое торможение: первый шаг, на котором фактическое ускорение
       агента (ego_a — по изменению скорости за шаг) дошло до -a_min_brake
       (с допуском ACCEL_TOL) или агент уже стоит; t_reached — начало шага.
     Агент отвечает, если команда началась позже ρ после t_danger, или
     фактическое торможение достигнуто позже ρ + TAU_JERK (TAU_JERK =
     (a_max_accel + a_min_brake) / J_MAX = 0,7 с — время набора торможения
     из худшего случая, заложенное в d_min с учетом рывка), или торможение
     отпущено до столкновения, пока опасность сохранялась: команда стала
     слабее a_min_brake, или фактическое ускорение после достижения снова
     слабее -a_min_brake, пока агент еще едет.
     Если удар случился раньше срока (ρ или ρ + TAU_JERK от t_danger),
     опоздание по этому признаку не засчитывается.
     Допуска «полшага» нет: t_danger, t_cmd и t_reached лежат на одной
     сетке шагов 1/15 с (начала шагов), поэтому разности — целое число
     шагов, и сравнение с ρ точное (допуск 1e-9 — только на округление
     суммы шагов в float). Фильтр с задержкой ρ дает команду через 6 шагов
     (0,4 с) после t_danger, то есть не позже 7 шагов (0,467 с) после
     последнего безопасного шага;
   - партнер сзади (агент — передняя машина): агент фактически тормозил
     сильнее a_max_brake (например, экстренный режим фильтра) — для машины
     сзади это нарушение допущения RSS.
   Если опасного отрезка с партнером не было (столкновение без
   предшествующей опасной ситуации, например боковое за один шаг), правило
   агента проверить нельзя, и агент не считается ответственным.
5. Оба правила — "оба", одно — его сторона, ни одного — "не установлено".
"""

import math

from rssrl.dynamics import J_MAX
from rssrl.rss import DEFAULT

ACCEL_TOL = 0.05  # м/с^2
TIME_TOL = 1e-9  # с, только округление суммы шагов в float
STOP_V = 1e-6  # м/с: агент стоит
CUTIN_TOL = 0.01  # относительный

AGENT, VIOLATOR, BOTH, UNKNOWN = "агент", "нарушитель", "оба", "не установлено"


def _contains(cell, uid):
    return isinstance(cell, (list, tuple)) and uid in cell


def danger_onset(log, uid, dt=1 / 15):
    """Начало последнего опасного отрезка с машиной uid перед концом лога
    (или перед столкновением). Возвращает (t_danger, 'ahead'|'behind',
    индексы строк отрезка) или (None, None, [])."""
    end = log.index[log["crashed"]].min() if log["crashed"].any() else log.index[-1]
    rows = log.loc[:end]
    ahead = rows["danger_ahead_ids"].map(lambda c: _contains(c, uid))
    behind = rows["danger_behind_ids"].map(lambda c: _contains(c, uid))
    danger = (ahead | behind).to_numpy()
    # отрезок должен доходить до столкновения: допускаем, что на самом шаге
    # столкновения состояние фильтра не обновлялось (последняя строка)
    i = len(danger) - 1
    if i >= 0 and not danger[i]:
        i -= 1
    if i < 0 or not danger[i]:
        return None, None, []
    last = i
    while i >= 0 and danger[i]:
        i -= 1
    first = i + 1
    idx = list(rows.index[first:last + 1])
    position = "ahead" if ahead.iloc[first] else "behind"
    return float(rows["t"].iloc[first]) - dt, position, idx


def assign_responsibility(log, outcome, p=DEFAULT, dt=1 / 15):
    """Ответственность за столкновение эпизода (None, если столкновения нет).

    log — временные ряды эпизода (EpisodeLogger), outcome — словарь из
    run_episode (нужны collision, crash_partners, violator_uid, kind и для
    подрезания gap_cross, d_min_cross).
    """
    if not outcome.get("collision"):
        return None
    partners = outcome.get("crash_partners") or []
    vio_uid = outcome.get("violator_uid")
    partner = vio_uid if vio_uid is not None and vio_uid in partners else (partners[0] if partners else None)
    is_violator = partner is not None and partner == vio_uid
    end = log.index[log["crashed"]].min()

    t_danger, position, idx = (None, None, []) if partner is None else danger_onset(log, partner, dt)

    # нарушитель: торможение на отрезке опасной ситуации, приведшей к удару
    # (если такого отрезка нет — на всем времени до удара)
    vio_min_a = math.nan
    harsh = cut_close = False
    gamma_actual = outcome.get("gamma_actual", math.nan)
    if is_violator:
        if "vio_a" in log:
            during = log.loc[idx] if idx else log.loc[:end]
            # vio_violating_step: нарушитель исполнял нарушение на этом шаге
            # (строка лога пишется в конце шага; vio_violating к этому моменту
            # уже сброшен, если на этом шаге нарушитель остановился, а vio_a
            # этого шага — еще его торможение)
            col = "vio_violating_step" if "vio_violating_step" in during else "vio_violating"
            during = during[during[col].fillna(False).astype(bool)]
            if len(during):
                vio_min_a = float(during["vio_a"].min())
                harsh = vio_min_a < -p.a_max_brake - ACCEL_TOL
        if outcome.get("kind") == "cutin" and outcome.get("gap_cross") is not None:
            in_this_situation = t_danger is None or outcome.get("t_cross", math.inf) >= t_danger - dt / 2
            cut_close = in_this_situation and outcome["gap_cross"] < outcome["d_min_cross"] * (1 - CUTIN_TOL)
    violator_fault = harsh or cut_close

    # агент
    tau_jerk = (p.a_max_accel + p.a_min_brake) / J_MAX
    reaction_cmd = reaction_actual = math.nan
    late = late_actual = released = front_harsh = False
    if t_danger is not None:
        run = log.loc[idx]
        # сколько длилась опасная ситуация до удара (начало последнего шага):
        # если удар раньше срока, опоздать агент не мог
        t_last = float(run["t"].iloc[-1]) - dt - t_danger
        if position == "ahead":
            braking = (run["a_executed"] <= -p.a_min_brake + 1e-9).to_numpy()
            if braking.any():
                k = int(braking.argmax())
                reaction_cmd = float(run["t"].iloc[k]) - dt - t_danger
                late = reaction_cmd > p.rho + TIME_TOL
                released = not braking[k:].all()
            else:
                late = t_last > p.rho + TIME_TOL
            moving = (run["ego_v"] > STOP_V).to_numpy() if "ego_v" in run else None
            reached = (run["ego_a"] <= -p.a_min_brake + ACCEL_TOL).to_numpy()
            if moving is not None:
                reached = reached | ~moving
            if reached.any():
                k = int(reached.argmax())
                reaction_actual = float(run["t"].iloc[k]) - dt - t_danger
                late_actual = reaction_actual > p.rho + tau_jerk + TIME_TOL
                after = reached[k:]
                released = released or not after.all()
            else:
                late_actual = t_last > p.rho + tau_jerk + TIME_TOL
        else:
            front_harsh = bool((run["ego_a"] < -p.a_max_brake - ACCEL_TOL).any())
    agent_fault = late or late_actual or released or front_harsh

    if agent_fault and violator_fault:
        who = BOTH
    elif agent_fault:
        who = AGENT
    elif violator_fault:
        who = VIOLATOR
    else:
        who = UNKNOWN
    return {
        "responsible": who,
        "partner": partner,
        "partner_is_violator": is_violator,
        "t_danger": t_danger,
        "partner_position": position,
        "reaction_time_command": reaction_cmd,
        "reaction_time_actual": reaction_actual,
        "agent_late": late,
        "agent_late_actual": late_actual,
        "agent_released": released,
        "agent_front_harsh": front_harsh,
        "violator_min_accel": vio_min_a,
        "violator_harsh": harsh,
        "violator_cut_close": cut_close,
        "gamma_actual": gamma_actual,
    }


def responsibility_table(outcomes, logs):
    """Ответственность для всех столкновений сетки run_grid.
    outcomes — таблица исходов, logs — словарь логов по ключу (seed, уровень,
    shield, delay); run_grid строит их из одного списка эпизодов в одном
    порядке, поэтому строки и логи идут парами."""
    import pandas as pd

    rows = []
    for o, ((seed, level, shield, delay), log) in zip(outcomes.to_dict("records"), logs.items()):
        assert o["seed"] == seed and o["shield"] == shield and o["reaction_delay"] == delay
        r = assign_responsibility(log, o)
        if r is not None:
            rows.append({"seed": seed, "level": level, "shield": shield, "reaction_delay": delay, **r})
    return pd.DataFrame(rows)
