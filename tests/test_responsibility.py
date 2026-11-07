"""Ответственность за столкновение (rssrl/responsibility.py)."""

import numpy as np
import pandas as pd
import pytest

from rssrl.dynamics import limit_jerk
from rssrl.responsibility import AGENT, BOTH, UNKNOWN, VIOLATOR, assign_responsibility, danger_onset
from rssrl.scenarios import make_scene, run_episode

DT = 1 / 15
VIO = 7  # uid нарушителя


def fake_log(n=60, danger_from=30, crash_at=50, brake_from=None, release_at=None,
             vio_a=-5.0, behind=False, ego_a=None):
    """Лог: опасность с нарушителем с шага danger_from, столкновение на шаге
    crash_at; агент командует -4 с шага brake_from (до release_at).
    Фактическое ускорение агента (ego_a) идет за командой с рывком J_MAX,
    если ego_a не задано числом."""
    t = np.arange(1, n + 1) * DT
    ids = [[VIO] if danger_from <= i <= crash_at else [] for i in range(n)]
    a_cmd = np.zeros(n)
    if brake_from is not None:
        a_cmd[brake_from:release_at] = -4.0
    if ego_a is None:
        ego_a, a = np.zeros(n), 0.0
        for i in range(n):
            a = limit_jerk(a_cmd[i], a, DT)
            ego_a[i] = a
    return pd.DataFrame({
        "t": t,
        "crashed": np.arange(n) >= crash_at,
        "danger_ahead_ids": [[] for _ in ids] if behind else ids,
        "danger_behind_ids": ids if behind else [[] for _ in ids],
        "a_executed": a_cmd,
        "ego_a": np.broadcast_to(ego_a, (n,)).astype(float),
        "ego_v": np.full(n, 20.0),
        "vio_a": np.full(n, vio_a),
        "vio_violating": np.arange(n) >= danger_from - 5,
        "vio_violating_step": np.arange(n) >= danger_from - 5,
    })


def outcome(**kw):
    return {"collision": True, "crash_partners": [VIO], "violator_uid": VIO, "kind": "brake", **kw}


def who(log, **kw):
    return assign_responsibility(log, outcome(**kw))["responsible"]


def test_no_collision_gives_none():
    assert assign_responsibility(fake_log(), {"collision": False}) is None


def test_danger_onset_is_start_of_last_dangerous_run():
    t, pos, idx = danger_onset(fake_log(danger_from=30, crash_at=50), VIO)
    assert t == pytest.approx(30 * DT)  # строка 30 записана в конце шага, решение — в его начале
    assert pos == "ahead" and idx[0] == 30


def test_violator_braked_too_hard_agent_reacted_in_time():
    # торможение 8 м/с^2 > 5; агент начал тормозить через 6 шагов = 0,4 с <= ρ
    assert who(fake_log(brake_from=36, vio_a=-8.0)) == VIOLATOR


def test_harsh_braking_in_earlier_resolved_situation_does_not_count():
    """Нарушитель тормозил 10 м/с^2, агент остановился без удара; позже агент
    снова поехал и въехал в стоящего нарушителя. Резкое торможение было в
    прошлой, уже закончившейся ситуации — нарушитель за удар не отвечает."""
    log = fake_log(danger_from=40, crash_at=50, brake_from=46, vio_a=0.0)
    log["vio_violating"] = log["vio_violating_step"] = False
    log.loc[5:20, ["vio_violating", "vio_violating_step"]] = True
    log.loc[5:20, "vio_a"] = -10.0  # резкое торможение давно, до последней опасной ситуации
    assert who(log) == UNKNOWN


def test_agent_did_not_react_violator_legal():
    assert who(fake_log(brake_from=None, vio_a=-5.0)) == AGENT


def test_both_responsible():
    assert who(fake_log(brake_from=None, vio_a=-8.0)) == BOTH


def test_nobody_responsible_by_rss_rules():
    """Нарушитель законен (5 м/с^2), агент отреагировал вовремя — правила RSS
    не нарушены никем, а столкновение было (например, из-за набора
    торможения с ограниченным рывком)."""
    assert who(fake_log(brake_from=36, vio_a=-5.0)) == UNKNOWN


def test_late_reaction_is_agent_fault():
    # 9 шагов = 0,6 с > ρ = 0,5 с
    r = assign_responsibility(fake_log(brake_from=39), outcome())
    assert r["responsible"] == AGENT and r["agent_late"]
    assert r["reaction_time_command"] == pytest.approx(0.6)


def test_command_and_actual_braking_are_different_moments():
    """Команда через 6 шагов (0,4 с), фактическое -4 м/с^2 с рывком 10 м/с^3
    от нуля — еще через 0,4 с (6 шагов): реакция вовремя, оба момента в логе."""
    r = assign_responsibility(fake_log(brake_from=36), outcome(gap_cross=None))
    assert r["reaction_time_command"] == pytest.approx(0.4)
    assert r["reaction_time_actual"] == pytest.approx(0.4 + 5 * DT)
    assert not r["agent_late"] and not r["agent_late_actual"]


def test_command_in_time_but_actual_braking_never_reached_is_agent_fault():
    """Команда вовремя, но машина фактически не тормозит (ego_a = 0)."""
    r = assign_responsibility(fake_log(brake_from=36, ego_a=0.0), outcome())
    assert not r["agent_late"] and r["agent_late_actual"]
    assert r["responsible"] == AGENT


def test_reaction_exactly_rho_is_in_time_one_step_more_is_late():
    """Допуска полшага нет: ровно ρ (7,5 шага не бывает — 7 шагов = 0,467 с)
    — вовремя, 8 шагов = 0,533 с — поздно."""
    assert not assign_responsibility(fake_log(brake_from=37), outcome())["agent_late"]
    assert assign_responsibility(fake_log(brake_from=38), outcome())["agent_late"]


def test_violator_harsh_braking_on_stopping_step_is_not_lost():
    """Пограничный случай: на шаге остановки нарушитель еще тормозит
    10 м/с^2, а vio_violating в строке этого шага уже сброшен. Признак
    vio_violating_step (нарушал на этом шаге) сохраняет этот шаг в проверке."""
    log = fake_log(brake_from=36, vio_a=-5.0)
    log["vio_violating"] = False
    log["vio_violating_step"] = False
    log.loc[45, "vio_violating_step"] = True  # шаг остановки
    log.loc[45, "vio_a"] = -10.0
    r = assign_responsibility(log, outcome())
    assert r["violator_harsh"] and r["responsible"] == VIOLATOR


def test_valid_profile_from_residual_acceleration_passes():
    """Команда при остатке разгона +3 м/с^2: ускорение убывает с рывком J_MAX и
    на этом пути еще положительно. Это допустимый профиль."""
    n, cmd = 60, 36
    a = np.zeros(n)
    a[:cmd] = 3.0
    for i in range(cmd, n):
        a[i] = limit_jerk(-4.0, a[i - 1], DT)
    r = assign_responsibility(fake_log(brake_from=cmd, ego_a=a), outcome())
    assert not r["agent_jerk_violation"] and not r["agent_accel_after_command"]
    assert r["responsible"] == UNKNOWN


def test_long_acceleration_then_illegal_jump_is_caught():
    """Письмо 07.11, п. 5: команда вовремя, ускорение держится +3 м/с^2 почти до
    срока, потом скачок сразу на -4. Сроки соблюдены, профиль недопустим."""
    a = np.zeros(60)
    a[30:48] = 3.0  # 36-й шаг — команда (0,4 с), 48-й — скачок (1,2 с: срок ρ + 0,7)
    a[48:] = -4.0
    r = assign_responsibility(fake_log(brake_from=36, ego_a=a, crash_at=55), outcome())
    assert not r["agent_late"] and not r["agent_late_actual"]
    assert r["agent_jerk_violation"] and r["agent_accel_after_command"]
    assert r["responsible"] == AGENT


def test_acceleration_growing_after_command_is_caught():
    a = np.zeros(60)
    a[37:40] = [0.3, 0.6, 0.9]
    for i in range(40, 60):
        a[i] = limit_jerk(-4.0, a[i - 1], DT)
    r = assign_responsibility(fake_log(brake_from=36, ego_a=a, crash_at=58), outcome())
    assert r["agent_accel_after_command"] and not r["agent_jerk_violation"]
    assert r["responsible"] == AGENT


def test_jump_to_braking_without_acceleration_is_jerk_violation():
    a = np.zeros(60)
    a[36:] = -4.0  # мгновенно от 0 до -4
    r = assign_responsibility(fake_log(brake_from=36, ego_a=a), outcome())
    assert r["agent_jerk_violation"] and not r["agent_accel_after_command"]
    assert r["responsible"] == AGENT


def test_transition_to_rest_is_exempt_from_jerk_check():
    """На шаге остановки ускорение скачком идет от торможения к малому значению,
    чтобы скорость стала ровно нулевой. Это не нарушение профиля."""
    log = fake_log(brake_from=36)
    log.loc[39, "ego_a"] = -1.0  # слабее, чем на прошлом шаге: дотормаживание до нуля
    log.loc[39:, "ego_v"] = 0.0
    log.loc[40:, "ego_a"] = 0.0
    r = assign_responsibility(log, outcome())
    assert not r["agent_jerk_violation"] and not r["agent_accel_after_command"]


def test_fallback_keeps_impact_step_for_violator_braking():
    """Письмо 07.11, п. 5: на шаге удара партнер уже не отмечен опасным (резервная
    ветвь), а единственное торможение сильнее a_max_brake — как раз на этом шаге."""
    log = fake_log(brake_from=36, vio_a=-5.0)
    log.at[50, "danger_ahead_ids"] = []
    log.loc[50, "vio_a"] = -10.0
    t, _, idx = danger_onset(log, VIO)
    assert t == pytest.approx(30 * DT) and idx[-1] == 49
    r = assign_responsibility(log, outcome())
    assert r["violator_harsh"] and r["violator_min_accel"] == -10.0
    assert r["responsible"] == VIOLATOR


def test_released_brake_while_dangerous_is_agent_fault():
    r = assign_responsibility(fake_log(brake_from=33, release_at=45), outcome())
    assert r["responsible"] == AGENT and r["agent_released"]


def test_cutin_closer_than_d_min_is_violator_fault():
    log = fake_log(brake_from=33)
    assert who(log, kind="cutin", gap_cross=30.0, d_min_cross=50.0, t_cross=2.5) == VIOLATOR
    # γ ≈ 0,995 — в пределах точности сцены, подрезание законно
    assert who(log, kind="cutin", gap_cross=49.75, d_min_cross=50.0) == UNKNOWN
    # подрезание было раньше начала опасной ситуации, приведшей к удару
    # (агент тогда остановился без удара) — нарушитель не отвечает
    assert who(log, kind="cutin", gap_cross=30.0, d_min_cross=50.0, t_cross=0.5) == UNKNOWN


def test_partner_behind_agent_braked_harder_than_assumed():
    """Агент спереди: если он сам тормозил сильнее a_max_brake (экстренный
    режим), для машины сзади это нарушение допущения RSS."""
    assert who(fake_log(behind=True, ego_a=-8.0)) == AGENT
    assert who(fake_log(behind=True, ego_a=-4.0)) == UNKNOWN


def test_crash_without_preceding_danger_is_not_agent_fault():
    log = fake_log(danger_from=100)  # опасности не было
    r = assign_responsibility(log, outcome())
    assert r["t_danger"] is None and r["responsible"] == UNKNOWN


def test_real_scene_beta_2_without_shield():
    """β = 2 без фильтра: нарушитель тормозит 10 м/с^2 (> 5), агент (IDLE)
    не тормозит вовсе — отвечают оба."""
    for seed in range(10):
        o, log = run_episode(make_scene(seed, beta=2.0, shield=False))
        if o["collision"] and o["hit_violator"]:
            r = assign_responsibility(log, o)
            assert r["responsible"] == BOTH
            assert r["violator_min_accel"] == pytest.approx(-10.0, abs=1e-6)
            return
    pytest.fail("нет столкновения с нарушителем на 10 сценах")
