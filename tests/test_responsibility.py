"""Ответственность за столкновение (rssrl/responsibility.py)."""

import numpy as np
import pandas as pd
import pytest

from rssrl.responsibility import AGENT, BOTH, UNKNOWN, VIOLATOR, assign_responsibility, danger_onset
from rssrl.scenarios import make_scene, run_episode

DT = 1 / 15
VIO = 7  # uid нарушителя


def fake_log(n=60, danger_from=30, crash_at=50, brake_from=None, release_at=None,
             vio_a=-5.0, behind=False, ego_a=0.0):
    """Лог: опасность с нарушителем с шага danger_from, столкновение на шаге
    crash_at; агент командует -4 с шага brake_from (до release_at)."""
    t = np.arange(1, n + 1) * DT
    ids = [[VIO] if danger_from <= i <= crash_at else [] for i in range(n)]
    a_cmd = np.zeros(n)
    if brake_from is not None:
        a_cmd[brake_from:release_at] = -4.0
    return pd.DataFrame({
        "t": t,
        "crashed": np.arange(n) >= crash_at,
        "danger_ahead_ids": [[] for _ in ids] if behind else ids,
        "danger_behind_ids": ids if behind else [[] for _ in ids],
        "a_executed": a_cmd,
        "ego_a": np.full(n, ego_a),
        "vio_a": np.full(n, vio_a),
        "vio_violating": np.arange(n) >= danger_from - 5,
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
    log["vio_violating"] = False
    log.loc[5:20, "vio_violating"] = True
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
    assert r["reaction_time"] == pytest.approx(0.6)


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
