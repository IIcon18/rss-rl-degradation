"""Этап 6: исходы эпизода, вмешательства фильтра и контрфактический прогон."""

import numpy as np
import pandas as pd
import pytest

from rssrl.pilot import _run_one, counterfactual, interventions
from rssrl.scenarios import STOP_DURATION, classify_outcome

DT = 1 / 15


def fake_log(n=150, v=20.0, crashed_at=None, offroad_at=None, stop=None):
    """Синтетический лог: скорость v, можно добавить столкновение,
    съезд с дороги и стоянку (stop = (начало, длина) в шагах)."""
    log = pd.DataFrame({
        "t": np.arange(1, n + 1) * DT,
        "ego_v": np.full(n, v),
        "crashed": np.zeros(n, bool),
        "on_road": np.ones(n, bool),
    })
    if stop:
        log.loc[stop[0]:stop[0] + stop[1] - 1, "ego_v"] = 0.0
    if crashed_at is not None:
        log.loc[crashed_at:, "crashed"] = True
    if offroad_at is not None:
        log.loc[offroad_at, "on_road"] = False
    return log


# ---------- исходы ----------

def test_outcome_safe():
    assert classify_outcome(fake_log())[0] == "safe"


def test_outcome_stopped_needs_long_enough_standing():
    n_stop = round(STOP_DURATION / DT)
    assert classify_outcome(fake_log(stop=(30, n_stop)))[0] == "stopped"
    assert classify_outcome(fake_log(stop=(30, n_stop - 1)))[0] == "safe"  # постоял меньше 2 с


def test_outcome_priority():
    # столкновение важнее съезда и стоянки, съезд важнее стоянки
    assert classify_outcome(fake_log(crashed_at=100, offroad_at=50, stop=(10, 60)))[0] == "collision"
    assert classify_outcome(fake_log(offroad_at=50, stop=(10, 60)))[0] == "offroad"


def test_standing_time_is_counted():
    _, t = classify_outcome(fake_log(stop=(10, 45)))
    assert t == pytest.approx(3.0)


# ---------- вмешательства ----------

def test_interventions_merge_short_gaps():
    n = 120
    active = np.zeros(n, bool)
    active[10:20] = True
    active[25:30] = True  # перерыв 5 шагов (0,33 с) < 1 с — то же вмешательство
    active[60:62] = True  # перерыв 2 с — новое вмешательство
    decision = np.array([None] * n, dtype=object)
    decision[100] = "blocked"  # запрет перестроения — тоже вмешательство
    log = pd.DataFrame({"t": np.arange(1, n + 1) * DT, "shield_active": active, "lc_decision": decision})
    iv = interventions(log)
    assert len(iv) == 3
    assert iv["brake_steps"].tolist() == [15, 2, 0]
    assert iv["lc_steps"].tolist() == [0, 0, 1]
    # начало — момент решения (начало шага), конец — конец последнего шага
    assert iv["t_start"].iloc[0] == pytest.approx(10 * DT)
    assert iv["t_end"].iloc[0] == pytest.approx(30 * DT)


# ---------- контрфактический прогон ----------

JOB = (3, 1.0, True, 0.0)  # сцена 3, номинальный режим, фильтр, задержка 0


@pytest.fixture(scope="module")
def original():
    return _run_one(JOB, "nominal", "idle")


def test_counterfactual_matches_original_before_switch_off(original):
    _, log = original
    iv = interventions(log).iloc[0]
    _, cf_log = _run_one(JOB, "nominal", "idle", shield_off_at=iv.t_start, t_stop=iv.t_end + 5)
    before = log["t"] <= iv.t_start + 1e-9
    cols = ["ego_x", "ego_y", "ego_v", "gap", "shield_active"]
    pd.testing.assert_frame_equal(log.loc[before, cols], cf_log.loc[before[: len(cf_log)], cols])
    # после отключения фильтр больше не вмешивается
    assert not cf_log.loc[cf_log["t"] > iv.t_start, "shield_active"].any()


def test_counterfactual_stops_at_window(original):
    _, log = original
    iv = interventions(log).iloc[0]
    _, cf_log = _run_one(JOB, "nominal", "idle", shield_off_at=iv.t_start, t_stop=iv.t_end + 5)
    # останавливаемся с точностью до шага политики (1 с) или раньше — из-за столкновения
    assert cf_log["t"].iloc[-1] <= iv.t_end + 5 + 1.0 + 1e-9


def test_idle_without_shield_needs_the_intervention(original):
    """IDLE не тормозит перед медленной машиной: если убрать первое длинное
    вмешательство фильтра, он врезается (как 97 из 100 без фильтра на этапе 5)."""
    _, log = original
    iv = interventions(log)
    longest = iv.loc[(iv.t_end - iv.t_start).idxmax()]
    cf = counterfactual(JOB, "nominal", "idle", longest.t_start, longest.t_end)
    assert cf["collision"]


def test_braking_violator_beta_1_ends_in_stop():
    """β = 1: лидер тормозит до остановки и стоит; IDLE с фильтром
    останавливается за ним без столкновения — исход «остановка»."""
    outcome, _ = _run_one((0, 1.0, True, 0.0), "brake", "idle")
    assert outcome["triggered"]
    assert outcome["outcome"] == "stopped"
    assert outcome["t_standing"] >= STOP_DURATION


def test_counterfactual_window_extends_past_episode_end():
    """Вмешательство в конце эпизода: окно контрфакта (+5 с) выходит за 40 с,
    и прогон продлевается, а сцена до момента отключения та же."""
    job = (12, 1.0, True, 0.0)
    _, log = _run_one(job, "nominal", "idle")
    iv = interventions(log).iloc[-1]
    assert iv.t_end + 5 > 40
    _, cf_log = _run_one(job, "nominal", "idle", shield_off_at=iv.t_start,
                         t_stop=iv.t_end + 5, duration=46)
    assert cf_log["t"].iloc[-1] > 40 or cf_log["crashed"].any()
    n = int((log["t"] <= iv.t_start + 1e-9).sum())
    pd.testing.assert_series_equal(log["ego_x"].iloc[:n], cf_log["ego_x"].iloc[:n])
