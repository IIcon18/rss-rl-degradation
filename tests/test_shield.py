"""Этап 3: RSS-фильтр на нижнем уровне (rssrl/shield.py, rssrl/scenarios.py)."""

import math

import numpy as np
import pytest

from rssrl.dynamics import A_PHYS_BRAKE, J_MAX
from rssrl.rss import DEFAULT, d_min_longitudinal
from rssrl.scenarios import make_env, make_scene, run_episode, violation_schedule
from rssrl.shield import SHIELD_DEFAULT, SIM_DT, ShieldedVehicle, find_front
from rssrl.violators import BoundedIDMVehicle

LANE_W = 4.0  # ширина полосы highway-env, м


def empty_road_with_ego(shield=True, speed=25.0, **kwargs):
    """Дорога без фоновых машин, агент в полосе 1 в точке x = 0."""
    env = make_env(0, vehicles_count=0)
    u = env.unwrapped
    ego = ShieldedVehicle.replace(u.vehicle, shield_enabled=shield, **kwargs)
    u.vehicle = ego
    ego.position[:] = [0.0, LANE_W]
    ego.heading, ego.speed = 0.0, speed
    ego.on_state_update()
    ego.target_lane_index = ego.lane_index  # иначе поедет перестраиваться в старую полосу
    return env, ego


def add_car(env, x, lane, speed):
    road = env.unwrapped.road
    car = BoundedIDMVehicle(road, np.array([x, lane * LANE_W]), heading=0.0, speed=speed)
    road.vehicles.append(car)
    return car


# ---------- что фильтр видит ----------

def test_find_front_nearest_in_own_lane():
    env, ego = empty_road_with_ego()
    far = add_car(env, 80.0, 1, 20.0)
    near = add_car(env, 40.0, 1, 20.0)
    add_car(env, 20.0, 0, 20.0)  # соседняя полоса — не «впереди»
    add_car(env, -30.0, 1, 20.0)  # сзади — не «впереди»
    front, gap = find_front(ego)
    assert front is near and front is not far
    assert gap == pytest.approx(40.0 - 5.0)  # между бамперами: минус две половины длины


def test_find_front_sees_beyond_observation_but_within_150m():
    env, ego = empty_road_with_ego()
    # 6 машин в соседних полосах ближе лидера: агент в наблюдении видит
    # только 5 ближайших, фильтр — всех в радиусе 150 м
    for x in [5.0, 15.0, 25.0, 35.0, 45.0, 55.0]:
        add_car(env, x, 0, 25.0)
    leader = add_car(env, 140.0, 1, 25.0)
    assert find_front(ego)[0] is leader
    leader.position[0] = 160.0
    assert find_front(ego)[0] is None


# ---------- решение фильтра ----------

def test_no_intervention_when_safe():
    env, ego = empty_road_with_ego()
    add_car(env, 120.0, 1, 25.0)  # d_min(25, 25) ≈ 38 м — далеко
    ego.act()
    assert not ego.state["dangerous"] and not ego.state["shield_active"]
    assert ego.state["a_executed"] == ego.state["a_nominal"]


def test_brakes_at_least_a_min_brake_when_dangerous():
    env, ego = empty_road_with_ego(reaction_delay=0.0)
    add_car(env, 30.0, 1, 25.0)  # зазор 25 м < d_min ≈ 38 м
    ego.act()
    assert ego.state["dangerous"] and ego.state["shield_active"]
    assert ego.state["a_executed"] <= -DEFAULT.a_min_brake
    # фильтр по умолчанию считает d_min по лемме 2 с учетом рывка
    assert ego.state["d_min"] == pytest.approx(d_min_longitudinal(25.0, 25.0, SHIELD_DEFAULT))


def test_without_shield_same_check_but_no_intervention():
    env, ego = empty_road_with_ego(shield=False)
    add_car(env, 30.0, 1, 25.0)
    ego.act()
    assert ego.state["dangerous"] and not ego.state["shield_active"]
    assert ego.state["a_executed"] == ego.state["a_nominal"]


def test_reaction_delay_is_counted_from_last_safe_step():
    """С задержкой ρ торможение начинается через 7 шагов (7/15 ≈ 0,467 с <= ρ)
    после последнего безопасного шага, а не через 8 (0,533 с > ρ)."""
    env, ego = empty_road_with_ego(reaction_delay=DEFAULT.rho)
    add_car(env, 30.0, 1, 25.0)
    ego.last_safe = ego.clock - SIM_DT  # шаг назад (в момент -1/15 с) было безопасно
    active = []
    for _ in range(10):
        ego.act()
        active.append(ego.state["shield_active"])
        ego.step(SIM_DT)
    # шаги 0..5 — команда политики, с шага 6 торможение: он начинается
    # в момент 6/15 с, то есть через 7/15 ≈ 0,467 с после безопасного момента
    assert active.index(True) == 6
    assert all(active[6:])


# ---------- главное свойство на сценах ----------

SEEDS = range(12)


@pytest.mark.parametrize("delay", [0.0, DEFAULT.rho])
def test_no_collisions_with_shield_when_assumption_holds(delay, monkeypatch):
    """β = 1 (лидер тормозит ровно с a_max_brake): с фильтром агент
    не должен ни во что врезаться, при любой из двух задержек.

    Вариант 1: d_min по лемме 2 как в статье — в условиях самой леммы, то
    есть без предела рывка (лемма считает, что торможение начинается
    мгновенно после ρ)."""
    monkeypatch.setattr("rssrl.dynamics.J_MAX", math.inf)
    for seed in SEEDS:
        outcome, _ = run_episode(make_scene(seed, beta=1.0, shield=True, reaction_delay=delay, dmin="lemma2"))
        assert outcome["triggered"], seed
        assert not outcome["collision"], (seed, outcome["crash_partners"])


@pytest.mark.parametrize("delay", [0.0, DEFAULT.rho])
def test_no_collisions_with_jerk_and_jerk_aware_dmin(delay):
    """Главный критерий готовности: рывок 10 м/с^3 включен у всех машин,
    фильтр считает d_min с учетом рывка (по умолчанию). При β = 1 столкновений
    быть не должно ни при одной задержке. Нарушение — по расписанию, чтобы
    оно наступало и у осторожного мгновенного фильтра."""
    for seed in SEEDS:
        sched = violation_schedule(seed)
        outcome, _ = run_episode(make_scene(seed, beta=1.0, shield=True, reaction_delay=delay, schedule=sched))
        assert outcome["triggered"], seed
        assert not outcome["collision"], (seed, outcome["crash_partners"], outcome["v_impact"])


@pytest.mark.parametrize("seed", [0, 3])
def test_ego_physical_limits_respected(seed):
    for shield in [True, False]:
        _, log = run_episode(make_scene(seed, beta=2.0, shield=shield))
        ok = log[~log["crashed"]]
        assert ok["ego_a"].max() <= DEFAULT.a_max_accel + 1e-9
        # сильнее 5 м/с^2 тормозит только экстренный режим фильтра
        floor = -A_PHYS_BRAKE if shield else -DEFAULT.a_max_brake
        assert ok["ego_a"].min() >= floor - 1e-9
        assert ok["ego_v"].min() >= 0.0
        # рывок: ускорение меняется не больше чем на J_MAX·dt за шаг
        # (кроме дотормаживания до нуля перед остановкой)
        jumps = ok["ego_a"].diff().abs()
        stopping = (ok["ego_v"] < 0.5) | (ok["ego_v"].shift() < 0.5)
        assert jumps[~stopping].max() <= J_MAX * SIM_DT + 1e-9


def test_scene_identical_until_violation_for_all_betas():
    """Общие сцены: до момента нарушения прогоны с разными β совпадают."""
    logs = [run_episode(make_scene(5, beta=b, shield=True))[1] for b in [1.0, 2.0]]
    t_trig = logs[0]["vio_v"].first_valid_index()
    assert t_trig == logs[1]["vio_v"].first_valid_index()
    cols = ["ego_x", "ego_v", "gap", "d_min"]
    np.testing.assert_array_equal(logs[0].loc[: t_trig - 1, cols], logs[1].loc[: t_trig - 1, cols])
