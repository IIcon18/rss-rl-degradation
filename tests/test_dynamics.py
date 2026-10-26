"""Предел рывка и экстренный режим фильтра (rssrl/dynamics.py, rssrl/shield.py)."""

import math

import numpy as np
import pytest

from rssrl.dynamics import A_PHYS_BRAKE, J_MAX, limit_jerk
from rssrl.rss import DEFAULT
from rssrl.scenarios import make_env
from rssrl.shield import SIM_DT, ShieldedVehicle, brake_insufficient
from rssrl.violators import BoundedIDMVehicle

LANE_W = 4.0


# ---------- предел рывка ----------

def test_limit_jerk_hand_examples():
    step = J_MAX * SIM_DT  # 10 · 1/15 ≈ 0,667 м/с^2 за шаг
    assert limit_jerk(-4.0, 0.0, SIM_DT) == pytest.approx(-step)
    assert limit_jerk(3.0, 0.0, SIM_DT) == pytest.approx(step)
    assert limit_jerk(-0.3, 0.0, SIM_DT) == pytest.approx(-0.3)  # в пределах шага — как есть
    assert limit_jerk(-4.0, 0.0, SIM_DT, j_max=math.inf) == -4.0


def test_limit_jerk_reaches_target_in_expected_time():
    """От 0 до -4 м/с^2 при J_MAX = 10 м/с^3: 0,4 с = 6 шагов."""
    a, n = 0.0, 0
    while a > -4.0 + 1e-12:
        a = limit_jerk(-4.0, a, SIM_DT)
        n += 1
    assert n == 6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_background_acceleration_changes_no_faster_than_jerk_limit(seed):
    env = make_env(seed)
    road, ego = env.unwrapped.road, env.unwrapped.vehicle
    prev = {}
    for _ in range(20 * 15):
        before = {id(v): v.speed for v in road.vehicles}
        road.act()
        road.step(SIM_DT)
        for v in road.vehicles:
            if v is ego or v.crashed or id(v) not in before:
                continue
            a = (v.speed - before[id(v)]) / SIM_DT
            stopping = v.speed < 1e-9 or before[id(v)] < 1.0
            if id(v) in prev and not stopping:
                assert abs(a - prev[id(v)]) <= J_MAX * SIM_DT + 1e-9
            prev[id(v)] = a
    env.close()


# ---------- прогноз «хватит ли a_min_brake» ----------

def test_brake_insufficient_hand_example():
    # задняя уже тормозит с 4 м/с^2, передняя стоит: путь задней 10²/(2·4) = 12,5 м
    # (+ v·dt/2 ≈ 0,33 м от явной схемы интегрирования)
    assert brake_insufficient(10.0, 10.0, 0.0, -4.0, 5.0, 4.0, j_max=math.inf)
    assert not brake_insufficient(14.0, 10.0, 0.0, -4.0, 5.0, 4.0, j_max=math.inf)


def test_brake_insufficient_accounts_for_jerk():
    """С пределом рывка набор торможения от +3 до -4 занимает 0,7 с,
    и зазор, которого хватало при мгновенном торможении, перестает хватать."""
    v, gap = 25.0, 25.0 ** 2 / 8 + 2.0  # без рывка запас 2 м
    assert not brake_insufficient(gap, v, 0.0, -4.0, 5.0, 4.0, j_max=math.inf)
    assert brake_insufficient(gap, v, 0.0, 3.0, 5.0, 4.0)


# ---------- режимы фильтра ----------

def road_with_ego(speed=25.0, **kwargs):
    env = make_env(0, vehicles_count=0)
    u = env.unwrapped
    ego = ShieldedVehicle.replace(u.vehicle, **kwargs)
    u.vehicle = ego
    ego.position[:] = [0.0, LANE_W]
    ego.heading, ego.speed = 0.0, speed
    ego.on_state_update()
    ego.target_lane_index = ego.lane_index
    return env, ego


def add_car(env, x, speed):
    car = BoundedIDMVehicle(env.unwrapped.road, np.array([x, LANE_W]), heading=0.0, speed=speed)
    env.unwrapped.road.vehicles.append(car)
    return car


def test_mode_brake_when_a_min_brake_is_enough():
    env, ego = road_with_ego(reaction_delay=0.0)
    add_car(env, 30.0, 25.0)  # зазор 25 м < d_min ≈ 38 м, но обе едут 25 м/с
    ego.act()
    assert ego.state["shield_mode"] == "brake"
    assert ego.state["a_executed"] == pytest.approx(-DEFAULT.a_min_brake)


def test_mode_emergency_when_a_min_brake_is_not_enough():
    env, ego = road_with_ego(reaction_delay=0.0)
    add_car(env, 30.0, 0.0)  # стоящая машина в 25 м: при 4 м/с^2 нужно 78 м
    ego.act()
    assert ego.state["shield_mode"] == "emergency"
    assert ego.state["a_executed"] == pytest.approx(-A_PHYS_BRAKE)


def test_emergency_is_held_until_danger_ends():
    env, ego = road_with_ego(reaction_delay=0.0)
    car = add_car(env, 30.0, 0.0)
    modes = []
    for _ in range(5):
        ego.act()
        modes.append(ego.state["shield_mode"])
        ego.step(SIM_DT)
    assert modes == ["emergency"] * 5
    car.position[0] = 500.0  # опасность исчезла
    ego.act()
    assert ego.state["shield_mode"] is None and not ego.emergency


def test_mode_delay_during_reaction_time():
    """С задержкой ρ: сначала режим "delay" (исполняется команда политики), потом торможение."""
    env, ego = road_with_ego(reaction_delay=DEFAULT.rho)
    add_car(env, 30.0, 25.0)
    ego.last_safe = ego.clock - SIM_DT
    modes = []
    for _ in range(8):
        ego.act()
        modes.append(ego.state["shield_mode"])
        ego.step(SIM_DT)
    assert modes[:6] == ["delay"] * 6
    assert modes[6] in ("brake", "emergency")


def test_default_reaction_delay_is_rho():
    _, ego = road_with_ego()
    assert ego.reaction_delay == DEFAULT.rho
