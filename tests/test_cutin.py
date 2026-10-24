"""Этап 4: подрезание и поперечная RSS в фильтре."""

import math

import numpy as np
import pytest

from rssrl.rss import DEFAULT, d_min_lateral
from rssrl.scenarios import LC_DURATION, make_env, make_scene, run_episode
from rssrl.shield import SIM_DT, ShieldedVehicle, relation
from rssrl.violators import BoundedIDMVehicle

LANE_W = 4.0


def road_with_ego(shield=True, lane=1, speed=25.0):
    """Пустая дорога, агент в полосе lane в точке x = 0, едет прямо."""
    env = make_env(0, vehicles_count=0)
    u = env.unwrapped
    ego = ShieldedVehicle.replace(u.vehicle, shield_enabled=shield)
    u.vehicle = ego
    ego.position[:] = [0.0, lane * LANE_W]
    ego.heading, ego.speed = 0.0, speed
    ego.on_state_update()
    ego.target_lane_index = ego.lane_index
    ego.target_speed = speed
    return env, ego


def add_car(env, x, y, speed, heading=0.0):
    road = env.unwrapped.road
    car = BoundedIDMVehicle(road, np.array([x, y]), heading=heading, speed=speed)
    road.vehicles.append(car)
    return car


# ---------- поперечная RSS в отношении машин ----------

def test_car_in_adjacent_lane_is_not_dangerous():
    env, ego = road_with_ego()
    car = add_car(env, 10.0, 0.0, 25.0)  # соседняя полоса, 10 м впереди
    r = relation(ego, car)
    assert r["long_unsafe"]  # продольно близко
    assert not r["lat_unsafe"]  # но поперек 2 м > d_lat(0, 0)
    assert not r["dangerous"]
    assert r["lat_gap"] == pytest.approx(LANE_W - 2.0)


def test_car_moving_towards_ego_becomes_dangerous():
    env, ego = road_with_ego()
    # та же машина, но сдвигается к агенту со скоростью 2 м/с
    vy = 2.0
    car = add_car(env, 10.0, 0.0, math.hypot(25.0, vy), heading=math.atan2(vy, 25.0))
    r = relation(ego, car)
    assert r["d_lat"] == pytest.approx(d_min_lateral(0.0, -vy), abs=1e-9)
    assert r["lat_unsafe"] and r["dangerous"]


# ---------- запрет и отмена перестроения ----------

def test_lane_change_blocked_into_unsafe_gap():
    env, ego = road_with_ego()
    add_car(env, 10.0, 0.0, 25.0)  # в целевой полосе 10 м впереди
    lane_before = ego.target_lane_index
    ego.act("LANE_LEFT")
    assert ego.state["lc_requested"] and ego.state["lc_decision"] == "blocked"
    assert ego.target_lane_index == lane_before


def test_lane_change_allowed_into_safe_gap():
    env, ego = road_with_ego()
    add_car(env, 120.0, 0.0, 25.0)  # далеко впереди
    lane_before = ego.target_lane_index
    ego.act("LANE_LEFT")
    assert ego.state["lc_decision"] is None
    assert ego.target_lane_index != lane_before


def test_without_shield_lane_change_is_not_blocked():
    env, ego = road_with_ego(shield=False)
    add_car(env, 10.0, 0.0, 25.0)
    lane_before = ego.target_lane_index
    ego.act("LANE_LEFT")
    assert ego.state["lc_decision"] == "blocked"  # фильтр бы запретил (видно в логе)
    assert ego.target_lane_index != lane_before  # но без фильтра агент едет


def test_started_lane_change_is_cancelled_when_it_becomes_dangerous():
    env, ego = road_with_ego()
    origin = ego.target_lane_index
    ego.act("LANE_LEFT")
    assert ego.target_lane_index != origin
    for _ in range(5):  # агент начал смещаться влево
        ego.step(SIM_DT)
        ego.act()
    # в целевой полосе рядом появляется машина
    add_car(env, 8.0, 0.0, 25.0)
    ego.act()
    assert ego.state["lc_decision"] == "cancelled"
    assert ego.target_lane_index == origin


# ---------- подрезающий нарушитель ----------

def triggered_cutin(gamma, shield=False, seeds=range(12)):
    """Эпизоды подрезания, где маневр состоялся: (outcome, log, cutter)."""
    out = []
    for seed in seeds:
        scene = make_scene(seed, shield=shield, kind="cutin", gamma=gamma)
        outcome, log = run_episode(scene)
        if outcome["triggered"]:
            out.append((outcome, log, scene.violator))
    return out


@pytest.fixture(scope="module")
def cutins_noshield():
    return triggered_cutin(0.25)


def test_cutin_happens_in_most_scenes(cutins_noshield):
    assert len(cutins_noshield) >= 6


def test_cutin_trajectory_is_smooth(cutins_noshield):
    for _, _, cutter in cutins_noshield:
        c = cutter.cut
        y, vy, ay = map(np.array, (c["y"], c["vy"], c["ay"]))
        dy_total = c["y1"] - c["y0"]
        # длительность и конечное положение — центр полосы агента
        assert c["t"][-1] - c["t_start"] == pytest.approx(LC_DURATION, abs=SIM_DT)
        assert y[-1] == pytest.approx(c["y1"], abs=1e-9)
        # монотонно, без скачков: смещение за шаг не больше vy_max·dt
        assert np.all(np.diff(y) * np.sign(dy_total) >= -1e-12)
        vy_max = 1.875 * abs(dy_total) / LC_DURATION
        assert np.abs(np.diff(np.r_[c["y0"], y])).max() <= vy_max * SIM_DT + 1e-9
        assert np.abs(vy).max() == pytest.approx(vy_max, rel=0.01)
        # поперечная скорость непрерывна и в конце нулевая
        ay_max = 5.774 * abs(dy_total) / LC_DURATION**2
        assert np.abs(np.diff(vy)).max() <= ay_max * SIM_DT + 1e-9
        assert vy[-1] == pytest.approx(0.0, abs=1e-9)
        assert np.abs(ay).max() <= ay_max + 1e-6


def test_gap_at_line_crossing_matches_gamma(cutins_noshield):
    """Агент без фильтра едет с постоянной скоростью, прогноз точный:
    зазор на разметке = γ·d_min с точностью до 1 м. Ошибка — дискретность
    шага: маневр стартует на первом шаге после порога, пересечение разметки
    фиксируется на шаге после пересечения; при сближении ~5 м/с один шаг
    1/15 с — это ~0,33 м."""
    for outcome, _, _ in cutins_noshield:
        target = 0.25 * outcome["d_min_cross"]
        assert abs(outcome["gap_cross"] - target) < 1.0, outcome


def test_cutter_brakes_from_line_crossing(cutins_noshield):
    for outcome, _, cutter in cutins_noshield:
        e = cutter.events[0]
        assert e["t_start"] == pytest.approx(outcome["t_cross"])
        assert e["a_commanded"] == pytest.approx(-DEFAULT.a_max_brake)  # β = 1


@pytest.mark.parametrize("delay", [0.0, DEFAULT.rho])
def test_legal_cutin_gamma_1_no_collisions_with_shield(delay):
    """γ = 1 (подрезание с зазором d_min, затем торможение 5 м/с^2) законно
    по RSS: с фильтром столкновений быть не должно."""
    n = 0
    for seed in range(12):
        outcome, _ = run_episode(make_scene(seed, shield=True, reaction_delay=delay, kind="cutin", gamma=1.0))
        n += outcome["triggered"]
        assert not outcome["collision"], (seed, outcome)
    assert n >= 4
