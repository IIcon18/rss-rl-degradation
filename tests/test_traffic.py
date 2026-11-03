"""Шаг 2 (письмо 29.10): сцена контролируемого следования traffic="following"."""

import numpy as np
import pytest

from rssrl.rss import d_min_longitudinal
from rssrl.scenarios import make_scene, run_episode
from rssrl.shield import SHIELD_DEFAULT
from rssrl.traffic import RSSFollowerVehicle, initial_gap_ratios

SEEDS = range(20)


@pytest.mark.parametrize("seed", SEEDS)
def test_initial_gaps_not_below_dmin(seed):
    """Правило расстановки: в каждой полосе все зазоры >= d_min с учетом рывка."""
    scene = make_scene(seed, kind="nominal", traffic="following")
    ratios = initial_gap_ratios(scene.env.unwrapped.road)
    assert len(ratios) > 30
    assert ratios.min() >= 1.0 - 1e-12
    scene.env.close()


def test_placement_is_deterministic_and_seed_dependent():
    def snapshot(seed):
        scene = make_scene(seed, kind="nominal", traffic="following")
        out = [(round(v.position[0], 9), round(v.position[1], 9), round(v.speed, 9))
               for v in scene.env.unwrapped.road.vehicles]
        scene.env.close()
        return out

    assert snapshot(3) == snapshot(3)
    assert snapshot(3) != snapshot(4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_background_never_changes_lane(seed):
    scene = make_scene(seed, kind="nominal", traffic="following")
    road = scene.env.unwrapped.road
    lanes = {id(v): v.lane_index for v in road.vehicles if v is not scene.ego}
    assert all(isinstance(v, RSSFollowerVehicle) for v in road.vehicles if v is not scene.ego)
    _, log = run_episode(scene)
    for v in road.vehicles:
        if v is not scene.ego:
            assert v.lane_index == lanes[id(v)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_first_logged_step_gap_not_below_dmin(seed):
    """Как в 05_nominal_check: на первом записанном шаге зазор агента до
    машины впереди не меньше d_min фильтра."""
    _, log = run_episode(make_scene(seed, kind="nominal", traffic="following"))
    first = log.iloc[0]
    assert not first["dangerous"]
    assert first["gap"] >= first["d_min"]


def test_follower_brakes_after_rho_when_too_close():
    """Фоновая машина: при зазоре меньше d_min через ρ — торможение не слабее
    a_min_brake (правило RSS для фона)."""
    scene = make_scene(0, kind="nominal", traffic="following")
    road = scene.env.unwrapped.road
    car = next(v for v in road.vehicles if isinstance(v, RSSFollowerVehicle))
    front, _ = road.neighbour_vehicles(car, car.lane_index)
    # подвинуть машину вплотную к передней
    s_f = car.lane.local_coordinates(front.position)[0]
    car.position = car.lane.position(s_f - car.LENGTH - 5.0, 0.0)
    car.speed = front.speed
    assert 5.0 < d_min_longitudinal(car.speed, front.speed, SHIELD_DEFAULT)
    accs = []
    for _ in range(10):
        car.act()
        accs.append(car.action["acceleration"])
        car.step(1 / 15)
        front.step(1 / 15)  # передняя едет дальше (act не вызываем: держит ускорение)
    assert accs[-1] <= -SHIELD_DEFAULT.a_min_brake + 1e-9
    assert all(a >= -SHIELD_DEFAULT.a_max_brake - 1e-9 for a in accs)
    scene.env.close()


def test_robustness_traffic_is_old_background():
    """traffic="robustness" — прежний фон highway-env с MOBIL."""
    scene = make_scene(0, kind="nominal", traffic="robustness")
    assert not any(isinstance(v, RSSFollowerVehicle) for v in scene.env.unwrapped.road.vehicles)
    ratios = initial_gap_ratios(scene.env.unwrapped.road)
    scene.env.close()
    assert np.isfinite(ratios).any()
