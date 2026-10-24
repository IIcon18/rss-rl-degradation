"""Этап 2: фоновые машины и тормозящий нарушитель (rssrl/violators.py)."""

import numpy as np
import pytest

from rssrl.scenarios import find_leader, make_env
from rssrl.violators import BoundedIDMVehicle, BrakingViolator

IDLE = 1
DT = 1 / 15
BETAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def violator_scene(beta, seed=0, trigger_time=1.0, steps=8):
    """Сцена, где лидер агента — нарушитель. Агент не сталкивается ни с кем
    (collidable = False), чтобы столкновение не прервало торможение лидера:
    здесь проверяем только самого нарушителя."""
    env = make_env(seed)
    env.unwrapped.vehicle.collidable = False
    vio = BrakingViolator.replace(find_leader(env), beta=beta, trigger_time=trigger_time)
    lane0 = vio.lane_index
    for _ in range(steps):
        env.step(IDLE)
    env.close()
    return vio, lane0


def test_background_vehicles_are_bounded():
    env = make_env(0)
    others = [v for v in env.unwrapped.road.vehicles if v is not env.unwrapped.vehicle]
    assert others and all(isinstance(v, BoundedIDMVehicle) for v in others)
    env.close()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_background_never_brakes_harder_than_5_and_never_reverses(seed):
    env = make_env(seed)
    road = env.unwrapped.road
    ego = env.unwrapped.vehicle
    # шагаем по шагам симуляции, чтобы измерить ускорение каждой машины
    for _ in range(20 * 15):
        before = {id(v): v.speed for v in road.vehicles}
        road.act()
        road.step(DT)
        for v in road.vehicles:
            if v is ego or v.crashed:
                continue
            a = (v.speed - before[id(v)]) / DT
            assert a >= -5.0 - 1e-9, (seed, a)
            assert v.speed >= 0.0
    env.close()


@pytest.mark.parametrize("beta", BETAS)
def test_violator_brakes_exactly_as_commanded(beta):
    vio, _ = violator_scene(beta)
    assert len(vio.events) == 1
    e = vio.events[0]
    assert e["t_start"] == pytest.approx(1.0)
    assert e["a_commanded"] == pytest.approx(-5.0 * beta)
    assert not e["crashed"] and e["t_stop"] is not None
    a = np.array(e["a_actual"])
    # все шаги, кроме последнего, — ровно заданное ускорение
    np.testing.assert_allclose(a[:-1], -5.0 * beta, atol=1e-9)
    # последний шаг — дотормаживание до нуля, не сильнее заданного
    assert -5.0 * beta - 1e-9 <= a[-1] <= 0.0


@pytest.mark.parametrize("beta", BETAS)
def test_violator_stops_without_reversing_and_keeps_lane(beta):
    vio, lane0 = violator_scene(beta)
    e = vio.events[0]
    assert min(e["speed"]) >= 0.0
    assert vio.speed >= 0.0
    # во время торможения полосу не меняет (после остановки снова едет по IDM/MOBIL)
    assert all(lane == lane0 for lane in e["lane"])
    # время торможения совпадает с v / (β·5) с точностью до шага
    assert e["t_stop"] - e["t_start"] == pytest.approx(e["v_start"] / (5.0 * beta), abs=DT)


def test_violator_is_identical_to_background_before_trigger():
    """Замена лидера нарушителем не меняет сцену до момента нарушения
    (основа для парных сравнений на общих сценах)."""

    def positions(replace):
        env = make_env(7)
        if replace:
            BrakingViolator.replace(find_leader(env), beta=2.0, trigger_time=3.0)
        track = []
        for _ in range(2):  # 2 с, нарушение начнется только на 3-й секунде
            env.step(IDLE)
            track.append(np.array([v.position for v in env.unwrapped.road.vehicles]))
        env.close()
        return np.array(track)

    np.testing.assert_array_equal(positions(False), positions(True))


def test_manual_trigger():
    env = make_env(0)
    env.unwrapped.vehicle.collidable = False
    vio = BrakingViolator.replace(find_leader(env), beta=1.6)
    env.step(IDLE)
    assert vio.events == []  # без trigger_time сам не начинает
    vio.start_violation()
    env.step(IDLE)
    assert vio.events[0]["a_commanded"] == pytest.approx(-8.0)
    assert vio.events[0]["a_actual"][0] == pytest.approx(-8.0)
    env.close()
