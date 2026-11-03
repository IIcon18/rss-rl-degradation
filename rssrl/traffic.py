"""Фон контролируемого следования (основной β-сценарий, traffic="following").

Зачем. В обычном фоне highway-env машины перестраиваются по MOBIL и могут
встать перед агентом ближе d_min, а стартовая расстановка не учитывает
d_min вовсе (в 05_nominal_check у 52 % эпизодов зазор на первом шаге был
меньше d_min). Тогда столкновение может случиться не из-за β, а из-за
подрезания фона, и после расхождения действий агентов эти взаимодействия
у разных вариантов разные. В основном β-сценарии это исключаем:

1. Фон не перестраивается (enable_lane_change = False).
2. Фон держит дистанцию по RSS (RSSFollowerVehicle): ускорение по IDM, но
   если зазор до машины впереди в своей полосе меньше d_min (лемма 2 с
   учетом рывка), то через ρ — торможение не слабее a_min_brake, как у
   базового фильтра агента. Торможение фона не сильнее a_max_brake, рывок
   не больше J_MAX (как у BoundedIDMVehicle). Поэтому машина позади агента
   не въедет в него, если агент тормозит не сильнее a_max_brake.
3. Начальные состояния — по общему правилу от seed (place_following),
   все начальные зазоры в каждой полосе не меньше d_min с учетом рывка.

Правило расстановки (одинаковое для всех вариантов агента и всех β):
rng = numpy.random.default_rng(seed) — отдельный генератор, не генератор
среды;
- полоса агента — равновероятно одна из LANES; агент в точке EGO_X,
  скорость EGO_SPEED;
- в каждой полосе N_AHEAD машин впереди и N_BEHIND позади «якоря»: в полосе
  агента якорь — сам агент, в других — первая машина в EGO_X + U[-40, 40] м;
- скорость каждой машины U[21, 24] м/с (как у highway-env: 0,7–0,8 от
  предела 30 м/с), целевая скорость IDM — та же;
- зазор между соседями в полосе (бампер к бамперу) = d_min(v_задней,
  v_передней) · U[GAP_FACTOR] — то есть не меньше d_min.
Поперек машины соседних полос стоят по центрам полос (поперечный зазор
2 м > поперечного d_min при нулевых поперечных скоростях).
"""

import numpy as np

from rssrl.rss import d_min_longitudinal
from rssrl.shield import SHIELD_DEFAULT, SIM_DT
from rssrl.violators import BoundedIDMVehicle

EGO_X = 200.0  # м от начала дороги
EGO_SPEED = 25.0  # м/с, как у агента highway-env
SPEED_RANGE = (21.0, 24.0)  # м/с
N_AHEAD, N_BEHIND = 10, 2  # машин в каждой полосе впереди и позади якоря
GAP_FACTOR = (1.0, 2.0)  # зазор = d_min · U[1, 2]
ANCHOR_SPREAD = 40.0  # м, разброс якоря в соседних полосах


class RSSFollowerVehicle(BoundedIDMVehicle):
    """Фоновая машина контролируемого следования: IDM без перестроений,
    с продольной надлежащей реакцией RSS на машину впереди в своей полосе."""

    params = SHIELD_DEFAULT

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("enable_lane_change", False)
        super().__init__(*args, **kwargs)
        self.clock = 0.0
        self.last_safe = 0.0

    def act(self, action=None):
        super().act(action)
        if self.crashed:
            return
        p = self.params
        a = min(self.action["acceleration"], p.a_max_accel)
        front, _ = self.road.neighbour_vehicles(self, self.lane_index)
        danger = False
        if front is not None:
            s_self = self.lane.local_coordinates(self.position)[0]
            s_front = self.lane.local_coordinates(front.position)[0]
            gap = s_front - s_self - (self.LENGTH + front.LENGTH) / 2
            danger = gap < d_min_longitudinal(max(0.0, self.speed), max(0.0, front.speed), p)
        if not danger:
            self.last_safe = self.clock
        elif self.clock - self.last_safe + SIM_DT > p.rho + 1e-9:
            a = min(a, -p.a_min_brake)
        self.action["acceleration"] = max(a, -p.a_max_brake)

    def step(self, dt):
        super().step(dt)
        self.clock += dt


def place_following(road, ego, seed, lanes=None):
    """Расставить фон контролируемого следования по правилу от seed.

    Убирает с дороги все машины, кроме ego, ставит ego в выбранную полосу
    в точку EGO_X со скоростью EGO_SPEED и добавляет RSSFollowerVehicle.
    Возвращает список новых машин (в порядке добавления на дорогу).
    """
    rng = np.random.default_rng(seed)
    p = SHIELD_DEFAULT
    from_ = next(iter(road.network.graph))  # прямая дорога highway-v0: одна пара узлов
    to_ = next(iter(road.network.graph[from_]))
    n_lanes = len(road.network.graph[from_][to_])
    lanes = list(range(n_lanes)) if lanes is None else lanes
    ego_lane = int(rng.integers(n_lanes))

    lane = road.network.get_lane((from_, to_, ego_lane))
    ego.position = lane.position(EGO_X, 0.0)
    ego.heading = lane.heading_at(EGO_X)
    ego.speed = EGO_SPEED
    ego.lane_index = ego.target_lane_index = (from_, to_, ego_lane)
    ego.lane = lane
    road.vehicles = [ego]

    def gap(v_rear, v_front):
        return d_min_longitudinal(v_rear, v_front, p) * rng.uniform(*GAP_FACTOR)

    out = []
    for i in lanes:
        lane = road.network.get_lane((from_, to_, i))
        length = ego.LENGTH
        if i == ego_lane:
            anchor = None
            x0, v0 = EGO_X, EGO_SPEED
        else:
            x0, v0 = EGO_X + rng.uniform(-ANCHOR_SPREAD, ANCHOR_SPREAD), rng.uniform(*SPEED_RANGE)
            anchor = (x0, v0)
        column = [anchor] if anchor else []
        # вперед: передняя машина впереди задней на зазор + длину
        x, v = x0, v0
        for _ in range(N_AHEAD):
            v_new = rng.uniform(*SPEED_RANGE)
            x = x + length + gap(v, v_new)
            v = v_new
            column.append((x, v))
        # назад: задняя позади передней
        x, v = x0, v0
        for _ in range(N_BEHIND):
            v_new = rng.uniform(*SPEED_RANGE)
            x = x - length - gap(v_new, v)
            v = v_new
            column.append((x, v))
        for x, v in column:
            car = RSSFollowerVehicle(road, lane.position(x, 0.0), heading=lane.heading_at(x), speed=v,
                                     target_speed=v)
            car.DELTA = rng.uniform(*car.DELTA_RANGE)
            road.vehicles.append(car)
            out.append(car)
    return out


def initial_gap_ratios(road, p=SHIELD_DEFAULT):
    """Для каждой пары соседних машин в каждой полосе: зазор / d_min
    (d_min с учетом рывка по умолчанию). Нужна проверка «все >= 1»."""
    by_lane = {}
    for v in road.vehicles:
        by_lane.setdefault(v.lane_index, []).append(v)
    ratios = []
    for lane_index, cars in by_lane.items():
        lane = road.network.get_lane(lane_index)
        cars.sort(key=lambda v: lane.local_coordinates(v.position)[0])
        for rear, front in zip(cars, cars[1:]):
            g = (lane.local_coordinates(front.position)[0] - lane.local_coordinates(rear.position)[0]
                 - (rear.LENGTH + front.LENGTH) / 2)
            d = d_min_longitudinal(max(0.0, rear.speed), max(0.0, front.speed), p)
            ratios.append(g / d if d > 0 else np.inf)
    return np.array(ratios)

