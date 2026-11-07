"""Запись временных рядов эпизода по шагам симуляции.

EpisodeLogger подключается к дороге и после каждого шага симуляции
(после движения машин и проверки столкновений) записывает строку:
- агент: положение, полоса, скорость, фактическое ускорение,
  ускорение от политики, исполненное ускорение, руль;
- решение фильтра: машина впереди, зазор, её скорость, d_min, опасно ли,
  вмешался ли фильтр;
- нарушитель (если есть): скорость, фактическое ускорение, зазор до агента,
  тормозит ли он сейчас;
- столкновение и с кем: партнер — машина, с которой у агента есть фактический
  контакт (см. _contacts), в том числе уже разбитая раньше.
Фактические ускорения считаются по изменению скорости за шаг.
"""

import pandas as pd

from rssrl.shield import velocity_in_lane


class EpisodeLogger:
    def __init__(self, env, ego, violator=None):
        self.env = env
        self.ego = ego
        self.violator = None
        self.dt = 1 / env.unwrapped.config["simulation_frequency"]
        self.road = env.unwrapped.road
        self.rows = []
        self.t = 0.0
        self.crash_partners = None
        self._prev_ego_speed = ego.speed
        self._prev_vio_speed = None
        if violator is not None:
            self.set_violator(violator)
        self._will_touch = []  # uid машин, с которыми контакт предсказан на прошлом шаге
        # дорога после своего шага вызывает record()
        original_step = self.road.step

        def step_and_record(dt):
            original_step(dt)
            self.record()

        self.road.step = step_and_record

    def set_violator(self, violator):
        """Начать записывать нарушителя (до этого его колонки пустые)."""
        self.violator = violator
        self._prev_vio_speed = velocity_in_lane(violator, self.ego.lane)[0]

    def _gap_to_ego(self, vehicle):
        """Продольный зазор между бамперами (впереди агента > 0)."""
        lane = self.ego.lane
        s_ego, _ = lane.local_coordinates(self.ego.position)
        s_v, _ = lane.local_coordinates(vehicle.position)
        dist = s_v - s_ego
        half = (self.ego.LENGTH + vehicle.LENGTH) / 2
        return dist - half if dist >= 0 else dist + half

    def _contacts(self):
        """Машины, с которыми у агента контакт сейчас (многоугольники кузовов
        пересекаются), и машины, с которыми он будет на следующем шаге.

        highway-env ставит crashed двумя путями: кузова уже пересеклись, или на
        прошлом шаге пересечение было предсказано, и на этом шаге машины
        раздвинуты импульсом impact (тогда сейчас кузова могут не пересекаться).
        Поэтому партнер на шаге удара — объединение текущего контакта и
        предсказанного на прошлом шаге. Был ли партнер разбит раньше, не важно."""
        now, soon = [], []
        for v in self.road.vehicles:
            if v is self.ego:
                continue
            uid = getattr(v, "uid", None)
            hit_a, will_a, _ = self.ego._is_colliding(v, self.dt)
            hit_b, will_b, _ = v._is_colliding(self.ego, self.dt)
            if hit_a or hit_b:
                now.append(uid)
            if will_a or will_b:
                soon.append(uid)
        return now, soon

    def record(self):
        self.t += self.dt
        ego = self.ego
        row = {
            "t": self.t,
            "ego_x": float(ego.position[0]),
            "ego_y": float(ego.position[1]),
            "ego_lane": ego.lane_index[2],
            "ego_v": ego.speed,
            "ego_a": (ego.speed - self._prev_ego_speed) / self.dt,
            "crashed": ego.crashed,
            "on_road": ego.on_road,
        }
        self._prev_ego_speed = ego.speed
        row.update(ego.state)

        vio = self.violator
        if vio is not None:
            vx, vy = velocity_in_lane(vio, ego.lane)
            row.update(
                {
                    "vio_v": vx,  # продольная скорость
                    "vio_a": (vx - self._prev_vio_speed) / self.dt,
                    "vio_y": ego.lane.local_coordinates(vio.position)[1],
                    "vio_vy": vy,
                    "vio_gap": self._gap_to_ego(vio),
                    "vio_violating": vio.violating,  # состояние после шага
                    # нарушал ли на этом шаге (к нему относится vio_a)
                    "vio_violating_step": getattr(vio, "violating_step", vio.violating),
                    "vio_is_front": ego.state.get("front_id") == getattr(vio, "uid", -1),
                }
            )
            self._prev_vio_speed = vx

        touching, will_touch = self._contacts()
        if ego.crashed and self.crash_partners is None:
            self.crash_partners = touching + [u for u in self._will_touch if u not in touching]
            row["crash_partners"] = self.crash_partners
        self._will_touch = will_touch
        self.rows.append(row)

    def frame(self):
        return pd.DataFrame(self.rows)
