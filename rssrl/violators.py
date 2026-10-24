"""Машины-участники: фоновые машины, тормозящий и подрезающий нарушители.

BoundedIDMVehicle — фоновая машина. Обычная IDM/MOBIL-машина highway-env,
но с двумя исправлениями, чтобы она соблюдала допущения RSS:
- тормозит не сильнее a_max_brake = 5 м/с^2 (в highway-env обрезка 6 м/с^2);
- не уезжает задом: скорость не опускается ниже нуля.

BrakingViolator — нарушитель. До момента нарушения ведет себя как фоновая
машина. В момент нарушения перестает слушать IDM и сам задает себе
ускорение -β·a_max_brake, держит полосу и тормозит до полной остановки.
После остановки стоит до конца эпизода (stay_stopped=True) или снова
едет как фоновая машина (stay_stopped=False). Это важно: если лидер сразу
трогается, догоняющий агент получает лишний запас дистанции.
При β = 1 это самое сильное торможение, которое RSS еще допускает;
при β > 1 — нарушение допущения. Потолок β = 2 (10 м/с^2) — выбранный
предел испытаний, а не физический закон.

Каждое торможение логируется: заданное ускорение и фактическое, посчитанное
по изменению скорости за шаг симуляции.
"""

import math

from highway_env.vehicle.behavior import IDMVehicle

from rssrl.rss import DEFAULT


def _no_reverse(acceleration, speed, dt):
    """Ускорение, при котором скорость за шаг dt не станет отрицательной."""
    return max(acceleration, -speed / dt)


class BoundedIDMVehicle(IDMVehicle):
    """Фоновая машина: IDM/MOBIL с торможением не сильнее a_max_brake."""

    BRAKE_LIMIT = DEFAULT.a_max_brake  # м/с^2

    def act(self, action=None):
        super().act(action)
        if not self.crashed:
            self.action["acceleration"] = max(self.action["acceleration"], -self.BRAKE_LIMIT)

    def step(self, dt):
        if not self.crashed:
            self.action["acceleration"] = _no_reverse(self.action["acceleration"], self.speed, dt)
        super().step(dt)


class BrakingViolator(BoundedIDMVehicle):
    """Лидер, который в заданный момент тормозит с β·a_max_brake до остановки.

    Нарушение запускается одним из способов:
    - по времени: trigger_time, с от начала существования машины;
    - по условию: trigger(нарушитель) -> bool, проверяется на каждом шаге
      (например, «агент подъехал на нужную дистанцию»);
    - вызовом start_violation() извне.
    """

    def __init__(self, *args, beta=1.0, trigger_time=None, trigger=None, stay_stopped=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.stay_stopped = stay_stopped
        self.trigger_time = trigger_time
        self.trigger = trigger
        self.clock = 0.0  # собственные часы, с
        self.violating = False  # сейчас тормозит
        self.finished = False  # уже отторможен до нуля
        self.events = []  # по одному словарю на каждое нарушение

    @classmethod
    def replace(cls, vehicle, beta=1.0, trigger_time=None, trigger=None, stay_stopped=True):
        """Заменить машину на дороге нарушителем в том же состоянии.

        Нарушитель встает на то же место в списке машин дороги, поэтому
        порядок обработки машин (и вся сцена) остается прежним.
        """
        new = cls(
            vehicle.road,
            vehicle.position.copy(),
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
            timer=getattr(vehicle, "timer", None),
            beta=beta,
            trigger_time=trigger_time,
            trigger=trigger,
            stay_stopped=stay_stopped,
        )
        new.DELTA = vehicle.DELTA  # характер, выпавший при генерации сцены
        if hasattr(vehicle, "uid"):
            new.uid = vehicle.uid
        new.lane_index, new.lane = vehicle.lane_index, vehicle.lane
        vehicles = vehicle.road.vehicles
        vehicles[vehicles.index(vehicle)] = new
        return new

    @property
    def commanded_acceleration(self):
        return -self.beta * DEFAULT.a_max_brake

    def start_violation(self):
        if self.violating or self.finished or self.crashed:
            return
        self.violating = True
        self.target_lane_index = self.lane_index  # во время торможения не перестраивается
        self.events.append(
            {
                "beta": self.beta,
                "t_start": self.clock,
                "v_start": self.speed,
                "x_start": float(self.position[0]),
                "a_commanded": self.commanded_acceleration,
                "t": [],  # время каждого шага торможения
                "speed": [],  # скорость в начале шага
                "a_set": [],  # ускорение, выставленное на шаге
                "a_actual": [],  # (v_после - v_до) / dt
                "lane": [],  # полоса на каждом шаге
                "t_stop": None,
                "crashed": False,
            }
        )

    def act(self, action=None):
        # допуск 1e-9: сумма шагов 1/15 дает 0.9999... вместо 1.0
        if self.trigger_time is not None and self.clock >= self.trigger_time - 1e-9:
            self.start_violation()
        if self.trigger is not None and not (self.violating or self.finished) and self.trigger(self):
            self.start_violation()
        if self.crashed:
            return
        if self.violating:
            # в обход IDM и его обрезок
            self._hold_lane(self.commanded_acceleration)
        elif self.finished and self.stay_stopped:
            self._hold_lane(0.0)
        else:
            super().act(action)

    def _hold_lane(self, acceleration):
        """Держать текущую полосу с заданным ускорением."""
        self.follow_road()
        steering = self.steering_control(self.target_lane_index)
        steering = max(-self.MAX_STEERING_ANGLE, min(self.MAX_STEERING_ANGLE, steering))
        self.action = {"steering": steering, "acceleration": acceleration}

    def step(self, dt):
        if not self.violating:
            super().step(dt)
            self.clock += dt
            return
        event = self.events[-1]
        v_before = self.speed
        if not self.crashed:
            # единственная поправка: на последнем шаге не уехать задом
            self.action["acceleration"] = _no_reverse(self.commanded_acceleration, v_before, dt)
        a_set = self.action["acceleration"]
        IDMVehicle.step(self, dt)  # без обрезки BoundedIDMVehicle
        event["t"].append(self.clock)
        event["speed"].append(v_before)
        event["a_set"].append(a_set)
        event["a_actual"].append((self.speed - v_before) / dt)
        event["lane"].append(self.lane_index)
        self.clock += dt
        if self.crashed:
            event["crashed"] = True
            self.violating = False
        elif self.speed <= 1e-9:
            self.speed = 0.0
            event["t_stop"] = self.clock
            self.violating = False
            self.finished = True


def _quintic(tau):
    """Гладкий профиль 0 -> 1 (полином 5-й степени) и его 1-я и 2-я производные
    по tau. Скорость и ускорение в начале и в конце равны нулю."""
    q = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    dq = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    ddq = 60 * tau - 180 * tau**2 + 120 * tau**3
    return q, dq, ddq


class CutInViolator(BrakingViolator):
    """Подрезающий нарушитель.

    До начала маневра — обычная фоновая машина в соседней полосе.
    start_cut(полоса агента) запускает перестроение длительностью
    lc_duration секунд по гладкой траектории: поперечное смещение меняется
    по полиному 5-й степени, поэтому положение, поперечная скорость и
    поперечное ускорение непрерывны (никакой телепортации). Пиковая
    поперечная скорость 1.875·Δ/T (при Δ = 4 м, T = 3 с — 2.5 м/с).
    Продольная скорость во время маневра постоянна.

    В момент пересечения разметки (середина маневра) вызывается on_cross —
    сцена измеряет там зазор до агента. Если brake_on_cross, нарушитель
    сразу начинает тормозить с β·a_max_brake (при β = 1 — самое сильное
    торможение, которое допускает RSS) и тормозит до остановки.
    """

    def __init__(self, *args, lc_duration=3.0, brake_on_cross=True, on_cross=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lc_duration = lc_duration
        self.brake_on_cross = brake_on_cross
        self.on_cross = on_cross
        self.cutting = False
        self.cut = None  # параметры и траектория маневра

    def start_cut(self, target_lane_index):
        base = self.lane  # координаты маневра — исходная полоса
        s, y = base.local_coordinates(self.position)
        target = self.road.network.get_lane(target_lane_index)
        _, y_target = base.local_coordinates(target.position(s, 0.0))
        self.cutting = True
        self.target_lane_index = target_lane_index
        self.cut = {
            "base": base,
            "s": s,
            "y0": y,
            "y1": y_target,
            "y_line": y_target / 2,  # разметка между центрами полос
            "vx": self.speed * math.cos(self.heading - base.heading_at(s)),
            "t_start": self.clock,
            "t_cross": None,
            "t": [], "y": [], "vy": [], "ay": [],
        }

    def act(self, action=None):
        if not self.cutting:
            super().act(action)
            return
        if self.crashed:
            return
        a = self.commanded_acceleration if self.violating else 0.0
        self.action = {"steering": 0.0, "acceleration": a}

    def step(self, dt):
        if not self.cutting:
            super().step(dt)
            return
        c = self.cut
        if self.crashed:
            self.cutting = False
            IDMVehicle.step(self, dt)
            self.clock += dt
            return
        tau = min(1.0, (self.clock + dt - c["t_start"]) / self.lc_duration)
        q, dq, ddq = _quintic(tau)
        dy_total = c["y1"] - c["y0"]
        y_new = c["y0"] + dy_total * q
        vy = dy_total * dq / self.lc_duration
        ay = dy_total * ddq / self.lc_duration**2

        vx = c["vx"]
        a = _no_reverse(self.commanded_acceleration, vx, dt) if self.violating else 0.0
        vx_new = max(0.0, vx + a * dt)
        c["s"] += (vx + vx_new) / 2 * dt
        y_old = self.cut["y"][-1] if self.cut["y"] else c["y0"]
        c["vx"] = vx_new

        base = c["base"]
        self.position = base.position(c["s"], y_new)
        self.heading = base.heading_at(c["s"]) + math.atan2(vy, max(vx_new, 1e-6))
        self.speed = math.hypot(vx_new, vy)
        if self.violating:
            e = self.events[-1]
            e["t"].append(self.clock)
            e["speed"].append(vx)
            e["a_set"].append(a)
            e["a_actual"].append((vx_new - vx) / dt)
            e["lane"].append(self.lane_index)
        self.clock += dt
        self.timer += dt
        self.on_state_update()
        c["t"].append(self.clock)
        c["y"].append(y_new)
        c["vy"].append(vy)
        c["ay"].append(ay)

        # пересечение разметки
        if c["t_cross"] is None and (y_old - c["y_line"]) * (y_new - c["y_line"]) <= 0 and y_new != y_old:
            c["t_cross"] = self.clock
            if self.on_cross is not None:
                self.on_cross(self)
            if self.brake_on_cross:
                self.start_violation()
        if self.violating and vx_new <= 1e-9:
            self.events[-1]["t_stop"] = self.clock
            self.violating = False
            self.finished = True
        if tau >= 1.0:
            self.cutting = False
            self.heading = base.heading_at(c["s"])
            self.speed = vx_new
