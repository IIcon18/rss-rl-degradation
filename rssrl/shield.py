"""RSS-фильтр на нижнем уровне управления.

ShieldedVehicle — машина агента. Политика (PPO или пилотная «постоянная
скорость») выбирает высокоуровневое действие раз в секунду, а машина
переводит его в ускорение и руль на каждом шаге симуляции (15 раз в секунду).
Фильтр стоит между этим переводом и физикой.

Что фильтр видит: все машины в радиусе 150 м по точным данным симулятора.
Для каждой считается отношение с агентом (relation): продольный и поперечный
зазоры между кузовами, d_min по леммам 2 и 4, небезопасно ли продольно и
поперечно. По определению RSS ситуация ОПАСНАЯ, только если небезопасно
и продольно, и поперечно. Например, машина в соседней полосе в 10 м впереди
продольно близко, но поперечно далеко — не опасна. Если она начинает
перестраиваться к агенту, поперечный d_min растет, и ситуация становится
опасной еще до пересечения разметки.

Правильная реакция:
1. Продольная. Если опасна машина впереди — тормозить не слабее a_min_brake
   (если политика сама тормозит сильнее, сильнее и тормозим), держать,
   пока опасных машин впереди не останется, затем вернуть управление.
   Экстренный режим: если по прогнозу даже торможение a_min_brake не спасает
   от столкновения, фильтр тормозит с физическим пределом машины
   A_PHYS_BRAKE = 8 м/с^2 (rssrl/dynamics.py) и держит его, пока опасность
   не пройдет. Прогноз (brake_insufficient): агент от своего текущего
   ускорения выходит на -a_min_brake с рывком J_MAX и тормозит до остановки;
   машина впереди с этого момента тормозит с max(a_max_brake, ее текущее
   фактическое торможение) — то есть по допущению RSS, а если она уже
   тормозит сильнее допущения, то так, как тормозит. Если зазор в прогнозе
   доходит до нуля — a_min_brake не хватает.
   Режим фильтра на каждом шаге пишется в лог (shield_mode): None — опасности
   нет; "delay" — опасно, но идет время реакции ρ; "brake" — торможение
   a_min_brake; "emergency" — экстренное торможение.
2. Перестроения. Новое перестроение политики запрещается, если в целевой
   полосе есть машина, с которой продольно небезопасно (после перестроения
   ситуация стала бы опасной). Начатое перестроение отменяется (агент
   возвращается в исходную полосу), если с машиной в целевой полосе стало
   опасно, — но только пока агент не пересек разметку и только если
   вернуться безопасно (в исходной полосе никто не оказался ближе d_min).
   После пересечения разметки перестроение считается завершенным, и на
   опасность впереди фильтр отвечает торможением.

Задержка реакции (reaction_delay), касается продольного торможения:
- ρ = 0,5 с (основной режим, по умолчанию): пока не прошло ρ после
  последнего безопасного момента, исполняется команда политики
  (не больше a_max_accel), потом торможение. Так ведет себя машина в
  предположении леммы 2;
- 0 (дополнительный вариант, «мгновенный фильтр»): фильтр тормозит уже на
  текущем шаге симуляции. ρ тогда остается только запасом внутри формулы
  d_min (лемма разрешает задней машине ρ секунд разгоняться, а фильтр
  сразу тормозит).
Отсчет идет от последнего безопасного шага, а не от шага, где опасность
заметили: лемма гарантирует безопасность, только если торможение началось
не позже чем через ρ после момента, когда зазор еще был >= d_min.
На сетке шагов 1/15 с это 7 шагов (0,467 с): 8 шагов было бы уже 0,533 > ρ.
Запрет и отмена перестроения действуют сразу (это не торможение, а отказ
от маневра).

Пределы машины агента:
- команда политики обрезается до [-a_max_brake, a_max_accel] = [-5, 3] м/с^2
  (с фильтром и без одинаково). Верхний предел 3 м/с^2 нужен, чтобы агент
  не разгонялся сильнее, чем предполагает формула d_min;
- сильнее 5 м/с^2 тормозит только экстренный режим фильтра (до 8 м/с^2);
- ускорение меняется не быстрее J_MAX = 10 м/с^3 (предел рывка): чтобы из
  движения с +3 м/с^2 выйти на -4 м/с^2, нужно 0,7 с. Это время добавляется
  к ρ, а лемма 2 как в статье его не учитывает (она считает, что торможение
  a_min_brake начинается мгновенно). Поэтому фильтр по умолчанию считает
  d_min по лемме 2 с учетом рывка (SHIELD_DEFAULT, rssrl/rss.py): с d_min
  как в статье (LEMMA2) агент при β = 1 въезжал в стоящую машину на 2–4 м/с;
- задом не едет.
"""

import math

from highway_env.vehicle.controller import MDPVehicle

from dataclasses import replace

from rssrl.dynamics import A_PHYS_BRAKE, J_MAX, limit_jerk
from rssrl.rss import DEFAULT, d_min_lateral, d_min_longitudinal

SENSE_RADIUS = 150.0  # м, фильтр видит все машины в этом радиусе
# Параметры фильтра по умолчанию: d_min по лемме 2 с учетом рывка J_MAX
# (машина агента набирает торможение не мгновенно). LEMMA2 — d_min как в
# статье, для сравнения.
SHIELD_DEFAULT = replace(DEFAULT, jerk=J_MAX)
LEMMA2 = DEFAULT
SIM_DT = 1 / 15  # шаг симуляции highway-v0, с


def velocity_in_lane(vehicle, lane):
    """Скорость машины в координатах полосы: (вдоль, поперек), м/с."""
    s, _ = lane.local_coordinates(vehicle.position)
    angle = vehicle.heading - lane.heading_at(s)
    return vehicle.speed * math.cos(angle), vehicle.speed * math.sin(angle)


def longitudinal_speed(vehicle, lane):
    """Проекция скорости машины на направление полосы, м/с."""
    return velocity_in_lane(vehicle, lane)[0]


def relation(ego, other, p=DEFAULT, y_offset=0.0):
    """RSS-отношение агента с другой машиной.

    Оси — координаты полосы агента. y_offset сдвигает агента поперек
    (нужно, чтобы оценить «а если бы агент уже был в целевой полосе»).
    Зазоры — между кузовами; отрицательный зазор значит перекрытие
    по этой оси.
    """
    lane = ego.lane
    s_e, y_e = lane.local_coordinates(ego.position)
    s_o, y_o = lane.local_coordinates(other.position)
    y_e += y_offset
    ds, dy = s_o - s_e, y_o - y_e
    vl_e, vt_e = velocity_in_lane(ego, lane)
    vl_o, vt_o = velocity_in_lane(other, lane)

    long_gap = abs(ds) - (ego.LENGTH + other.LENGTH) / 2
    lat_gap = abs(dy) - (ego.WIDTH + other.WIDTH) / 2
    ahead = ds > 0

    if ahead:  # агент сзади
        d_long = d_min_longitudinal(max(0.0, vl_e), max(0.0, vl_o), p)
    else:  # агент спереди
        d_long = d_min_longitudinal(max(0.0, vl_o), max(0.0, vl_e), p)
    # ось поперечной леммы — от агента к другой машине
    sign = 1.0 if dy >= 0 else -1.0
    d_lat = d_min_lateral(sign * vt_e, sign * vt_o, p)

    long_unsafe = long_gap < d_long
    lat_unsafe = lat_gap < d_lat
    return {
        "vehicle": other,
        "ahead": ahead,
        "dy": dy,
        "long_gap": long_gap,
        "lat_gap": lat_gap,
        "d_long": d_long,
        "d_lat": d_lat,
        "v_long": vl_o,
        "long_unsafe": long_unsafe,
        "lat_unsafe": lat_unsafe,
        "dangerous": long_unsafe and lat_unsafe,
    }


def neighbours(ego, radius=SENSE_RADIUS):
    """Все машины дороги в радиусе radius, кроме агента."""
    r2 = radius * radius
    out = []
    for v in ego.road.vehicles:
        if v is ego:
            continue
        dx, dy = v.position[0] - ego.position[0], v.position[1] - ego.position[1]
        if dx * dx + dy * dy <= r2:
            out.append(v)
    return out


def critical_ahead(relations):
    """Самая критичная машина впереди среди поперечно небезопасных:
    с наименьшим запасом long_gap - d_long. None, если таких нет."""
    candidates = [r for r in relations if r["ahead"] and r["lat_unsafe"]]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["long_gap"] - r["d_long"])


def brake_insufficient(gap, v_rear, v_front, a_rear, b_front, b_rear, dt=SIM_DT,
                       j_max=None, horizon=15.0):
    """Прогноз: хватит ли задней машине торможения b_rear.

    Задняя машина (скорость v_rear, текущее ускорение a_rear) выходит на
    ускорение -b_rear с рывком j_max (по умолчанию J_MAX) и тормозит до остановки. Передняя
    (v_front) с этого момента тормозит с постоянным b_front до остановки.
    Интегрирование — как в highway-env (сначала положение по старой
    скорости, потом скорость). True, если зазор gap доходит до нуля.
    """
    x_r, x_f = 0.0, gap
    v_r, v_f, a = v_rear, v_front, a_rear
    for _ in range(int(horizon / dt)):
        a = limit_jerk(-b_rear, a, dt, j_max)
        a_eff = max(a, -v_r / dt)
        x_r += v_r * dt
        x_f += v_f * dt
        v_r += a_eff * dt
        v_f = max(0.0, v_f - b_front * dt)
        if x_f - x_r <= 0:
            return True
        if v_r <= 1e-9:
            return False
    return False


def find_front(ego, radius=SENSE_RADIUS, p=DEFAULT):
    """Критичная машина впереди и зазор до нее (или None, inf)."""
    r = critical_ahead([relation(ego, v, p) for v in neighbours(ego, radius)])
    return (None, math.inf) if r is None else (r["vehicle"], r["long_gap"])


class ShieldedVehicle(MDPVehicle):
    """Машина агента с RSS-фильтром (shield_enabled=False — тот же код без фильтра)."""

    def __init__(self, *args, shield_enabled=True, params=SHIELD_DEFAULT, reaction_delay=DEFAULT.rho, **kwargs):
        super().__init__(*args, **kwargs)
        self.shield_enabled = shield_enabled
        self.params = params
        self.reaction_delay = reaction_delay
        self.clock = 0.0  # собственные часы, с
        # для контрфактического прогона: с этого момента фильтр не вмешивается
        self.shield_off_at = math.inf
        self.last_safe = 0.0  # время начала последнего шага, на котором было безопасно
        self.emergency = False  # фильтр в экстренном режиме (до конца опасности)
        self.a_prev = 0.0  # ускорение, исполненное на прошлом шаге, м/с^2
        self.lc_origin = None  # полоса, из которой начато текущее перестроение
        self.state = {}  # решение на текущем шаге, читает логгер
        self._last_act_clock = None
        self._last_lc = (False, None)

    @classmethod
    def replace(cls, vehicle, shield_enabled=True, params=SHIELD_DEFAULT, reaction_delay=DEFAULT.rho):
        """Заменить машину агента в среде на ShieldedVehicle в том же состоянии."""
        new = cls(
            vehicle.road,
            vehicle.position.copy(),
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            target_speeds=vehicle.target_speeds,
            route=vehicle.route,
            shield_enabled=shield_enabled,
            params=params,
            reaction_delay=reaction_delay,
        )
        new.lane_index, new.lane = vehicle.lane_index, vehicle.lane
        vehicles = vehicle.road.vehicles
        vehicles[vehicles.index(vehicle)] = new
        return new

    @property
    def filtering(self):
        """Вмешивается ли фильтр сейчас."""
        return self.shield_enabled and self.clock < self.shield_off_at - 1e-9

    # ---------- перестроения ----------

    def _lane_center_offset(self, lane_index):
        """Поперечное смещение центра полосы lane_index от агента, м."""
        lane = self.road.network.get_lane(lane_index)
        _, y_self = self.lane.local_coordinates(self.position)
        _, y_lane = self.lane.local_coordinates(lane.position(0.0, 0.0))
        return y_lane - y_self

    def _in_lane(self, rel, lane_index, half_width=2.0):
        """Находится ли машина из rel (хотя бы частично) в полосе lane_index."""
        offset = self._lane_center_offset(lane_index)
        return abs(rel["dy"] - offset) < half_width + rel["vehicle"].WIDTH / 2

    def _lane_change_blocked(self, target, relations):
        """Можно ли начать перестроение в target: продольно безопасно со всеми
        машинами целевой полосы (оценка — как будто агент уже в ней)."""
        offset = self._lane_center_offset(target)
        for r in relations:
            if not self._in_lane(r, target):
                continue
            moved = relation(self, r["vehicle"], self.params, y_offset=offset)
            if moved["long_unsafe"]:
                return True
        return False

    def _lane_change_dangerous(self, target, relations):
        """Опасно ли продолжать начатое перестроение: есть опасная машина
        в целевой полосе со стороны движения."""
        side = 1.0 if self._lane_center_offset(target) > 0 else -1.0
        for r in relations:
            if r["dangerous"] and r["dy"] * side > 0 and self._in_lane(r, target):
                return True
        return False

    def _supervise_lane_change(self, previous_target, relations):
        """Запрет нового и отмена начатого опасного перестроения.
        Возвращает 'blocked', 'cancelled' или None."""
        target = self.target_lane_index
        decision = None
        if target != previous_target and target != self.lane_index:
            # политика только что попросила перестроиться
            if self._lane_change_blocked(target, relations):
                decision = "blocked"
            else:
                self.lc_origin = previous_target
        elif self.lc_origin is not None and target != self.lc_origin and self.lane_index != target:
            # перестроение идет, агент еще не пересек разметку. Отменяем, только
            # если возвращаться безопасно: уход в исходную полосу не должен сам
            # создавать опасную ситуацию (туда мог перестроиться кто-то другой)
            if self._lane_change_dangerous(target, relations) and not self._lane_change_blocked(self.lc_origin, relations):
                decision = "cancelled"
        if decision and self.filtering:
            back = previous_target if decision == "blocked" else self.lc_origin
            self.target_lane_index = back
            self.lc_origin = None
            steering = self.steering_control(back)
            self.action["steering"] = max(-self.MAX_STEERING_ANGLE, min(self.MAX_STEERING_ANGLE, steering))
        if self.lc_origin is not None and self.lane_index == self.target_lane_index:
            # центр машины уже в целевой полосе (пересек разметку): перестроение
            # завершено, дальше на опасность впереди — только торможение
            self.lc_origin = None
        return decision

    # ---------- главный цикл ----------

    def act(self, action=None):
        previous_target = self.target_lane_index
        # 1. Политика -> ускорение и руль (стандартный перевод highway-env)
        super().act(action)
        if self.crashed:
            return
        p = self.params
        nominal = min(max(self.action["acceleration"], -p.a_max_brake), p.a_max_accel)
        nominal_target = self.target_lane_index

        # 2. Отношения со всеми машинами в радиусе (считаем и без фильтра — для логов)
        relations = [relation(self, v, p) for v in neighbours(self)]

        # 3. Перестроения
        lc_decision = self._supervise_lane_change(previous_target, relations)

        # 4. Продольная реакция на опасные машины впереди
        crit = critical_ahead(relations)
        dangerous = any(r["dangerous"] and r["ahead"] for r in relations)
        if not dangerous:
            self.last_safe = self.clock
        # тормозить, если иначе следующий шаг начнется позже, чем через
        # reaction_delay после последнего безопасного момента
        reacting = dangerous and self.clock - self.last_safe + SIM_DT > self.reaction_delay + 1e-9
        executed = nominal
        active = self.filtering and reacting
        mode = "delay" if self.filtering and dangerous else None
        if not dangerous:
            self.emergency = False
        if active:
            if not self.emergency:
                self.emergency = self._needs_emergency(relations)
            if self.emergency:
                executed = -A_PHYS_BRAKE
                mode = "emergency"
            else:
                executed = min(nominal, -p.a_min_brake)
                mode = "brake"
        self.action["acceleration"] = executed

        self.state = {
            "a_nominal": nominal,
            "a_executed": executed,
            "steering": float(self.action["steering"]),
            "front_id": getattr(crit["vehicle"], "uid", None) if crit else None,
            "gap": crit["long_gap"] if crit else math.inf,
            "v_front": crit["v_long"] if crit else math.nan,
            "d_min": crit["d_long"] if crit else 0.0,
            "lat_gap": crit["lat_gap"] if crit else math.nan,
            "d_lat": crit["d_lat"] if crit else math.nan,
            "dangerous": dangerous,
            "shield_active": active,
            "shield_mode": mode,
            "danger_ahead_ids": [getattr(r["vehicle"], "uid", None) for r in relations if r["dangerous"] and r["ahead"]],
            "danger_behind_ids": [getattr(r["vehicle"], "uid", None) for r in relations if r["dangerous"] and not r["ahead"]],
            "lc_requested": nominal_target != previous_target,
            "lc_decision": lc_decision,
        }
        # На первом шаге каждого решения highway-env вызывает act() дважды:
        # с действием политики и затем без него. Чтобы второй вызов не стер
        # запись о запрошенном/запрещенном перестроении, объединяем их.
        if self._last_act_clock == self.clock:
            self.state["lc_requested"] |= self._last_lc[0]
            self.state["lc_decision"] = self.state["lc_decision"] or self._last_lc[1]
        self._last_act_clock = self.clock
        self._last_lc = (self.state["lc_requested"], self.state["lc_decision"])

    def _needs_emergency(self, relations):
        """Не хватит ли a_min_brake хотя бы для одной опасной машины впереди."""
        p = self.params
        v_e = max(0.0, velocity_in_lane(self, self.lane)[0])
        for r in relations:
            if not (r["dangerous"] and r["ahead"]):
                continue
            b_front = max(p.a_max_brake, -getattr(r["vehicle"], "a_prev", 0.0))
            if brake_insufficient(r["long_gap"], v_e, max(0.0, r["v_long"]), self.a_prev,
                                  b_front, p.a_min_brake):
                return True
        return False

    def step(self, dt):
        if not self.crashed:
            a = limit_jerk(self.action["acceleration"], self.a_prev, dt)
            # задом не едем: на последнем шаге торможения дотормаживаем до нуля
            self.action["acceleration"] = max(a, -self.speed / dt)
        super().step(dt)
        self.a_prev = self.action["acceleration"]
        self.state["a_applied"] = self.a_prev
        self.clock += dt
