"""Сборка сцен highway-v0 с заданным seed.

make_env    — среда с фоновыми машинами BoundedIDMVehicle;
make_scene  — сцена эксперимента: агент (с фильтром или без), нарушитель
              (тормозящий лидер или подрезающий из соседней полосы), логгер;
run_episode — прогон эпизода и его исход.

Все случайности сцены задаются seed сцены. Одна и та же сцена (seed)
используется для всех β и для вариантов с фильтром и без (общие сцены).
"""

import math
from dataclasses import dataclass

import gymnasium as gym
import highway_env  # noqa: F401  регистрирует highway-v0

from rssrl.logger import EpisodeLogger
from rssrl.rss import DEFAULT, d_min_longitudinal
from rssrl.shield import LEMMA2, SHIELD_DEFAULT, ShieldedVehicle, longitudinal_speed, neighbours, relation
from rssrl.violators import BrakingViolator, CutInViolator

BACKGROUND = "rssrl.violators.BoundedIDMVehicle"
RHO = DEFAULT.rho  # основная задержка реакции фильтра, с
IDLE = 1  # DiscreteMetaAction: держать полосу и целевую скорость

# Исход эпизода «остановка»: агент стоял (скорость < STOP_SPEED) не меньше STOP_DURATION
STOP_SPEED = 0.5  # м/с
STOP_DURATION = 2.0  # с

# Когда начинается нарушение (выбор сцены, а не часть RSS):
TRIGGER_T_MIN = 3.0  # не раньше 3 с от начала эпизода
TRIGGER_RATIO = 1.1  # агент идет за машиной впереди с зазором <= 1.1·d_min
LC_DURATION = 3.0  # длительность перестроения подрезающего, с
ROOM_MARGIN = 5.0  # м, запас между подрезающим и лидером агента
CUTIN_FREE_AHEAD = 250.0  # в сцене подрезания полоса агента впереди свободна на столько м


def make_env(seed, render_mode=None, **config):
    """Создать highway-v0 и сбросить ее с seed сцены.

    По умолчанию фоновые машины — BoundedIDMVehicle (торможение <= 5 м/с^2).
    Любой параметр конфигурации highway-env можно переопределить через config.
    """
    env = gym.make("highway-v0", render_mode=render_mode)
    env.unwrapped.configure({"other_vehicles_type": BACKGROUND, **config})
    env.reset(seed=seed)
    return env


def find_leader(env):
    """Машина непосредственно перед агентом в его полосе (или None)."""
    ego = env.unwrapped.vehicle
    front, _ = env.unwrapped.road.neighbour_vehicles(ego, ego.lane_index)
    return front


class ViolationDirector:
    """Назначает тормозящего нарушителя в момент нарушения.

    Два способа выбрать момент:
    1. По зазору (schedule=None, как в пилотах): перед каждым шагом симуляции
       проверяется условие: прошло >= t_min секунд, перед агентом есть машина
       и зазор до нее <= trigger_ratio·d_min. Как только условие выполнено,
       эта машина заменяется нарушителем в том же состоянии, и он сразу
       начинает тормозить к β·5 м/с^2. Момент и машина зависят от того, как
       ехал агент, поэтому у разных вариантов агента они могут различаться.
    2. По расписанию (schedule={"t": момент, "uid": машина}): в момент t
       нарушителем становится машина uid, где бы ни был агент. Расписание
       берется из эталонного прогона (violation_schedule), и тогда момент и
       машина одинаковы для всех вариантов агента на этой сцене.
    До момента нарушения в сцене только обычные фоновые машины, поэтому
    сцена одинакова для всех β.
    """

    def __init__(self, env, ego, logger, beta, t_min, trigger_ratio=TRIGGER_RATIO, schedule=None):
        self.schedule = dict(schedule) if schedule else None
        self.ego, self.logger, self.beta = ego, logger, beta
        self.trigger_ratio, self.t_min = trigger_ratio, t_min
        self.road = env.unwrapped.road
        self.dt = 1 / env.unwrapped.config["simulation_frequency"]
        self.t = 0.0  # время эпизода на начало текущего шага
        self.violator = None
        self.t_trigger = None
        self.info = {}  # что сцена намерила (для таблицы исходов)
        original_act = self.road.act

        def check_and_act():
            if self.violator is None and not self.ego.crashed and self.t >= self.t_min - 1e-9:
                self.check()
            original_act()
            self.t += self.dt

        self.road.act = check_and_act

    def _vehicle(self, uid):
        return next(v for v in self.road.vehicles if getattr(v, "uid", None) == uid)

    def _assign(self, vio):
        vio.clock = self.t  # часы нарушителя = время эпизода
        self.violator = vio
        self.t_trigger = self.t
        self.logger.set_violator(vio)

    def _scheduled_vehicle(self):
        """Машина из расписания, если ее момент настал (иначе None). Если
        машины уже нет на дороге или она разбита, нарушения не будет."""
        if self.t < self.schedule["t"] - 1e-9:
            return None
        v = next((v for v in self.road.vehicles if getattr(v, "uid", None) == self.schedule["uid"]), None)
        self.info["violator_was_front"] = self.ego.state.get("front_id") == self.schedule["uid"]
        if v is None or v.crashed:
            self.schedule["t"] = math.inf  # больше не проверять
            self.info["schedule_missed"] = True
            return None
        return v

    def check(self):
        if self.schedule is not None:
            v = self._scheduled_vehicle()
            if v is not None:
                self._assign(BrakingViolator.replace(v, beta=self.beta, trigger_time=0.0))
            return
        s = self.ego.state
        if s.get("front_id") is None:
            return
        # условие сцены — по d_min леммы 2 (DEFAULT), а не по d_min фильтра:
        # так сцена одинакова для любого варианта формулы в фильтре
        v_e = max(0.0, longitudinal_speed(self.ego, self.ego.lane))
        if s["gap"] > self.trigger_ratio * d_min_longitudinal(v_e, max(0.0, s["v_front"]), DEFAULT):
            return
        front = self._vehicle(s["front_id"])
        self._assign(BrakingViolator.replace(front, beta=self.beta, trigger_time=0.0))


class CutInDirector(ViolationDirector):
    """Назначает подрезающего нарушителя.

    Перед каждым шагом перебирает машины в соседних с агентом полосах
    впереди него (в радиусе 150 м, не перестраивающиеся) и прогнозирует
    зазор в момент пересечения разметки — через lc_duration/2 секунд —
    при постоянных скоростях:
        зазор_прогноз = зазор_сейчас + (v_машины - v_агента) · lc_duration/2.
    Маневр начинается, когда прогноз впервые опускается до γ·d_min
    (d_min по лемме 2: агент сзади, подрезающий спереди). Дополнительно
    требуется, чтобы перед агентом в его полосе было место: в момент
    пересечения разметки лидер агента впереди подрезающего минимум на длину
    машины + ROOM_MARGIN.

    Фактический зазор измеряется в момент пересечения разметки (on_cross):
    если агент за время маневра затормозил, фактическое γ будет больше
    заданного — это записывается.
    """

    def __init__(self, env, ego, logger, beta, t_min, gamma, lc_duration=LC_DURATION, schedule=None):
        super().__init__(env, ego, logger, beta, t_min, schedule=schedule)
        self.gamma, self.lc_duration = gamma, lc_duration
        self.prev_margin = {}  # uid -> (прогноз - γ·d_min) на прошлом шаге

    def _start(self, v, lane_index):
        cutter = CutInViolator.replace(v, beta=self.beta)
        cutter.lc_duration = self.lc_duration
        cutter.on_cross = self.on_cross
        self._assign(cutter)
        cutter.start_cut(lane_index)

    def check(self):
        if self.schedule is not None:
            # по расписанию: машина uid в момент t перестраивается в полосу lane
            # (полоса агента в эталонном прогоне), если она в соседней с ней полосе
            v = self._scheduled_vehicle()
            if v is None:
                return
            lane = self.schedule["lane"]
            if v.lane_index[:2] != lane[:2] or abs(v.lane_index[2] - lane[2]) != 1:
                self.schedule["t"] = math.inf
                self.info["schedule_missed"] = True
                return
            self._start(v, lane)
            return
        ego, p = self.ego, DEFAULT  # γ·d_min — по лемме 2, независимо от фильтра
        lane = ego.lane
        s_e, _ = lane.local_coordinates(ego.position)
        vx_e = longitudinal_speed(ego, lane)
        leader, _ = self.road.neighbour_vehicles(ego, ego.lane_index)
        if leader is not None:  # где будет лидер агента в момент пересечения разметки
            s_leader = lane.local_coordinates(leader.position)[0] + longitudinal_speed(leader, lane) * (self.lc_duration / 2)
        else:
            s_leader = math.inf
        t_cross = self.lc_duration / 2
        for v in neighbours(ego):
            if v.lane_index[:2] != ego.lane_index[:2] or abs(v.lane_index[2] - ego.lane_index[2]) != 1:
                continue
            if v.lane_index != getattr(v, "target_lane_index", v.lane_index) or v.crashed:
                continue
            s_v, _ = lane.local_coordinates(v.position)
            if s_v <= s_e:
                continue
            vx_v = longitudinal_speed(v, lane)
            gap_pred = s_v - s_e - (ego.LENGTH + v.LENGTH) / 2 + (vx_v - vx_e) * t_cross
            d_pred = d_min_longitudinal(max(0.0, vx_e), max(0.0, vx_v), p)
            margin = gap_pred - self.gamma * d_pred
            prev = self.prev_margin.get(v.uid)
            self.prev_margin[v.uid] = margin
            room = s_leader - (s_v + vx_v * t_cross) >= v.LENGTH + ROOM_MARGIN
            if prev is not None and prev > 0 >= margin and gap_pred > 0 and room:
                self._start(v, ego.lane_index)
                self.info.update({"gap_pred": gap_pred, "d_min_pred": d_pred})
                return

    def on_cross(self, cutter):
        r = relation(self.ego, cutter, DEFAULT)
        self.info.update({
            "t_cross": cutter.clock,
            "gap_cross": r["long_gap"],
            "d_min_cross": r["d_long"],
            "gamma_actual": r["long_gap"] / r["d_long"] if r["d_long"] > 0 else math.inf,
        })


class NoViolationDirector(ViolationDirector):
    """Номинальный режим: нарушителя нет, все машины — обычный фон."""

    def check(self):
        pass


@dataclass
class Scene:
    env: gym.Env
    ego: ShieldedVehicle
    director: ViolationDirector
    logger: EpisodeLogger
    seed: int
    kind: str
    beta: float
    gamma: float
    shield: bool
    reaction_delay: float

    @property
    def violator(self):
        return self.director.violator


def make_scene(seed, beta=1.0, shield=True, reaction_delay=RHO, kind="brake", gamma=1.0,
               t_min=TRIGGER_T_MIN, schedule=None, dmin="jerk", **config):
    """Сцена эксперимента.

    - агент: ShieldedVehicle с фильтром (shield=True) или без,
      reaction_delay — задержка реакции фильтра, с (основной режим ρ = 0,5 с,
      дополнительный — 0);
    - kind="brake": нарушитель — машина непосредственно перед агентом в момент
      нарушения (ViolationDirector); тормозит с β·5 м/с^2 до остановки и стоит;
    - kind="cutin": нарушитель — машина из соседней полосы, подрезает агента
      с зазором γ·d_min на разметке и тормозит с β·5 м/с^2 (CutInDirector).
      Чтобы было куда подрезать, на старте из полосы агента убираются машины
      на CUTIN_FREE_AHEAD м вперед (одинаково для всех вариантов агента и всех γ);
    - kind="nominal": нарушителя нет (проверка номинального режима);
    - dmin — формула d_min в фильтре: "jerk" (по умолчанию, лемма 2 с учетом
      рывка) или "lemma2" (как в статье);
    - schedule — нарушение по расписанию (см. ViolationDirector и
      violation_schedule) вместо условия по зазору.
    """
    env = make_env(seed, **config)
    u = env.unwrapped
    for i, v in enumerate(u.road.vehicles):
        v.uid = i  # постоянные номера машин для логов

    params = {"jerk": SHIELD_DEFAULT, "lemma2": LEMMA2}[dmin]
    ego = ShieldedVehicle.replace(u.vehicle, shield_enabled=shield, reaction_delay=reaction_delay, params=params)
    ego.uid = 0
    u.vehicle = ego
    if kind == "cutin":
        lane = ego.lane
        s_e, _ = lane.local_coordinates(ego.position)
        u.road.vehicles = [
            v for v in u.road.vehicles
            if v is ego or v.lane_index != ego.lane_index
            or not 0 < lane.local_coordinates(v.position)[0] - s_e <= CUTIN_FREE_AHEAD
        ]

    logger = EpisodeLogger(env, ego)
    if kind == "brake":
        director = ViolationDirector(env, ego, logger, beta, t_min, schedule=schedule)
    elif kind == "cutin":
        director = CutInDirector(env, ego, logger, beta, t_min, gamma, schedule=schedule)
    elif kind == "nominal":
        director = NoViolationDirector(env, ego, logger, beta, t_min)
    else:
        raise ValueError(kind)
    return Scene(env, ego, director, logger, seed, kind, beta, gamma, shield, reaction_delay)


def violation_schedule(seed, kind="brake", gamma=1.0, shield=True, reaction_delay=RHO,
                       policy=None, **config):
    """Расписание нарушения для сцены seed из эталонного прогона.

    Эталонный агент (по умолчанию IDLE с фильтром и задержкой ρ) едет, пока
    сцена не назначит нарушителя по условию на зазор. Возвращает
    {"t": момент, "uid": машина, "lane": полоса агента в этот момент} или
    None, если нарушения не было. Момент до нарушения от β не зависит.
    """
    scene = make_scene(seed, kind=kind, gamma=gamma, shield=shield, reaction_delay=reaction_delay, **config)
    env, director = scene.env, scene.director
    policy = policy or (lambda obs: IDLE)
    obs = env.unwrapped.observation_type.observe()
    done = False
    lane = None
    original_assign = director._assign

    def assign_and_remember(vio):
        nonlocal lane
        lane = scene.ego.lane_index
        original_assign(vio)

    director._assign = assign_and_remember
    while not done and director.violator is None:
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        done = terminated or truncated
    env.close()
    if director.violator is None:
        return None
    return {"t": director.t_trigger, "uid": director.violator.uid, "lane": lane}


def classify_outcome(log, dt=1 / 15):
    """Исход эпизода по его логу (в порядке приоритета):
    - "collision" — столкновение агента;
    - "offroad"   — агент хотя бы раз оказался вне дороги;
    - "stopped"   — агент стоял (скорость < STOP_SPEED) не меньше STOP_DURATION подряд;
    - "safe"      — доехал до конца времени без всего перечисленного.
    Возвращает (исход, суммарное время стоянки, с).
    """
    standing = (log["ego_v"] < STOP_SPEED).to_numpy()
    longest = run = 0
    for s in standing:
        run = run + 1 if s else 0
        longest = max(longest, run)
    t_standing = float(standing.sum() * dt)
    if log["crashed"].any():
        return "collision", t_standing
    if not log["on_road"].all():
        return "offroad", t_standing
    if longest * dt >= STOP_DURATION - 1e-9:
        return "stopped", t_standing
    return "safe", t_standing


def run_episode(scene, policy=None, t_stop=None):
    """Прогнать эпизод до конца времени или столкновения.

    policy(obs) -> действие; по умолчанию IDLE (пилот без PPO:
    агент держит полосу и целевую скорость).
    t_stop — остановить раньше, как только время эпизода >= t_stop
    (с точностью до шага политики, 1 с; для контрфактических прогонов).
    Возвращает словарь с исходом эпизода и таблицу временных рядов.
    """
    env = scene.env
    policy = policy or (lambda obs: IDLE)
    obs = env.unwrapped.observation_type.observe()
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        if t_stop is not None and scene.logger.t >= t_stop - 1e-9:
            break
    log = scene.logger.frame()
    vio = scene.violator
    events = vio.events if vio is not None else []
    partners = scene.logger.crash_partners or []
    result, t_standing = classify_outcome(log)
    outcome = {
        "seed": scene.seed,
        "kind": scene.kind,
        "beta": scene.beta,
        "gamma": scene.gamma,
        "shield": scene.shield,
        "reaction_delay": scene.reaction_delay,
        "outcome": result,
        "collision": bool(scene.ego.crashed),
        "t_crash": float(log.loc[log["crashed"], "t"].min()) if log["crashed"].any() else None,
        # скорость агента на последнем шаге перед ударом, м/с
        "v_impact": float(log.loc[: log.index[log["crashed"]].min(), "ego_v"].iloc[-2])
        if log["crashed"].any() and log["crashed"].idxmax() > 0 else None,
        "violator_uid": vio.uid if vio is not None else None,
        "hit_violator": vio is not None and vio.uid in partners,
        "crash_partners": partners,
        "t_end": float(log["t"].iloc[-1]),
        "distance_km": (log["ego_x"].iloc[-1] - log["ego_x"].iloc[0]) / 1000,
        "t_standing": t_standing,
        "triggered": vio is not None,
        "t_trigger": scene.director.t_trigger,
        "t_brake": events[0]["t_start"] if events else None,
        "scheduled": scene.director.schedule is not None,
        "dmin": "jerk" if scene.ego.params.jerk is not None else "lemma2",
        "shield_steps": int(log["shield_active"].sum()),
        **scene.director.info,
    }
    env.close()
    return outcome, log
