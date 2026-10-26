"""Формулы минимальной безопасной дистанции RSS.

Источник: Shalev-Shwartz, Shammah, Shashua. On a Formal Model of Safe and
Scalable Self-driving Cars. arXiv:1708.06374, 2017.
- лемма 2 — продольная дистанция (машины едут в одну сторону друг за другом);
- лемма 4 — поперечная дистанция (машины едут рядом в соседних полосах).

Вариант леммы 2 с учетом рывка (RSSParams.jerk не None): после времени
реакции задняя машина не может мгновенно перейти от разгона a_max_accel к
торможению a_min_brake — ускорение меняется с рывком jerk, на это уходит
(a_max_accel + a_min_brake) / jerk секунд (при 3, 4 и 10 м/с^3 — 0,7 с).
Этот путь добавляется к пути задней машины. Передняя машина, как и в
лемме, тормозит с a_max_brake сразу (худший случай для задней).

Все величины в СИ: метры, секунды, м/с, м/с^2. Ускорения и торможения
задаются положительными числами (модуль).
Дистанция — это зазор между кузовами (от заднего бампера передней машины
до переднего бампера задней), а не между центрами.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RSSParams:
    """Параметры RSS.

    Продольные значения взяты из CLAUDE.md (согласованы в статье).
    Поперечные значения — значения по умолчанию библиотеки Intel ad-rss-lib;
    их нужно сверить с текстом статьи.
    """

    rho: float = 0.5  # время реакции, с
    a_max_accel: float = 3.0  # макс. разгон задней машины за время реакции
    a_min_brake: float = 4.0  # торможение, которое гарантирует задняя машина
    a_max_brake: float = 5.0  # макс. торможение передней машины (допущение RSS)

    a_lat_max_accel: float = 0.2  # макс. поперечный разгон за время реакции
    a_lat_min_brake: float = 0.8  # поперечное торможение до нулевой поперечной скорости
    mu: float = 0.1  # поперечный запас, м

    jerk: float | None = None  # рывок задней машины, м/с^3; None — лемма 2 как в статье


DEFAULT = RSSParams()


def d_min_longitudinal(v_rear, v_front, p=DEFAULT):
    """Лемма 2: минимальная безопасная продольная дистанция, м.

    Худший случай, который закладывает RSS:
    - передняя машина сразу тормозит с a_max_brake до остановки;
    - задняя машина ρ секунд еще разгоняется с a_max_accel (не успела
      отреагировать), потом тормозит с a_min_brake до остановки.
    d_min — сколько нужно, чтобы после остановки обеих машин зазор был >= 0.

        d_min = [v_r ρ + a_acc ρ^2 / 2 + (v_r + ρ a_acc)^2 / (2 a_brake_min)
                 - v_f^2 / (2 a_brake_max)]_+

    v_rear, v_front — скорости задней и передней машины (>= 0), м/с.
    """
    if v_rear < 0 or v_front < 0:
        raise ValueError("скорости в лемме 2 должны быть неотрицательными")
    rho, a_acc = p.rho, p.a_max_accel
    v_rear_rho = v_rear + rho * a_acc  # скорость задней в конце времени реакции
    if p.jerk is None:
        rear_path = v_rear * rho + 0.5 * a_acc * rho**2 + v_rear_rho**2 / (2 * p.a_min_brake)
    else:
        rear_path = v_rear * rho + 0.5 * a_acc * rho**2 + _jerk_brake_path(v_rear_rho, a_acc, p.a_min_brake, p.jerk)
    front_path = v_front**2 / (2 * p.a_max_brake)
    return max(0.0, rear_path - front_path)


def _jerk_brake_path(v, a0, b, j):
    """Путь до остановки машины со скоростью v и ускорением a0 (> 0, разгон),
    которая уменьшает ускорение с рывком j до -b и дальше тормозит с b.

    Фаза перехода длится tau = (a0 + b) / j: a(t) = a0 - j t,
    v(t) = v + a0 t - j t^2/2, x(t) = v t + a0 t^2/2 - j t^3/6.
    Если машина останавливается еще во время перехода (v(t) = 0 раньше tau),
    путь — x в момент остановки.
    """
    tau = (a0 + b) / j
    t_stop = (a0 + (a0 * a0 + 2 * j * v) ** 0.5) / j  # корень v(t) = 0
    if t_stop <= tau:
        t = t_stop
        return v * t + a0 * t * t / 2 - j * t**3 / 6
    x_tau = v * tau + a0 * tau * tau / 2 - j * tau**3 / 6
    v_tau = v + a0 * tau - j * tau * tau / 2
    return x_tau + v_tau * v_tau / (2 * b)


def _lateral_path(v, a_acc, a_brake, rho):
    """Поперечное смещение машины: ρ секунд разгон a_acc, затем торможение
    a_brake до нулевой поперечной скорости. Знак сохраняется (v|v|), чтобы
    формула оставалась верной и при движении от соседа."""
    v_rho = v + rho * a_acc
    return (v + v_rho) / 2 * rho + v_rho * abs(v_rho) / (2 * a_brake)


def d_min_lateral(v1, v2, p=DEFAULT):
    """Лемма 4: минимальная безопасная поперечная дистанция, м.

    Машина 1 слева, машина 2 справа (или наоборот, важна только ось).
    Ось направлена от машины 1 к машине 2:
    - v1 > 0 значит машина 1 сдвигается к машине 2;
    - v2 < 0 значит машина 2 сдвигается к машине 1.
    Худший случай: обе ρ секунд разгоняются навстречу друг другу с
    a_lat_max_accel, затем тормозят поперек с a_lat_min_brake до нуля.

        d_min = μ + [путь_1 - путь_2]_+

    Квадраты скоростей записаны как v|v| (так в поздних версиях статьи):
    в первой версии стоит v^2, и при движении машин в разные стороны
    формула дает лишний запас неверного знака.
    """
    rho = p.rho
    path_1 = _lateral_path(v1, p.a_lat_max_accel, p.a_lat_min_brake, rho)
    path_2 = _lateral_path(v2, -p.a_lat_max_accel, p.a_lat_min_brake, rho)
    return p.mu + max(0.0, path_1 - path_2)


def is_safe_longitudinal(gap, v_rear, v_front, p=DEFAULT):
    """Безопасна ли продольная ситуация: зазор не меньше d_min."""
    return gap >= d_min_longitudinal(v_rear, v_front, p)


def is_safe_lateral(gap, v1, v2, p=DEFAULT):
    """Безопасна ли поперечная ситуация: зазор не меньше d_min."""
    return gap >= d_min_lateral(v1, v2, p)
