"""Этап 1: проверка формул RSS (rssrl/rss.py).

Два вида проверок:
1. Ручные примеры: значения посчитаны на бумаге (вычисления в комментариях).
2. Проверка симуляцией: худший случай из леммы проигрывается мелкими
   шагами по времени, независимо от формулы. Если начать ровно с d_min,
   машины должны остановиться, почти касаясь (зазор около 0, но не меньше).
   Если начать с зазора меньше d_min, должно быть столкновение.
"""

import numpy as np
import pytest

from rssrl.rss import (
    DEFAULT,
    RSSParams,
    d_min_lateral,
    d_min_longitudinal,
    is_safe_longitudinal,
)

DT = 1e-3  # шаг симуляции для проверки, с
TOL = 0.05  # допуск из-за дискретного шага, м


# ---------- 1. ручные примеры ----------

def test_longitudinal_equal_speeds_25():
    # задняя: 25*0.5 + 3*0.25/2 + (25 + 1.5)^2 / 8 = 12.5 + 0.375 + 87.78125 = 100.65625
    # передняя: 25^2 / 10 = 62.5
    assert d_min_longitudinal(25, 25) == pytest.approx(38.15625)


def test_longitudinal_both_stopped():
    # даже стоя RSS требует зазор: задняя может тронуться на ρ секунд
    # 0 + 0.375 + 1.5^2 / 8 = 0.65625
    assert d_min_longitudinal(0, 0) == pytest.approx(0.65625)


def test_longitudinal_front_much_faster_is_zero():
    # задняя: 10 + 0.375 + 21.5^2 / 8 = 68.15625 < передняя 30^2 / 10 = 90
    assert d_min_longitudinal(20, 30) == 0.0


def test_longitudinal_grows_with_rear_speed_and_falls_with_front_speed():
    v = np.linspace(0, 40, 41)
    by_rear = [d_min_longitudinal(x, 25) for x in v]
    by_front = [d_min_longitudinal(25, x) for x in v]
    assert np.all(np.diff(by_rear) >= 0)
    assert np.all(np.diff(by_front) <= 0)


def test_longitudinal_rejects_negative_speed():
    with pytest.raises(ValueError):
        d_min_longitudinal(-1, 10)


def test_is_safe_longitudinal():
    d = d_min_longitudinal(25, 25)
    assert is_safe_longitudinal(d, 25, 25)
    assert not is_safe_longitudinal(d - 0.01, 25, 25)


def test_lateral_both_still():
    # каждая за ρ сдвигается на 0.1*0.5/2 + 0.1^2/1.6 = 0.03125 к соседу
    # d = 0.1 + 2 * 0.03125 = 0.1625
    assert d_min_lateral(0, 0) == pytest.approx(0.1625)


def test_lateral_one_moves_towards():
    # машина 1: v_ρ = 1.1; 2.1/2*0.5 + 1.21/1.6 = 0.525 + 0.75625 = 1.28125
    # машина 2: -0.03125;  d = 0.1 + 1.28125 + 0.03125 = 1.4125
    assert d_min_lateral(1.0, 0.0) == pytest.approx(1.4125)


def test_lateral_symmetric():
    # машина 1 едет к 2 со скоростью 1 — то же, что машина 2 едет к 1
    assert d_min_lateral(1.0, 0.0) == pytest.approx(d_min_lateral(0.0, -1.0))


def test_lateral_moving_apart_needs_only_margin():
    assert d_min_lateral(-2.0, 2.0) == pytest.approx(DEFAULT.mu)


# ---------- 2. проверка симуляцией худшего случая ----------

def simulate_longitudinal(gap0, v_rear, v_front, p=DEFAULT, front_brake=None):
    """Проиграть худший случай леммы 2, вернуть минимальный зазор за всё время.

    front_brake — торможение передней машины (по умолчанию a_max_brake;
    больше этого — нарушение допущения, как у нашего нарушителя с β > 1).
    """
    front_brake = p.a_max_brake if front_brake is None else front_brake
    x_r, x_f = 0.0, gap0
    vr, vf = float(v_rear), float(v_front)
    t, min_gap = 0.0, gap0
    while vr > 0 or vf > 0 or t < p.rho:
        a_r = p.a_max_accel if t < p.rho else -p.a_min_brake
        a_f = -front_brake
        # точное перемещение за шаг с остановкой на нуле скорости
        for_rear = vr * DT + 0.5 * a_r * DT**2
        for_front = vf * DT + 0.5 * a_f * DT**2
        if vr + a_r * DT < 0:
            for_rear, new_vr = vr**2 / (2 * -a_r), 0.0
        else:
            new_vr = vr + a_r * DT
        if vf + a_f * DT < 0:
            for_front, new_vf = vf**2 / (2 * -a_f), 0.0
        else:
            new_vf = vf + a_f * DT
        x_r += for_rear
        x_f += for_front
        vr, vf = new_vr, new_vf
        t += DT
        min_gap = min(min_gap, x_f - x_r)
    return min_gap


def simulate_lateral(gap0, v1, v2, p=DEFAULT):
    """Худший случай леммы 4 (ось от машины 1 к машине 2), минимальный зазор."""
    y1, y2 = 0.0, gap0
    t, min_gap = 0.0, gap0
    while True:
        if t < p.rho:
            a1, a2 = p.a_lat_max_accel, -p.a_lat_max_accel
        else:
            a1 = -np.sign(v1) * p.a_lat_min_brake
            a2 = -np.sign(v2) * p.a_lat_min_brake
            if v1 == 0 and v2 == 0:
                break
        n1, n2 = v1 + a1 * DT, v2 + a2 * DT
        if t >= p.rho and np.sign(n1) != np.sign(v1):
            n1 = 0.0
        if t >= p.rho and np.sign(n2) != np.sign(v2):
            n2 = 0.0
        y1 += (v1 + n1) / 2 * DT
        y2 += (v2 + n2) / 2 * DT
        v1, v2 = n1, n2
        t += DT
        min_gap = min(min_gap, y2 - y1)
    return min_gap


SPEED_PAIRS = [(25, 25), (30, 20), (20, 25), (30, 30), (10, 0), (0, 0), (35, 15)]


@pytest.mark.parametrize("v_rear,v_front", SPEED_PAIRS)
def test_longitudinal_dmin_is_enough(v_rear, v_front):
    d = d_min_longitudinal(v_rear, v_front)
    assert simulate_longitudinal(d, v_rear, v_front) >= -TOL


@pytest.mark.parametrize("v_rear,v_front", [(25, 25), (30, 20), (30, 30), (35, 15), (10, 0)])
def test_longitudinal_dmin_is_tight(v_rear, v_front):
    # формула не перестраховывается: в худшем случае зазор доходит почти до нуля
    d = d_min_longitudinal(v_rear, v_front)
    assert simulate_longitudinal(d, v_rear, v_front) == pytest.approx(0.0, abs=TOL)
    # и если зазор чуть меньше d_min — столкновение
    assert simulate_longitudinal(d - 0.5, v_rear, v_front) < 0


def test_longitudinal_grid_never_collides():
    # сетка скоростей шоссе: старт с d_min никогда не ведет к столкновению
    for v_rear in range(0, 41, 5):
        for v_front in range(0, 41, 5):
            d = d_min_longitudinal(v_rear, v_front)
            assert simulate_longitudinal(d, v_rear, v_front) >= -TOL, (v_rear, v_front)


def test_violation_beta_2_breaks_dmin():
    # если передняя тормозит 10 м/с2 вместо 5 (β = 2), d_min уже мало
    d = d_min_longitudinal(25, 25)
    assert simulate_longitudinal(d, 25, 25, front_brake=10.0) < -10


@pytest.mark.parametrize("v1,v2", [(0, 0), (1, 0), (0, -1), (0.5, -0.5), (1.5, 0.3), (-0.5, 0.5)])
def test_lateral_dmin_is_enough_and_tight(v1, v2):
    d = d_min_lateral(v1, v2)
    min_gap = simulate_lateral(d, v1, v2)
    assert min_gap >= DEFAULT.mu - TOL
    if d > DEFAULT.mu:
        # без запаса μ машины сошлись бы ровно впритык
        assert min_gap == pytest.approx(DEFAULT.mu, abs=TOL)


def test_custom_params_are_used():
    p = RSSParams(rho=1.0)
    assert d_min_longitudinal(25, 25, p) > d_min_longitudinal(25, 25)
