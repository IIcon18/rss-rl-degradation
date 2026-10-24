"""Этап 5: номинальный режим — нарушителя нет, допущения RSS по торможению
соблюдаются (фон тормозит не сильнее 5 м/с^2). С фильтром столкновений
быть не должно ни при какой задержке и ни при какой политике агента."""

import pytest

from rssrl.pilot import run_grid, unsafe_entries

SEEDS = range(16)
VARIANTS = [(True, 0.0), (True, 0.5)]


@pytest.fixture(scope="module", params=["idle", "random"])
def nominal(request):
    return run_grid(SEEDS, [1.0], VARIANTS, kind="nominal", policy=request.param)


def test_no_violator_in_nominal_mode(nominal):
    outcomes, _ = nominal
    assert not outcomes["triggered"].any()


def test_no_collisions_with_shield(nominal):
    outcomes, _ = nominal
    assert not outcomes["collision"].any(), outcomes[outcomes.collision]


def test_episodes_run_to_time_limit_and_agent_keeps_moving(nominal):
    outcomes, logs = nominal
    assert (outcomes["t_end"] > 39.9).all()
    for log in logs.values():
        assert log["ego_v"].min() > 5.0  # фильтр не останавливает агента на пустом месте


def test_shield_commands_stay_within_physical_limits(nominal):
    _, logs = nominal
    for log in logs.values():
        assert log["ego_a"].min() >= -5.0 - 1e-6
        assert log["ego_a"].max() <= 3.0 + 1e-6


def test_unsafe_entries_detects_background_cut_in():
    """Синтетический лог: агент в своей полосе, впереди сменилась машина,
    и зазор до новой меньше d_min — это одно событие."""
    import pandas as pd

    log = pd.DataFrame({
        "t": [4.0, 4.1, 4.2, 4.3, 4.4],
        "front_id": [5, 5, 7, 7, 8],
        "ego_lane": [1, 1, 1, 1, 1],
        "gap": [30.0, 29.0, 10.0, 10.5, 40.0],
        "d_min": [20.0, 20.0, 20.0, 20.0, 20.0],
    })
    ev = unsafe_entries(log)
    assert list(ev.index) == [2]  # машина 8 появилась на безопасном зазоре
    assert ev["ratio"].iloc[0] == pytest.approx(0.5)


def test_unsafe_entries_ignores_episode_start():
    import pandas as pd

    log = pd.DataFrame({
        "t": [0.1, 0.2, 0.3],
        "front_id": [5, 7, 7],
        "ego_lane": [1, 1, 1],
        "gap": [30.0, 10.0, 10.0],
        "d_min": [20.0, 20.0, 20.0],
    })
    assert unsafe_entries(log).empty
