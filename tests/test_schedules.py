"""Шаг 3 (письмо 29.10): эталон IDM/MOBIL + RSS и расписания в файле."""

import math

import pandas as pd
import pytest

from rssrl.pilot import _run_one
from rssrl.scenarios import REFERENCE, make_scene, run_episode, violation_schedule
from rssrl.schedules import build_schedules, get_schedule, load_schedules, save_schedules
from rssrl.shield import ShieldedIDMVehicle

SEEDS = [0, 1, 2, 3]


@pytest.fixture(scope="module")
def sched(tmp_path_factory):
    df = build_schedules(SEEDS)
    path = save_schedules(df, tmp_path_factory.mktemp("s") / "schedules.csv")
    return df, path


def test_reference_is_idm_mobil_with_base_shield():
    assert REFERENCE == {"driver": "idm", "shield": True, "reaction_delay": 0.5,
                         "emergency_brake": False, "dmin": "jerk"}
    scene = make_scene(0, kind="nominal", **REFERENCE)
    assert isinstance(scene.ego, ShieldedIDMVehicle)
    assert scene.ego.shield_enabled and not scene.ego.emergency_brake
    scene.env.close()


def test_idm_reference_drives_without_collision_in_nominal_following():
    for seed in SEEDS:
        outcome, log = run_episode(make_scene(seed, kind="nominal", **REFERENCE))
        assert not outcome["collision"], seed
        assert outcome["driver"] == "idm"


def test_file_has_parameters_and_code_version(sched):
    df, path = sched
    back = pd.read_csv(path)
    assert list(back["seed"]) == SEEDS
    for col in ["t", "uid", "lane_id", "reference", "trigger_ratio", "t_min", "code_version", "traffic"]:
        assert col in back
    assert back["code_version"].iloc[0] != ""


def test_file_matches_direct_reference_run(sched):
    _, path = sched
    for seed in SEEDS:
        direct = violation_schedule(seed)
        from_file = get_schedule(seed, "brake", 1.0, "following", path)
        if direct is None:
            assert math.isinf(from_file["t"])
        else:
            assert from_file["t"] == pytest.approx(direct["t"])
            assert from_file["uid"] == direct["uid"]


def test_schedules_are_not_overwritten(sched):
    df, path = sched
    with pytest.raises(FileExistsError):
        save_schedules(df, path)


def test_missing_scene_is_an_error_not_regenerated(sched):
    _, path = sched
    with pytest.raises(KeyError):
        get_schedule(99, "brake", 1.0, "following", path)


def test_all_variants_use_the_same_schedule_and_log_role(sched):
    """Все варианты агента берут одно расписание; для каждого эпизода видно,
    состоялось ли воздействие, роль назначенной машины и зазор."""
    _, path = sched
    table = load_schedules(str(path))
    for seed in SEEDS:
        s = table[(seed, "brake", 1.0, "following")]
        for shield, delay in [(True, 0.5), (True, 0.0), (False, 0.5)]:
            o, _ = _run_one((seed, 1.5, shield, delay), "brake", "idle", trigger="schedule", schedule_file=path)
            assert o["impact"] in {"состоялось", "столкновение до воздействия",
                                   "назначенной машины нет на дороге или не в той полосе",
                                   "не назначено (эталон не дошел до условия)"}
            if o["triggered"]:
                assert s is not None
                assert o["t_trigger"] == pytest.approx(s["t"])
                assert o["violator_uid"] == s["uid"]
                assert o["violator_role"] in {"лидер", "впереди в полосе агента, не лидер",
                                              "впереди в другой полосе", "позади или рядом"}
                assert math.isfinite(o["gap_at_trigger"])


def test_cutin_lane_survives_file_round_trip(tmp_path):
    """Полоса подрезания из файла совпадает с прямым прогоном эталона, даже
    если рядом лежит файл, где у части сцен полосы нет (пропуски в CSV)."""
    save_schedules(build_schedules([0], kind="cutin", gammas=(1.0,)), tmp_path / "cutin.csv")
    df = build_schedules([0])
    df.loc[0, ["t", "uid", "lane_from", "lane_to", "lane_id"]] = None  # «не назначено»
    save_schedules(df, tmp_path / "brake.csv")
    direct = violation_schedule(0, kind="cutin", gamma=1.0)
    assert get_schedule(0, "cutin", 1.0, "following", tmp_path)["lane"] == direct["lane"]
