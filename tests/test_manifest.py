"""Манифест моделей (rssrl/manifest.py): хеши, фактическое число переходов, архив."""

import zipfile

import pytest

from rssrl.manifest import build_manifest, model_num_timesteps, sha256_file


@pytest.fixture(scope="module")
def dummy_models(tmp_path_factory):
    """Две маленькие PPO-модели на CartPole (env-агностичная проверка
    манифеста: не нужен highway-env, важна только структура zip-файла
    stable-baselines3)."""
    from stable_baselines3 import PPO

    base = tmp_path_factory.mktemp("models")
    runs = [("a_seed0", True, 0), ("b_seed1", False, 1)]
    for name, _, seed in runs:
        model = PPO("MlpPolicy", "CartPole-v1", n_steps=32, batch_size=32, seed=seed, verbose=0, device="cpu")
        model.learn(64)
        d = base / name
        d.mkdir(parents=True)
        model.save(str(d / "final_model"))
    return base, runs


def test_sha256_file_is_deterministic_and_sensitive_to_content(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    h1 = sha256_file(f)
    h2 = sha256_file(f)
    assert h1 == h2
    f.write_bytes(b"hello world!")
    assert sha256_file(f) != h1


def test_model_num_timesteps_matches_requested_training(dummy_models):
    base, runs = dummy_models
    n = model_num_timesteps(base / "a_seed0" / "final_model.zip")
    assert n >= 64  # PPO переходит границу total_timesteps только вверх (полный rollout)


def test_build_manifest_writes_csv_with_hashes_and_timesteps(tmp_path, dummy_models):
    models_dir, runs = dummy_models
    results_dir = tmp_path / "results"
    for name, _, _ in runs:
        (results_dir / name).mkdir(parents=True)
        (results_dir / name / "config.json").write_text("{}")
    archive = tmp_path / "archive" / "models.zip"

    df = build_manifest(runs, models_dir, results_dir, archive)

    assert len(df) == 2
    assert set(df["run"]) == {"a_seed0", "b_seed1"}
    assert (results_dir / "models_manifest.csv").exists()
    for _, row in df.iterrows():
        assert row["sha256"] == sha256_file(models_dir / row["run"] / "final_model.zip")
        assert row["num_timesteps"] >= 64
    # архив: один хеш на все строки, совпадает с реально записанным файлом
    assert df["archive_sha256"].nunique() == 1
    assert df["archive_sha256"].iloc[0] == sha256_file(archive)


def test_build_manifest_archive_contains_all_models_with_matching_hashes(tmp_path, dummy_models):
    models_dir, runs = dummy_models
    results_dir = tmp_path / "results"
    for name, _, _ in runs:
        (results_dir / name).mkdir(parents=True)
    archive = tmp_path / "models.zip"

    df = build_manifest(runs, models_dir, results_dir, archive)

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        assert names == {f"{name}/final_model.zip" for name, _, _ in runs}
        for _, row in df.iterrows():
            extracted = zf.read(f"{row['run']}/final_model.zip")
            import hashlib
            assert hashlib.sha256(extracted).hexdigest() == row["sha256"]


def test_build_manifest_does_not_touch_config_files(tmp_path, dummy_models):
    """Старые конфиги пилотного обучения не переписываются: только путь к ним
    попадает в манифест, содержимое не читается и не трогается."""
    models_dir, runs = dummy_models
    results_dir = tmp_path / "results"
    original = '{"marker": "do not touch"}'
    for name, _, _ in runs:
        (results_dir / name).mkdir(parents=True)
        (results_dir / name / "config.json").write_text(original)
    build_manifest(runs, models_dir, results_dir, tmp_path / "models.zip")
    for name, _, _ in runs:
        assert (results_dir / name / "config.json").read_text() == original
