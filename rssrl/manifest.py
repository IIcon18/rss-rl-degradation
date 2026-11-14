"""Манифест финальных моделей пробного обучения (письмо 14.11.2026, п. 1).

Контрольная сумма (SHA-256) и привязка к конфигу/seed для каждой из 4
финальных моделей (`final_model.zip`, не `best_model.zip`: правило «последний
чекпойнт» для основной серии, см. CLAUDE.md). Фактическое число переходов
(`num_timesteps`) читается из самой модели, а не из конфига: полный rollout
при TOTAL_TIMESTEPS=100000 и N_ENVS=12 заканчивается на 102000 (12*170*50),
а не ровно на 100000. Модели упаковываются в один архив для передачи вне git
(models/ в .gitignore); хеш архива тоже попадает в манифест.

Исторические results/pilot_training/<прогон>/config.json не переписываются.
"""

import hashlib
import zipfile
from pathlib import Path

import pandas as pd


def sha256_file(path, chunk_size=1 << 20):
    """SHA-256 файла, читая по частям (модели небольшие, но на будущее)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def model_num_timesteps(path):
    """Фактическое число переходов среды, пройденных при обучении этой модели."""
    from stable_baselines3 import PPO

    return int(PPO.load(str(path), device="cpu").num_timesteps)


def build_manifest(runs, models_dir, results_dir, archive_path, model_filename="final_model.zip"):
    """Собрать манифест и архив моделей.

    runs — список (имя, shield, seed). Для каждого прогона модель ожидается
    в models_dir/<имя>/model_filename, конфиг — в results_dir/<имя>/config.json
    (путь только записывается в манифест, конфиг не читается и не меняется).
    Возвращает DataFrame (та же таблица, что пишется в
    results_dir/models_manifest.csv), архив — в archive_path.
    """
    models_dir, results_dir, archive_path = Path(models_dir), Path(results_dir), Path(archive_path)
    rows = []
    for name, shield, seed in runs:
        model_path = models_dir / name / model_filename
        rows.append({
            "run": name,
            "shield": bool(shield),
            "seed": seed,
            "config_path": str((results_dir / name / "config.json").as_posix()),
            "model_path": str(model_path.as_posix()),
            "num_timesteps": model_num_timesteps(model_path),
            "sha256": sha256_file(model_path),
        })

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in rows:
            zf.write(r["model_path"], arcname=f"{Path(r['run']).as_posix()}/{model_filename}")
    archive_hash = sha256_file(archive_path)
    for r in rows:
        r["archive_path"] = str(archive_path.as_posix())
        r["archive_sha256"] = archive_hash

    df = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "models_manifest.csv", index=False)
    return df


if __name__ == "__main__":
    from rssrl.train_pilot import MODELS_DIR, RESULTS_DIR, SEEDS, SHIELD_VARIANTS, run_name

    runs = [(run_name(sh, s), sh, s) for sh in SHIELD_VARIANTS for s in SEEDS]
    archive = MODELS_DIR / "final_models.zip"
    out = build_manifest(runs, MODELS_DIR, RESULTS_DIR, archive)
    print(out[["run", "shield", "seed", "num_timesteps", "sha256"]].to_string(index=False))
    print("архив:", archive, out["archive_sha256"].iloc[0])
