"""Собирает и выполняет notebooks/09_diagnostics.ipynb.

Запуск из корня проекта: .venv/Scripts/python notebooks/make_09_diagnostics.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Этап 9. Диагностика замороженных финальных моделей пробного обучения\n\n"
        "Письмо 14.11.2026, п. 3. Код: `rssrl/diagnostics.py` (прогон и сводка), "
        "`rssrl/manifest.py` (контрольные суммы и привязка моделей к конфигам). "
        "Проверки: `tests/test_diagnostics.py`, `tests/test_manifest.py`.\n\n"
        "**Что это и что нет.** 4 финальные модели пробного обучения "
        "(`results/pilot_training/models_manifest.csv`: 2 типа — «с базовым фильтром» / "
        "«без фильтра» — × 2 seed обучения) исполняются в двух режимах — с фильтром и "
        "без — независимо от того, как они обучались: перекрестное исполнение отделяет "
        "поведение самой политики от непосредственной работы защиты. Это **описательная "
        "диагностика** (как `describe_2x2` в `rssrl/stats.py`): без p-значений и знаков, "
        "выводов о механизме привыкания не делает.\n\n"
        "Сцены — **номинальный режим** (нарушителя нет, фон контролируемого следования, "
        "то же, на чем модели обучались и оценивались), отдельный общий набор из 100 "
        "сцен, не пересекающийся ни со сценами обучения, ни со сценами оценки во время "
        "обучения (`DIAG_SEED_OFFSET`). В номинальном режиме с фильтром частота "
        "столкновений ожидаемо около нуля (`tests/test_nominal.py`) — здесь это не "
        "измеряется заново, а непрерывные показатели (скорость, зазор/d_min, доля "
        "времени вмешательства фильтра, частоты действий политики) — вот что действительно "
        "различает политики и режимы исполнения.\n\n"
        "**Две оговорки для статьи (письмо 14.11), которые нужно сохранить и здесь:** "
        "досрочное окончание эпизода не равнозначно столкновению (это разные категории "
        "исхода, см. `classify_outcome`); полный эпизод длиной 40 с сам по себе не "
        "исключает столкновение на последнем шаге — `collision` в таблице ниже проверяется "
        "отдельно, а не выводится из длины эпизода."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "import subprocess\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.diagnostics import ACTION_NAMES, summarize\n\n"
        "RES = Path('..') / 'results'\n"
        "FIG = RES / 'figures'\n"
        "DIAG = RES / 'pilot_diag'\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "manifest = pd.read_csv(RES / 'pilot_training' / 'models_manifest.csv')\n"
        "manifest[['run', 'shield', 'seed', 'num_timesteps', 'sha256']]"
    ),
    md(
        "## Прогон диагностики\n\n"
        "100 номинальных сцен × 4 модели × 2 режима исполнения = 800 эпизодов. "
        "Отдельным процессом (`python -m rssrl.diagnostics`): параллельные среды из "
        "ядра Jupyter зависают, как и в этапе 8. Готовый результат не перезаписываем."
    ),
    code(
        "if not (DIAG / 'episodes.csv').exists():\n"
        "    subprocess.run([sys.executable, '-m', 'rssrl.diagnostics'], cwd='..', check=True)\n"
        "episodes = pd.read_csv(DIAG / 'episodes.csv')\n"
        "len(episodes)"
    ),
    md("## Сводка по (модель, режим исполнения)"),
    code(
        "summary = summarize(episodes)\n"
        "summary['вариант'] = ['обучена ' + ('с фильтром' if ts else 'без фильтра') + f', seed {sd}, исполняется '\n"
        "                      + ('с фильтром' if ex else 'без фильтра')\n"
        "                      for ts, sd, ex in zip(summary.trained_shield, summary.trained_seed, summary.exec_shield)]\n"
        "show = summary[['вариант', 'episodes', 'collision_rate', 'offroad_rate', 'mean_speed',\n"
        "                'shield_frac', 'gap_ratio_median', 'gap_ratio_frac_below_1']].round(4)\n"
        "show"
    ),
    md(
        "Как читать:\n"
        "- `collision_rate` / `offroad_rate` — доля из 100 сцен. Ожидаемо около нуля при "
        "исполнении с фильтром (номинальный режим); без фильтра на исполнении зависит "
        "от того, как эта конкретная политика ведет себя без защиты.\n"
        "- `shield_frac` — доля времени эпизода, когда фильтр реально тормозил (0 при "
        "исполнении без фильтра по определению).\n"
        "- `gap_ratio_median`, `gap_ratio_frac_below_1` — медиана и доля времени, когда "
        "зазор до критичной машины впереди меньше d_min (быстрее формула фильтра, см. "
        "`rssrl/shield.py`): насколько консервативно или впритык едет политика."
    ),
    md("## Скорость и запас по зазору"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "labels = show['вариант'].str.replace('обучена ', '').str.replace(', исполняется', '\\nиспол.')\n"
        "x = np.arange(len(summary))\n"
        "axes[0].bar(x, summary['mean_speed'], color='#2c6fbb')\n"
        "axes[0].set_ylabel('средняя скорость, м/с')\n"
        "axes[1].bar(x, summary['gap_ratio_frac_below_1'], color='#e67e22')\n"
        "axes[1].set_ylabel('доля времени зазор < d_min')\n"
        "for ax in axes:\n"
        "    ax.set_xticks(x, labels, fontsize=6, rotation=90)\n"
        "    ax.grid(axis='y', alpha=0.3)\n"
        "fig.suptitle('Диагностика 4 моделей x 2 режима исполнения, 100 номинальных сцен', fontsize=10)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '09_speed_gap.png', dpi=150)"
    ),
    md("## Частоты действий политики"),
    code(
        "act_cols = [f'action_frac_{n}' for n in ACTION_NAMES.values()]\n"
        "act = episodes.groupby(['run', 'exec_shield'])[act_cols].mean()\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "bottom = np.zeros(len(act))\n"
        "colors = {'LANE_LEFT': '#8e44ad', 'IDLE': '#bdc3c7', 'LANE_RIGHT': '#16a085',\n"
        "          'FASTER': '#2c6fbb', 'SLOWER': '#c0392b'}\n"
        "xl = [f'{r}\\nисп.{\"+щ\" if e else \"-щ\"}' for r, e in act.index]\n"
        "for name in ACTION_NAMES.values():\n"
        "    vals = act[f'action_frac_{name}'].to_numpy()\n"
        "    ax.bar(range(len(act)), vals, bottom=bottom, label=name, color=colors[name])\n"
        "    bottom += vals\n"
        "ax.set_xticks(range(len(act)), xl, fontsize=7)\n"
        "ax.set_ylabel('доля решений политики')\n"
        "ax.legend(fontsize=7, frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.15))\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '09_actions.png', dpi=150)"
    ),
    md(
        "## Ограничения (для статьи)\n\n"
        "1. Только номинальный режим: диагностика не оценивает частоту столкновений при "
        "нарушении и не заменяет основной β-эксперимент.\n"
        "2. Перекрестное исполнение описывает поведение, но само по себе не доказывает "
        "механизм привыкания политики к фильтру — для этого нужна отдельная причинная "
        "проверка на сценах с нарушением (основной эксперимент).\n"
        "3. 2 seed обучения на тип — различия между seed одного типа в этой таблице "
        "дают только грубое представление о разбросе между обучениями, не точную оценку."
    ),
    md(
        "## Объем для семьи A (письмо 14.11, п. 4)\n\n"
        "`rssrl/volume.py`: `family_a_power` — симуляцией, вероятность, что **всё** правило "
        "семьи A (поправка Холма по 6 сравнениям + подтверждение «хуже» на следующем β, "
        "`rssrl.stats.decide_family` + `family_threshold`) найдет именно заданный истинный "
        "порог β*. `projected_halfwidth` считает полуширину ОДНОГО интервала и такой "
        "мощности не заменяет (это и есть замечание преподавателя).\n\n"
        "**Откуда берутся числа для симуляции.** Диагностика выше — номинальный режим, где "
        "столкновений с фильтром практически нет; она не может дать оценку разброса между "
        "обучениями для вероятности столкновения ПРИ НАРУШЕНИИ — такой величины в проекте "
        "пока ни у кого нет (основной β-эксперимент не проводился). Поэтому `policy_spread` "
        "(стандартное отклонение вероятности столкновения между политиками одного типа) и "
        "сам эффект (`baseline`, `delta`, `beta_onset`) — ПРЕДПОЛАГАЕМЫЕ параметры модели, "
        "а не оценки из данных; ниже — сетка по ним, а не одно число. Диагностика дает "
        "«полезную первичную информацию» в другом смысле (замечание преподавателя): по двум "
        "seed на тип видно, что политики одного типа уже по-разному ведут себя в номинале "
        "(видно в таблице выше) — качественное основание, что разброс между обучениями "
        "вообще есть, но не его величину для столкновений при нарушении."
    ),
    code(
        "from rssrl.volume import family_a_power, family_a_precision, step_effect\n\n"
        "N_REP, N_BOOT = 150, 1000  # повторов симуляции и бутстрепа внутри каждого повтора\n"
        "N_POLICIES = [2, 3, 5]\n"
        "N_SCENES = [100, 300, 500]"
    ),
    md(
        "### Мощность при обнаружимом эффекте (β* = 1,4, не верхняя точка сетки)\n\n"
        "Допуск ±2 п.п. (`MARGIN`) — заранее заданная конвенция проекта для этого "
        "симуляционного исследования, не нормативный предел дорожного риска (письмо 14.11); "
        "здесь ее роль — порог практической величины внутри `decide_family`, на обнаружение "
        "статистического знака впрямую не влияет."
    ),
    code(
        "rows = []\n"
        "for delta in (0.02, 0.05, 0.10):\n"
        "    for spread in (0.0, 0.01):\n"
        "        eff = step_effect(1.4, delta, baseline=0.03)\n"
        "        df = family_a_power(eff, 1.4, N_POLICIES, N_SCENES, policy_spread=spread,\n"
        "                            n_rep=N_REP, n_boot=N_BOOT, seed=hash((delta, spread)) % 2**31)\n"
        "        df['эффект (дельта), п.п.'] = round(100 * delta, 1)\n"
        "        rows.append(df)\n"
        "power_table = pd.concat(rows, ignore_index=True)\n"
        "power_table.to_csv(DIAG / 'family_a_power.csv', index=False)\n"
        "power_table.pivot_table(index=['эффект (дельта), п.п.', 'разброс политик (std)'],\n"
        "                        columns=['политик на тип', 'сцен'], values='мощность').round(2)"
    ),
    md(
        "### Исход «не установлено» под нулевым эффектом и при ухудшении только на верхней точке сетки\n\n"
        "Оба случая — ожидаемый, допустимый результат правила (письмо 07.11, `beta_star`): "
        "если «хуже» впервые появляется только на β = 2, подтвердить его на следующем "
        "уровне нечем. Мощность здесь — доля повторов, где правило КОРРЕКТНО возвращает "
        "«не установлено»."
    ),
    code(
        "no_effect = step_effect(10.0, 0.0, baseline=0.03)  # эффекта нет ни на одном уровне сетки\n"
        "top_only = step_effect(2.0, 0.08, baseline=0.03)   # «хуже» только на верхней точке\n"
        "rows = []\n"
        "for name, eff in [('нет эффекта', no_effect), ('эффект только при β = 2', top_only)]:\n"
        "    df = family_a_power(eff, None, [3, 5], [300, 500], policy_spread=0.0, n_rep=N_REP, n_boot=N_BOOT, seed=7)\n"
        "    df['случай'] = name\n"
        "    rows.append(df)\n"
        "none_table = pd.concat(rows, ignore_index=True)[['случай', 'политик на тип', 'сцен', 'мощность']]\n"
        "none_table.to_csv(DIAG / 'family_a_power_none.csv', index=False)\n"
        "none_table"
    ),
    md(
        "### Точность (не мощность): полуширина совместного интервала семьи\n\n"
        "Под нулевым эффектом — медианная полуширина СОВМЕСТНОГО интервала "
        "(`ci_simult`, уровень 1 − α/6), а не обычного 95% интервала по умолчанию "
        "(так его считает `projected_halfwidth`, и это другая величина)."
    ),
    code(
        "rows = []\n"
        "for n_pol in N_POLICIES:\n"
        "    for n_sc in N_SCENES:\n"
        "        w = family_a_precision(n_pol, n_sc, n_rep=N_REP, n_boot=N_BOOT, seed=11)\n"
        "        rows.append({'политик на тип': n_pol, 'сцен': n_sc, 'точность (± п.п.)': round(100 * w, 2)})\n"
        "precision_table = pd.DataFrame(rows)\n"
        "precision_table.to_csv(DIAG / 'family_a_precision.csv', index=False)\n"
        "precision_table.pivot(index='политик на тип', columns='сцен', values='точность (± п.п.)')"
    ),
    md(
        "## Выводы этапа 9\n\n"
        "1. **Диагностика** (`results/pilot_diag/episodes.csv`, `summary.csv`): 4 финальные "
        "модели заметно различаются по непрерывным показателям (скорость, запас по зазору, "
        "частоты действий) даже в номинальном режиме, причем по-разному реагируют на "
        "отключение фильтра при исполнении — это описательный результат, не доказательство "
        "механизма привыкания.\n"
        "2. **Манифест моделей** (`results/pilot_training/models_manifest.csv`): SHA-256 и "
        "фактическое число переходов (102000 у всех четырех — полный rollout после 100000) "
        "для `final_model.zip`; правило «последний чекпойнт» закреплено в CLAUDE.md для "
        "основной серии, исторические конфиги пилота не менялись.\n"
        "3. **Причина завершения эпизода и вмешательства фильтра** — теперь в info "
        "`TrainingEnv` и (через `info_keywords`) в `monitor.csv` для обучения и оценки; "
        "это инфраструктура для основной серии, уже прошедший пилот не перезапускался.\n"
        "4. **Объем для семьи A**: мощность всего правила (Холм + подтверждение на "
        "следующем β) и отдельно точность на уровне совместного интервала — сеткой по "
        "предполагаемому эффекту и разбросу между политиками (не оценка из данных: "
        "диагностика в номинальном режиме не может дать вероятность столкновения при "
        "нарушении). Итоговые числа и чувствительность — в тексте письма преподавателю, "
        "исход «не установлено» заранее принят как допустимый."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("09_diagnostics.ipynb")
ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
