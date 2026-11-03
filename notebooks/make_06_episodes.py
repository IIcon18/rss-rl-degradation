"""Собирает и выполняет notebooks/06_episodes.ipynb.

Запуск из корня проекта: .venv/bin/python notebooks/make_06_episodes.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Этап 6. Эпизоды: исходы, пробег, контрфактическая проверка вмешательств фильтра\n\n"
        "Код: `classify_outcome`, `run_episode` в `rssrl/scenarios.py`; `interventions`, "
        "`counterfactual`, `counterfactual_grid` в `rssrl/pilot.py`. "
        "Проверки: `tests/test_episodes.py`.\n\n"
        "**Исход эпизода** (highway-v0 ограничен 40 с), в порядке приоритета:\n"
        "- *столкновение* — агент столкнулся (эпизод на этом заканчивается);\n"
        "- *съезд с дороги* — агент хотя бы раз оказался вне дороги;\n"
        "- *остановка* — агент стоял (скорость < 0,5 м/с) не меньше 2 с подряд;\n"
        "- *безопасно* — доехал до конца времени без всего перечисленного.\n\n"
        "**Пробег** — сколько агент проехал по факту, до конца времени или до столкновения.\n\n"
        "**Контрфактическая проверка.** Каждое вмешательство фильтра (торможение или "
        "запрет/отмена перестроения) проверяем так: заново прогоняем ту же сцену с той же "
        "политикой, но с момента вмешательства фильтр отключен — на все время вмешательства "
        "и еще 5 с после. До этого момента прогон совпадает с исходным (сцена детерминирована "
        "своим seed — это проверяет тест). Результат — диагностика на выбранном горизонте: "
        "«без вмешательства — столкновение на горизонте 5 с» или «нет столкновения на "
        "горизонте 5 с». Второе не доказывает, что вмешательство было ненужным или нарушало "
        "правила RSS: RSS страхует от худшего случая, который в этом окне мог не наступить.\n\n"
        "Вмешательства с перерывом меньше 1 с считаем одним: иначе «дребезг» фильтра "
        "на границе d_min (+3/−4 м/с² через шаг) дал бы сотни коротких вмешательств.\n\n"
        "Повтор после письма от 29.10. Сцена — контролируемое следование (фон не "
        "перестраивается, держит d_min). Фильтр — базовая надлежащая реакция с учетом рывка "
        "(без аварийного торможения), d_min — лемма 2 с учетом рывка. Нарушение (тормозящий "
        "лидер, подрезание) — по расписанию из файлов `results/schedules/` (эталон IDM/MOBIL "
        "+ RSS), одинаковое для всех вариантов агента."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.pilot import run_grid, counterfactual_grid, _run_one, MERGE_GAP, CF_HORIZON\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_06'  # первичные данные: CSV в git, PNG — нет\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "SEEDS = range(20)\n"
        "# основной режим — фильтр с задержкой ρ, дополнительный — мгновенный фильтр\n"
        "VARIANTS = [(True, 0.5), (True, 0.0), (False, 0.5)]\n"
        "NAMES = {(True, 0.5): 'базовая, задержка ρ (основной)', (True, 0.0): 'базовая, задержка 0 (доп.)',\n"
        "         (False, 0.5): 'без фильтра'}\n"
        "# (сценарий, уровни, политика, подпись)\n"
        "SETUPS = [\n"
        "    ('nominal', [1.0], 'idle', 'номинальный, IDLE'),\n"
        "    ('nominal', [1.0], 'random', 'номинальный, случайная'),\n"
        "    ('brake', [1.0, 2.0], 'idle', 'тормозящий лидер, IDLE'),\n"
        "    ('cutin', [1.0, 0.25], 'idle', 'подрезание, IDLE'),\n"
        "]"
    ),
    md("## Исходы эпизодов и пробег"),
    code(
        "frames, LOGS = [], {}\n"
        "for kind, levels, policy, label in SETUPS:\n"
        "    o, logs = run_grid(SEEDS, levels, VARIANTS, kind=kind, policy=policy, trigger='schedule')\n"
        "    o['сценарий'] = label\n"
        "    o['уровень'] = o['gamma'] if kind == 'cutin' else o['beta']\n"
        "    frames.append(o)\n"
        "    LOGS[label] = (kind, policy, logs)\n"
        "outcomes = pd.concat(frames, ignore_index=True)\n"
        "outcomes['вариант'] = [NAMES[(s, d)] for s, d in zip(outcomes['shield'], outcomes['reaction_delay'])]\n"
        "outcomes.drop(columns=['crash_partners']).to_csv(PILOT / 'outcomes.csv', index=False)\n"
        "\n"
        "table = (outcomes.groupby(['сценарий', 'уровень', 'вариант'], sort=False)['outcome']\n"
        "         .value_counts().unstack(fill_value=0)\n"
        "         .reindex(columns=['safe', 'stopped', 'offroad', 'collision'], fill_value=0)\n"
        "         .rename(columns={'safe': 'безопасно', 'stopped': 'остановка',\n"
        "                          'offroad': 'съезд', 'collision': 'столкновение'}))\n"
        "table['пробег, км (среднее)'] = outcomes.groupby(['сценарий', 'уровень', 'вариант'], sort=False)['distance_km'].mean().round(3)\n"
        "table"
    ),
    md(
        "Как читать таблицу (уровень — β для тормозящего лидера, γ для подрезания, "
        "в номинальном режиме не используется; 20 сцен):\n"
        "- **Номинальный режим**: с фильтром 0 столкновений, все эпизоды «безопасно», "
        "пробег ≈ 0,87–0,89 км. Без фильтра — 20 и 19 из 20.\n"
        "- **Тормозящий лидер**: при β = 1 с фильтром столкновений нет, главный исход — "
        "«остановка» (агент стоит за остановившимся нарушителем: 18 и 14 из 20). При β = 2 "
        "у основного варианта (задержка ρ) 10 из 20 столкновений, у мгновенного фильтра 0.\n"
        "- **Подрезание**: при γ = 1 — 0 столкновений, все эпизоды «остановка». При "
        "γ = 0,25 базовая реакция без аварийного торможения не спасает: 20 из 20 (ρ) и "
        "17 из 20 (задержка 0).\n"
        "- **Остановка** — отдельная категория: без нее эти эпизоды попали бы в "
        "«безопасно» и приукрасили бы результат (агент стоит и никуда не едет).\n"
        "- **Съезд с дороги** не встретился ни разу.\n"
        "- Без фильтра у тормозящего лидера в 5 сценах из 20 IDLE врезается до "
        "воздействия; эпизоды остаются в выборке."
    ),
    md("## Контрфактическая проверка вмешательств"),
    code(
        "cf_frames = []\n"
        "for label, (kind, policy, logs) in LOGS.items():\n"
        "    cf = counterfactual_grid(logs, kind=kind, policy=policy, trigger='schedule')\n"
        "    cf['сценарий'] = label\n"
        "    cf_frames.append(cf)\n"
        "cf = pd.concat(cf_frames, ignore_index=True)\n"
        "cf['вариант'] = [NAMES[(True, d)] for d in cf['reaction_delay']]\n"
        "cf['тип'] = np.where(cf['brake_steps'] > 0, 'торможение', 'только перестроение')\n"
        "cf.drop(columns=['crash_partners']).to_csv(PILOT / 'counterfactual.csv', index=False)\n"
        "\n"
        "cf_table = cf.groupby(['сценарий', 'level', 'вариант'], sort=False).agg(\n"
        "    вмешательств=('collision', 'size'),\n"
        "    столкновение_без_вмешательства_5с=('collision', 'sum'),\n"
        "    длительность_с=('duration', 'mean'),\n"
        ")\n"
        "cf_table['доля со столкновением без вмешательства'] = (cf_table['столкновение_без_вмешательства_5с'] / cf_table['вмешательств']).round(2)\n"
        "cf_table.round(1)"
    ),
    code(
        "by_type = cf.groupby(['сценарий', 'тип'], sort=False).agg(\n"
        "    вмешательств=('collision', 'size'), столкновение_без_вмешательства_5с=('collision', 'sum'))\n"
        "by_type['доля'] = (by_type['столкновение_без_вмешательства_5с'] / by_type['вмешательств']).round(2)\n"
        "by_type"
    ),
    code(
        "# по сценариям и задержкам: со столкновением без вмешательства / без столкновения на 5 с\n"
        "g = cf.groupby(['сценарий', 'level', 'reaction_delay'], sort=False)['collision'].agg(['sum', 'size']).reset_index()\n"
        "SHORT = {'номинальный, IDLE': 'номин.\\nIDLE', 'номинальный, случайная': 'номин.\\nслучайная',\n"
        "         'тормозящий лидер, IDLE': 'тормоз.\\nβ = {:g}', 'подрезание, IDLE': 'подрез.\\nγ = {:g}'}\n"
        "g['подпись'] = [SHORT[s].format(l) + f\"\\nзадержка {'ρ' if d else '0'}\"\n"
        "              for s, l, d in zip(g['сценарий'], g['level'], g['reaction_delay'])]\n"
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "x = np.arange(len(g))\n"
        "ax.bar(x, g['sum'], color='#2c6fbb', label='без вмешательства — столкновение на горизонте 5 с')\n"
        "ax.bar(x, g['size'] - g['sum'], bottom=g['sum'], color='#bdc3c7', label='нет столкновения на горизонте 5 с')\n"
        "for xi, (s, n) in enumerate(zip(g['sum'], g['size'])):\n"
        "    ax.text(xi, n + 1, f'{s}/{n}', ha='center', fontsize=8)\n"
        "ax.set_xticks(x, g['подпись'], fontsize=8)\n"
        "ax.set_ylabel('вмешательств фильтра')\n"
        "ax.legend(frameon=False, fontsize=8)\n"
        "ax.set_title(f'Контрфактическая проверка: фильтр отключен на время вмешательства + {CF_HORIZON:g} с (20 сцен)', fontsize=10)\n"
        "ax.grid(axis='y', alpha=0.3)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '06_counterfactual.png', dpi=150)"
    ),
    md(
        "Как читать:\n"
        "- Результат каждой строки — диагностика на горизонте 5 с: «без вмешательства — "
        "столкновение на горизонте 5 с» или «нет столкновения на горизонте 5 с». Второе "
        "**не** значит, что вмешательство было ненужным или нарушало правила RSS.\n"
        "- Исход сильно зависит от длины вмешательства и правила слияния (1 с). С задержкой "
        "0 у IDLE фильтр тормозит почти непрерывно, вмешательства сливаются в одно на эпизод "
        "(≈ 30 с), и без него агент врезается в 100 % случаев. С задержкой ρ фильтр тормозит "
        "короткими отрезками (≈ 1 с в номинальном режиме), и без одного такого отрезка агент "
        "за 5 с не успевает врезаться (номинальный IDLE: 0 из 298). Это не значит, что "
        "фильтр с задержкой ρ не нужен: без фильтра IDLE врезается в 20 из 20.\n"
        "- У тормозящего лидера и подрезания доля «столкновение без вмешательства» выше "
        "(33–100 %).\n"
        "Вывод для методики: эту метрику считаем с одним и тем же правилом слияния и окном "
        "для всех вариантов, оба параметра указываем в статье, а прямо сравнивать ее между "
        "задержками 0 и ρ некорректно."
    ),
    md(
        "Когда начинаются и сколько длятся вмешательства IDLE — с исходом контрфакта и без."
    ),
    code(
        "idle = cf[cf['сценарий'].str.contains('IDLE')]\n"
        "idle.groupby('collision').agg(\n"
        "    вмешательств=('t_start', 'size'),\n"
        "    начались_в_первые_3_с=('t_start', lambda t: int((t < 3).sum())),\n"
        "    медиана_длительности_с=('duration', 'median'),\n"
        ").rename(index={False: 'нет столкновения на 5 с', True: 'столкновение без вмешательства'})"
    ),
    md(
        "У IDLE вмешательства без столкновения на горизонте 5 с — в основном короткие "
        "торможения при задержке ρ (медиана ≈ 0,9 с), со столкновением — длинные (медиана "
        "≈ 11 с). В первые 3 с эпизода начинается малая часть вмешательств (18 и 12): на "
        "контролируемом следовании старт уже не создает опасных ситуаций."
    ),
    md("## Пример торможения, без которого нет столкновения на горизонте 5 с"),
    code(
        "extra = cf[(cf.no_collision_5s) & (cf['тип'] == 'торможение') & (cf.duration > 0.5)].sort_values('duration', ascending=False)\n"
        "ex = extra.iloc[0]\n"
        "kind, policy, logs = LOGS[ex['сценарий']]\n"
        "job = (int(ex.seed), ex.level, True, ex.reaction_delay)\n"
        "orig = logs[job]\n"
        "_, cf_log = _run_one(job, kind, policy, shield_off_at=ex.t_start, t_stop=ex.t_end + CF_HORIZON, trigger='schedule')\n"
        "lo, hi = ex.t_start - 3, ex.t_end + CF_HORIZON\n"
        "fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)\n"
        "for log, style, name in [(orig, '-', 'с фильтром (как было)'), (cf_log, '--', 'без вмешательства')]:\n"
        "    w = log[(log.t >= lo) & (log.t <= hi)]\n"
        "    axes[0].plot(w.t, w.gap.where(w.gap < 1e6), style, color='#2c6fbb', label=f'зазор, {name}')\n"
        "    axes[0].plot(w.t, w.d_min.where(w.gap < 1e6), style, color='#e67e22', label=f'$d_{{min}}$, {name}')\n"
        "    axes[1].plot(w.t, w.ego_v, style, color='#2c6fbb', label=name)\n"
        "for ax in axes:\n"
        "    ax.axvspan(ex.t_start, ex.t_end, color='0.85', linewidth=0)\n"
        "    ax.grid(alpha=0.3)\n"
        "axes[0].set_ylabel('м'); axes[0].set_ylim(bottom=0)\n"
        "axes[1].set_ylabel('скорость агента, м/с')\n"
        "axes[1].set_xlabel('время, с (серое — вмешательство фильтра)')\n"
        "axes[0].legend(fontsize=7, frameon=False, ncol=2)\n"
        "axes[1].legend(fontsize=8, frameon=False)\n"
        "axes[0].set_title(f\"{ex['сценарий']}, сцена {int(ex.seed)}, {NAMES[(True, ex.reaction_delay)]}: без вмешательства нет столкновения на 5 с\", fontsize=10)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '06_extra_intervention.png', dpi=150)\n"
        "ex[['сценарий', 'seed', 't_start', 't_end', 'brake_steps', 'lc_steps', 'min_gap']]"
    ),
    md(
        "Без вмешательства зазор опускается ниже d_min (по RSS ситуация опасная), но "
        "на горизонте 5 с столкновения нет: в этот раз худший случай, от которого страхует "
        "RSS, не наступил. Это диагностический результат на выбранном окне, а не "
        "доказательство, что торможение было лишним."
    ),
    md(
        "## Ограничения этой проверки (для статьи)\n\n"
        f"1. Окно — время вмешательства + 5 с. Столкновение позже окна не учитывается; "
        "при большем окне столкновений без вмешательства было бы больше.\n"
        "2. Вмешательства с перерывом меньше 1 с объединены. У IDLE это дает длинные "
        "вмешательства (десятки секунд), и исход контрфакта относится ко всему отрезку.\n"
        "3. Контрфакт отвечает на вопрос «было бы столкновение», а не «была бы опасная "
        "по RSS ситуация». Второе почти всегда «да» — фильтр срабатывает только при опасности.\n"
        "4. Политика в контрфакте та же (IDLE или та же последовательность случайных "
        "действий). Для PPO надо будет запрашивать политику по наблюдениям контрфактического "
        "прогона — `run_episode` это уже позволяет."
    ),
    md(
        "## Выводы этапа 6\n\n"
        "1. Исходы эпизодов разделены на четыре категории; «остановка» — главный исход с "
        "фильтром в сценах с нарушителем при β = 1 и γ = 1.\n"
        "2. Пробег считается по факту и есть в таблице исходов.\n"
        "3. С основным фильтром (базовая надлежащая реакция) столкновения появляются там, "
        "где допущение нарушено сильно: β = 2 (10 из 20 при задержке ρ) и γ = 0,25 (20 из 20). "
        "При соблюденном допущении (номинальный режим, β = 1, γ = 1) — 0.\n"
        "4. Контрфактическая проверка называется теперь «нет столкновения на горизонте 5 с» "
        "и трактуется как диагностика. Ее результат зависит от правила слияния (1 с) и окна "
        "(5 с); эти параметры фиксируем до основного эксперимента.\n"
        "5. Первичные данные: `results/pilot_06/outcomes.csv`, `counterfactual.csv`."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("06_episodes.ipynb")
ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
