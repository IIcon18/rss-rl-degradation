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
        "# Этап 6. Эпизоды: исходы, пробег, «лишние» срабатывания фильтра\n\n"
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
        "своим seed — это проверяет тест). Если в этом окне агент столкнулся бы — "
        "вмешательство **нужное**, если нет — **лишнее**.\n\n"
        "Вмешательства с перерывом меньше 1 с считаем одним: иначе «дребезг» фильтра "
        "на границе d_min (+3/−4 м/с² через шаг) дал бы сотни коротких вмешательств.\n\n"
        "Фильтр: d_min — лемма 2 с учетом рывка 10 м/с³. Нарушение (тормозящий лидер, "
        "подрезание) — по расписанию из эталонного прогона, одинаковое для всех вариантов агента."
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
        "PILOT = Path('..') / 'results' / 'pilot_06'  # таблицы пилота, в git не идут\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "SEEDS = range(20)\n"
        "# основной режим — фильтр с задержкой ρ, дополнительный — мгновенный фильтр\n"
        "VARIANTS = [(True, 0.5), (True, 0.0), (False, 0.5)]\n"
        "NAMES = {(True, 0.5): 'фильтр, задержка ρ (основной)', (True, 0.0): 'фильтр, задержка 0 (доп.)',\n"
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
        "в номинальном режиме не используется):\n"
        "- **С фильтром столкновений нет ни в одном сценарии**, ни при одной задержке "
        "(d_min с учетом рывка). Без фильтра — 16–20 из 20.\n"
        "- **Остановка** — главный исход с фильтром в сценах с нарушителем: нарушитель "
        "останавливается и стоит, IDLE с фильтром останавливается за ним и стоит до конца "
        "эпизода (17 из 20 у тормозящего лидера, 13–15 из 20 у подрезания). Это безопасно, "
        "но агент никуда не едет. Без категории «остановка» такие эпизоды попали бы в "
        "«безопасно» и приукрасили бы результат. С d_min леммы 2 как в статье на месте "
        "«остановок» при задержке ρ были столкновения (агент трогался и въезжал в стоящего).\n"
        "- **Съезд с дороги** не встретился ни разу: агент highway-env перестраивается "
        "только между существующими полосами.\n"
        "- **Пробег** меньше там, где агент врезается рано (без фильтра) или стоит."
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
        "    вмешательств=('needed', 'size'),\n"
        "    нужных=('needed', 'sum'),\n"
        "    длительность_с=('duration', 'mean'),\n"
        ")\n"
        "cf_table['доля нужных'] = (cf_table['нужных'] / cf_table['вмешательств']).round(2)\n"
        "cf_table.round(1)"
    ),
    code(
        "by_type = cf.groupby(['сценарий', 'тип'], sort=False).agg(\n"
        "    вмешательств=('needed', 'size'), нужных=('needed', 'sum'))\n"
        "by_type['доля нужных'] = (by_type['нужных'] / by_type['вмешательств']).round(2)\n"
        "by_type"
    ),
    code(
        "# доля нужных по сценариям и задержкам — столбики «нужные / лишние»\n"
        "g = cf.groupby(['сценарий', 'level', 'reaction_delay'], sort=False)['needed'].agg(['sum', 'size']).reset_index()\n"
        "SHORT = {'номинальный, IDLE': 'номин.\\nIDLE', 'номинальный, случайная': 'номин.\\nслучайная',\n"
        "         'тормозящий лидер, IDLE': 'тормоз.\\nβ = {:g}', 'подрезание, IDLE': 'подрез.\\nγ = {:g}'}\n"
        "g['подпись'] = [SHORT[s].format(l) + f\"\\nзадержка {'ρ' if d else '0'}\"\n"
        "              for s, l, d in zip(g['сценарий'], g['level'], g['reaction_delay'])]\n"
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "x = np.arange(len(g))\n"
        "ax.bar(x, g['sum'], color='#2c6fbb', label='нужное (без него — столкновение)')\n"
        "ax.bar(x, g['size'] - g['sum'], bottom=g['sum'], color='#bdc3c7', label='лишнее (без него — без столкновения)')\n"
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
        "- **Доля нужных сильно зависит от длины вмешательства.** С задержкой 0 у IDLE "
        "вмешательства длинные (14–22 с: слитый «дребезг» за машиной впереди), и без них "
        "агент врезается — нужных 70–80 %. С задержкой ρ фильтр тормозит короткими "
        "отрезками (≈ 1 с в номинальном режиме) с перерывами больше 1 с: за ρ агент успевает "
        "отъехать от границы d_min, и вмешательства не сливаются. Без одного такого отрезка "
        "агент за окно 5 с обычно не успевает врезаться — в номинальном режиме нужных всего "
        "2 из 320. Это не значит, что фильтр с задержкой ρ не нужен (без фильтра "
        "столкновений 18 из 20): контрфакт отключает фильтр только на одно вмешательство + 5 с.\n"
        "- **Случайная политика** — нужных около 20 %, в основном лишние запреты "
        "перестроений и короткие торможения.\n"
        "- **Тормозящий лидер и подрезание** при задержке ρ — нужных 20–40 %, при задержке 0 — 70–80 %.\n"
        "Вывод для методики: метрику «доля лишних срабатываний» нужно считать с одним и тем "
        "же правилом слияния и окном для всех вариантов и указывать оба параметра в статье; "
        "сравнивать ее между задержками 0 и ρ напрямую некорректно."
    ),
    md(
        "Откуда у IDLE лишние вмешательства: смотрим, когда они начинаются и сколько длятся."
    ),
    code(
        "idle = cf[cf['сценарий'].str.contains('IDLE')]\n"
        "idle.groupby('needed').agg(\n"
        "    вмешательств=('t_start', 'size'),\n"
        "    начались_в_первые_3_с=('t_start', lambda t: int((t < 3).sum())),\n"
        "    медиана_длительности_с=('duration', 'median'),\n"
        ").rename(index={False: 'лишние', True: 'нужные'})"
    ),
    md(
        "Лишних у IDLE много за счет коротких торможений при задержке ρ (медиана длительности "
        "лишнего ≈ 0,9 с, нужного — ≈ 17 с). В первые 3 с начинается только малая часть "
        "лишних (92 из 745): это не столько стартовая расстановка, сколько повторяющиеся "
        "короткие торможения на границе d_min. Для экспериментов по-прежнему разумно не "
        "считать вмешательства в первые 3 с."
    ),
    md("## Пример «лишнего» торможения: исходный прогон и прогон без вмешательства"),
    code(
        "extra = cf[(~cf.needed) & (cf['тип'] == 'торможение') & (cf.duration > 0.5)].sort_values('duration', ascending=False)\n"
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
        "axes[0].set_title(f\"{ex['сценарий']}, сцена {int(ex.seed)}, {NAMES[(True, ex.reaction_delay)]}: лишнее торможение\", fontsize=10)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '06_extra_intervention.png', dpi=150)\n"
        "ex[['сценарий', 'seed', 't_start', 't_end', 'brake_steps', 'lc_steps', 'min_gap']]"
    ),
    md(
        "Без вмешательства зазор опускается ниже d_min (по RSS ситуация опасная), но "
        "столкновения не происходит: машина впереди не тормозит изо всех сил. Это и есть "
        "смысл «лишнего» срабатывания: RSS страхует от худшего случая, который в этот раз "
        "не наступил. «Лишнее» здесь — не ошибка фильтра, а цена гарантии."
    ),
    md(
        "## Ограничения этой проверки (для статьи)\n\n"
        f"1. Окно — время вмешательства + 5 с. Столкновение позже окна не учитывается; "
        "при большем окне доля нужных была бы выше.\n"
        "2. Вмешательства с перерывом меньше 1 с объединены. У IDLE это дает длинные "
        "вмешательства (десятки секунд), и вердикт «нужное» относится ко всему отрезку.\n"
        "3. Контрфакт отвечает на вопрос «было бы столкновение», а не «была бы опасная "
        "по RSS ситуация». Второе почти всегда «да» — фильтр срабатывает только при опасности.\n"
        "4. Политика в контрфакте та же (IDLE или та же последовательность случайных "
        "действий). Для PPO надо будет запрашивать политику по наблюдениям контрфактического "
        "прогона — `run_episode` это уже позволяет."
    ),
    md(
        "## Выводы этапа 6\n\n"
        "1. Исходы эпизодов разделены на четыре категории. С фильтром (d_min с учетом рывка) "
        "столкновений нет ни в одном сценарии; «остановка» — главный исход в сценах с "
        "нарушителем (агент стоит за остановившимся нарушителем).\n"
        "2. Пробег считается по факту и есть в таблице исходов.\n"
        "3. Контрфактическая проверка работает (с расписанием нарушения сцена до момента "
        "отключения фильтра совпадает с исходной — это проверяет тест), но доля «нужных» "
        "сильно зависит от правила слияния вмешательств (1 с) и окна (5 с): при задержке ρ "
        "вмешательства короткие, и большинство выглядят «лишними». Эти параметры надо "
        "зафиксировать до основного эксперимента и обсудить с преподавателем."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("06_episodes.ipynb")
ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
