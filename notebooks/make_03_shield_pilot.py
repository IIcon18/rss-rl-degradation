"""Собирает и выполняет notebooks/03_shield_pilot.ipynb.

Запуск из корня проекта: .venv/bin/python notebooks/make_03_shield_pilot.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Этап 3. RSS-фильтр на нижнем уровне: пилот β = 1 и β = 2\n\n"
        "Код: `rssrl/shield.py` (фильтр), `rssrl/traffic.py` (фон), `rssrl/scenarios.py` "
        "(сцена), `rssrl/schedules.py` (расписания), `rssrl/responsibility.py` "
        "(ответственность). Проверки: `tests/test_shield.py`, `tests/test_dynamics.py`, "
        "`tests/test_traffic.py`, `tests/test_schedules.py`, `tests/test_responsibility.py`.\n\n"
        "Повтор технического пилота после письма от 29.10.\n\n"
        "**Агент без PPO**: всегда IDLE — держит полосу и целевую скорость 25 м/с, на машину "
        "впереди сам не реагирует. Вся безопасность — только от фильтра.\n\n"
        "**Сцена — контролируемое следование** (`traffic='following'`): фон не "
        "перестраивается и сам держит d_min по RSS (тормозит не слабее 4 и не сильнее "
        "5 м/с²), все начальные зазоры не меньше d_min по общему правилу от seed. "
        "40 с, сцены 0–39.\n\n"
        "**Нарушение — по расписанию из файла** `results/schedules/brake_following_0-499.csv`, "
        "построенному заранее эталоном IDM/MOBIL + RSS-фильтр: момент — по seed на "
        "5–30 с, нарушитель — машина, за которой эталон ехал в этот момент. Расписание "
        "одно для всех вариантов агента и всех β; ничего не исключаем. Нарушитель "
        "набирает торможение β·5 м/с² с рывком ≤ 10 м/с³, тормозит до остановки и стоит.\n\n"
        "**Варианты агента** (одни и те же 40 сцен):\n"
        "- базовая надлежащая реакция с учетом рывка, задержка ρ = 0,5 с — **основной**;\n"
        "- базовая надлежащая реакция, задержка 0 — дополнительный;\n"
        "- тот же фильтр с дополнительным аварийным торможением до 8 м/с², задержка ρ — "
        "сравнение;\n"
        "- без фильтра."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.pilot import run_variants, export_pilot, plot_episode, variant_slug\n"
        "from rssrl.responsibility import responsibility_table\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_03'  # первичные данные: CSV в git, PNG — нет\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "SEEDS = range(40)\n"
        "VARIANTS = [\n"
        "    ('базовая, задержка ρ (основной)', True, 0.5, False),\n"
        "    ('базовая, задержка 0 (доп.)', True, 0.0, False),\n"
        "    ('с аварийным торможением, ρ (сравнение)', True, 0.5, True),\n"
        "    ('без фильтра', False, 0.5, False),\n"
        "]\n"
        "ORDER = [v[0] for v in VARIANTS]"
    ),
    code(
        "outcomes, logs = run_variants(SEEDS, [1.0, 2.0], VARIANTS, trigger='schedule')\n"
        "for mode in ['brake', 'emergency']:\n"
        "    outcomes[f'доля_{mode}'] = [(log['shield_mode'] == mode).mean() for log in logs.values()]\n"
        "table = outcomes.pivot_table(index='вариант', columns='beta', values='collision', aggfunc='sum')\n"
        "table.columns = [f'β = {b:g}' for b in table.columns]\n"
        "print(f'Число эпизодов со столкновением из {len(SEEDS)}:')\n"
        "table.loc[ORDER]"
    ),
    md(
        "## Состоялось ли воздействие и какую роль играла назначенная машина\n\n"
        "Все эпизоды остаются в выборке. Отдельно показываем: столкновения до момента "
        "воздействия, несостоявшиеся воздействия и роль назначенной машины для этого "
        "агента в момент воздействия (совпадение времени и номера машины не гарантирует, "
        "что она лидер именно этого агента)."
    ),
    code(
        "imp = outcomes.pivot_table(index=['вариант', 'beta'], columns='impact', values='seed',\n"
        "                           aggfunc='size', fill_value=0, sort=False)\n"
        "role = outcomes.pivot_table(index=['вариант', 'beta'], columns='violator_role', values='seed',\n"
        "                            aggfunc='size', fill_value=0, sort=False)\n"
        "gap = outcomes.groupby(['вариант', 'beta'], sort=False)['gap_at_trigger'].describe()[['min', '50%', 'max']]\n"
        "gap.columns = ['зазор мин, м', 'зазор медиана, м', 'зазор макс, м']\n"
        "pd.concat([imp, role, gap.round(1)], axis=1)"
    ),
    code(
        "summary = outcomes.groupby(['вариант', 'beta'], sort=False).agg(\n"
        "    столкновений=('collision', 'sum'),\n"
        "    из_них_с_нарушителем=('hit_violator', 'sum'),\n"
        "    скорость_удара_медиана=('v_impact', 'median'),\n"
        "    исход_остановка=('outcome', lambda x: int((x == 'stopped').sum())),\n"
        "    доля_шагов_торможение=('доля_brake', 'mean'),\n"
        "    доля_шагов_аварийно_8=('доля_emergency', 'mean'),\n"
        "    пробег_км=('distance_km', 'mean'),\n"
        ").round(3)\n"
        "summary"
    ),
    md(
        "Что видно (40 сцен):\n"
        "- **Без фильтра** IDLE врезается всегда, на полной скорости 25 м/с: он не тормозит "
        "сам перед более медленной машиной. В 11 сценах из 40 — еще до момента воздействия "
        "(в лидера, который едет медленнее). Эти эпизоды остаются в выборке с пометкой "
        "«столкновение до воздействия»; там, где воздействие состоялось (29), нарушителем "
        "был лидер агента.\n"
        "- **С фильтром при β = 1** — 0 столкновений у всех трех вариантов фильтра. Это "
        "критерий готовности: допущение RSS соблюдается, и базовая надлежащая реакция "
        "без аварийного торможения его выдерживает.\n"
        "- **При β = 2 базовая реакция с задержкой ρ (основной вариант)** — 24 столкновения "
        "из 40, все с нарушителем, скорость удара 4,7–10,5 м/с (медиана 8,8). С задержкой 0 "
        "и с аварийным торможением — 0. То есть нарушение допущения (лидер тормозит 10 м/с² "
        "вместо 5) снимает гарантию RSS, и у основного варианта это видно уже на пилоте; "
        "запас мгновенной реакции или аварийное торможение 8 м/с² покрывают его.\n"
        "- Столкновения при β = 2 — в сценах, где агент шел ближе к лидеру в момент "
        "воздействия (зазор 38–60 м, медиана 45 м); без столкновения — зазор 40–93 м "
        "(медиана 55 м).\n"
        "- Во всех сценах с состоявшимся воздействием назначенная машина была лидером агента "
        "(у вариантов с фильтром — 40 из 40).\n"
        "- Аварийный режим у варианта сравнения включался только при β = 2 (3,5 % шагов)."
    ),
    md(
        "## Ответственность за столкновения\n\n"
        "По формализации RSS (`rssrl/responsibility.py`). Момент опасности — начало "
        "последнего опасного отрезка с партнером перед ударом. Нарушитель отвечает, если "
        "в этой ситуации тормозил сильнее 5 м/с². Агент отвечает, если команда "
        "торможения ≥ 4 м/с² началась позже ρ после момента опасности, или фактическое "
        "торможение дошло до 4 м/с² позже ρ + 0,7 с (0,7 с — набор торможения от +3 до "
        "−4 м/с² с рывком 10 м/с³, как в d_min с учетом рывка), или торможение отпущено до "
        "удара. Все моменты — на одной сетке шагов 1/15 с, без допуска «полшага»."
    ),
    code(
        "resp = responsibility_table(outcomes, logs)\n"
        "n_logs = export_pilot(outcomes, logs, PILOT, responsibility=resp,\n"
        "                      params={'notebook': '03_shield_pilot', 'kind': 'brake', 'traffic': 'following',\n"
        "                              'trigger': 'schedule', 'policy': 'idle'})\n"
        "print('сохранено покадровых журналов столкновений:', n_logs)\n"
        "if len(resp):\n"
        "    display(resp.pivot_table(index=['вариант', 'level'], columns='responsible', values='seed',\n"
        "                             aggfunc='size', fill_value=0, sort=False)\n"
        "                .rename_axis(index=['вариант', 'β'], columns='отвечает'))"
    ),
    code(
        "# время реакции агента: команда торможения и фактическое торможение 4 м/с² (от момента опасности)\n"
        "if len(resp):\n"
        "    display(resp.groupby(['вариант', 'level'], sort=False)[['reaction_time_command', 'reaction_time_actual']]\n"
        "                .agg(['count', 'min', 'max']).round(2))"
    ),
    md(
        "## Одна сцена, два β (общие сцены)\n\n"
        "До момента нарушения прогоны с β = 1 и β = 2 совпадают полностью: сцена одна "
        "и та же. Разница появляется только после того, как лидер начинает тормозить."
    ),
    code(
        "MAIN = (True, 0.5, False)\n"
        "seed = int(outcomes[outcomes.triggered].seed.min())\n"
        "fig, ax = plt.subplots(figsize=(8, 3.5))\n"
        "for beta, color in [(1.0, '#f5b041'), (2.0, '#a04000')]:\n"
        "    log = logs[(seed, beta, *MAIN)]\n"
        "    ax.plot(log['t'], log['vio_gap'].fillna(log['gap']).clip(lower=0), color=color,\n"
        "            label=f'зазор до лидера, β = {beta:g}')\n"
        "    ax.plot(log['t'], log['d_min'].where(log['gap'] < 1e6), color=color, linewidth=1, linestyle='--')\n"
        "log1 = logs[(seed, 1.0, *MAIN)]\n"
        "t0 = log1.loc[log1['vio_v'].notna(), 't'].min()\n"
        "ax.axvline(t0, color='#c0392b', linewidth=1, linestyle=':')\n"
        "ax.text(t0 + 0.2, ax.get_ylim()[1] * 0.92, 'начало нарушения', fontsize=8, color='#c0392b')\n"
        "ax.set_xlim(0, min(t0 + 15, log1['t'].max()))\n"
        "ax.set_ylim(bottom=0)\n"
        "ax.set_xlabel('время, с'); ax.set_ylabel('м')\n"
        "ax.set_title(f'Сцена seed = {seed}, базовая реакция, задержка ρ (пунктир — $d_{{min}}$)', fontsize=10)\n"
        "ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)\n"
        "fig.savefig(FIG / '03_same_scene_two_betas.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
    md(
        "## Временные ряды столкновений\n\n"
        "Покадровые журналы всех столкновений лежат в `results/pilot_03/collisions/` "
        "(CSV, шаг 1/15 с: зазор, d_min, скорости, ускорения, команды и режим фильтра, "
        "нарушитель). Графики к ним — `results/pilot_03/*.png` (не коммитятся, "
        "строятся заново из журналов). Ниже — первые столкновения с фильтром (если их нет — "
        "пример без фильтра)."
    ),
    code(
        "for old in PILOT.glob('*.png'):\n"
        "    old.unlink()\n"
        "crashes = outcomes[outcomes.collision]\n"
        "for o, (key, log) in zip(outcomes.to_dict('records'), logs.items()):\n"
        "    if not o['collision']:\n"
        "        continue\n"
        "    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)\n"
        "    plot_episode(log, f\"{o['вариант']}, β = {o['beta']:g}, seed = {o['seed']}\", axes)\n"
        "    fig.savefig(PILOT / f\"{variant_slug(*key[2:])}_l{key[1]:g}_s{key[0]}.png\", dpi=110, bbox_inches='tight')\n"
        "    plt.close(fig)\n"
        "print('графиков:', len(crashes))"
    ),
    code(
        "show = crashes[crashes.shield].head(4)\n"
        "if show.empty:\n"
        "    print('С фильтром столкновений нет. Пример столкновения без фильтра:')\n"
        "    show = crashes[~crashes.shield].head(1)\n"
        "for i, o in enumerate(show.itertuples()):\n"
        "    log = logs[(o.seed, o.beta, o.shield, o.reaction_delay, o.emergency_brake)]\n"
        "    t0 = log.loc[log['vio_v'].notna(), 't'].min()\n"
        "    part = log[log.t >= (t0 - 2 if pd.notna(t0) else 0)]\n"
        "    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)\n"
        "    plot_episode(part, f'{o.вариант}, β = {o.beta:g}, seed = {o.seed}', axes)\n"
        "    if i == 0:\n"
        "        fig.savefig(FIG / '03_collision_example.png', dpi=150, bbox_inches='tight')\n"
        "    plt.show()"
    ),
    md(
        "## Вывод пилота\n\n"
        "1. Основной вариант теперь — базовая надлежащая реакция с учетом рывка: торможение "
        "не слабее 4 и не сильнее 5 м/с², без аварийного режима. Аварийное торможение до "
        "8 м/с² — отдельный вариант для сравнения (`emergency_brake=True`).\n"
        "2. При β = 1 (допущение соблюдается) — 0 столкновений у всех вариантов фильтра на "
        "сцене контролируемого следования.\n"
        "3. При β = 2 у основного варианта 24 из 40 столкновений. По формализации RSS все 24 — "
        "ответственность нарушителя: он тормозил 10 м/с² (> 5), а агент выполнил надлежащую "
        "реакцию — команда торможения через 0,4 с после момента опасности (≤ ρ), фактическое "
        "торможение 4 м/с² через 0,93–1,07 с (≤ ρ + 0,7 с) и без отпускания до удара. Это "
        "ожидаемое следствие нарушения допущения, а не ошибка реализации.\n"
        "4. Мгновенный фильтр (задержка 0) и вариант с аварийным торможением при β = 2 "
        "столкновений не дали. Как частота растет между β = 1 и 2 — ноутбук 07.\n"
        "5. Сцены не ужесточали и сетку не расширяли: столкновения при β = 2 получились на "
        "заранее заданной сцене и заранее сохраненном расписании.\n"
        "6. Первичные данные: `results/pilot_03/outcomes.csv` (все эпизоды), "
        "`responsibility.csv`, `collisions/*.csv.gz` (покадровые журналы всех 104 "
        "столкновений), `params.json` (параметры, seed, файлы расписаний, версия кода)."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("03_shield_pilot.ipynb")
ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
