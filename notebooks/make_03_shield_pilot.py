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
        "Код: `rssrl/shield.py` (фильтр), `rssrl/scenarios.py` (сцена и момент нарушения), "
        "`rssrl/logger.py` (временные ряды), `rssrl/responsibility.py` (ответственность). "
        "Проверки: `tests/test_shield.py`, `tests/test_dynamics.py`, `tests/test_responsibility.py`.\n\n"
        "**Пилот без PPO**: агент всегда выбирает IDLE — держит полосу и целевую "
        "скорость 25 м/с, на машину впереди сам не реагирует. Вся безопасность — "
        "только от фильтра.\n\n"
        "**Сцена**: highway-v0, 50 фоновых машин (торможение ≤ 5 м/с²), 40 с. Нарушение — "
        "по расписанию: момент и машина берутся из эталонного прогона (IDLE + фильтр ρ: "
        "первый момент после 3 с, когда агент идет за машиной с зазором ≤ 1,1·d_min "
        "леммы 2) и одинаковы для всех вариантов агента. Нарушитель набирает торможение "
        "β·5 м/с² (с рывком ≤ 10 м/с³), тормозит до остановки и стоит.\n\n"
        "**Динамика**: у всех машин ускорение меняется не быстрее J_max = 10 м/с³. "
        "Фильтр тормозит не слабее 4 м/с², а если по прогнозу 4 м/с² не хватает — "
        "экстренно, с физическим пределом 8 м/с². **d_min фильтра — лемма 2 с учетом "
        "рывка** (`rssrl/rss.py`): к пути задней машины добавлено время перехода от "
        "разгона +3 м/с² к торможению −4 м/с² с рывком 10 м/с³ (0,7 с). Сравнение с "
        "d_min как в статье — в ноутбуке 07.\n\n"
        "**Варианты агента** (все на одних и тех же 20 сценах):\n"
        "- фильтр с задержкой реакции ρ = 0,5 с — основной режим (предположение леммы 2);\n"
        "- фильтр без задержки (мгновенный) — дополнительный вариант;\n"
        "- без фильтра."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.pilot import run_grid, plot_episode\n"
        "from rssrl.responsibility import responsibility_table\n"
        "from rssrl.rss import DEFAULT\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_03'  # все графики столкновений, в git не идут\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "SEEDS = range(20)\n"
        "# основной режим — фильтр с задержкой ρ, дополнительный — мгновенный фильтр\n"
        "VARIANTS = [(True, 0.5), (True, 0.0), (False, 0.5)]\n"
        "NAMES = {(True, 0.5): 'фильтр, задержка ρ (основной)', (True, 0.0): 'фильтр, задержка 0 (доп.)',\n"
        "         (False, 0.5): 'без фильтра'}"
    ),
    code(
        "outcomes, logs = run_grid(SEEDS, [1.0, 2.0], VARIANTS, trigger='schedule')\n"
        "outcomes['вариант'] = [NAMES[(s, d)] for s, d in zip(outcomes['shield'], outcomes['reaction_delay'])]\n"
        "# доля шагов в каждом режиме фильтра\n"
        "for mode in ['brake', 'emergency']:\n"
        "    outcomes[f'доля_{mode}'] = [(log['shield_mode'] == mode).mean() for log in logs.values()]\n"
        "outcomes.drop(columns=['crash_partners']).to_csv(PILOT / 'outcomes.csv', index=False)\n"
        "table = outcomes.pivot_table(index='вариант', columns='beta', values='collision', aggfunc='sum')\n"
        "table.columns = [f'β = {b:g}' for b in table.columns]\n"
        "table = table.loc[list(NAMES.values())]\n"
        "print('Число эпизодов со столкновением из 20:')\n"
        "table"
    ),
    code(
        "summary = outcomes.groupby(['вариант', 'beta'], sort=False).agg(\n"
        "    столкновений=('collision', 'sum'),\n"
        "    из_них_с_нарушителем=('hit_violator', 'sum'),\n"
        "    скорость_удара_медиана=('v_impact', 'median'),\n"
        "    нарушение_было=('triggered', 'sum'),\n"
        "    доля_шагов_торможение_4=('доля_brake', 'mean'),\n"
        "    доля_шагов_экстренно_8=('доля_emergency', 'mean'),\n"
        "    пробег_км=('distance_km', 'mean'),\n"
        ").round(3)\n"
        "summary"
    ),
    md(
        "Что видно:\n"
        "- **Без фильтра** агент врезается почти всегда, на полной скорости 25 м/с "
        "(IDLE не тормозит сам), и при β = 1 тоже; в 3 сценах из 20 он врезается еще до "
        "момента нарушения.\n"
        "- **С фильтром** (d_min с учетом рывка) — 0 столкновений из 20 и при β = 1, и при "
        "β = 2, при обеих задержках. При β = 1 это критерий готовности: допущение RSS "
        "соблюдается, и агент с правильной реакцией не должен попадать в аварии.\n"
        "- **Экстренный режим** почти не нужен (около 1 % шагов): новая d_min заранее "
        "оставляет место на время набора торможения. Большую долю шагов фильтр командует "
        "торможение 4 м/с², потому что агент стоит за остановившимся нарушителем, а IDLE "
        "все время просит разгон.\n"
        "- Для сравнения: с d_min леммы 2 как в статье было 16 и 17 столкновений из 20 "
        "при задержке ρ — медленные въезды в стоящего нарушителя (см. ноутбук 07, где обе "
        "формулы идут рядом на 50 сценах)."
    ),
    md(
        "## Ответственность за столкновения\n\n"
        "По формализации RSS (`rssrl/responsibility.py`): момент опасности — начало "
        "последнего опасного отрезка с партнером перед ударом. Нарушитель отвечает, если "
        "тормозил сильнее 5 м/с²; агент — если не начал торможение ≥ 4 м/с² за ρ после "
        "момента опасности или отпустил его до удара. «Не установлено» — правила RSS "
        "не нарушил никто, а столкновение было."
    ),
    code(
        "resp = responsibility_table(outcomes, logs)\n"
        "resp['вариант'] = [NAMES[(s, d)] for s, d in zip(resp['shield'], resp['reaction_delay'])]\n"
        "resp.to_csv(PILOT / 'responsibility.csv', index=False)\n"
        "(resp.pivot_table(index=['вариант', 'level'], columns='responsible', values='seed',\n"
        "                  aggfunc='size', fill_value=0, sort=False)\n"
        "     .rename_axis(index=['вариант', 'β'], columns='отвечает'))"
    ),
    code(
        "# время реакции агента (от момента опасности до команды торможения ≥ 4 м/с²)\n"
        "resp.groupby(['вариант', 'level'], sort=False)['reaction_time'].describe()[['count', 'min', 'max']].round(2)"
    ),
    md(
        "## Механизм столкновений с фильтром\n\n"
        "Для каждого столкновения с фильтром: скорость агента в момент удара и стоял ли "
        "агент (скорость < 0,5 м/с) после начала нарушения, но до удара. «Стоял, потом "
        "врезался» значит: агент уже остановился за стоящим нарушителем, фильтр отпустил "
        "тормоз, IDLE снова поехал, и торможение не успело набраться. Так было с d_min "
        "леммы 2 как в статье (при нулевых скоростях она всего 0,66 м). С d_min с учетом "
        "рывка (1,75 м при нулевых скоростях) столкновений с фильтром нет, и таблица пустая."
    ),
    code(
        "rows = []\n"
        "for o in outcomes[outcomes.collision & outcomes.shield].itertuples():\n"
        "    log = logs[(o.seed, o.beta, o.shield, o.reaction_delay)]\n"
        "    i = log.index[log.crashed].min()\n"
        "    after = log.loc[:i - 1]\n"
        "    after = after[after['vio_v'].notna()]\n"
        "    rows.append({'вариант': o.вариант, 'β': o.beta, 'seed': o.seed,\n"
        "                 'скорость удара, м/с': o.v_impact,\n"
        "                 'стоял до удара': bool((after['ego_v'] < 0.5).any()),\n"
        "                 'режим фильтра перед ударом': log.loc[i - 1, 'shield_mode']})\n"
        "mech = pd.DataFrame(rows, columns=['вариант', 'β', 'seed', 'скорость удара, м/с',\n"
        "                                   'стоял до удара', 'режим фильтра перед ударом'])\n"
        "mech.groupby(['вариант', 'β'], sort=False).agg(\n"
        "    столкновений=('seed', 'size'),\n"
        "    стоял_до_удара=('стоял до удара', 'sum'),\n"
        "    скорость_удара_медиана=('скорость удара, м/с', 'median'),\n"
        "    скорость_удара_макс=('скорость удара, м/с', 'max'),\n"
        ").round(2)"
    ),
    md(
        "## Одна сцена, два β (общие сцены)\n\n"
        "До момента нарушения прогоны с β = 1 и β = 2 совпадают полностью: сцена одна "
        "и та же. Разница появляется только после того, как лидер начинает тормозить."
    ),
    code(
        "sub = outcomes[outcomes.shield & (outcomes.reaction_delay == 0.5)].pivot(index='seed', columns='beta', values='collision')\n"
        "cand = sub.index[(~sub[1.0].astype(bool)) & sub[2.0].astype(bool)]\n"
        "seed = int(cand[0]) if len(cand) else int(sub.index[0])\n"
        "fig, ax = plt.subplots(figsize=(8, 3.5))\n"
        "for beta, color in [(1.0, '#f5b041'), (2.0, '#a04000')]:\n"
        "    log = logs[(seed, beta, True, 0.5)]\n"
        "    ax.plot(log['t'], log['vio_gap'].fillna(log['gap']).clip(lower=0), color=color,\n"
        "            label=f'зазор до лидера, β = {beta:g}')\n"
        "    ax.plot(log['t'], log['d_min'], color=color, linewidth=1, linestyle='--')\n"
        "t0 = logs[(seed, 1.0, True, 0.5)]['vio_v'].first_valid_index() / 15\n"
        "ax.axvline(t0, color='#c0392b', linewidth=1, linestyle=':')\n"
        "ax.text(t0 + 0.2, ax.get_ylim()[1] * 0.92, 'начало нарушения', fontsize=8, color='#c0392b')\n"
        "ax.set_xlim(0, min(25, logs[(seed, 1.0, True, 0.5)]['t'].max()))\n"
        "ax.set_ylim(bottom=0)\n"
        "ax.set_xlabel('время, с'); ax.set_ylabel('м')\n"
        "ax.set_title(f'Сцена seed = {seed}, фильтр с задержкой ρ (пунктир — $d_{{min}}$)', fontsize=10)\n"
        "ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)\n"
        "fig.savefig(FIG / '03_same_scene_two_betas.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
    md(
        "## Временные ряды столкновений с фильтром\n\n"
        "Для каждого столкновения: зазор и d_min, скорости, ускорения (команда политики, "
        "фактическое ускорение агента и нарушителя). Серым показаны шаги, где фильтр "
        "тормозит. Графики всех столкновений, включая варианты без фильтра, сохраняются "
        "в `results/pilot_03/`. Ниже — первые 4 при задержке ρ и все при задержке 0 "
        "(если с фильтром столкновений нет — пример без фильтра)."
    ),
    code(
        "crashes = outcomes[outcomes.collision].sort_values(['shield', 'reaction_delay', 'beta', 'seed'],\n"
        "                                                  ascending=[False, False, True, True])\n"
        "for _, o in crashes.iterrows():\n"
        "    key = (o.seed, o.beta, o.shield, o.reaction_delay)\n"
        "    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)\n"
        "    plot_episode(logs[key], f\"{o['вариант']}, β = {o.beta:g}, seed = {o.seed}\", axes)\n"
        "    name = f\"{'shield' if o.shield else 'noshield'}_d{o.reaction_delay:g}_b{o.beta:g}_s{o.seed}.png\"\n"
        "    fig.savefig(PILOT / name, dpi=110, bbox_inches='tight')\n"
        "    plt.close(fig)\n"
        "print('сохранено графиков:', len(crashes))"
    ),
    code(
        "show = pd.concat([\n"
        "    crashes[(crashes.shield) & (crashes.reaction_delay == 0.5)].head(4),\n"
        "    crashes[(crashes.shield) & (crashes.reaction_delay == 0)],\n"
        "])\n"
        "if show.empty:  # с фильтром столкновений нет — показываем пример без фильтра\n"
        "    print('С фильтром столкновений нет. Пример столкновения без фильтра:')\n"
        "    show = crashes[~crashes.shield].head(1)\n"
        "for i, (_, o) in enumerate(show.iterrows()):\n"
        "    key = (o.seed, o.beta, o.shield, o.reaction_delay)\n"
        "    log = logs[key]\n"
        "    t0 = log.loc[log['vio_v'].notna(), 't'].min()\n"
        "    part = log[(log.t >= t0 - 2) & (log.t <= log.t.max())]\n"
        "    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)\n"
        "    plot_episode(part, f\"{o['вариант']}, β = {o.beta:g}, seed = {o.seed}\", axes)\n"
        "    if i == 0:\n"
        "        fig.savefig(FIG / '03_collision_example.png', dpi=150, bbox_inches='tight')\n"
        "    plt.show()"
    ),
    md(
        "## Вывод пилота\n\n"
        "1. Рывок для торможения фильтра — экстренный, 10 м/с³ (один предел для всех "
        "машин, `rssrl/dynamics.py`); комфортный рывок нигде не используется.\n"
        "2. Причиной прежних аварий при β = 1 была формула: лемма 2 как в статье не "
        "учитывает 0,7 с набора торможения с рывком. С d_min, где этот путь учтен, при "
        "β = 1 и задержке ρ столкновений нет (20 сцен здесь, 12 сцен в тесте "
        "`test_no_collisions_with_jerk_and_jerk_aware_dmin`), в номинальном режиме — тоже "
        "(ноутбук 05). Критерий готовности выполнен.\n"
        "3. При β = 2 на этих 20 сценах столкновений тоже нет: большая дистанция и "
        "экстренное торможение 8 м/с² покрывают торможение лидера 10 м/с². Как частота "
        "растет с β на 50 сценах — в ноутбуке 07.\n"
        "4. Нарушение здесь — по расписанию (одинаковый момент для всех вариантов): с новой "
        "формулой осторожный мгновенный фильтр никогда не подходит на 1,1·d_min, и нарушение "
        "«по зазору» у него не наступало бы вовсе."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("03_shield_pilot.ipynb")
ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
