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
        "`rssrl/logger.py` (временные ряды). Проверки: `tests/test_shield.py`.\n\n"
        "**Пилот без PPO**: агент всегда выбирает IDLE — держит полосу и целевую "
        "скорость 25 м/с, на машину впереди сам не реагирует. Вся безопасность — "
        "только от фильтра.\n\n"
        "**Сцена**: highway-v0, 50 фоновых машин (торможение ≤ 5 м/с²), 40 с. Когда "
        "прошло ≥ 3 с и агент идет за машиной впереди с зазором ≤ 1,1·d_min, эта машина "
        "становится нарушителем: тормозит с β·5 м/с² до остановки и стоит.\n\n"
        "**Варианты агента** (все на одних и тех же 20 сценах):\n"
        "- без фильтра;\n"
        "- фильтр, реакция сразу (задержка 0);\n"
        "- фильтр, задержка реакции ρ = 0,5 с (ровно предположение леммы 2)."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.pilot import run_grid, plot_episode\n"
        "from rssrl.rss import DEFAULT\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_03'  # все графики столкновений, в git не идут\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "SEEDS = range(20)\n"
        "VARIANTS = [(False, 0.0), (True, 0.0), (True, 0.5)]\n"
        "NAMES = {(False, 0.0): 'без фильтра', (True, 0.0): 'фильтр, задержка 0',\n"
        "         (True, 0.5): 'фильтр, задержка ρ'}"
    ),
    code(
        "outcomes, logs = run_grid(SEEDS, [1.0, 2.0], VARIANTS)\n"
        "outcomes['вариант'] = [NAMES[(s, d)] for s, d in zip(outcomes['shield'], outcomes['reaction_delay'])]\n"
        "outcomes.to_csv(PILOT / 'outcomes.csv', index=False)\n"
        "table = outcomes.pivot_table(index='вариант', columns='beta', values='collision', aggfunc='sum')\n"
        "table.columns = [f'β = {b:g}' for b in table.columns]\n"
        "table = table.loc[list(NAMES.values())]\n"
        "print('Число эпизодов со столкновением из 20:')\n"
        "table"
    ),
    code(
        "summary = outcomes.groupby(['вариант', 'beta']).agg(\n"
        "    столкновений=('collision', 'sum'),\n"
        "    из_них_с_нарушителем=('hit_violator', 'sum'),\n"
        "    нарушение_было=('triggered', 'sum'),\n"
        "    шагов_с_торможением_фильтра=('shield_steps', 'mean'),\n"
        "    пробег_км=('distance_km', 'mean'),\n"
        ").round(2)\n"
        "summary"
    ),
    md(
        "Что видно:\n"
        "- **Без фильтра** агент врезается всегда, причем почти всегда в машину впереди, "
        "и при β = 1 тоже. Один раз агент врезался еще до начала нарушения.\n"
        "- **С фильтром при β = 1** столкновений нет ни при одной задержке. Это и "
        "обещает RSS, когда допущение соблюдается. Пилот проверяет, что фильтр "
        "реализован правильно.\n"
        "- **С фильтром при β = 2** столкновения появляются. Их доля сильно зависит от "
        "задержки реакции: при задержке ρ столкновения во всех сценах, при задержке 0 "
        "только в части."
    ),
    md(
        "## Одна сцена, два β (общие сцены)\n\n"
        "До момента нарушения прогоны с β = 1 и β = 2 совпадают полностью: сцена одна "
        "и та же. Разница появляется только после того, как лидер начинает тормозить."
    ),
    code(
        "seed = int(outcomes[(outcomes.shield) & (outcomes.reaction_delay == 0) & (outcomes.beta == 2)\n"
        "                    & outcomes.collision]['seed'].iloc[0])\n"
        "fig, ax = plt.subplots(figsize=(8, 3.5))\n"
        "for beta, color in [(1.0, '#f5b041'), (2.0, '#a04000')]:\n"
        "    log = logs[(seed, beta, True, 0.0)]\n"
        "    ax.plot(log['t'], log['vio_gap'].fillna(log['gap']).clip(lower=0), color=color,\n"
        "            label=f'зазор до лидера, β = {beta:g}')\n"
        "    ax.plot(log['t'], log['d_min'], color=color, linewidth=1, linestyle='--')\n"
        "t0 = logs[(seed, 1.0, True, 0.0)]['vio_v'].first_valid_index() / 15\n"
        "ax.axvline(t0, color='#c0392b', linewidth=1, linestyle=':')\n"
        "ax.text(t0 + 0.2, ax.get_ylim()[1] * 0.92, 'начало нарушения', fontsize=8, color='#c0392b')\n"
        "ax.set_xlim(0, min(25, logs[(seed, 1.0, True, 0.0)]['t'].max()))\n"
        "ax.set_ylim(bottom=0)\n"
        "ax.set_xlabel('время, с'); ax.set_ylabel('м')\n"
        "ax.set_title(f'Сцена seed = {seed}, фильтр с задержкой 0 (пунктир — $d_{{min}}$)', fontsize=10)\n"
        "ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)\n"
        "fig.savefig(FIG / '03_same_scene_two_betas.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
    md(
        "## Временные ряды каждого столкновения с фильтром\n\n"
        "Для каждого столкновения: зазор и d_min, скорости, ускорения (команда политики, "
        "фактическое ускорение агента и нарушителя). Серым показаны шаги, где фильтр "
        "тормозит. Графики всех столкновений, включая варианты без фильтра, сохраняются "
        "в `results/pilot_03/`. Ниже показаны все столкновения при задержке 0 и первые "
        "4 при задержке ρ."
    ),
    code(
        "crashes = outcomes[outcomes.collision].sort_values(['shield', 'reaction_delay', 'beta', 'seed'])\n"
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
        "    crashes[(crashes.shield) & (crashes.reaction_delay == 0)],\n"
        "    crashes[(crashes.shield) & (crashes.reaction_delay == 0.5)].head(4),\n"
        "])\n"
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
        "## Сходится ли это с физикой?\n\n"
        "При задержке 0 фильтр начинает тормозить с 4 м/с² практически сразу после "
        "начала нарушения (d_min мгновенно вырастает выше зазора). Тогда упрощенный "
        "прогноз: столкновение будет, если\n\n"
        "$$\\text{зазор}_0 < \\frac{v_{агент}^2}{2\\cdot 4} - \\frac{v_{лидер}^2}{2\\cdot 5\\beta}.$$\n\n"
        "Сравним прогноз с тем, что получилось в симуляции, для всех 40 эпизодов с "
        "фильтром и задержкой 0."
    ),
    code(
        "rows = []\n"
        "for (seed, beta, shield, delay), log in logs.items():\n"
        "    if not shield or delay != 0:\n"
        "        continue\n"
        "    i0 = log['vio_v'].first_valid_index()\n"
        "    before = log.loc[i0 - 1]  # последний шаг до нарушения\n"
        "    v_e, v_l, gap0 = before['ego_v'], before['vio_v'] if pd.notna(before['vio_v']) else log.loc[i0, 'v_front'], before['gap']\n"
        "    need = v_e**2 / (2 * DEFAULT.a_min_brake) - v_l**2 / (2 * 5 * beta)\n"
        "    o = outcomes[(outcomes.seed == seed) & (outcomes.beta == beta) & outcomes.shield & (outcomes.reaction_delay == 0)].iloc[0]\n"
        "    rows.append({'seed': seed, 'β': beta, 'v агента': v_e, 'v лидера': v_l, 'зазор': gap0,\n"
        "                 'нужно (прогноз)': need, 'прогноз: столкновение': gap0 < need,\n"
        "                 'симуляция: столкновение': o.collision})\n"
        "check = pd.DataFrame(rows).sort_values(['β', 'seed']).round(2)\n"
        "agree = (check['прогноз: столкновение'] == check['симуляция: столкновение'])\n"
        "print(f'прогноз совпал с симуляцией в {agree.sum()} из {len(check)} эпизодов')\n"
        "check[check['β'] == 2.0]"
    ),
    md(
        "Прогноз и симуляция расходятся только в пограничных сценах, где запас по "
        "прогнозу меньше 2 м (seed 0 и 19). Прогноз этого не учитывает, но в момент "
        "нарушения агент в этих сценах еще разгонялся с +3 м/с², и фильтр начал "
        "тормозить на шаг позже (через 1/15 с). Это стоит порядка метра.\n\n"
        "**Дребезг фильтра.** Перед нарушением, пока агент идет ровно на границе d_min, "
        "фильтр через шаг переключает ускорение: политика просит +3, фильтр отвечает −4 "
        "(пила на графике ускорения). На безопасность это не влияет, но плавной такую "
        "езду не назовешь. Это будет важно, когда на этапе 6 будем считать «лишние» "
        "срабатывания фильтра."
    ),
    md(
        "## Вывод пилота и решение, которое нужно принять\n\n"
        "1. Фильтр реализован верно: при соблюдении допущения (β = 1) столкновений нет "
        "ни в одной сцене, а без фильтра они есть во всех.\n"
        "2. При β = 2 столкновения есть, но их число **очень сильно зависит от задержки "
        "реакции фильтра**. При задержке 0 ρ остается только запасом в формуле: лемма "
        "разрешает агенту ρ секунд разгоняться, а фильтр тормозит сразу. Этот запас "
        "частично покрывает лишнее торможение лидера.\n"
        "3. Поэтому до основного эксперимента нужно зафиксировать, что моделируем:\n"
        "   - **задержка ρ** (реакция ровно как в лемме): чистая проверка допущения RSS, "
        "результат не зависит от того, насколько быстрый у нас фильтр;\n"
        "   - **задержка 0** (мгновенный фильтр): ближе к реальной автоматике, но "
        "часть эффекта β съедается запасом ρ, и вывод будет про «наш фильтр», а не про RSS.\n"
        "4. Время эпизода: ~3 с на одном ядре, на 8 ядрах 120 эпизодов идут ~70 с "
        "(подробно на этапе 7)."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("03_shield_pilot.ipynb")
ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
