"""Собирает и выполняет notebooks/04_cutin.ipynb.

Запуск из корня проекта: .venv/bin/python notebooks/make_04_cutin.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Этап 4. Подрезание и поперечная RSS\n\n"
        "Код: `rssrl/violators.py` (CutInViolator), `rssrl/scenarios.py` (CutInDirector), "
        "`rssrl/shield.py` (поперечная RSS, запрет и отмена перестроений). "
        "Проверки: `tests/test_cutin.py`.\n\n"
        "**Подрезающий**: машина из соседней полосы перестраивается перед агентом за 3 с "
        "по гладкой траектории (полином 5-й степени, без телепортации). Маневр "
        "начинается, когда прогнозный зазор в момент пересечения разметки равен "
        "γ·d_min. Фактический зазор измеряется на разметке. С этого момента подрезающий "
        "тормозит с 5 м/с² (β = 1, самое сильное торможение, которое допускает RSS).\n\n"
        "**Сцена подрезания**: чтобы было куда подрезать, на старте из полосы агента "
        "убраны машины на 250 м вперед. Это одинаково для всех вариантов агента и всех γ.\n\n"
        "**Фильтр**: ситуация опасна, только если небезопасно и продольно (лемма 2), "
        "и поперечно (лемма 4). Подрезающий становится опасным еще до разметки: как "
        "только он начинает сдвигаться к агенту, поперечный d_min растет."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.pilot import run_grid, plot_episode\n"
        "from rssrl.scenarios import LC_DURATION\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_04'  # все графики столкновений, в git не идут\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "SEEDS = range(20)\n"
        "VARIANTS = [(False, 0.0), (True, 0.0), (True, 0.5)]\n"
        "NAMES = {(False, 0.0): 'без фильтра', (True, 0.0): 'фильтр, задержка 0',\n"
        "         (True, 0.5): 'фильтр, задержка ρ'}"
    ),
    code(
        "outcomes, logs = run_grid(SEEDS, [1.0, 0.25], VARIANTS, kind='cutin')\n"
        "outcomes['вариант'] = [NAMES[(s, d)] for s, d in zip(outcomes['shield'], outcomes['reaction_delay'])]\n"
        "outcomes.to_csv(PILOT / 'outcomes.csv', index=False)\n"
        "trig = outcomes[outcomes.triggered]\n"
        "summary = outcomes.groupby(['вариант', 'gamma'], sort=False).agg(\n"
        "    подрезание_было=('triggered', 'sum'),\n"
        "    столкновений_всего=('collision', 'sum'),\n"
        "    с_подрезающим=('hit_violator', 'sum'),\n"
        ")\n"
        "summary['доля с подрезающим среди подрезаний'] = (summary['с_подрезающим'] / summary['подрезание_было']).round(2)\n"
        "summary"
    ),
    md(
        "Как читать таблицу:\n"
        "- **Без фильтра** агент врезается во всех эпизодах: он ни на кого не реагирует. "
        "Часть столкновений — с другими машинами, еще до подрезания.\n"
        "- **С фильтром при γ = 1** столкновений нет ни при одной задержке. Подрезание "
        "с зазором d_min и торможение 5 м/с² законны по RSS, и фильтр с ними справляется.\n"
        "- **С фильтром при γ = 0,25** столкновения с подрезающим есть. Их доля снова "
        "сильно зависит от задержки реакции.\n"
        "- Подрезание случилось не во всех сценах: нужна подходящая машина в соседней "
        "полосе и место перед агентом. При γ = 1 нужно место побольше, поэтому таких "
        "сцен меньше. Число подрезаний немного различается и между вариантами агента, "
        "потому что момент маневра зависит от того, как ехал сам агент."
    ),
    md("## Траектория подрезания: гладкая, без телепортации"),
    code(
        "o = trig[(~trig.shield) & (trig.gamma == 0.25)].iloc[0]\n"
        "log = logs[(o.seed, 0.25, False, 0.0)]\n"
        "t0 = o.t_trigger\n"
        "# до начала маневра нарушитель еще не назначен — его колонки пустые\n"
        "part = log[(log.t <= t0 + LC_DURATION + 0.5) & log.vio_y.notna()]\n"
        "# первая запись — после первого шага маневра; добавляем состояние в момент старта\n"
        "# (поперечная скорость 0), чтобы производная на краю считалась честно\n"
        "start = pd.DataFrame({'t': [t0], 'vio_y': [part.vio_y.iloc[0]], 'vio_vy': [0.0]})\n"
        "part = pd.concat([start, part[['t', 'vio_y', 'vio_vy']]], ignore_index=True)\n"
        "ay = np.gradient(part['vio_vy'], part['t'])\n"
        "fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)\n"
        "axes[0].plot(part.t - t0, part.vio_y, color='#c0392b')\n"
        "for y in (-2, 2):\n"
        "    axes[0].axhline(y, color='0.5', linewidth=1, linestyle='--')\n"
        "axes[0].axvline(o.t_cross - t0, color='0.3', linewidth=1, linestyle=':')\n"
        "axes[0].text(o.t_cross - t0 + 0.05, -3.6, 'пересечение разметки', fontsize=8)\n"
        "axes[0].set_ylabel('смещение от центра\\nполосы агента, м')\n"
        "axes[1].plot(part.t - t0, part.vio_vy, color='#c0392b')\n"
        "axes[1].set_ylabel('поперечная\\nскорость, м/с')\n"
        "axes[2].plot(part.t - t0, ay, color='#c0392b')\n"
        "axes[2].set_ylabel('поперечное\\nускорение, м/с²')\n"
        "axes[2].set_xlabel('время от начала маневра, с')\n"
        "axes[0].set_title(f'Подрезание, seed = {o.seed}, γ = 0,25 (агент без фильтра)', fontsize=10)\n"
        "for a in axes: a.grid(alpha=0.3)\n"
        "fig.savefig(FIG / '04_cutin_trajectory.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'наибольшая поперечная скорость {part.vio_vy.abs().max():.2f} м/с, '\n"
        "      f'наибольшее поперечное ускорение {np.abs(ay).max():.2f} м/с²')"
    ),
    md(
        "Положение, поперечная скорость и поперечное ускорение непрерывны, маневр длится "
        "3 с. Пиковая поперечная скорость — 1,875·4 м / 3 с = 2,5 м/с, пиковое "
        "поперечное ускорение — около 2,6 м/с². Это резкое, но физически возможное "
        "перестроение. Ускорение здесь посчитано численно по записанной скорости, "
        "поэтому в точках излома (0 и 3 с) видна небольшая погрешность конечной разности; "
        "по формуле оно в начале и в конце маневра равно нулю."
    ),
    md(
        "## Фактический зазор на разметке\n\n"
        "Задано: зазор на разметке = γ·d_min. Без фильтра агент едет с постоянной "
        "скоростью, и фактическое γ совпадает с заданным до долей процента. С фильтром "
        "фактическое γ больше: агент начинает тормозить еще до разметки, когда "
        "подрезающий только сдвигается к нему (поперечная RSS)."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
        "colors = {'без фильтра': '#7f8c8d', 'фильтр, задержка 0': '#2c6fbb', 'фильтр, задержка ρ': '#8e44ad'}\n"
        "for i, g in enumerate([0.25, 1.0]):\n"
        "    for j, name in enumerate(colors):\n"
        "        vals = trig[(trig.gamma == g) & (trig['вариант'] == name)]['gamma_actual']\n"
        "        x = i * 4 + j + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))\n"
        "        ax.scatter(x, vals, s=30, color=colors[name], edgecolor='white', linewidth=1,\n"
        "                   label=name if i == 0 else None, zorder=3)\n"
        "    ax.hlines(g, i * 4 - 0.4, i * 4 + 2.4, color='black', linewidth=1, linestyle='--')\n"
        "ax.set_xticks([1, 5]); ax.set_xticklabels(['задано γ = 0,25', 'задано γ = 1'])\n"
        "ax.set_ylabel('фактическое γ на разметке')\n"
        "ax.legend(frameon=False, fontsize=8, loc='upper left'); ax.grid(alpha=0.3, axis='y')\n"
        "ax.set_title('Зазор на разметке / d_min (пунктир — заданное)', fontsize=10)\n"
        "fig.savefig(FIG / '04_gamma_actual.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "trig.groupby(['вариант', 'gamma'], sort=False)['gamma_actual'].agg(['mean', 'min', 'max']).round(3)"
    ),
    md(
        "## Временные ряды каждого столкновения с фильтром\n\n"
        "Четвертая панель — поперечное положение подрезающего относительно центра "
        "полосы агента (штрихи — разметка). Графики всех столкновений сохраняются в "
        "`results/pilot_04/`, ниже — первые три для каждой задержки."
    ),
    code(
        "crashes = outcomes[outcomes.collision].sort_values(['shield', 'reaction_delay', 'gamma', 'seed'])\n"
        "for _, o in crashes.iterrows():\n"
        "    key = (o.seed, o.gamma, o.shield, o.reaction_delay)\n"
        "    log = logs[key]\n"
        "    axes = plot_episode(log, f\"{o['вариант']}, γ = {o.gamma:g}, seed = {o.seed}\",\n"
        "                        lateral='vio_y' in log and log['vio_y'].notna().any())\n"
        "    fig = axes[0].figure\n"
        "    name = f\"{'shield' if o.shield else 'noshield'}_d{o.reaction_delay:g}_g{o.gamma:g}_s{o.seed}.png\"\n"
        "    fig.savefig(PILOT / name, dpi=110, bbox_inches='tight')\n"
        "    plt.close(fig)\n"
        "print('сохранено графиков:', len(crashes))\n\n"
        "first = True\n"
        "for delay in [0.0, 0.5]:\n"
        "    sel = crashes[crashes.shield & (crashes.reaction_delay == delay)].head(3)\n"
        "    for _, o in sel.iterrows():\n"
        "        log = logs[(o.seed, o.gamma, o.shield, o.reaction_delay)]\n"
        "        part = log[log.t >= o.t_trigger - 2]\n"
        "        fig, axes = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True)\n"
        "        plot_episode(part, f\"{o['вариант']}, γ = {o.gamma:g}, seed = {o.seed}\", axes, lateral=True)\n"
        "        if first:\n"
        "            fig.savefig(FIG / '04_collision_example.png', dpi=150, bbox_inches='tight')\n"
        "            first = False\n"
        "        plt.show()"
    ),
    md(
        "## Проверка запрета перестроений: агент со случайными действиями\n\n"
        "Пилотный агент (IDLE) никогда не перестраивается, поэтому запрет и отмену "
        "перестроений он не проверяет. Здесь агент каждую секунду выбирает случайное "
        "действие (влево, ничего, вправо, быстрее, медленнее). У случайной политики свой "
        "генератор (seed сцены + 1000), он не связан со сценой. Сцена тормозящего лидера, "
        "β = 1, допущение соблюдается. С фильтром столкновений быть не должно."
    ),
    code(
        "rand, _ = run_grid(SEEDS, [1.0], VARIANTS, kind='brake', policy='random')\n"
        "rand['вариант'] = [NAMES[(s, d)] for s, d in zip(rand['shield'], rand['reaction_delay'])]\n"
        "rand.groupby('вариант', sort=False).agg(\n"
        "    столкновений=('collision', 'sum'),\n"
        "    запросов_перестроения=('lc_requested', 'sum'),\n"
        "    запрещено=('lc_blocked', 'sum'),\n"
        "    отменено_начатых=('lc_cancelled', 'sum'),\n"
        ")"
    ),
    md(
        "Без фильтра случайный агент разбивается почти всегда. С фильтром столкновений "
        "нет: опасные перестроения запрещаются, начатые — отменяются. Без фильтра "
        "в колонках «запрещено» и «отменено» стоит то, что фильтр *запретил бы*: те же "
        "проверки считаются и пишутся в лог, но не исполняются.\n\n"
        "## Вывод этапа 4\n\n"
        "1. Подрезание реализовано гладко, зазор на разметке совпадает с заданным γ·d_min "
        "(у агента без фильтра — до долей процента).\n"
        "2. Поперечная RSS работает: законное подрезание (γ = 1) и случайные перестроения "
        "самого агента не приводят к столкновениям с фильтром.\n"
        "3. При γ = 0,25 столкновения есть, и, как и для β, их доля сильно зависит от "
        "задержки реакции фильтра.\n"
        "4. С фильтром фактическое γ больше заданного: агент начинает тормозить до "
        "разметки. Это особенность RSS (поперечная опасность видна раньше), а не ошибка. "
        "В статье стоит указывать и заданное, и фактическое γ."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("04_cutin.ipynb")
ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
