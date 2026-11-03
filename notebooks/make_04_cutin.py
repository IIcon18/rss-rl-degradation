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
        "**Сцена подрезания** (повтор после письма от 29.10): контролируемое следование — фон "
        "не перестраивается и держит d_min (`rssrl/traffic.py`); чтобы было куда подрезать, "
        "на старте из полосы агента убраны машины на 250 м вперед. Это одинаково для всех "
        "вариантов агента и всех γ. Сцены 0–39.\n\n"
        "**Момент подрезания — по расписанию из файла** `results/schedules/cutin_following_0-49.csv`: "
        "момент, машина и полоса построены заранее эталоном IDM/MOBIL + RSS-фильтр, маневр "
        "начинается по прогнозу зазора γ·d_min (d_min — лемма 2 как в статье). У всех "
        "вариантов агента подрезание начинается в тот же момент, а фактический зазор на "
        "разметке зависит от того, как ехал агент, — он записывается (фактическое γ).\n\n"
        "**Фильтр** — базовая надлежащая реакция с учетом рывка (без аварийного торможения), "
        "d_min — лемма 2 с учетом рывка 10 м/с³.\n\n"
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
        "from rssrl.pilot import run_grid, plot_episode, export_pilot\n"
        "from rssrl.scenarios import LC_DURATION\n"
        "from rssrl.responsibility import responsibility_table\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_04'  # первичные данные: CSV в git, PNG — нет\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "SEEDS = range(40)\n"
        "# основной режим — фильтр с задержкой ρ, дополнительный — мгновенный фильтр\n"
        "VARIANTS = [(True, 0.5), (True, 0.0), (False, 0.5)]\n"
        "NAMES = {(True, 0.5): 'базовая, задержка ρ (основной)', (True, 0.0): 'базовая, задержка 0 (доп.)',\n"
        "         (False, 0.5): 'без фильтра'}"
    ),
    code(
        "outcomes, logs = run_grid(SEEDS, [1.0, 0.25], VARIANTS, kind='cutin', trigger='schedule')\n"
        "outcomes['вариант'] = [NAMES[(s, d)] for s, d in zip(outcomes['shield'], outcomes['reaction_delay'])]\n"
        
        "trig = outcomes[outcomes.triggered]\n"
        "summary = outcomes.groupby(['вариант', 'gamma'], sort=False).agg(\n"
        "    подрезание_было=('triggered', 'sum'),\n"
        "    столкновений_всего=('collision', 'sum'),\n"
        "    с_подрезающим=('hit_violator', 'sum'),\n"
        "    до_подрезания=('impact', lambda x: int((x == 'столкновение до воздействия').sum())),\n"
        ")\n"
        "summary['доля с подрезающим среди подрезаний'] = (summary['с_подрезающим'] / summary['подрезание_было']).round(2)\n"
        "summary"
    ),
    md(
        "Как читать таблицу (40 сцен, подрезание состоялось во всех: на контролируемом "
        "следовании фон не мешает маневру):\n"
        "- **γ = 1 (законное подрезание)** — с фильтром 0 столкновений при обеих задержках.\n"
        "- **γ = 0,25** — базовая надлежащая реакция без аварийного торможения: 40 из 40 "
        "столкновений с подрезающим при задержке ρ и 35 из 40 при задержке 0 (скорость удара "
        "медиана 12,9 и 6,5 м/с). Зазор на разметке в четыре раза меньше d_min: торможения "
        "4–5 м/с² не хватает. В прошлом пилоте (фильтр с аварийным торможением до 8 м/с²) "
        "здесь было 0 — так что этот результат — следствие выбора основного варианта "
        "фильтра, и он ожидаем.\n"
        "- **Без фильтра** IDLE врезается всегда (в подрезающего — во всех 40)."
    ),
    md("## Траектория подрезания: гладкая, без телепортации"),
    code(
        "o = trig[(~trig.shield) & (trig.gamma == 0.25)].iloc[0]\n"
        "log = logs[(o.seed, 0.25, False, 0.5)]\n"
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
        "Задано: зазор на разметке = γ·d_min (d_min — лемма 2 как в статье) по прогнозу "
        "эталонного прогона при постоянных скоростях. Фактическое γ зависит от того, как ехал "
        "агент: с фильтром оно в среднем больше заданного (при γ = 1: 1,04 у задержки ρ и "
        "1,34 у задержки 0; минимум 0,82), без фильтра — меньше (0,88). Поэтому без фильтра "
        "при «законном» γ = 1 подрезание фактически оказывается ближе d_min, и за него "
        "отвечает и нарушитель. В статье нужно указывать и заданное, и фактическое γ."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
        "colors = {NAMES[(True, 0.5)]: '#8e44ad', NAMES[(True, 0.0)]: '#2c6fbb', NAMES[(False, 0.5)]: '#7f8c8d'}\n"
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
        "`results/pilot_04/`, ниже — первые три для каждой задержки (если с фильтром "
        "столкновений нет — пример без фильтра)."
    ),
    code(
        "for old in PILOT.glob('*.png'):\n"
        "    old.unlink()\n"
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
        "groups = [crashes[crashes.shield & (crashes.reaction_delay == d)].head(3) for d in [0.5, 0.0]]\n"
        "if all(g.empty for g in groups):  # с фильтром столкновений нет — пример без фильтра\n"
        "    print('С фильтром столкновений нет. Пример столкновения без фильтра с подрезающим:')\n"
        "    groups = [crashes[~crashes.shield & crashes.hit_violator].head(1)]\n"
        "for sel in groups:\n"
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
        "## Ответственность за столкновения\n\n"
        "Правила те же, что на этапе 3 (`rssrl/responsibility.py`). Для подрезания нарушитель "
        "отвечает еще и тогда, когда зазор на разметке меньше d_min (допуск 1 %: при γ = 1 "
        "сцена попадает в d_min с точностью до долей процента)."
    ),
    code(
        "resp = responsibility_table(outcomes, logs)\n"
        "resp['вариант'] = [NAMES[(s, d)] for s, d in zip(resp['shield'], resp['reaction_delay'])]\n"
        "n_logs = export_pilot(outcomes, logs, PILOT, responsibility=resp,\n"
        "                      params={'notebook': '04_cutin', 'kind': 'cutin', 'traffic': 'following',\n"
        "                              'trigger': 'schedule', 'policy': 'idle'})\n"
        "print('сохранено покадровых журналов столкновений:', n_logs)\n"
        "(resp.pivot_table(index=['вариант', 'level'], columns='responsible', values='seed',\n"
        "                  aggfunc='size', fill_value=0, sort=False)\n"
        "     .rename_axis(index=['вариант', 'γ'], columns='отвечает'))"
    ),
    code(
        "# столкновения с фильтром: скорость удара и режим фильтра\n"
        "outcomes[outcomes.collision & outcomes.shield].groupby(['вариант', 'gamma'], sort=False).agg(\n"
        "    столкновений=('seed', 'size'),\n"
        "    скорость_удара_медиана=('v_impact', 'median'),\n"
        "    скорость_удара_макс=('v_impact', 'max')).round(2)"
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
        "rand, rand_logs = run_grid(SEEDS, [1.0], VARIANTS, kind='brake', policy='random', trigger='schedule')\n"
        "rand['вариант'] = [NAMES[(s, d)] for s, d in zip(rand['shield'], rand['reaction_delay'])]\n"
        "rand.groupby('вариант', sort=False).agg(\n"
        "    столкновений=('collision', 'sum'),\n"
        "    запросов_перестроения=('lc_requested', 'sum'),\n"
        "    запретил_бы=('lc_would_block', 'sum'),\n"
        "    запрещено=('lc_blocked', 'sum'),\n"
        "    отменено_начатых=('lc_cancelled', 'sum'),\n"
        ")"
    ),
    code(
        "# столкновения случайного агента с фильтром: скорость удара и ответственность\n"
        "rr = responsibility_table(rand, rand_logs)\n"
        "rr = rr[rr.shield].merge(rand[['seed', 'shield', 'reaction_delay', 'v_impact', 'hit_violator']],\n"
        "                         on=['seed', 'shield', 'reaction_delay'])\n"
        "rr[['seed', 'reaction_delay', 'v_impact', 'hit_violator', 'responsible', 'partner_position',\n"
        "    'reaction_time_command', 'reaction_time_actual', 'agent_late', 'agent_late_actual',\n"
        "    'agent_released']].round(2)"
    ),
    md(
        "Без фильтра случайный агент в сцене тормозящего лидера (β = 1) разбивается в 38 "
        "из 40 эпизодов. С фильтром — 0 при обеих задержках: опасные перестроения "
        "запрещаются. Колонка «запретил бы» — гипотетическое решение фильтра (считается и "
        "без фильтра: 112), «запрещено» и «отменено» — реально выполненные вмешательства "
        "(без фильтра 0).\n\n"
        "## Вывод этапа 4\n\n"
        "1. Подрезание реализовано гладко: поперечная скорость до 2,5 м/с, поперечное "
        "ускорение до ≈ 2,6 м/с², без телепортации.\n"
        "2. При законном γ = 1 у базовой надлежащей реакции 0 столкновений из 40 при обеих "
        "задержках.\n"
        "3. При γ = 0,25 базовая реакция не спасает: 40 из 40 (ρ) и 35 из 40 (задержка 0). "
        "По формализации RSS все 75 — ответственность подрезающего (зазор на разметке "
        "меньше d_min); у агента ни одного опоздания команды (0–0,4 с после момента "
        "опасности) или фактического торможения (0,33–0,73 с) и ни одного отпускания.\n"
        "4. Подрезание — по расписанию эталона IDM/MOBIL + RSS из файла; фактическое γ у "
        "вариантов агента разное, его указываем рядом с заданным.\n"
        "5. Запрет и отмена перестроения — не полная поперечная надлежащая реакция RSS: "
        "на машину, которая сама сдвигается к агенту, фильтр отвечает только продольным "
        "торможением.\n"
        "6. Первичные данные: `results/pilot_04/outcomes.csv`, `responsibility.csv`, "
        "`collisions/*.csv.gz` (155 журналов), `params.json`."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("04_cutin.ipynb")
ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
