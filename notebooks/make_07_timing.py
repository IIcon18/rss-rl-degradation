"""Собирает и выполняет notebooks/07_timing.ipynb.

Запуск из корня проекта: .venv/bin/python notebooks/make_07_timing.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Этап 7. Пилот по сетке β и время эпизода\n\n"
        "Код: `rssrl/pilot.py`, `rssrl/volume.py`. Проверки: `tests/test_volume.py`.\n\n"
        "Повтор после письма от 29.10: основной β-сценарий — контролируемое следование, "
        "нарушение по расписанию из файла (эталон IDM/MOBIL + RSS), основной фильтр — "
        "базовая надлежащая реакция с учетом рывка (без аварийного торможения).\n\n"
        "Вопросы этапа:\n"
        "1. Сколько считается один эпизод на одном ядре и сколько эпизодов в час дает "
        "параллельный счет?\n"
        "2. Куда уходит время?\n"
        "3. Пилот по сетке β: IDLE вместо PPO, 50 общих сцен.\n\n"
        "Объем основного эксперимента, правило решения и замер обучения PPO на целевой "
        "среде — в ноутбуке 08: расчет объема по этому пилоту делать нельзя (IDLE не "
        "оценивает разброс будущих политик PPO).\n\n"
        "Числа времени зависят от компьютера и его загрузки: это порядок величин, а не константы."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "import os, time, cProfile, pstats\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.pilot import run_variants, export_pilot\n"
        "from rssrl.responsibility import responsibility_table\n"
        "from rssrl.scenarios import make_scene, run_episode\n"
        "from rssrl.stats import zero_upper_bound\n"
        "from rssrl.volume import time_episodes\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_07'  # первичные данные: CSV в git\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "print('ядер (логических):', os.cpu_count(), '| загрузка (1 мин):', round(os.getloadavg()[0], 1))"
    ),
    md("## 1. Время эпизода на одном ядре"),
    code(
        "setups = [\n"
        "    ('nominal', (1.0, False, 0.0), 'номинальный, без фильтра'),\n"
        "    ('nominal', (1.0, True, 0.5), 'номинальный, базовая реакция ρ'),\n"
        "    ('brake', (2.0, True, 0.5), 'тормозящий β = 2, базовая реакция ρ'),\n"
        "]\n"
        "frames = []\n"
        "for kind, (level, shield, delay), label in setups:\n"
        "    t = time_episodes([(s, level, shield, delay) for s in range(8)], kind)\n"
        "    t['вариант'] = label\n"
        "    frames.append(t)\n"
        "single = pd.concat(frames)\n"
        "single['с счета на 1 с симуляции'] = single.wall_s / single.sim_s\n"
        "single.groupby('вариант', sort=False).agg(\n"
        "    эпизодов=('wall_s', 'size'),\n"
        "    счет_с=('wall_s', 'mean'),\n"
        "    симуляция_с=('sim_s', 'mean'),\n"
        "    счет_на_1с_симуляции=('с счета на 1 с симуляции', 'mean'),\n"
        ").round(3)"
    ),
    md(
        "Полный эпизод (40 с симуляции) — около 6 с счета на одном ядре (≈ 0,15 с счета на "
        "1 с симуляции; эпизод без фильтра короче, потому что заканчивается столкновением). "
        "Замер сделан при высокой загрузке компьютера (см. первую ячейку): это порядок "
        "величины."
    ),
    md("## 2. Куда уходит время"),
    code(
        "pr = cProfile.Profile()\n"
        "pr.enable()\n"
        "run_episode(make_scene(5, shield=True, kind='nominal'))\n"
        "pr.disable()\n"
        "st = pstats.Stats(pr)\n"
        "total = sum(v[2] for v in st.stats.values())  # собственное время всех функций\n"
        "def group(path):\n"
        "    if 'rssrl' in path: return 'наш код (rssrl: фильтр, фон, нарушители, логгер)'\n"
        "    if 'highway_env' in path: return 'highway-env (IDM, дорога, столкновения)'\n"
        "    if 'numpy' in path: return 'numpy (вызывается из highway-env и нашего кода)'\n"
        "    return 'прочее (Python, gymnasium)'\n"
        "share = {}\n"
        "for (path, _, _), v in st.stats.items():\n"
        "    share[group(path)] = share.get(group(path), 0) + v[2]\n"
        "(pd.Series(share) / total).sort_values(ascending=False).round(3).to_frame('доля времени')"
    ),
    md(
        "Основное время — сам highway-env (≈ 70 %): IDM каждой из ~49 фоновых машин ищет "
        "соседей на каждом из 600 шагов, плюс проверка столкновений. Наш код (фильтр, фон "
        "с RSS, нарушители, логгер) — около 2 %. Ускорять его бессмысленно, а уменьшать "
        "число машин или частоту симуляции нельзя: это изменит эксперимент."
    ),
    md(
        "## 3. Пилот по сетке β\n\n"
        "50 общих сцен × 6 уровней β × 4 варианта агента (IDLE вместо PPO): базовая "
        "реакция с задержкой ρ (основной) и 0, тот же фильтр с аварийным торможением "
        "(ρ), без фильтра. Расписание — `results/schedules/brake_following_0-499.csv`, "
        "одно для всех. Это и замер пропускной способности параллельного счета."
    ),
    code(
        "BETAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]\n"
        "VARIANTS = [\n"
        "    ('базовая, задержка ρ (основной)', True, 0.5, False),\n"
        "    ('базовая, задержка 0 (доп.)', True, 0.0, False),\n"
        "    ('с аварийным торможением, ρ (сравнение)', True, 0.5, True),\n"
        "    ('без фильтра', False, 0.5, False),\n"
        "]\n"
        "ORDER = [v[0] for v in VARIANTS]\n"
        "t0 = time.perf_counter()\n"
        "grid, grid_logs = run_variants(range(50), BETAS, VARIANTS, kind='brake', trigger='schedule')\n"
        "wall = time.perf_counter() - t0\n"
        "EP_PER_HOUR = len(grid) / wall * 3600\n"
        "SIM_PER_WALL = grid.t_end.sum() / wall  # секунд симуляции за секунду счета всей пачки\n"
        "print(f'{len(grid)} эпизодов за {wall / 60:.1f} мин на {os.cpu_count()} ядрах: '\n"
        "      f'{EP_PER_HOUR:.0f} эпизодов в час, {SIM_PER_WALL:.1f} с симуляции за 1 с счета пачки')\n"
        "import json\n"
        "(PILOT / 'timing.json').write_text(json.dumps({'episodes': len(grid), 'wall_s': wall, 'cpu': os.cpu_count(),\n"
        "    'episodes_per_hour': EP_PER_HOUR, 'sim_s_per_wall_s': SIM_PER_WALL,\n"
        "    'mean_episode_sim_s': float(grid.t_end.mean()), 'load_1min': os.getloadavg()[0]}, indent=1))\n"
        "grid.pivot_table(index='вариант', columns='beta', values='collision', aggfunc='sum').loc[ORDER]"
    ),
    md(
        "Пропускная способность — эпизодов в час для **всей параллельной пачки** на этом "
        "компьютере. Ее уже не нужно делить на число ядер: параллельность в ней учтена."
    ),
    code(
        "imp = grid.pivot_table(index=['вариант'], columns='impact', values='seed',\n"
        "                       aggfunc='size', fill_value=0, sort=False)\n"
        "role = grid.pivot_table(index=['вариант'], columns='violator_role', values='seed',\n"
        "                        aggfunc='size', fill_value=0, sort=False)\n"
        "pd.concat([imp, role], axis=1).loc[ORDER]  # эпизодов по всем 6 уровням β (300 на вариант)"
    ),
    code(
        "# доля столкновений: точка — доля из 50 сцен; полоса — 95 % бутстреп по сценам;\n"
        "# если столкновений 0 — точная верхняя граница (Клоппер — Пирсон), не бутстреп\n"
        "rng = np.random.default_rng(0)\n"
        "fig, ax = plt.subplots(figsize=(8, 4.8))\n"
        "colors = dict(zip(ORDER, ['#e67e22', '#2c6fbb', '#8e44ad', '#7f8c8d']))\n"
        "style = dict(zip(ORDER, [(11, 3.5), (6, 1.5), (8, 2), (8, 2)]))\n"
        "for name in ORDER:\n"
        "    sub = grid[grid['вариант'] == name]\n"
        "    mat = sub.pivot(index='seed', columns='beta', values='collision').astype(float)[BETAS].to_numpy()\n"
        "    m = mat.mean(axis=0)\n"
        "    ms, lw = style[name]\n"
        "    ax.plot(BETAS, m, 'o-', color=colors[name], linewidth=lw, markersize=ms, label=name)\n"
        "    if mat.sum() == 0:\n"
        "        ax.fill_between(BETAS, 0, zero_upper_bound(len(mat)), color=colors[name], alpha=0.12, linewidth=0)\n"
        "    else:\n"
        "        boot = mat[rng.integers(0, len(mat), (2000, len(mat)))].mean(axis=1)\n"
        "        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)\n"
        "        ax.fill_between(BETAS, lo, hi, color=colors[name], alpha=0.15, linewidth=0)\n"
        "ax.set_xlabel('β — во сколько раз торможение лидера больше a_max,brake = 5 м/с²')\n"
        "ax.set_ylabel('доля эпизодов со столкновением')\n"
        "ax.set_ylim(-0.04, 1.04)\n"
        "ax.grid(alpha=0.3)\n"
        "ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)\n"
        "ax.set_title('Пилот: IDLE вместо PPO, контролируемое следование, 50 общих сцен,\\n'\n"
        "             'нарушение по расписанию эталона (полоса — 95 % бутстреп или точная граница при 0)', fontsize=9)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '07_pilot_beta_grid.png', dpi=150)"
    ),
    code(
        "resp = responsibility_table(grid, grid_logs)\n"
        "n_logs = export_pilot(grid, grid_logs, PILOT, responsibility=resp,\n"
        "                      params={'notebook': '07_timing', 'kind': 'brake', 'traffic': 'following',\n"
        "                              'trigger': 'schedule', 'policy': 'idle'})\n"
        "print('сохранено покадровых журналов столкновений:', n_logs)\n"
        "if len(resp):\n"
        "    display(resp.pivot_table(index=['вариант', 'responsible'], columns='level', values='seed',\n"
        "                             aggfunc='size', fill_value=0, sort=False)\n"
        "                .rename_axis(index=['вариант', 'отвечает'], columns='β'))"
    ),
    md(
        "## Выводы пилота по сетке β\n\n"
        "1. **Основной вариант (базовая надлежащая реакция, задержка ρ)**: столкновений "
        "0 / 12 / 18 / 23 / 26 / 26 из 50 при β = 1 / 1,2 / 1,4 / 1,6 / 1,8 / 2. Частота растет с "
        "β, и рост монотонен по каждой сцене: нет ни одной сцены, где столкновение было бы "
        "при меньшем β и пропало при большем. Скорость удара растет вместе с β (медиана "
        "3,5 м/с при β = 1,2 и 8,8 м/с при β = 2).\n"
        "2. **По формализации RSS все 105 столкновений основного варианта — ответственность "
        "нарушителя**: агент выполнил надлежащую реакцию в каждом случае (команда торможения "
        "через 0,4 с после момента опасности, фактическое торможение 4 м/с² через 0,93–1,07 с, "
        "ни одного опоздания и отпускания). Нарушение допущения снимает гарантию — и здесь "
        "столкновения действительно появляются; при β = 1 их нет.\n"
        "3. **Мгновенный фильтр (задержка 0) и вариант с аварийным торможением** — 0 "
        "столкновений на всех уровнях: они перекрывают превышение β·5 м/с² за счет более "
        "ранней реакции или торможения до 8 м/с². На графике их линии совпадают на нуле "
        "(синяя «задержка 0» закрыта фиолетовой); светлая полоса у нуля — точная верхняя "
        "95 % граница при 0 из 50 (5,8 %), а не бутстреп.\n"
        "4. **Без фильтра** IDLE врезается всегда; в 96 эпизодах из 300 — еще до воздействия "
        "(в более медленного лидера). Эпизоды не исключаем: они помечены «столкновение до "
        "воздействия». Во всех эпизодах, где воздействие состоялось (у всех вариантов), "
        "назначенная машина была лидером агента.\n"
        "5. **Счет**: 1200 эпизодов за 34 мин на 8 ядрах — около 2100 эпизодов в час для "
        "всей параллельной пачки (уже с учетом параллельности, на ядра не делим), при "
        "загрузке компьютера. Сохранено в `results/pilot_07/timing.json` для плана в "
        "ноутбуке 08.\n"
        "6. Это IDLE, а не PPO: числа описывают фильтр с неподвижной политикой, а не "
        "будущую систему RL + RSS, и не годятся для расчета объема (ноутбук 08).\n"
        "7. Первичные данные: `results/pilot_07/outcomes.csv`, `responsibility.csv`, "
        "`collisions/*.csv.gz` (405 журналов), `params.json`."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("07_timing.ipynb")
ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
