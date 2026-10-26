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
        "# Этап 7. Время эпизода и реальный объем прогонов\n\n"
        "Код: `rssrl/volume.py`. Проверки: `tests/test_volume.py`.\n\n"
        "Вопросы этапа:\n"
        "1. Сколько считается один эпизод на одном ядре и при параллельном счете?\n"
        "2. Куда уходит время — в наш код или в highway-env?\n"
        "3. Сколько сцен нужно на уровень β, чтобы классифицировать исходы "
        "(хуже / лучше / эквивалентно / неопределенно) с поправкой Холма?\n"
        "4. Сколько займут оценка и обучение PPO на этом компьютере?\n\n"
        "Числа ниже зависят от компьютера и его загрузки: это порядок величин, а не константы."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "import os, time, cProfile, pstats\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.pilot import run_grid\n"
        "from rssrl.responsibility import responsibility_table\n"
        "from rssrl.scenarios import make_scene, run_episode\n"
        "from rssrl.volume import (time_episodes, discordance, episodes_needed,\n"
        "                          min_detectable, holm_z, evaluation_plan)\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_07'  # таблицы пилота, в git не идут\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "print('ядер (логических):', os.cpu_count(), '| загрузка (1 мин):', round(os.getloadavg()[0], 1))"
    ),
    md("## 1. Время эпизода на одном ядре"),
    code(
        "setups = [\n"
        "    ('nominal', (1.0, False, 0.0), 'номинальный, без фильтра'),\n"
        "    ('nominal', (1.0, True, 0.5), 'номинальный, фильтр ρ'),\n"
        "    ('brake', (2.0, True, 0.5), 'тормозящий β = 2, фильтр, задержка ρ'),\n"
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
        "Время пропорционально длине эпизода в секундах симуляции: эпизод без фильтра "
        "короче, потому что заканчивается столкновением. Полный эпизод 40 с — около 5 с "
        "счета на одном ядре (прошлый замер ≈ 4 с; этот сделан при сильной посторонней "
        "загрузке компьютера, см. загрузку в первой ячейке)."
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
        "    if 'rssrl' in path: return 'наш код (rssrl: фильтр, нарушители, логгер)'\n"
        "    if 'highway_env' in path: return 'highway-env (фон, столкновения, дорога)'\n"
        "    if 'numpy' in path: return 'numpy (вызывается из highway-env и нашего кода)'\n"
        "    return 'прочее (Python, gymnasium)'\n"
        "share = {}\n"
        "for (path, _, _), v in st.stats.items():\n"
        "    share[group(path)] = share.get(group(path), 0) + v[2]\n"
        "(pd.Series(share) / total).sort_values(ascending=False).round(3).to_frame('доля времени')"
    ),
    md(
        "Основное время уходит на сам highway-env: каждая из 50 фоновых машин на каждом "
        "из 600 шагов ищет соседей (IDM, MOBIL) и проверяет столкновения со всеми. Наш "
        "фильтр, нарушители и логгер — малая доля. Поэтому ускорять наш код бессмысленно, "
        "а уменьшать число машин или частоту симуляции нельзя: это изменит сам эксперимент."
    ),
    md(
        "## 3. Параллельный счет и пилот по сетке β\n\n"
        "Основной эксперимент в миниатюре: 50 сцен × 6 уровней β × 3 варианта (фильтр с "
        "задержкой ρ и 0, без фильтра), агент IDLE вместо PPO, нарушение по расписанию "
        "(одинаковый момент для всех вариантов). d_min фильтра — лемма 2 с учетом рывка. "
        "Рядом — тот же пилот с d_min как в статье (старый вариант). Это одновременно "
        "замер скорости параллельного счета и источник оценки разброса для расчета объема."
    ),
    code(
        "BETAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]\n"
        "# основной режим — фильтр с задержкой ρ, дополнительный — мгновенный фильтр\n"
        "VARIANTS = [(True, 0.5), (True, 0.0), (False, 0.5)]\n"
        "NAMES = {(True, 0.5): 'фильтр, задержка ρ (основной)', (True, 0.0): 'фильтр, задержка 0 (доп.)',\n"
        "         (False, 0.5): 'без фильтра'}\n"
        "t0 = time.perf_counter()\n"
        "grid, grid_logs = run_grid(range(50), BETAS, VARIANTS, kind='brake', trigger='schedule')\n"
        "wall = time.perf_counter() - t0\n"
        "grid['вариант'] = [NAMES[(s, d)] for s, d in zip(grid.shield, grid.reaction_delay)]\n"
        "grid.drop(columns=['crash_partners']).to_csv(PILOT / 'beta_grid.csv', index=False)\n"
        "THROUGHPUT = grid.t_end.sum() / wall  # секунд симуляции за секунду счета\n"
        "print(f'{len(grid)} эпизодов за {wall / 60:.1f} мин; '\n"
        "      f'{THROUGHPUT:.1f} с симуляции за 1 с счета; '\n"
        "      f'полный эпизод 40 с ≈ {40 / THROUGHPUT:.2f} с при параллельном счете')"
    ),
    code(
        "# старый вариант: d_min по лемме 2 как в статье, те же сцены и то же расписание\n"
        "old, old_logs = run_grid(range(50), BETAS, VARIANTS[:2], kind='brake', trigger='schedule', dmin='lemma2')\n"
        "old['вариант'] = [NAMES[(s, d)] for s, d in zip(old.shield, old.reaction_delay)]\n"
        "old.drop(columns=['crash_partners']).to_csv(PILOT / 'beta_grid_lemma2.csv', index=False)\n"
        "pd.concat({'d_min с учетом рывка (новый)': grid.pivot_table(index='вариант', columns='beta', values='collision', aggfunc='sum', sort=False),\n"
        "           'd_min леммы 2 как в статье (старый)': old.pivot_table(index='вариант', columns='beta', values='collision', aggfunc='sum', sort=False)},\n"
        "          names=['формула d_min в фильтре'])"
    ),
    code(
        "# доля столкновений с 95 % бутстреп-интервалом по сценам (одна «политика» — IDLE)\n"
        "rng = np.random.default_rng(0)\n"
        "fig, ax = plt.subplots(figsize=(8, 4.8))\n"
        "colors = {(False, 0.5): '#7f8c8d', (True, 0.0): '#2c6fbb', (True, 0.5): '#e67e22'}\n"
        "# ρ рисуем крупнее и раньше, чтобы на нуле была видна и она, и линия задержки 0\n"
        "style = {(True, 0.5): (11, 3.5), (True, 0.0): (6, 1.5), (False, 0.5): (8, 2)}\n"
        "for key, name in NAMES.items():\n"
        "    sub = grid[(grid.shield == key[0]) & (grid.reaction_delay == key[1])]\n"
        "    mat = sub.pivot(index='seed', columns='beta', values='collision').astype(float)[BETAS].to_numpy()\n"
        "    boot = mat[rng.integers(0, len(mat), (2000, len(mat)))].mean(axis=1)\n"
        "    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)\n"
        "    m = mat.mean(axis=0)\n"
        "    ms, lw = style[key]\n"
        "    ax.plot(BETAS, m, 'o-', color=colors[key], linewidth=lw, markersize=ms,\n"
        "            label=name + (', d_min с рывком' if key[0] else ''))\n"
        "    ax.fill_between(BETAS, lo, hi, color=colors[key], alpha=0.15, linewidth=0)\n"
        "for key, name in list(NAMES.items())[:2]:\n"
        "    sub = old[(old.shield == key[0]) & (old.reaction_delay == key[1])]\n"
        "    m = sub.pivot(index='seed', columns='beta', values='collision').astype(float)[BETAS].mean().to_numpy()\n"
        "    ax.plot(BETAS, m, 's--', color=colors[key], linewidth=1.5, markersize=6, alpha=0.75,\n"
        "            label=name + ', d_min леммы 2 (старый)')\n"
        "ax.set_xlabel('β — во сколько раз торможение лидера больше a_max,brake = 5 м/с²')\n"
        "ax.set_ylabel('доля эпизодов со столкновением')\n"
        "ax.set_ylim(-0.04, 1.04)\n"
        "ax.grid(alpha=0.3)\n"
        "ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)\n"
        "ax.set_title('Пилот: IDLE вместо PPO, 50 общих сцен, нарушение по расписанию\\n'\n"
        "             '(сплошные — d_min с учетом рывка, полоса — 95 % бутстреп; пунктир — d_min леммы 2)', fontsize=9)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '07_pilot_beta_grid.png', dpi=150)"
    ),
    md(
        "Это пилот без RL, но он показывает главное изменение. **С d_min, учитывающей "
        "рывок**, у фильтра 0 столкновений из 50 на каждом уровне β от 1 до 2, при обеих "
        "задержках: агент держит дистанцию с запасом на набор торможения, а при β > 1 "
        "добавляется экстренное торможение 8 м/с². **С d_min леммы 2 как в статье** при "
        "задержке ρ столкновения 36–42 из 50 на каждом уровне, включая β = 1: медленные "
        "въезды в стоящего нарушителя, по правилам RSS почти все «не установлено». Без "
        "фильтра IDLE врезается всегда.\n\n"
        "Отрицательный результат допустим: при агенте IDLE + фильтр превышение торможения "
        "лидера до β = 2 (10 м/с²) в этой сцене к столкновениям не приводит, β* на сетке "
        "не достигнуто. С PPO, которая сама выбирает дистанцию и скорость, картина может "
        "быть другой. Если и с PPO столкновений не будет, вопрос к преподавателю — "
        "расширять ли сетку (10 м/с² — выбранный предел испытаний) или делать сцену "
        "жестче (например, нарушение при меньшем зазоре)."
    ),
    code(
        "# ответственность за столкновения в пилоте по сетке β\n"
        "resp = pd.concat([responsibility_table(grid, grid_logs).assign(формула='с рывком'),\n"
        "                  responsibility_table(old, old_logs).assign(формула='лемма 2')], ignore_index=True)\n"
        "resp['вариант'] = [NAMES[(s, d)] for s, d in zip(resp['shield'], resp['reaction_delay'])]\n"
        "resp.to_csv(PILOT / 'beta_grid_responsibility.csv', index=False)\n"
        "(resp.pivot_table(index=['формула', 'вариант', 'responsible'], columns='level', values='seed',\n"
        "                  aggfunc='size', fill_value=0, sort=False)\n"
        "     .rename_axis(index=['формула d_min', 'вариант', 'отвечает'], columns='β'))"
    ),
    md(
        "## 3а. Парность сцен: одинаков ли момент нарушения у разных вариантов агента\n\n"
        "Сейчас нарушение начинается по условию на зазор (агент идет за машиной с зазором "
        "≤ 1,1·d_min). Зазор зависит от того, как ехал агент: с фильтром агент мог раньше "
        "притормозить, без фильтра — нет. Тогда у вариантов агента на одной и той же сцене "
        "момент нарушения (и даже машина-нарушитель) могут различаться, и сравнение "
        "перестает быть строго парным. Проверяем на 50 сценах (β = 1; до нарушения β ни на что "
        "не влияет), сравнивая с основным вариантом (фильтр, задержка ρ).\n\n"
        "Альтернатива — нарушение **по расписанию** (`trigger='schedule'`): момент и машина "
        "берутся из эталонного прогона (IDLE + фильтр ρ) и одинаковы для всех вариантов агента."
    ),
    code(
        "del grid_logs, old_logs  # освобождаем память\n"
        "pair = {}\n"
        "for trig in ['gap', 'schedule']:\n"
        "    o, _ = run_grid(range(50), [1.0, 2.0], VARIANTS, kind='brake', trigger=trig)\n"
        "    o['вариант'] = [NAMES[(s, d)] for s, d in zip(o.shield, o.reaction_delay)]\n"
        "    pair[trig] = o\n"
        "rows = []\n"
        "for trig, o in pair.items():\n"
        "    b1 = o[o.beta == 1.0]\n"
        "    t = b1.pivot(index='seed', columns='вариант', values='t_trigger')\n"
        "    u = b1.pivot(index='seed', columns='вариант', values='violator_uid')\n"
        "    ref = NAMES[(True, 0.5)]\n"
        "    for name in [NAMES[(True, 0.0)], NAMES[(False, 0.5)]]:\n"
        "        same_t = ((t[name] - t[ref]).abs() < 1e-6) | (t[name].isna() & t[ref].isna())\n"
        "        same_u = (u[name] == u[ref]) | (u[name].isna() & u[ref].isna())\n"
        "        rows.append({'момент нарушения': 'по зазору' if trig == 'gap' else 'по расписанию',\n"
        "                     'вариант (сравнение с основным)': name,\n"
        "                     'тот же момент, сцен из 50': int(same_t.sum()),\n"
        "                     'та же машина, сцен из 50': int(same_u.sum()),\n"
        "                     'медиана |Δt|, с': round(float((t[name] - t[ref]).abs().median()), 2),\n"
        "                     'макс |Δt|, с': round(float((t[name] - t[ref]).abs().max()), 2)})\n"
        "pd.DataFrame(rows)"
    ),
    code(
        "# влияет ли способ выбора момента на число столкновений\n"
        "pd.concat({('по зазору' if k == 'gap' else 'по расписанию'):\n"
        "           o.pivot_table(index='вариант', columns='beta', values='collision', aggfunc='sum', sort=False)\n"
        "           for k, o in pair.items()}, names=['момент нарушения'])"
    ),
    code(
        "# по расписанию: был ли нарушитель в момент нарушения машиной впереди агента\n"
        "s = pair['schedule']\n"
        "s[s.beta == 1.0].groupby('вариант', sort=False)['violator_was_front'].agg(['sum', 'size'])"
    ),
    md(
        "Итог по парности:\n"
        "- **По зазору** с новой формулой момент нарушения у мгновенного фильтра и у агента "
        "без фильтра совпадает с основным вариантом только в 2 сценах из 50, расхождение "
        "до ≈ 20 с. Причина: агент с новой d_min держится дальше, и условие «зазор "
        "≤ 1,1·d_min» наступает в другое время или с другой машиной. Сравнение по зазору "
        "уже не парное.\n"
        "- **По расписанию** момент и машина одинаковы во всех сценах, где нарушение "
        "вообще случилось (без фильтра в 8 сценах агент врезался раньше момента нарушения). "
        "Нарушитель в этот момент был машиной впереди агента в 50 / 47 / 41 сценах из 50 "
        "(основной вариант / мгновенный фильтр / без фильтра).\n\n"
        "Плюсы расписания: одно и то же воздействие на все варианты агента (настоящие "
        "общие случайные числа), разница в исходах — только от агента; агент, который "
        "держит большую дистанцию, получает за это честный выигрыш. Минусы: если агент "
        "едет иначе, чем эталон (перестроится, отстанет), нарушитель может оказаться не "
        "перед ним, и испытание для этого варианта будет мягче; «трудность» сцены зависит "
        "от выбора эталонного агента (для основного эксперимента предлагаем IDM/MOBIL + RSS — "
        "он есть во всех сравнениях и не зависит от seed обучения). Плюсы условия по "
        "зазору: каждый вариант испытывается в одинаково критической ситуации. Минусы: "
        "пары не строгие, а у осторожного агента нарушение может не наступить вовсе. "
        "Рекомендация: основной эксперимент — по расписанию, долю сцен, где нарушитель "
        "оказался машиной впереди, — в отчет."
    ),
    md(
        "## 4. Сколько сцен нужно на уровень\n\n"
        "Сравнения парные: уровень β и β = 1 на одних и тех же сценах. Разброс парной "
        "разности определяется долей **несовпадающих пар** d — сцен, где на одном уровне "
        "столкновение есть, а на другом нет. Половина ДИ разности ≈ z·√(d/n).\n\n"
        "Поправка Холма: сравнений в основной части 5 уровней (β > 1 против β = 1) × "
        "5 вариантов (RL без фильтра; IDM/MOBIL+RSS и RL+RSS — каждый при двух задержках) = 25. "
        "Для объема берем самый строгий шаг Холма (уровень 0,05/25)."
    ),
    code(
        "M = 25\n"
        "Z = holm_z(M)\n"
        "print(f'z для 1 сравнения: {holm_z(1):.2f}; z с поправкой Холма (m = {M}, худший шаг): {Z:.2f}')\n"
        "disc = []\n"
        "for (sh, d), name in NAMES.items():\n"
        "    if not sh:\n"
        "        continue\n"
        "    sub = grid[grid.shield & (grid.reaction_delay == d)].pivot(index='seed', columns='beta', values='collision')\n"
        "    disc.append(pd.Series({b: discordance(sub[b], sub[1.0]) for b in BETAS[1:]}, name=name))\n"
        "disc = pd.DataFrame(disc)\n"
        "disc.columns = [f'β = {b:g}' for b in BETAS[1:]]\n"
        "disc.round(2)  # доля несовпадающих пар с β = 1"
    ),
    md(
        "Два разных требования к объему:\n"
        "- **«Хуже»** (весь ДИ разности > 0) — легко, когда эффект большой. Наименьший "
        "эффект, который при n сценах можно назвать «хуже»: δ_min ≈ z²/(n + z²).\n"
        "- **«Эквивалентно»** (весь ДИ в пределах ±2 п.п.) — трудно: нужно n ≈ z²·d / 0,02². "
        "Достижимо только при малом d, то есть там, где эффекта почти нет."
    ),
    code(
        "rows = []\n"
        "for n in (100, 300, 500, 1000, 2000):\n"
        "    rows.append({'сцен на уровень': n,\n"
        "                 'δ_min «хуже», п.п. (без поправки)': round(100 * min_detectable(n), 1),\n"
        "                 'δ_min «хуже», п.п. (Холм)': round(100 * min_detectable(n, Z), 1),\n"
        "                 'эквивалентность ±2 п.п. возможна при d ≤, п.п. (Холм)': round(100 * n * 0.02 ** 2 / Z ** 2, 2)})\n"
        "pd.DataFrame(rows).set_index('сцен на уровень')"
    ),
    code(
        "need = pd.DataFrame({\n"
        "    f'±{w * 100:g} п.п.': [episodes_needed(d, w, Z) for d in (0.01, 0.02, 0.05, 0.1, 0.2)]\n"
        "    for w in (0.02, 0.03, 0.05)\n"
        "}, index=[f'd = {d:g}' for d in (0.01, 0.02, 0.05, 0.1, 0.2)])\n"
        "need.index.name = 'несовпадающих пар'\n"
        "need  # сцен на уровень, чтобы половина ДИ (с Холмом) была не шире указанной"
    ),
    md(
        "Вывод для плана: **500 сцен на уровень** — разумный компромисс. При нем:\n"
        "- «хуже» обнаруживается начиная с разности ≈ 2 п.п. (с поправкой Холма);\n"
        "- «эквивалентно» в пределах ±2 п.п. можно показать, если несовпадающих пар не "
        "больше ≈ 2 %. В пилоте с новой формулой их 0 (столкновений нет ни на одном уровне), "
        "то есть для IDLE + фильтр результат был бы «эквивалентно» на всех уровнях.\n\n"
        "**Оговорка.** Это расчет для одной политики. По замечанию 8 бутстреп двухуровневый: "
        "сначала политики (seed обучения), затем сцены. Разброс между политиками PPO до "
        "обучения неизвестен; если он большой, сужать ДИ нужно числом политик, а не сцен."
    ),
    md("## 5. Сколько займет оценка"),
    code(
        "mean_sim = grid.t_end.mean()  # в пилоте часть эпизодов короче 40 с из-за столкновений\n"
        "rows = []\n"
        "for k in (3, 5):\n"
        "    for n in (300, 500, 1000):\n"
        "        plan = evaluation_plan(n_scenes=n, n_policies=k)\n"
        "        ep = plan['эпизодов всего']\n"
        "        rows.append({'политик на тип (seed обучения)': k, 'сцен на уровень': n,\n"
        "                     'эпизодов на сцену и уровень': plan['эпизодов на сцену и уровень'],\n"
        "                     'эпизодов всего': ep,\n"
        "                     'часов (все эпизоды по 40 с)': round(ep * 40 / THROUGHPUT / 3600, 1),\n"
        "                     f'часов (средняя длина как в пилоте, {mean_sim:.0f} с)': round(ep * mean_sim / THROUGHPUT / 3600, 1)})\n"
        "pd.DataFrame(rows).set_index(['политик на тип (seed обучения)', 'сцен на уровень'])"
    ),
    md(
        "Что входит: 6 уровней β + 4 уровня γ; на каждую сцену и уровень — 2 типа политик "
        "PPO × k seed обучения, каждая с фильтром и без (схема 2×2), плюс IDM/MOBIL+RSS "
        "без RL; варианты с фильтром — при двух задержках. Дополнительный эксперимент 2 "
        "и контрфактические прогоны сюда не входят (их делаем на ~100 сценах).\n\n"
        "Вариант «5 политик × 500 сцен» при замеренных 23 с симуляции за 1 с счета (замер "
        "при посторонней загрузке; в прошлый раз 26–35) — 61–77 часов, то есть около "
        "3 суток. Его можно разбить на ночные прогоны по уровням."
    ),
    md("## 6. Сколько займет обучение PPO"),
    code(
        "from rssrl.volume import ppo_steps_per_second\n"
        "speed = {n: ppo_steps_per_second(n) for n in (1, 8)}\n"
        "print({f'{k} сред': f'{v:.1f} шагов/с' for k, v in speed.items()})"
    ),
    code(
        "rows = []\n"
        "for budget in (2e4, 1e5, 2e5):\n"
        "    per_policy_h = budget / speed[8] / 3600\n"
        "    rows.append({'шагов обучения на политику': f'{budget:.0e}',\n"
        "                 'часов на 1 политику': round(per_policy_h, 1),\n"
        "                 'часов на 2 типа × 3 seed': round(6 * per_policy_h, 1),\n"
        "                 'часов на 2 типа × 5 seed': round(10 * per_policy_h, 1)})\n"
        "pd.DataFrame(rows).set_index('шагов обучения на политику')"
    ),
    md(
        "Шаг обучения — одно решение политики (1 с симуляции, 15 шагов физики). "
        "Замер сделан на обычной highway-v0 с нашим фоном: стоимость шага определяет "
        "highway-env, а фильтр ее почти не меняет (раздел 2). Бюджет шагов выберем по "
        "кривой обучения первой политики; для highway-env обычно хватает 10⁴–10⁵ шагов."
    ),
    md(
        "## Выводы этапа 7\n\n"
        "1. Полный эпизод (40 с) — около 5 с на одном ядре и ≈ 1,7 с при параллельном "
        "счете (при сильной посторонней загрузке). Время уходит на highway-env (≈ 66 %), "
        "наш код — 1,5 %.\n"
        "2. Пилот по сетке β (IDLE, 50 сцен, нарушение по расписанию): с d_min, "
        "учитывающей рывок, у фильтра 0 столкновений на всех уровнях β ≤ 2 при обеих "
        "задержках; с d_min леммы 2 как в статье при задержке ρ — 36–42 из 50 на каждом "
        "уровне (въезды в стоящего нарушителя). β* на сетке не достигнуто.\n"
        "3. Момент нарушения «по зазору» с новой формулой почти никогда не совпадает "
        "между вариантами агента (2 сцены из 50); «по расписанию» — совпадает. Пилоты и "
        "основной эксперимент — по расписанию.\n"
        "4. Предлагаемый объем: 500 общих сцен на уровень (δ_min «хуже» ≈ 2 п.п. с Холмом).\n"
        "5. Оценка «5 политик × 500 сцен» — около 3 суток счета; обучение 10 политик по "
        "10⁵ шагов — около 9 часов (30,6 шага/с на 8 средах)."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("07_timing.ipynb")
ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
