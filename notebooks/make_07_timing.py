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
        "    ('nominal', (1.0, True, 0.0), 'номинальный, фильтр'),\n"
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
        "короче, потому что заканчивается столкновением. Полный эпизод 40 с — около 4 с "
        "счета на одном ядре."
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
        "Основной эксперимент в миниатюре: 50 сцен × 6 уровней β × 3 варианта (без фильтра, "
        "фильтр с задержкой 0 и ρ), агент IDLE вместо PPO. Это одновременно замер "
        "скорости параллельного счета и источник оценки разброса для расчета объема."
    ),
    code(
        "BETAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]\n"
        "VARIANTS = [(False, 0.0), (True, 0.0), (True, 0.5)]\n"
        "NAMES = {(False, 0.0): 'без фильтра', (True, 0.0): 'фильтр, задержка 0',\n"
        "         (True, 0.5): 'фильтр, задержка ρ'}\n"
        "t0 = time.perf_counter()\n"
        "grid, _ = run_grid(range(50), BETAS, VARIANTS, kind='brake')\n"
        "wall = time.perf_counter() - t0\n"
        "grid['вариант'] = [NAMES[(s, d)] for s, d in zip(grid.shield, grid.reaction_delay)]\n"
        "grid.drop(columns=['crash_partners']).to_csv(PILOT / 'beta_grid.csv', index=False)\n"
        "THROUGHPUT = grid.t_end.sum() / wall  # секунд симуляции за секунду счета\n"
        "print(f'{len(grid)} эпизодов за {wall / 60:.1f} мин; '\n"
        "      f'{THROUGHPUT:.1f} с симуляции за 1 с счета; '\n"
        "      f'полный эпизод 40 с ≈ {40 / THROUGHPUT:.2f} с при параллельном счете')"
    ),
    code(
        "grid.pivot_table(index='вариант', columns='beta', values='collision', aggfunc='sum', sort=False)"
    ),
    code(
        "# доля столкновений с 95 % бутстреп-интервалом по сценам (одна «политика» — IDLE)\n"
        "rng = np.random.default_rng(0)\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "colors = {(False, 0.0): '#7f8c8d', (True, 0.0): '#2c6fbb', (True, 0.5): '#e67e22'}\n"
        "for key, name in NAMES.items():\n"
        "    sub = grid[(grid.shield == key[0]) & (grid.reaction_delay == key[1])]\n"
        "    mat = sub.pivot(index='seed', columns='beta', values='collision').astype(float)[BETAS].to_numpy()\n"
        "    boot = mat[rng.integers(0, len(mat), (2000, len(mat)))].mean(axis=1)\n"
        "    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)\n"
        "    m = mat.mean(axis=0)\n"
        "    ax.plot(BETAS, m, 'o-', color=colors[key], linewidth=2, markersize=8, label=name)\n"
        "    ax.fill_between(BETAS, lo, hi, color=colors[key], alpha=0.15, linewidth=0)\n"
        "ax.set_xlabel('β — во сколько раз торможение лидера больше a_max,brake = 5 м/с²')\n"
        "ax.set_ylabel('доля эпизодов со столкновением')\n"
        "ax.set_ylim(-0.02, 1.02)\n"
        "ax.grid(alpha=0.3)\n"
        "ax.legend(frameon=False, fontsize=9)\n"
        "ax.set_title('Пилот: IDLE вместо PPO, 50 сцен, общие сцены для всех β (полоса — 95 % бутстреп)', fontsize=9)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '07_pilot_beta_grid.png', dpi=150)"
    ),
    md(
        "Это пилот без RL, но он уже показывает форму зависимости. С задержкой 0 "
        "столкновения начинаются с β ≈ 1,6–1,8, с задержкой ρ — уже с β = 1,2. "
        "Без фильтра IDLE врезается всегда (он не тормозит сам). С PPO картина будет "
        "другой: политика сама держит дистанцию, поэтому разница между вариантами "
        "станет содержательной."
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
        "- «эквивалентно» в пределах ±2 п.п. можно показать, только если несовпадающих "
        "пар не больше ≈ 1,9 %. Там, где d больше, а эффект мал, результат честно будет "
        "«неопределенно».\n\n"
        "**Оговорка.** Это расчет для одной политики. По замечанию 8 бутстреп двухуровневый: "
        "сначала политики (seed обучения), затем сцены. Разброс между политиками PPO до "
        "обучения неизвестен; если он большой, сужать ДИ нужно числом политик, а не сцен. "
        "Число сцен и политик окончательно уточним по первым обученным политикам."
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
        "(сочетания β и γ) и контрфактические прогоны сюда не входят: их делаем на меньшем "
        "числе сцен (например, 100).\n\n"
        "Вариант «5 политик × 500 сцен» — по двум замерам скорости (26 и 35 с симуляции за 1 с счета) "
        "от 1,5 до 3 суток счета на этом компьютере, в зависимости от его загрузки. "
        "Его можно разбить на ночные прогоны по уровням: результаты по уровням независимы."
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
        "1. Полный эпизод (40 с) — около 4 с на одном ядре и 1,2–1,5 с при параллельном "
        "счете на этом компьютере (зависит от загрузки). Время уходит на highway-env, а не на наш код.\n"
        "2. Пилот по сетке β (IDLE, 50 сцен) показывает четкую зависимость: с задержкой 0 "
        "столкновения с β ≈ 1,6–1,8, с задержкой ρ — с β = 1,2.\n"
        "3. Предлагаемый объем: 500 общих сцен на уровень. С поправкой Холма это позволяет "
        "назвать «хуже» разность от ≈ 2 п.п.; «эквивалентно ±2 п.п.» — только при малой "
        "доле несовпадающих пар.\n"
        "4. Оценка «5 политик × 500 сцен» — 1,5–3 суток счета, обучение 10 политик по "
        "10⁵ шагов — 7–12 часов (24–39 шагов/с на 8 средах). Оба числа уточним после первой обученной политики.\n"
        "5. Неизвестное до обучения: разброс между политиками PPO (seed обучения). От него "
        "зависит, сколько политик нужно для двухуровневого бутстрепа."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("07_timing.ipynb")
ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
