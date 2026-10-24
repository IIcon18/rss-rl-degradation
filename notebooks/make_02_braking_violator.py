"""Собирает и выполняет notebooks/02_braking_violator.ipynb.

Запуск из корня проекта: .venv/bin/python notebooks/make_02_braking_violator.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Этап 2. Фоновые машины и тормозящий нарушитель\n\n"
        "Код: `rssrl/violators.py`, проверки: `tests/test_violators.py`.\n\n"
        "1. Фоновые машины highway-env тормозят до 6 м/с², хотя RSS допускает только "
        "5 м/с². Показываем это и исправляем.\n"
        "2. Нарушитель в момент нарушения сам ставит себе ускорение −β·5 м/с² "
        "в обход IDM. Проверяем, что фактическое торможение совпадает с заданным."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.scenarios import make_env, find_leader\n"
        "from rssrl.violators import BrakingViolator\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,\n"
        "                     'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 2})\n"
        "DT = 1 / 15\n"
        "BETAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]"
    ),
    md(
        "## 1. Торможение фоновых машин: стандартный IDM и наш BoundedIDMVehicle\n\n"
        "10 сцен по 40 с, у каждой машины на каждом шаге считаем фактическое "
        "ускорение по изменению скорости."
    ),
    code(
        "def background_accels(vehicle_type, seeds=range(10), seconds=40):\n"
        "    out = []\n"
        "    for s in seeds:\n"
        "        env = make_env(s, other_vehicles_type=vehicle_type)\n"
        "        road, ego = env.unwrapped.road, env.unwrapped.vehicle\n"
        "        for _ in range(seconds * 15):\n"
        "            before = {id(v): v.speed for v in road.vehicles}\n"
        "            road.act(); road.step(DT)\n"
        "            out += [(v.speed - before[id(v)]) / DT for v in road.vehicles\n"
        "                    if v is not ego and not v.crashed]\n"
        "        env.close()\n"
        "    return np.array(out)\n\n"
        "stock = background_accels('highway_env.vehicle.behavior.IDMVehicle')\n"
        "bounded = background_accels('rssrl.violators.BoundedIDMVehicle')\n"
        "summary = pd.DataFrame({\n"
        "    'машина': ['IDMVehicle (как в highway-env)', 'BoundedIDMVehicle (наша)'],\n"
        "    'шагов всего': [stock.size, bounded.size],\n"
        "    'шагов с торможением > 5 м/с²': [(stock < -5 - 1e-9).sum(), (bounded < -5 - 1e-9).sum()],\n"
        "    'самое сильное торможение, м/с²': [-stock.min(), -bounded.min()],\n"
        "})\n"
        "summary"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)\n"
        "bins = np.arange(-6.375, -1.874, 0.25)  # столбцы центрированы на -6, -5.75, ..., -2\n"
        "for ax, data, name in zip(axes, [stock, bounded], ['IDMVehicle (как в highway-env)', 'BoundedIDMVehicle (наша)']):\n"
        "    hard = data[data < -2]  # интересны только сильные торможения\n"
        "    ax.hist(hard, bins=bins, color='#4a7ab5', edgecolor='white', linewidth=1)\n"
        "    ax.axvline(-5, color='#c0392b', linewidth=1.5, linestyle='--')\n"
        "    ax.text(-4.9, 1.5e3, 'a_max,brake = 5', color='#c0392b', fontsize=9)\n"
        "    ax.set_title(name, fontsize=10)\n"
        "    ax.set_xlabel('ускорение, м/с² (показаны торможения сильнее 2 м/с²)')\n"
        "    ax.set_yscale('log')\n"
        "axes[0].set_ylabel('число шагов (лог. шкала)')\n"
        "fig.suptitle('Фоновые машины: торможение до и после ограничения', y=1.02)\n"
        "fig.savefig(FIG / '02_background_braking.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
    md(
        "Стандартные IDM-машины иногда тормозят сильнее 5 м/с², максимум до 6 м/с². "
        "Это нарушает допущение RSS ещё до появления нарушителя, и смешивалось бы "
        "с эффектом β. После ограничения таких шагов нет: те шаги, где IDM хотел "
        "тормозить сильнее, теперь срезаны ровно до 5 м/с² (высокий столбец у −5).\n\n"
        "Не стали ли фоновые машины из-за этого сталкиваться между собой? Проверяем "
        "на 30 сценах по 40 с (агент «бесплотный», чтобы считать только фоновые машины):"
    ),
    code(
        "def background_crashes(vehicle_type, seeds=range(30), seconds=40):\n"
        "    crashed = 0\n"
        "    for s in seeds:\n"
        "        env = make_env(s, other_vehicles_type=vehicle_type)\n"
        "        road, ego = env.unwrapped.road, env.unwrapped.vehicle\n"
        "        ego.collidable = False\n"
        "        for _ in range(seconds * 15):\n"
        "            road.act(); road.step(DT)\n"
        "        crashed += sum(v.crashed for v in road.vehicles if v is not ego)\n"
        "        env.close()\n"
        "    return crashed\n\n"
        "for name, typ in [('IDMVehicle', 'highway_env.vehicle.behavior.IDMVehicle'),\n"
        "                  ('BoundedIDMVehicle', 'rssrl.violators.BoundedIDMVehicle')]:\n"
        "    print(f'{name:18s} столкнувшихся фоновых машин за 30 сцен: {background_crashes(typ)}')"
    ),
    md(
        "## 2. Нарушитель: задано / фактически\n\n"
        "Лидера агента заменяем нарушителем, нарушение начинается через 1 с. Чтобы "
        "столкновение с агентом не оборвало торможение, агента на время этой "
        "проверки делаем «бесплотным» (`collidable = False`). Здесь проверяется "
        "только сам нарушитель. Для каждого β берем 10 сцен (seed 0–9)."
    ),
    code(
        "rows, traces = [], {}\n"
        "for beta in BETAS:\n"
        "    for seed in range(10):\n"
        "        env = make_env(seed)\n"
        "        env.unwrapped.vehicle.collidable = False\n"
        "        vio = BrakingViolator.replace(find_leader(env), beta=beta, trigger_time=1.0)\n"
        "        for _ in range(8):\n"
        "            env.step(1)\n"
        "        env.close()\n"
        "        e = vio.events[0]\n"
        "        a = np.array(e['a_actual'])\n"
        "        if seed == 0:\n"
        "            traces[beta] = e\n"
        "        rows.append({'β': beta, 'seed': seed, 'задано, м/с²': e['a_commanded'],\n"
        "                     'фактически (среднее без последнего шага)': a[:-1].mean(),\n"
        "                     'макс. отклонение': np.abs(a[:-1] - e['a_commanded']).max(),\n"
        "                     'последний шаг': a[-1], 'v в начале, м/с': e['v_start'],\n"
        "                     'время торможения, с': e['t_stop'] - e['t_start'],\n"
        "                     'v/(β·5), с': e['v_start'] / (5 * beta)})\n"
        "events = pd.DataFrame(rows)\n"
        "events.groupby('β')[['задано, м/с²', 'фактически (среднее без последнего шага)',\n"
        "                     'макс. отклонение', 'v в начале, м/с']].agg(['mean']).round(6)"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "colors = plt.cm.Oranges(np.linspace(0.35, 0.95, len(BETAS)))\n\n"
        "ax = axes[0]\n"
        "for beta, c in zip(BETAS, colors):\n"
        "    e = traces[beta]\n"
        "    # фаза торможения: от начала нарушения до остановки, дальше 0\n"
        "    t = np.r_[e['t'], e['t'][-1] + DT, 6.0]\n"
        "    a = np.r_[e['a_actual'], 0.0, 0.0]\n"
        "    ax.step(t, a, where='post', color=c)\n"
        "    ax.annotate(f'β = {beta:g}', (1.05, e['a_commanded']),\n"
        "                xytext=(0, 3), textcoords='offset points', fontsize=8)\n"
        "ax.set_xlabel('время от начала сцены, с'); ax.set_ylabel('фактическое ускорение, м/с²')\n"
        "ax.set_title('Нарушитель, seed 0: фактическое ускорение по шагам\\n'\n"
        "             '(нарушение с 1 с, ступенька вверх — остановка)', fontsize=10)\n"
        "ax.set_xlim(0.9, 6.0)\n\n"
        "ax = axes[1]\n"
        "lim = [-10.8, -4.2]\n"
        "ax.plot(lim, lim, color='gray', linewidth=1, linestyle='--')\n"
        "ax.text(-5.0, -5.6, 'фактически = задано', color='gray', fontsize=8, rotation=45)\n"
        "for beta, c in zip(BETAS, colors):\n"
        "    sub = events[events['β'] == beta]\n"
        "    ax.scatter(sub['задано, м/с²'], sub['фактически (среднее без последнего шага)'],\n"
        "               color=c, s=60, edgecolor='white', linewidth=2, zorder=3)\n"
        "ax.set_xlim(lim); ax.set_ylim(lim)\n"
        "ax.set_xlabel('задано −β·5, м/с²'); ax.set_ylabel('фактически, м/с²')\n"
        "ax.set_title('Задано / фактически: 6 уровней β × 10 сцен', fontsize=10)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '02_violator_commanded_vs_actual.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
    md(
        "Фактическое ускорение совпадает с заданным на всех шагах торможения, кроме "
        "последнего. На последнем шаге машина дотормаживает до нуля, чтобы не уехать "
        "назад, поэтому там ускорение по модулю меньше. Время торможения совпадает с "
        "v/(β·5) с точностью до шага симуляции:"
    ),
    code(
        "events[['β', 'seed', 'v в начале, м/с', 'время торможения, с', 'v/(β·5), с', 'последний шаг']]"
        ".head(12).round(3)"
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("02_braking_violator.ipynb")
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
