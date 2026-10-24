"""Собирает и выполняет notebooks/01_rss_formulas.ipynb.

Запуск из корня проекта: .venv/bin/python notebooks/make_01_rss_formulas.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Этап 1. Формулы минимальной безопасной дистанции RSS\n\n"
        "Код формул: `rssrl/rss.py`, проверки: `tests/test_rss.py`.\n"
        "Здесь смотрим на формулы глазами: как d_min зависит от скоростей и что "
        "происходит в худшем случае, если передняя машина тормозит сильнее допущения."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.rss import DEFAULT, d_min_longitudinal, d_min_lateral\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,\n"
        "                     'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 2})\n"
        "print(DEFAULT)"
    ),
    md(
        "## 1. Продольная дистанция (лемма 2)\n\n"
        "$$d_{min} = \\Big[v_r\\rho + \\tfrac12 a_{max,accel}\\rho^2 + "
        "\\frac{(v_r+\\rho\\, a_{max,accel})^2}{2a_{min,brake}} - "
        "\\frac{v_f^2}{2a_{max,brake}}\\Big]_+$$\n\n"
        "$v_r$ — скорость задней машины (агента), $v_f$ — передней (лидера)."
    ),
    code(
        "v_r = np.linspace(0, 40, 401)\n"
        "fronts = [15, 20, 25, 30]\n"
        "colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(fronts)))\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "for v_f, c in zip(fronts, colors):\n"
        "    d = [d_min_longitudinal(x, v_f) for x in v_r]\n"
        "    ax.plot(v_r, d, color=c)\n"
        "    ax.annotate(f'$v_f$ = {v_f} м/с', (40, d[-1]), xytext=(4, 0),\n"
        "                textcoords='offset points', va='center', fontsize=9)\n"
        "ax.set_xlabel('скорость задней машины $v_r$, м/с')\n"
        "ax.set_ylabel('$d_{min}$, м')\n"
        "ax.set_title('Минимальная безопасная продольная дистанция')\n"
        "ax.set_xlim(0, 40); ax.set_ylim(bottom=0)\n"
        "fig.savefig(FIG / '01_dmin_longitudinal.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "for v in [20, 25, 30]:\n"
        "    d = d_min_longitudinal(v, v)\n"
        "    print(f'обе по {v} м/с ({v*3.6:.0f} км/ч): d_min = {d:.1f} м, это {d/v:.2f} с хода')"
    ),
    md(
        "При равных скоростях 25 м/с (90 км/ч) RSS требует около 38 м, примерно "
        "1,5 с хода. Это совпадает с привычным правилом «держи 1,5–2 секунды»."
    ),
    md(
        "## 2. Худший случай: что ломается при β > 1\n\n"
        "Старт ровно с $d_{min}$, обе машины 25 м/с. Задняя ведет себя как в лемме "
        "(ρ секунд разгоняется, потом тормозит с $a_{min,brake}$ = 4 м/с²), а передняя "
        "тормозит с $\\beta \\cdot 5$ м/с². При β = 1 машины останавливаются впритык. "
        "При β > 1 зазор уходит ниже нуля, то есть происходит столкновение.\n\n"
        "Это *худший* случай: задняя ведет себя как можно хуже. Настоящий фильтр "
        "начнет тормозить раньше и, возможно, сильнее, поэтому реальная частота "
        "столкновений — вопрос эксперимента, а не этой формулы."
    ),
    code(
        "def worst_case_gap(gap0, v_rear, v_front, front_brake, p=DEFAULT, dt=1e-3):\n"
        "    \"\"\"Зазор во времени в худшем случае леммы 2 (машины не едут назад).\"\"\"\n"
        "    t, x_r, x_f, vr, vf = 0.0, 0.0, gap0, v_rear, v_front\n"
        "    ts, gaps = [0.0], [gap0]\n"
        "    while vr > 0 or vf > 0 or t < p.rho:\n"
        "        a_r = p.a_max_accel if t < p.rho else -p.a_min_brake\n"
        "        nvr = max(0.0, vr + a_r * dt)\n"
        "        nvf = max(0.0, vf - front_brake * dt)\n"
        "        x_r += (vr + nvr) / 2 * dt\n"
        "        x_f += (vf + nvf) / 2 * dt\n"
        "        vr, vf, t = nvr, nvf, t + dt\n"
        "        ts.append(t); gaps.append(x_f - x_r)\n"
        "    return np.array(ts), np.array(gaps)\n\n"
        "betas = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]\n"
        "v = 25.0\n"
        "d0 = d_min_longitudinal(v, v)\n"
        "colors = plt.cm.Oranges(np.linspace(0.35, 0.95, len(betas)))\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "for b, c in zip(betas, colors):\n"
        "    t, g = worst_case_gap(d0, v, v, b * DEFAULT.a_max_brake)\n"
        "    ax.plot(t, g, color=c)\n"
        "    ax.annotate(f'β = {b:g}', (t[-1], g[-1]), xytext=(4, 0),\n"
        "                textcoords='offset points', va='center', fontsize=9)\n"
        "ax.axhline(0, color='black', linewidth=1)\n"
        "ax.text(0.2, 1.5, 'зазор 0: касание', fontsize=9)\n"
        "ax.set_xlabel('время, с'); ax.set_ylabel('зазор, м')\n"
        "ax.set_title(f'Худший случай со старта $d_{{min}}$ = {d0:.1f} м, обе по {v:.0f} м/с')\n"
        "fig.savefig(FIG / '01_worst_case_beta.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
    md(
        "Проверка формулой: машины в худшем случае не расходятся до остановки, поэтому "
        "итоговый зазор равен $d_{min}$ плюс путь передней минус путь задней. "
        "Нехватка при β > 1 равна разнице тормозных путей передней машины: "
        "$\\frac{v_f^2}{2\\cdot 5} - \\frac{v_f^2}{2\\cdot 5\\beta}$."
    ),
    code(
        "print(' β   нехватка (симуляция), м   нехватка (формула), м')\n"
        "for b in betas:\n"
        "    _, g = worst_case_gap(d0, v, v, b * DEFAULT.a_max_brake)\n"
        "    formula = v**2 / (2 * 5) - v**2 / (2 * 5 * b)\n"
        "    print(f'{b:.1f}   {-g[-1]:10.2f}               {formula:10.2f}')"
    ),
    md(
        "## 3. Поперечная дистанция (лемма 4)\n\n"
        "Поперечные параметры: $a^{lat}_{max,accel}$ = 0,2 м/с², "
        "$a^{lat}_{min,brake}$ = 0,8 м/с², μ = 0,1 м (значения по умолчанию "
        "Intel ad-rss-lib, **сверить со статьей**). Скорость положительна, если "
        "машина 1 сдвигается к машине 2."
    ),
    code(
        "v1 = np.linspace(-1, 2, 301)\n"
        "others = [0.0, -0.5, -1.0]\n"
        "colors = plt.cm.Greens(np.linspace(0.5, 0.95, len(others)))\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "for v2, c in zip(others, colors):\n"
        "    d = [d_min_lateral(x, v2) for x in v1]\n"
        "    ax.plot(v1, d, color=c)\n"
        "    label = 'не сдвигается' if v2 == 0 else f'сдвигается навстречу {-v2:g} м/с'\n"
        "    ax.annotate(f'машина 2 {label}', (2, d[-1]), xytext=(4, 0),\n"
        "                textcoords='offset points', va='center', fontsize=9)\n"
        "ax.axhline(DEFAULT.mu, color='gray', linewidth=1, linestyle='--')\n"
        "ax.text(-0.95, DEFAULT.mu + 0.1, 'запас μ', fontsize=9, color='gray')\n"
        "ax.set_xlabel('поперечная скорость машины 1 к машине 2, м/с')\n"
        "ax.set_ylabel('$d^{lat}_{min}$, м')\n"
        "ax.set_title('Минимальная безопасная поперечная дистанция')\n"
        "ax.set_xlim(-1, 2); ax.set_ylim(bottom=0)\n"
        "fig.savefig(FIG / '01_dmin_lateral.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "print('полоса 4 м, машина 2 м: зазор между машинами в соседних полосах ≈ 2 м')\n"
        "for v in [0.5, 1.0, 1.5]:\n"
        "    print(f'машина 1 сдвигается со скоростью {v} м/с: d_lat_min = {d_min_lateral(v, 0):.2f} м')"
    ),
    md(
        "При перестроении поперечная скорость составляет порядка 1–1,5 м/с, и "
        "$d^{lat}_{min}$ быстро превышает 2 м — зазор между машинами в соседних "
        "полосах. Значит, перестроение рядом с машиной в соседней полосе поперечная "
        "RSS будет запрещать, если ее нельзя обогнать или пропустить. Это проверим "
        "на этапе 4."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("01_rss_formulas.ipynb")
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
