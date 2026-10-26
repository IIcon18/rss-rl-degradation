"""Собирает и выполняет notebooks/05_nominal_check.ipynb.

Запуск из корня проекта: .venv/bin/python notebooks/make_05_nominal_check.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Этап 5. Проверка номинального режима\n\n"
        "Код: `make_scene(kind='nominal')` в `rssrl/scenarios.py`, `unsafe_entries` в "
        "`rssrl/pilot.py`. Проверки: `tests/test_nominal.py`.\n\n"
        "**Номинальный режим** — нарушителя нет. Все фоновые машины тормозят не сильнее "
        "5 м/с² (a_max,brake), то есть продольное допущение RSS соблюдается. "
        "Если фильтр сделан правильно, с ним агент не должен попадать в столкновения.\n\n"
        "Это проверка до всех экспериментов: если фильтр не справляется даже здесь, "
        "столкновения при β > 1 нельзя было бы приписать нарушителю.\n\n"
        "**Агенты** (PPO еще не обучен):\n"
        "- IDLE — держит полосу и целевую скорость, никого не объезжает;\n"
        "- случайная политика — каждую секунду случайное действие из пяти, "
        "в том числе перестроения (seed политики = seed сцены + 1000).\n\n"
        "Сцены 0–99, для каждой три варианта: фильтр с задержкой ρ = 0,5 с (основной), "
        "фильтр с задержкой 0 (дополнительный), без фильтра. У всех машин ускорение "
        "меняется не быстрее 10 м/с³; фильтр в экстренном режиме тормозит до 8 м/с²; "
        "d_min фильтра — лемма 2 с учетом рывка."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.pilot import run_grid, plot_episode, unsafe_entries\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_05'  # таблицы пилота, в git не идут\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "SEEDS = range(100)\n"
        "# основной режим — фильтр с задержкой ρ, дополнительный — мгновенный фильтр\n"
        "VARIANTS = [(True, 0.5), (True, 0.0), (False, 0.5)]\n"
        "NAMES = {(True, 0.5): 'фильтр, задержка ρ (основной)', (True, 0.0): 'фильтр, задержка 0 (доп.)',\n"
        "         (False, 0.5): 'без фильтра'}"
    ),
    code(
        "frames, LOGS = [], {}\n"
        "for policy in ['idle', 'random']:\n"
        "    o, logs = run_grid(SEEDS, [1.0], VARIANTS, kind='nominal', policy=policy)\n"
        "    frames.append(o)\n"
        "    LOGS[policy] = logs\n"
        "outcomes = pd.concat(frames, ignore_index=True)\n"
        "outcomes['вариант'] = [NAMES[(s, d)] for s, d in zip(outcomes['shield'], outcomes['reaction_delay'])]\n"
        "\n"
        "def per_episode(policy, key):\n"
        "    log = LOGS[policy][key]\n"
        "    return {'доля времени фильтр тормозит': log['shield_active'].mean(),\n"
        "            'мин. скорость, м/с': log['ego_v'].min(),\n"
        "            'доля времени экстренно': (log['shield_mode'] == 'emergency').mean(),\n"
        "            'опасных событий от фона': len(unsafe_entries(log))}\n"
        "\n"
        "extra = pd.DataFrame([per_episode(p, (s, 1.0, sh, d))\n"
        "                      for p, s, sh, d in zip(outcomes.policy, outcomes.seed, outcomes.shield, outcomes.reaction_delay)])\n"
        "outcomes = pd.concat([outcomes, extra], axis=1)\n"
        "outcomes.drop(columns=['crash_partners']).to_csv(PILOT / 'outcomes.csv', index=False)\n"
        "\n"
        "summary = outcomes.groupby(['policy', 'вариант'], sort=False).agg(\n"
        "    эпизодов=('collision', 'size'),\n"
        "    столкновений=('collision', 'sum'),\n"
        "    пробег_км=('distance_km', 'mean'),\n"
        "    фильтр_тормозит=('доля времени фильтр тормозит', 'mean'),\n"
        "    из_них_экстренно=('доля времени экстренно', 'mean'),\n"
        "    скорость_удара_медиана=('v_impact', 'median'),\n"
        "    мин_скорость=('мин. скорость, м/с', 'min'),\n"
        "    перестроений_запрещено=('lc_blocked', 'sum'),\n"
        "    перестроений_отменено=('lc_cancelled', 'sum'),\n"
        ").round(3)\n"
        "summary"
    ),
    md(
        "Как читать таблицу:\n"
        "- **Без фильтра** агент врезается почти всегда (93–96 из 100): IDLE едет с "
        "целевой скоростью 25 м/с и не тормозит перед более медленной машиной. Без фильтра "
        "колонки «запрещено/отменено» показывают, что фильтр *запретил бы*.\n"
        "- **С фильтром** — 0 столкновений в каждом из четырех вариантов (2 политики × "
        "2 задержки) по 100 эпизодов, с рывком 10 м/с³ у всех машин.\n"
        "- **Пробег** с фильтром около 0,85–0,87 км за 40 с (≈ 21–22 м/с), минимальная "
        "скорость не ниже ≈ 8 м/с.\n"
        "- **Фильтр тормозит** 36–42 % времени у IDLE и около 21 % у случайной политики — "
        "больше, чем с d_min леммы 2 как в статье (27–39 % и 15–19 %): новая формула "
        "требует дистанцию больше. Зато **экстренные торможения почти исчезли**: около "
        "1–2 % всего времени (колонка «из_них_экстренно» — доля всего времени эпизода), "
        "а со старой формулой при задержке ρ было 21 %. Значит, новая d_min заранее "
        "оставляет место на набор торможения, и обычного торможения 4 м/с² хватает."
    ),
    md(
        "## Насколько мала частота столкновений\n\n"
        "0 из 100 не значит «ровно ноль». Точная (Клоппера — Пирсона) односторонняя "
        "95 % верхняя граница частоты (при 0 столкновений из n эпизодов она равна 1 − 0,05^(1/n)). "
        "Считаем отдельно для каждого варианта (объединять разные варианты нельзя)."
    ),
    code(
        "from math import comb\n"
        "def upper95(k, n):\n"
        "    # точная (Клоппер — Пирсон) односторонняя граница: p, при котором P(X <= k) = 0,05\n"
        "    cdf = lambda p: sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))\n"
        "    lo, hi = 0.0, 1.0\n"
        "    for _ in range(60):\n"
        "        mid = (lo + hi) / 2\n"
        "        lo, hi = (mid, hi) if cdf(mid) > 0.05 else (lo, mid)\n"
        "    return hi\n"
        "\n"
        "bound = summary[['эпизодов', 'столкновений']].copy()\n"
        "bound['верхняя граница 95 %'] = [round(upper95(k, n), 3) for k, n in zip(bound['столкновений'], bound['эпизодов'])]\n"
        "bound"
    ),
    md(
        "С фильтром при 100 эпизодах можно утверждать только, что частота столкновений "
        "в номинальном режиме меньше примерно 3 % на эпизод. Чтобы сузить эту границу "
        "до 1 %, понадобилось бы около 300 эпизодов на вариант."
    ),
    md("## Как часто фильтр вмешивается"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)\n"
        "colors = {0.0: '#2c6fbb', 0.5: '#e67e22'}\n"
        "bins = np.linspace(0, 0.5, 26)\n"
        "for ax, policy, title in zip(axes, ['idle', 'random'], ['IDLE', 'случайная политика']):\n"
        "    for d in (0.0, 0.5):\n"
        "        x = outcomes[(outcomes.policy == policy) & outcomes.shield & (outcomes.reaction_delay == d)]\n"
        "        ax.hist(x['доля времени фильтр тормозит'], bins=bins, histtype='step', linewidth=2,\n"
        "                color=colors[d], label=NAMES[(True, d)])\n"
        "    ax.set_title(title, fontsize=10)\n"
        "    ax.set_xlabel('доля времени эпизода, когда фильтр тормозит')\n"
        "    ax.grid(alpha=0.3)\n"
        "axes[0].set_ylabel('эпизодов')\n"
        "axes[0].legend(fontsize=8, frameon=False)\n"
        "fig.suptitle('Номинальный режим: вмешательства фильтра (100 сцен, столкновений 0)', fontsize=10)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '05_shield_share.png', dpi=150)"
    ),
    md(
        "У IDLE фильтр тормозит больше, чем у случайной политики: случайная политика "
        "часто сама нажимает SLOWER или уходит в свободную полосу, а IDLE всегда давит на газ. "
        "С задержкой 0 фильтр включается и выключается чаще (≈ 63 раза за эпизод против "
        "≈ 33 при задержке ρ): мгновенный фильтр реагирует на каждое касание границы d_min."
    ),
    code(
        "def switches(log):\n"
        "    return int((log['shield_active'].astype(int).diff().abs() == 1).sum())\n"
        "\n"
        "pd.DataFrame({\n"
        "    NAMES[(True, d)]: [np.mean([switches(LOGS['idle'][(s, 1.0, True, d)]) for s in SEEDS])]\n"
        "    for d in (0.0, 0.5)\n"
        "}, index=['IDLE: включений/выключений фильтра за эпизод (среднее)']).round(1)"
    ),
    md(
        "## Фоновые машины нарушают поперечное допущение RSS\n\n"
        "Фоновые машины перестраиваются по модели MOBIL, а не по поперечной RSS, и "
        "сдвигаются вбок быстро (до ~5 м/с). Иногда машина впереди агента начинает "
        "сдвигаться к нему, когда продольно она уже ближе d_min: в его полосу или в "
        "соседнюю. По RSS это опасная ситуация, и создает ее фоновая машина. "
        "Ищем такие моменты: впереди появилась новая опасная машина (продольно и "
        "поперечно ближе d_min), "
"а сам агент последние 3 с полосу не менял (`unsafe_entries`). Первые 3 с эпизода "
        "не считаем: highway-env расставляет машины на старте случайно, без учета d_min "
        "(сколько эпизодов начинаются с зазора меньше d_min — ниже). У IDLE агент "
        "полосу не меняет никогда, поэтому для него подсчет точный."
    ),
    code(
        "ue = outcomes[outcomes.shield].groupby(['policy', 'вариант'], sort=False).agg(\n"
        "    эпизодов_с_событием=('опасных событий от фона', lambda x: int((x > 0).sum())),\n"
        "    событий_всего=('опасных событий от фона', 'sum'),\n"
        ")\n"
        "ue"
    ),
    code(
        "start_unsafe = np.mean([\n"
        "    (LOGS['idle'][(s, 1.0, True, 0.5)]['gap'].iloc[0] < LOGS['idle'][(s, 1.0, True, 0.5)]['d_min'].iloc[0])\n"
        "    for s in SEEDS])\n"
        "print(f'Эпизодов, где на первом шаге зазор до машины впереди меньше d_min: {start_unsafe:.0%}')"
    ),
    code(
        "rows = []\n"
        "for s in SEEDS:\n"
        "    ev = unsafe_entries(LOGS['idle'][(s, 1.0, True, 0.5)])\n"
        "    rows += [(s, r.t, r.ratio) for r in ev.itertuples()]\n"
        "ev_idle = pd.DataFrame(rows, columns=['seed', 't', 'зазор / d_min'])\n"
        "# пример: самое близкое событие с положительным зазором\n"
        "seed_ex, t_ex = ev_idle[ev_idle['зазор / d_min'] > 0].sort_values('зазор / d_min').iloc[0][['seed', 't']]\n"
        "seed_ex = int(seed_ex)\n"
        "log = LOGS['idle'][(seed_ex, 1.0, True, 0.5)]\n"
        "win = log[(log.t > t_ex - 4) & (log.t < t_ex + 8)]\n"
        "axes = plot_episode(win, f'Сцена {seed_ex}, IDLE, фильтр, задержка ρ: '\n"
        "                    f'фоновая машина резко сдвигается к агенту')\n"
        "for ax in axes:\n"
        "    ax.axvline(t_ex, color='#8e44ad', linewidth=1, linestyle='--')\n"
        "axes[0].set_ylim(-2, None)\n"
        "plt.gcf().tight_layout()\n"
        "plt.gcf().savefig(FIG / '05_background_cutin.png', dpi=150)\n"
        "ev_idle.round(2)"
    ),
    md(
        "На графике (фиолетовый штрих — момент события) зазор скачком падает почти "
        "до нуля: фоновая машина резко сдвигается к агенту, когда продольно она уже "
        "ближе d_min. Фильтр тормозит. Столкновения не было.\n\n"
        "По строгой RSS в такой ситуации (продольно машины уже были ближе d_min, когда "
        "возникла поперечная опасность) правильный ответ — поперечный, продольное "
        "торможение не обязательно. Наш фильтр тормозит всегда — это осторожнее, чем "
        "требует RSS.\n\n"
        "«Номинальный режим» номинален по продольному допущению (торможение ≤ 5 м/с²), "
        "но фон иногда нарушает поперечное. Варианты: (а) оставить как есть и описать "
        "как свойство фона, одинаковое для всех β и γ; (б) запретить фону перестраиваться "
        "ближе d_min к агенту. Вопрос для письма преподавателю."
    ),
    md(
        "## Выводы этапа 5\n\n"
        "1. Критерий готовности выполнен: в номинальном режиме с фильтром 0 столкновений "
        "из 100 для каждой из двух политик и каждой из двух задержек, с рывком 10 м/с³ "
        "и d_min с учетом рывка. Верхняя 95 % граница частоты — около 3 % на вариант. "
        "Без фильтра — 93–96 из 100.\n"
        "2. По дороге найдена и исправлена ошибка реализации: фильтр мог «отменить» уже "
        "фактически законченное перестроение и увести агента в занятую исходную полосу "
        "(один боковой удар у случайной политики). Теперь перестроение завершено, как "
        "только агент пересек разметку, а отмена — только если возвращаться безопасно.\n"
        "3. Фильтр не «покупает» безопасность остановкой: агент едет ≈ 21–22 м/с, все "
        "эпизоды доходят до конца времени. Экстренных торможений около 1–2 % времени.\n"
        "4. Фоновые машины иногда резко сдвигаются к агенту, когда продольно уже ближе "
        "d_min (MOBIL не знает RSS). Это одинаково для всех уровней β и γ.\n"
        "5. Наш фильтр на любую опасность впереди отвечает продольным торможением; "
        "строгая RSS в части случаев требует только поперечного ответа.\n"
        "6. highway-env расставляет машины на старте без учета d_min: с новой (большей) "
        "d_min таких эпизодов 52 %. Нарушения начинаются не раньше 3 с."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("05_nominal_check.ipynb")
ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
