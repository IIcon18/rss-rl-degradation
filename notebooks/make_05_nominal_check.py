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
        "Код: `make_scene(kind='nominal')` в `rssrl/scenarios.py`, фон — `rssrl/traffic.py`, "
        "`unsafe_entries` в `rssrl/pilot.py`. Проверки: `tests/test_nominal.py`, "
        "`tests/test_traffic.py`.\n\n"
        "**Номинальный режим** — нарушителя нет. Повтор после письма от 29.10 на двух фонах:\n"
        "- **контролируемое следование** (`traffic='following'`, основной β-сценарий): "
        "фон не перестраивается, держит d_min по RSS (торможение ≥ 4 через ρ, но ≤ 5 м/с²), "
        "начальные зазоры не меньше d_min по общему правилу от seed;\n"
        "- **прежний фон highway-env с MOBIL** (`traffic='robustness'`) — отдельное "
        "испытание устойчивости. Допущения RSS о других машинах тут не гарантированы "
        "(перестроения MOBIL, случайная стартовая расстановка), поэтому это не «режим "
        "соблюдения допущений».\n\n"
        "**Агенты** (PPO еще не обучен): IDLE и случайная политика (каждую секунду случайное "
        "действие из пяти, seed политики = seed сцены + 1000). Сцены 0–49.\n\n"
        "**Варианты фильтра**: базовая надлежащая реакция с учетом рывка (задержка ρ — "
        "основной, задержка 0 — дополнительный), тот же фильтр с дополнительным аварийным "
        "торможением до 8 м/с² (сравнение), без фильтра."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from rssrl.pilot import run_variants, export_pilot, plot_episode, unsafe_entries\n"
        "from rssrl.scenarios import make_scene, IDLE\n"
        "from rssrl.stats import zero_upper_bound\n"
        "from rssrl.traffic import initial_gap_ratios\n\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "PILOT = Path('..') / 'results' / 'pilot_05'  # первичные данные: CSV в git, PNG — нет\n"
        "PILOT.mkdir(parents=True, exist_ok=True)\n"
        "plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\n"
        "SEEDS = range(50)\n"
        "VARIANTS = [\n"
        "    ('базовая, задержка ρ (основной)', True, 0.5, False),\n"
        "    ('базовая, задержка 0 (доп.)', True, 0.0, False),\n"
        "    ('с аварийным торможением, ρ (сравнение)', True, 0.5, True),\n"
        "    ('без фильтра', False, 0.5, False),\n"
        "]"
    ),
    md(
        "## Начальные состояния\n\n"
        "Доля эпизодов, где на первом записанном шаге зазор агента до машины впереди "
        "меньше d_min, и где хотя бы одна пара соседних машин в какой-либо полосе стоит "
        "ближе d_min (d_min с учетом рывка). 100 сцен, только расстановка и один шаг."
    ),
    code(
        "rows = []\n"
        "for traffic in ['following', 'robustness']:\n"
        "    for s in range(100):\n"
        "        sc = make_scene(s, kind='nominal', traffic=traffic)\n"
        "        pairs_bad = initial_gap_ratios(sc.env.unwrapped.road).min() < 1\n"
        "        sc.env.step(IDLE)\n"
        "        r = sc.logger.rows[0]\n"
        "        rows.append({'фон': traffic, 'агент ближе d_min на 1-м шаге': r['gap'] < r['d_min'],\n"
        "                     'есть пара фона ближе d_min': pairs_bad})\n"
        "        sc.env.close()\n"
        "start = pd.DataFrame(rows).groupby('фон', sort=False).mean()\n"
        "start.to_csv(PILOT / 'start_states.csv')\n"
        "(start * 100).round(0).astype(int).astype(str) + ' %'"
    ),
    md(
        "## Контролируемое следование (основной сценарий)"
    ),
    code(
        "frames, LOGS = [], {}\n"
        "for traffic in ['following', 'robustness']:\n"
        "    for policy in ['idle', 'random']:\n"
        "        o, logs = run_variants(SEEDS, [1.0], VARIANTS, kind='nominal', policy=policy, traffic=traffic)\n"
        "        frames.append(o)\n"
        "        LOGS[(traffic, policy)] = logs\n"
        "outcomes = pd.concat(frames, ignore_index=True)\n"
        "\n"
        "def per_episode(log):\n"
        "    return {'доля времени фильтр тормозит': log['shield_active'].mean(),\n"
        "            'доля времени аварийно': (log['shield_mode'] == 'emergency').mean(),\n"
        "            'мин. скорость, м/с': log['ego_v'].min(),\n"
        "            'опасных событий от фона': len(unsafe_entries(log))}\n"
        "\n"
        "extra = pd.DataFrame([per_episode(log) for k in LOGS for log in LOGS[k].values()])\n"
        "outcomes = pd.concat([outcomes, extra], axis=1)\n"
        "all_logs = {(*key, t, p): log for (t, p), logs in LOGS.items() for key, log in logs.items()}\n"
        "n_logs = export_pilot(outcomes, all_logs, PILOT,\n"
        "                      params={'notebook': '05_nominal_check', 'kind': 'nominal'})\n"
        "print('сохранено покадровых журналов столкновений:', n_logs)\n"
        "\n"
        "def summary(df):\n"
        "    return df.groupby(['policy', 'вариант'], sort=False).agg(\n"
        "        эпизодов=('collision', 'size'),\n"
        "        столкновений=('collision', 'sum'),\n"
        "        пробег_км=('distance_km', 'mean'),\n"
        "        фильтр_тормозит=('доля времени фильтр тормозит', 'mean'),\n"
        "        аварийно=('доля времени аварийно', 'mean'),\n"
        "        мин_скорость=('мин. скорость, м/с', 'min'),\n"
        "        запретил_бы=('lc_would_block', 'sum'),\n"
        "        запретил=('lc_blocked', 'sum'),\n"
        "        отменил=('lc_cancelled', 'sum'),\n"
        "    ).round(3)\n"
        "\n"
        "summary(outcomes[outcomes.traffic == 'following'])"
    ),
    md(
        "Колонки перестроений: **«запретил бы»** — решение фильтра, которое считается и "
        "без фильтра (для сравнения); **«запретил»** и **«отменил»** — реально "
        "выполненные вмешательства, без фильтра они всегда 0."
    ),
    md(
        "## Насколько мала частота столкновений\n\n"
        "0 из n не значит «ровно ноль». Для каждого варианта отдельно — точная "
        "(Клоппера — Пирсона) односторонняя 95 % верхняя граница; при 0 столкновений "
        "из n она равна 1 − 0,05^(1/n). Граница относится к одному варианту при "
        "независимых сценах, а не ко всем вариантам вместе."
    ),
    code(
        "b = summary(outcomes[outcomes.traffic == 'following'])[['эпизодов', 'столкновений']].copy()\n"
        "b['верхняя граница 95 % (если 0)'] = [round(zero_upper_bound(n), 3) if k == 0 else None\n"
        "                                      for k, n in zip(b['столкновений'], b['эпизодов'])]\n"
        "b"
    ),
    md("## Как часто фильтр вмешивается (контролируемое следование)"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)\n"
        "colors = ['#2c6fbb', '#e67e22', '#8e44ad']\n"
        "bins = np.linspace(0, 0.6, 31)\n"
        "f = outcomes[(outcomes.traffic == 'following') & outcomes.shield]\n"
        "for ax, policy, title in zip(axes, ['idle', 'random'], ['IDLE', 'случайная политика']):\n"
        "    for (name, *_), c in zip(VARIANTS[:3], colors):\n"
        "        x = f[(f.policy == policy) & (f['вариант'] == name)]\n"
        "        ax.hist(x['доля времени фильтр тормозит'], bins=bins, histtype='step', linewidth=2,\n"
        "                color=c, label=name)\n"
        "    ax.set_title(title, fontsize=10)\n"
        "    ax.set_xlabel('доля времени эпизода, когда фильтр тормозит')\n"
        "    ax.grid(alpha=0.3)\n"
        "axes[0].set_ylabel('эпизодов')\n"
        "axes[0].legend(fontsize=7, frameon=False)\n"
        "fig.suptitle(f'Номинальный режим, контролируемое следование: вмешательства фильтра ({len(SEEDS)} сцен)', fontsize=10)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '05_shield_share.png', dpi=150)"
    ),
    md(
        "## Испытание устойчивости: прежний фон с MOBIL\n\n"
        "Те же варианты и сцены 0–49, но фон highway-env: машины перестраиваются по MOBIL "
        "(не по поперечной RSS) и стартуют без учета d_min. Здесь допущения RSS о других "
        "машинах могут нарушаться без всякого нарушителя."
    ),
    code(
        "summary(outcomes[outcomes.traffic == 'robustness'])"
    ),
    code(
        "ue = outcomes[outcomes.shield].groupby(['traffic', 'policy', 'вариант'], sort=False).agg(\n"
        "    эпизодов_с_событием=('опасных событий от фона', lambda x: int((x > 0).sum())),\n"
        "    событий_всего=('опасных событий от фона', 'sum'),\n"
        ")\n"
        "ue"
    ),
    md(
        "Опасное событие от фона (`unsafe_entries`): впереди агента появилась новая "
        "машина продольно и поперечно ближе d_min, хотя сам агент 3 с не менял полосу "
        "(первые 3 с не считаем). На контролируемом следовании фон не перестраивается, "
        "и таких событий у IDLE быть не может."
    ),
    code(
        "rob = LOGS[('robustness', 'idle')]\n"
        "rows = []\n"
        "for key, log in rob.items():\n"
        "    if key[2] and key[3] == 0.5 and not key[4]:\n"
        "        rows += [(key[0], r.t, r.ratio) for r in unsafe_entries(log).itertuples()]\n"
        "ev = pd.DataFrame(rows, columns=['seed', 't', 'зазор / d_min'])\n"
        "if len(ev[ev['зазор / d_min'] > 0]):\n"
        "    seed_ex, t_ex = ev[ev['зазор / d_min'] > 0].sort_values('зазор / d_min').iloc[0][['seed', 't']]\n"
        "    log = rob[(int(seed_ex), 1.0, True, 0.5, False)]\n"
        "    win = log[(log.t > t_ex - 4) & (log.t < t_ex + 8)]\n"
        "    axes = plot_episode(win, f'Фон с MOBIL, сцена {int(seed_ex)}, IDLE, базовая реакция ρ: '\n"
        "                        f'фоновая машина резко сдвигается к агенту')\n"
        "    for ax in axes:\n"
        "        ax.axvline(t_ex, color='#8e44ad', linewidth=1, linestyle='--')\n"
        "    axes[0].set_ylim(-2, None)\n"
        "    plt.gcf().tight_layout()\n"
        "    plt.gcf().savefig(FIG / '05_background_cutin.png', dpi=150)\n"
        "ev.round(2).head(10)"
    ),
    md(
        "## Выводы этапа 5\n\n"
        "1. **Начальные состояния.** На контролируемом следовании зазор агента на первом "
        "шаге меньше d_min — в 0 % сцен из 100, и ни одна пара соседних машин не стоит ближе "
        "d_min. На прежнем фоне highway-env — 52 % и 100 % соответственно (именно эти 52 % "
        "указаны в письме; прежнее «примерно треть» было неверно). Отсрочка нарушения на 3 с "
        "эту проблему не устраняла — теперь она устранена правилом расстановки.\n"
        "2. **Номинальный режим, контролируемое следование**: с фильтром 0 столкновений из 50 "
        "у каждого из трех вариантов фильтра и обеих политик (точная верхняя 95 % граница — "
        "5,8 % на вариант, отдельно для каждого). Без фильтра — 50 из 50 (IDLE) и 49 из 50 "
        "(случайная политика). Агент едет: пробег ≈ 0,87–0,88 км за 40 с, минимальная "
        "скорость ≈ 18 м/с. Фильтр тормозит 35–41 % времени у IDLE и ≈ 19 % у случайной "
        "политики. Аварийный режим не включался ни разу: на этом фоне базовой реакции "
        "хватает всегда, поэтому варианты «базовая» и «с аварийным торможением» совпадают.\n"
        "3. **Счетчики перестроений разделены**: «запретил бы» (решение фильтра, считается "
        "и без фильтра — 158 у случайной политики без фильтра) и реально выполненные запреты "
        "и отмены (без фильтра — 0).\n"
        "4. **Опасных событий от фона** на контролируемом следовании нет (фон не "
        "перестраивается). На прежнем фоне с MOBIL они есть: 5 эпизодов из 50 у IDLE и "
        "20–22 у случайной политики.\n"
        "5. **Испытание устойчивости (прежний фон)**: с фильтром тоже 0 столкновений из 50 "
        "во всех вариантах, включая базовую реакцию без аварийного торможения; вариант с "
        "аварийным торможением включал его 1–2 % времени. Это отдельное испытание, а не "
        "режим соблюдения допущений.\n"
        "6. Фильтр на опасность от машины, сдвигающейся сбоку, отвечает только продольным "
        "торможением: полной поперечной надлежащей реакции RSS в нем нет (см. docstring "
        "`rssrl/shield.py`). Поэтому многополосный фильтр не называем точной реализацией "
        "всех правил RSS."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("05_nominal_check.ipynb")
ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
