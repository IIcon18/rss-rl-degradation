"""Собирает и выполняет notebooks/00_environment.ipynb.

Запуск из корня проекта: .venv/bin/python notebooks/make_00_environment.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Этап 0. Окружение и пустая highway-v0\n\n"
        "Проверяем, что среда запускается, и записываем ее параметры по умолчанию. "
        "Здесь ничего не меняем, только смотрим, с чем будем работать."
    ),
    code(
        "import sys\n"
        "import gymnasium as gym\n"
        "import highway_env\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n\n"
        "print('Python', sys.version.split()[0])\n"
        "print('highway-env', highway_env.__version__ if hasattr(highway_env, '__version__') else '?')\n"
        "print('gymnasium', gym.__version__)\n"
        "FIG = Path('..') / 'results' / 'figures'\n"
        "FIG.mkdir(parents=True, exist_ok=True)"
    ),
    md("## Параметры highway-v0 по умолчанию"),
    code(
        "env = gym.make('highway-v0', render_mode='rgb_array')\n"
        "cfg = env.unwrapped.config\n"
        "keys = ['lanes_count', 'vehicles_count', 'vehicles_density', 'duration',\n"
        "        'simulation_frequency', 'policy_frequency', 'other_vehicles_type',\n"
        "        'collision_reward', 'high_speed_reward', 'right_lane_reward',\n"
        "        'reward_speed_range', 'offroad_terminal']\n"
        "for k in keys:\n"
        "    print(f'{k:22s} {cfg[k]}')\n"
        "print('observation', cfg['observation'])\n"
        "print('action     ', cfg['action'])"
    ),
    md(
        "Что важно для следующих этапов:\n"
        "- **simulation_frequency = 15 Гц**: машины двигаются шагами по 1/15 с ≈ 0,067 с. "
        "Время реакции RSS ρ = 0,5 с — это 7–8 таких шагов.\n"
        "- **policy_frequency = 1 Гц**: агент выбирает действие раз в секунду, "
        "а между решениями симулятор делает 15 шагов. Поэтому фильтр нельзя ставить "
        "только между решениями агента: он должен работать внутри этих 15 шагов.\n"
        "- **duration = 40 с**: эпизод ограничен по времени (пункт 9 письма).\n"
        "- Наблюдение `Kinematics` видит только 5 ближайших машин; фильтр будет видеть "
        "все машины в радиусе ~150 м напрямую из симулятора (пункт 2 письма)."
    ),
    md("## Константы машин"),
    code(
        "from highway_env.vehicle.kinematics import Vehicle\n"
        "from highway_env.vehicle.behavior import IDMVehicle\n"
        "print('Длина / ширина машины, м:', Vehicle.LENGTH, '/', Vehicle.WIDTH)\n"
        "print('Скорость, м/с: от', Vehicle.MIN_SPEED, 'до', Vehicle.MAX_SPEED)\n"
        "print('IDM: желаемое торможение COMFORT_ACC_MIN =', IDMVehicle.COMFORT_ACC_MIN)\n"
        "print('IDM: обрезка ускорения ACC_MAX = ±', IDMVehicle.ACC_MAX)"
    ),
    md(
        "IDM-машина может тормозить до 6 м/с² (обрезка `ACC_MAX`), а RSS предполагает, "
        "что никто не тормозит сильнее a_max,brake = 5 м/с². На этапе 2 фоновым машинам "
        "ограничим торможение 5 м/с² (пункт 3 письма).\n\n"
        "`MIN_SPEED = -40`: симулятор сам не останавливает машину на нуле, при сильном "
        "торможении она может поехать назад. Для нарушителя это нужно будет учесть."
    ),
    md("## Пустой эпизод: агент держит полосу и скорость (действие IDLE)"),
    code(
        "obs, info = env.reset(seed=0)\n"
        "frame0 = env.render()\n"
        "steps, terminated, truncated = 0, False, False\n"
        "while not (terminated or truncated):\n"
        "    obs, reward, terminated, truncated, info = env.step(1)  # 1 = IDLE\n"
        "    steps += 1\n"
        "print('шагов агента:', steps, '| время, с:', env.unwrapped.time)\n"
        "print('столкновение:', env.unwrapped.vehicle.crashed, '| по времени:', truncated)\n"
        "print('машин на дороге:', len(env.unwrapped.road.vehicles))\n\n"
        "fig, ax = plt.subplots(figsize=(10, 2.5))\n"
        "ax.imshow(frame0); ax.axis('off')\n"
        "ax.set_title('highway-v0, seed = 0, начало эпизода (агент — зеленый)')\n"
        "fig.savefig(FIG / '00_highway_frame.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "env.close()"
    ),
    md("## Как часто «тупой» агент сталкивается без всяких нарушителей"),
    code(
        "env = gym.make('highway-v0')\n"
        "crash_times = []\n"
        "for seed in range(30):\n"
        "    env.reset(seed=seed)\n"
        "    terminated = truncated = False\n"
        "    while not (terminated or truncated):\n"
        "        _, _, terminated, truncated, _ = env.step(1)\n"
        "    if terminated:\n"
        "        crash_times.append(env.unwrapped.time)\n"
        "env.close()\n"
        "print(f'столкновений: {len(crash_times)} из 30')\n"
        "print('время столкновения, с:', crash_times)"
    ),
    md(
        "Агент с постоянным действием IDLE не подстраивается под машину впереди и почти "
        "всегда во что-то врезается, хотя нарушителей в сцене нет. Значит:\n"
        "- в пилоте этапа 3 (агент без PPO) столкновения будут и при β = 1, причем по вине "
        "самого агента. Именно эти случаи RSS-фильтр обязан предотвращать;\n"
        "- проверка номинального режима (этап 5) должна показать, что с фильтром "
        "таких столкновений нет. Если они останутся, фильтр реализован неверно."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("00_environment.ipynb")
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("ok:", out)
