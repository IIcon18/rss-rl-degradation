"""Целевая среда обучения PPO (письмо 29.10, п. 9).

TrainingEnv — та же система, что и в оценке: машина агента ShieldedVehicle
(с фильтром или без), фон контролируемого следования, номинальный режим
(нарушителя нет: обучение только в номинальном режиме, решение 4 письма от
21.10). Каждый reset строит новую сцену make_scene с seed обучения.

Две политики схемы 2×2:
- «обучена с фильтром»: shield=True, базовая надлежащая реакция с учетом
  рывка (основной фильтр, emergency_brake=False), задержка ρ;
- «обучена без фильтра»: shield=False. Ограничения машины у нее те же:
  команда обрезается до [-a_max_brake, a_max_accel] = [-5, 3] м/с², рывок
  ≤ J_MAX, задом не едет.
Наблюдения, награда (стандартная highway-v0), бюджет шагов — одинаковые.

Seed сцен обучения не пересекаются с seed сцен оценки: TRAIN_SEED_OFFSET.
"""

import gymnasium as gym

from rssrl.scenarios import RHO, make_scene

TRAIN_SEED_OFFSET = 1_000_000  # сцены оценки — 0..N, сцены обучения — от миллиона


class TrainingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, shield=True, reaction_delay=RHO, emergency_brake=False, traffic="following", worker=0):
        self.shield, self.reaction_delay = shield, reaction_delay
        self.emergency_brake, self.traffic = emergency_brake, traffic
        self.worker = worker
        self.episode = 0
        self.scene = self._new_scene(TRAIN_SEED_OFFSET)
        u = self.scene.env.unwrapped
        self.observation_space = u.observation_space
        self.action_space = u.action_space

    def _new_scene(self, seed):
        return make_scene(seed, kind="nominal", shield=self.shield, reaction_delay=self.reaction_delay,
                          emergency_brake=self.emergency_brake, traffic=self.traffic)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.base = int(seed)
        base = getattr(self, "base", 0)
        self.scene.env.close()
        # seed сцены обучения: смещение + (база, номер среды, номер эпизода)
        scene_seed = TRAIN_SEED_OFFSET + base * 100_000 + self.worker * 10_000 + self.episode
        self.episode += 1
        self.scene = self._new_scene(scene_seed)
        obs = self.scene.env.unwrapped.observation_type.observe()
        return obs, {}

    def step(self, action):
        return self.scene.env.step(action)

    def close(self):
        self.scene.env.close()


def make_training_env(shield=True, worker=0, **kwargs):
    """Фабрика для векторных сред stable-baselines3 (функция модуля, чтобы
    ее можно было передать в дочерний процесс)."""
    return TrainingEnv(shield=shield, worker=worker, **kwargs)
