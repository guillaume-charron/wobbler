from typing import Tuple

import numpy as np
from gymnasium.core import Wrapper

class RandomActResetEnv(Wrapper):
    def __init__(self, env, max_num_random_act=4):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.max_num_random_act=max_num_random_act

    def _sample_num_repeat(self):
        return int(np.random.randint(0, self.max_num_random_act + 1))

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, ORIG_INFO = super(RandomActResetEnv, self).reset(**kwargs)

        num_random_act = self._sample_num_repeat()
        # TODO: sample random action and step, possibly resetting env in the process
        for _ in range(num_random_act):
            obs, _, terminated, truncated, _ = self.step([0] * self.env.action_space.shape[0])
            if terminated or truncated:
                obs, _ = super(RandomActResetEnv, self).reset(**kwargs)
        return obs, ORIG_INFO

    def step(self, ac):
        return self.env.step(ac)