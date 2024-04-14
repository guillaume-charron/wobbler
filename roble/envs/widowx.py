import numpy as np
from gym import Wrapper
from gym.wrappers import FilterObservation, FlattenObservation

import roboverse


def create_widow_env(env="Widow250BallBalancing-v0", **kwargs):
    env = roboverse.make(env, **kwargs)
    env = FlattenObservation(
        FilterObservation(
            RoboverseWrapper(env),
            [
                "state",
                # The object is located at the goal location. For
                # non goal conditioned observations, only state
                # should be used.
                # "object_position",
                # "object_orientation"
            ],
        )
    )
    return env


class RoboverseWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.render_mode = 'rgb_array'

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated or truncated, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return obs

    def render(self, mode="rgb_array", **kwargs):
        img = self.env.render_obs()
        img = np.transpose(img, (1, 2, 0))
        return img
