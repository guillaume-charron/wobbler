import gymnasium
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
        self.action_space = gymnasium.spaces.Box(
            low=env.action_space.low[:-2], high=env.action_space.high[:-2]
        )

    def step(self, action):
        # Append default gripper values:
        action = np.append(action, [0.015, -0.015])
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated or truncated, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return obs

    def render(self, mode="rgb_array", **kwargs):
        img = self.env.render_obs()
        img = np.transpose(img, (1, 2, 0))
        return img
