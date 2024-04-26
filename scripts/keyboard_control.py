import numpy as np
import time
import argparse

import roboverse
import roboverse.bullet as bullet
import pybullet as p
import hydra
from hydra.core.global_hydra import GlobalHydra
from roble.envs.widowx import create_widow_env
from roboverse.envs.widow250_real import Widow250EnvROSARob

KEY_TO_ACTION_MAPPING = {
    bullet.p.B3G_LEFT_ARROW: np.array([0.1, 0, 0, 0, 0, 0, 0]),
    bullet.p.B3G_RIGHT_ARROW: np.array([-0.1, 0, 0, 0, 0, 0, 0]),
    bullet.p.B3G_UP_ARROW: np.array([0, -0.1, 0, 0, 0, 0, 0]),
    bullet.p.B3G_DOWN_ARROW: np.array([0, 0.1, 0, 0, 0, 0, 0]),
    ord('j'): np.array([0, 0, 0.2, 0, 0, 0, 0]),
    ord('k'): np.array([0, 0, -0.2, 0, 0, 0, 0]),
    ord('1'): np.array([0, 0, 0, 0.2, 0, 0, 0]),
    ord('2'): np.array([0, 0, 0, -0.2, 0, 0, 0]),
    ord('3'): np.array([0, 0, 0, 0, 0.2, 0, 0]),
    ord('4'): np.array([0, 0, 0, 0, -0.2, 0, 0]),
    ord('5'): np.array([0, 0, 0, 0, 0, 0.2, 0]),
    ord('6'): np.array([0, 0, 0, 0, 0, -0.2, 0]),
    ord('h'): np.array([0, 0, 0, 0, 0, 0, -0.7]),
    ord('l'): np.array([0, 0, 0, 0, 0, 0, 0.7])
}

ENV_COMMANDS = {
    ord('r'): lambda env: env.reset(),
    ord('m'): lambda env: p.saveBullet('test.bullet'),
    ord('p'): lambda env: p.restoreState(fileName='test.bullet'),
    ord('d'): lambda env: env.delete_box()
}

def load_config():
    if not GlobalHydra.instance().is_initialized():
        hydra.initialize(config_path="../conf")
    cfg = hydra.compose(config_name="config")
    return cfg

def keyboard_control(args):
    cfg = load_config()
    env_config = cfg.environment
    # env = create_widow_env(observation_mode='state', cfg=env_config, gui=True)
    # env = Widow250EnvROSARob(rs_address='192.168.1.101:50051')
    env = roboverse.make(args.env_name, gui=True)
    # env.update_randomization(True)
    env.reset()

    while True:
        take_action = False
        action = np.array([0, 0, 0, 0, 0, 0, 0], dtype='float32')
        keys = bullet.p.getKeyboardEvents()
        for qKey in keys:
            if qKey in KEY_TO_ACTION_MAPPING.keys():
                action += KEY_TO_ACTION_MAPPING[qKey]
                take_action = True
            elif qKey in ENV_COMMANDS.keys():
                ENV_COMMANDS[qKey](env)
                take_action = False

        if take_action:
            obs, rew, done, info = env.step(action)
            print(rew)
        time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-name", type=str,
                        default='Widow250EnvROSARob-v0')
    args = parser.parse_args()
    keyboard_control(args)
