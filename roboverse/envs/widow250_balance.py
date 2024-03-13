from roboverse.envs.widow250 import Widow250Env
import roboverse
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
import numpy as np
import itertools
from roboverse.assets.shapenet_object_lists import CONTAINER_CONFIGS
import pybullet as p


class Widow250BalanceEnv(Widow250Env):
    def __init__(self,
                 **kwargs):
        super(Widow250BalanceEnv, self).__init__(**kwargs)

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.plate_id = objects.plate()

        self.objects = {}
        object_positions = object_utils.generate_object_positions(
            self.object_position_low, self.object_position_high,
            self.num_objects,
        )
        self.original_object_positions = object_positions

        for object_name, object_position in zip(self.object_names,
                                                object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            if object_name == 'ball':
                self.ball_id = self.objects[object_name]
            bullet.step_simulation(self.num_sim_steps_reset)
        
        # Make the ball bounce on the table
        p.changeDynamics(self.ball_id, -1, restitution=0.9, mass=1.0, lateralFriction=0.9, rollingFriction=0.9, spinningFriction=0.9)
        p.changeDynamics(self.table_id, -1, restitution=0.8, lateralFriction=0.9, rollingFriction=0.9, spinningFriction=0.9)

    def get_info(self):
        info = super(Widow250BalanceEnv, self).get_info()
        return info

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        if self.reward_type == "balance":
            return -1
        else:
            return super(Widow250BalanceEnv, self).get_reward(info)
        
    def reset(self, target=None, seed=None, options=None):
        bullet.reset()
        bullet.setup_headless()
        self._load_meshes()
        bullet.reset_robot(
            self.robot_id,
            self.reset_joint_indices,
            self.reset_joint_values)
        self.is_gripper_open = self.default_gripper_state 
        self.ee_pos_init, self.ee_quat_init = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        p.createConstraint(self.end_effector_index, -1, self.plate_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])


        return self.get_observation(), self.get_info()


if __name__ == "__main__":
    env = roboverse.make('Widow250BallBalancing-v0',
                         gui=True, transpose_image=False)
    import time
    env.reset()
    # import IPython; IPython.embed()

    for j in range(5):
        for i in range(20):
            obs, rew, done, info = env.step(
                np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0., 0.]))
            print("reward", rew, "info", info)
            time.sleep(0.1)
        env.reset()