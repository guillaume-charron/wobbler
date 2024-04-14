from roboverse.envs.widow250 import END_EFFECTOR_INDEX, Widow250Env
import roboverse
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
import numpy as np
import pybullet as p
import os.path as osp

OBJECT_IN_GRIPPER_PATH = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                'assets/bullet-objects/bullet_saved_states/objects_in_gripper/')


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
        
    def drop_ball(self):
        if self.ball_id is not None:
            p.removeBody(self.ball_id)
            self.ball_id = None
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
    
    def generate_dynamics(self):
        p.changeDynamics(self.ball_id, -1, mass=0.00005, rollingFriction=0.01, spinningFriction=0.01, lateralFriction=0.4)
        p.changeDynamics(self.plate_id, -1, lateralFriction=0.2)

    def get_info(self):
        info = super(Widow250BalanceEnv, self).get_info()
        info['ball_pos'] = self.get_ball_pos()
        info['plate_pos'] = self.get_plate_pos()
        info['distance_from_center'] = object_utils.get_distance_from_center(
            info['ball_pos'], info['plate_pos'])
        # print(info)
        return info
    
    def get_ball_pos(self):
        return object_utils.get_ball_pos(self.ball_id)
    
    def get_plate_pos(self):
        return object_utils.get_plate_pos(self.plate_id)

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        if self.reward_type == "balance":
            return -info['distance_from_center']
        else:
            return super(Widow250BalanceEnv, self).get_reward(info)
        
    def reset(self, target=None, seed=None, options=None):
        super(Widow250BalanceEnv, self).reset()
        bullet.load_state(osp.join(OBJECT_IN_GRIPPER_PATH,
            'plate_in_gripper_reset.bullet'))
        self.is_gripper_open = False

        self.drop_ball()
        self.generate_dynamics()

        return self.get_observation()


class Widow250BalanceKeyboardEnv(Widow250Env):
    def __init__(self,
                 **kwargs):
        super(Widow250BalanceKeyboardEnv, self).__init__(**kwargs)

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.plate_id = objects.plate(pos=[0.6, 0.25, 0])
        self.box_id = objects.box()
        
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


    def get_reward(self, info):
        if not info:
            info = self.get_info()
        if self.reward_type == "balance":
            return 0.0
        else:
            return super(Widow250BalanceKeyboardEnv, self).get_reward(info)
    
    def delete_box(self):
        p.removeBody(self.box_id)
        
    def reset(self, target=None, seed=None, options=None):
        bullet.reset()
        bullet.setup_headless()
        self._load_meshes()
        bullet.reset_robot(
            self.robot_id,
            self.reset_joint_indices,
            self.reset_joint_values)
        self.is_gripper_open = self.default_gripper_state         
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