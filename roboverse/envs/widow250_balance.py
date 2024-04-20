import gym
import numpy as np
import pybullet as p
import os.path as osp
import time
from roboverse.envs.widow250 import END_EFFECTOR_INDEX, Widow250Env
import roboverse
import roboverse.bullet as bullet
from roboverse.bullet import object_utils
from roboverse.envs import objects

OBJECT_IN_GRIPPER_PATH = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                                 'assets/bullet-objects/bullet_saved_states/objects_in_gripper/')

class Widow250BalanceEnv(Widow250Env):
    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg
        self.should_randomize = False
        super().__init__(**kwargs)
        
    def update_randomization(self, should_randomize):
        self.should_randomize = should_randomize

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.plate_id = objects.plate()
        self.objects = {}
        object_positions = object_utils.generate_object_positions(self.object_position_low,
                                                                  self.object_position_high,
                                                                  self.num_objects)
        self.original_object_positions = object_positions

        for object_name, object_position in zip(self.object_names, object_positions):
            loaded_object = object_utils.load_object(object_name,
                                                     object_position,
                                                     object_quat=self.object_orientations[object_name],
                                                     scale=self.object_scales[object_name])
            self.objects[object_name] = loaded_object
            if object_name == 'ball':
                self.ball_id = loaded_object
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
        if self.should_randomize:
            self.randomize_ball('reset')
        else:
            self.ball_mass = 0.00005
            p.changeDynamics(self.ball_id, -1, mass=self.ball_mass, rollingFriction=0.01, spinningFriction=0.01, lateralFriction=0.4)
            p.changeDynamics(self.plate_id, -1, lateralFriction=0.2)
            
        # Attach plate to end effector:
        const_id = p.createConstraint(self.robot_id, END_EFFECTOR_INDEX, self.plate_id, -1, p.JOINT_POINT2POINT, [0,0,0], [0,-0.1,0], [0,0,0], [0,0,0], [0,0,0])
        p.changeConstraint(const_id, maxForce=1e4, erp=1e-20)

    def randomize_ball(self, step_or_reset: str = 'reset' or 'step'):
        if self.cfg['randomize_ball_drop'] and step_or_reset == 'reset':
            # Randomize ball size
            ball_size = np.random.uniform(self.cfg['ball_size']['min'], self.cfg['ball_size']['max'])
            self.object_scales['ball'] = ball_size
            b_min, b_max = self.cfg['ball_mass'][step_or_reset]
            self.ball_mass = np.random.uniform(b_min, b_max)
            self.drop_ball()
        
        r_min, r_max = self.cfg['ball_rolling_friction'][step_or_reset]
        s_min, s_max = self.cfg['ball_spinning_friction'][step_or_reset]
        l_min, l_max = self.cfg['ball_lateral_friction'][step_or_reset]
        p_min, p_max = self.cfg['plate_lateral_friction'][step_or_reset]

        rollingFriction = np.random.uniform(r_min, r_max)
        spinningFriction = np.random.uniform(s_min, s_max)
        lateralFriction = np.random.uniform(l_min, l_max)
        plate_friction = np.random.uniform(p_min, p_max)

        p.changeDynamics(self.ball_id, -1, mass=self.ball_mass,
                            rollingFriction=rollingFriction,
                            spinningFriction=spinningFriction,
                            lateralFriction=lateralFriction)
        p.changeDynamics(self.plate_id, -1, lateralFriction=plate_friction)

    def get_info(self):
        info = super(Widow250BalanceEnv, self).get_info()
        info['ball_pos'] = self.get_ball_pos()
        info['plate_pos'] = self.get_plate_pos()
        info['distance_from_center'] = object_utils.get_distance_from_center(
            info['ball_pos'], info['plate_pos'])
        info['height_distance'] = np.abs(info['ball_pos'][2] - info['plate_pos'][2]) # TODO is the ball_pos the center of the ball?
        return info
    
    def get_ball_pos(self):
        return object_utils.get_ball_pos(self.ball_id)
    
    def get_plate_pos(self):
        return object_utils.get_plate_pos(self.plate_id)

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        if self.reward_type == "balance":
            distance_reward = -info['distance_from_center']
            duration_reward = self.duration * self.cfg['duration_weight']
            height_reward = -info['height_distance'] * self.cfg['height_weight']
            return distance_reward + duration_reward + height_reward
        else:
            return super().get_reward(info)
        
    def reset(self, target=None, seed=None, options=None):
        super().reset()
        bullet.load_state(osp.join(OBJECT_IN_GRIPPER_PATH, 'plate_in_gripper_reset.bullet'))
        self.is_gripper_open = False
        self.duration = 0

        self.drop_ball()
        self.generate_dynamics()

        return self.get_observation()

    def step(self, action):
        
        obs, reward, done, truncated, info = super().step(action)

        if self.reward_type == "balance":
            reward = self.get_reward(info)
            if info['ball_pos'][2] < -0.35: # TODO put this in config or something
                truncated = True
            else:
                self.duration += 1
                
        if self.cfg.get("randomize_every_step", False):
            self.randomize_ball('step')
        
        return obs, reward, done, truncated, info   

    def _set_observation_space(self):
        robot_state_dim = 9  # XYZ + QUAT + XY_BALL
        obs_bound = 100
        obs_high = np.ones(robot_state_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)
        object_position = gym.spaces.Box(-np.ones(3), np.ones(3))
        object_orientation = gym.spaces.Box(-np.ones(4), np.ones(4))
        spaces = {'state': state_space, 'object_position': object_position,
                    'object_orientation': object_orientation}
        self.observation_space = gym.spaces.Dict(spaces)

    def get_observation(self):
        ee_pos, ee_quat = bullet.get_link_state(self.robot_id, self.end_effector_index)
        object_position, object_orientation = bullet.get_object_position(self.objects[self.target_object])
        ball_pos = self.get_ball_pos()
        plate_pos = self.get_plate_pos()
        ball_relative_pos = np.array(plate_pos)[:2] - np.array(ball_pos)[:2]
        return {
            'object_position': object_position,
            'object_orientation': object_orientation,
            'state': np.concatenate((ee_pos, ee_quat, ball_relative_pos)),
        }

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
    env.reset()

    for j in range(5):
        for i in range(20):
            obs, rew, done, info = env.step(
                np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0., 0.]))
            print("reward", rew, "info", info)
            time.sleep(0.1)
        env.reset()