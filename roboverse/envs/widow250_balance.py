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
        self.arm_min_radius = 0.100
        self.ee_distance_threshold = self.cfg['ee_distance_threshold']
        self.ee_target_pose = None
        super().__init__(**kwargs)
        
    def update_randomization(self, should_randomize):
        self.should_randomize = should_randomize

    def _load_meshes(self, target_position=None):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.plate_id = objects.plate()
        self.debug_sphere_id = None
        self.objects = {}
        object_positions = object_utils.generate_object_positions(self.object_position_low,
                                                                  self.object_position_high,
                                                                  self.num_objects)
        self.original_object_positions = object_positions

        for object_name, object_position in zip(self.object_names, object_positions):
            self.objects[object_name] = object_utils.load_object(object_name,
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
    
    def render_goal_sphere(self):
        if self.debug_sphere_id:
            p.removeBody(self.debug_sphere_id)
            self.debug_sphere_id = None
            
        self.debug_sphere_id = object_utils.create_debug_sphere(self.ee_target_pose, self.ee_distance_threshold)

        bullet.step_simulation(self.num_sim_steps)
    
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
        
        # Balance info
        info['ball_pos'] = self.get_ball_pos()
        info['plate_pos'], plate_quat = self.get_plate_pos_quat()
        info['distance_from_center'] = object_utils.get_distance_from_center(
            info['ball_pos'], info['plate_pos'], self.cfg['center_radius'])
        info['height_distance'] = np.abs(info['ball_pos'][2] - info['plate_pos'][2])
        info['plate_angle'] = p.getEulerFromQuaternion(plate_quat)[0] # X angle -> plate tilt angle
        # ----------------
        
        # GCRL info
        ee_pos, ee_quat = bullet.get_link_state(self.robot_id, self.end_effector_index)
        target_coord = np.array(self.ee_target_pose)
        ee_coord = np.array(ee_pos)
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)
        info['ee_pose_success'] = euclidean_dist_3d <= self.ee_distance_threshold
        info['euclidean_distance'] = euclidean_dist_3d
        info['target_coord'] = self.ee_target_pose
        # ----------------
        
        return info
    
    def get_ball_pos(self):
        return object_utils.get_ball_pos(self.ball_id)
    
    def get_plate_pos_quat(self):
        return object_utils.get_plate_pos_quat(self.plate_id)
    
    def get_reward(self, info):
        if not info:
            info = self.get_info()
        reward = 0

        distance_reward = -np.exp(info['distance_from_center']) * self.cfg['distance_center_weight']
        duration_reward = self.duration * self.cfg['duration_weight']
        height_reward = -info['height_distance'] * self.cfg['height_weight']
        tilt_reward = -np.abs(info['plate_angle']) * self.cfg['tilt_weight']
        
        reward += distance_reward + duration_reward + height_reward + tilt_reward
        
        if self.cfg["gcrl"]:
            g_w = self.cfg['goal_reached_weight']
            d_w = self.cfg['distance_goal_weight']

            target_distance_reward = -np.exp(info['euclidean_distance']) * d_w
            reward += target_distance_reward

            if info['ee_pose_success']:
                reward += g_w * 1
                self.done = True
    
        return reward

        
    def reset(self, target=None, seed=None, options=None):
        if target:
            assert len(target) == 6
            self.ee_target_pose = target
        else:
            self.ee_target_pose = self._get_target_pose()
            
        super().reset()
        bullet.load_state(osp.join(OBJECT_IN_GRIPPER_PATH, 'plate_in_gripper_reset.bullet'))
        self.is_gripper_open = False
        self.duration = 0
           
        if self.cfg["gcrl"]:
            self.render_goal_sphere()
            
        self.drop_ball()
        self.generate_dynamics()
        self.done = False

        return self.get_observation()

    def step(self, action):
        
        obs, reward, _, truncated, info = super().step(action)

        if self.reward_type == "balance":
            reward = self.get_reward(info)
            if info['ball_pos'][2] < -0.35: # TODO put this in config or something
                truncated = True
            else:
                self.duration += 1
                
        if self.cfg.get("randomize_every_step", False):
            self.randomize_ball('step')
        
        return obs, reward, self.done, truncated, info   

    def _set_observation_space(self):
        if self.cfg["gcrl"]:
            robot_state_dim = 12 # XYZ + QUAT + XY_BALL + XY_TARGET
        else:
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
        plate_pos, _ = self.get_plate_pos_quat()
        ball_relative_pos = np.array(plate_pos)[:2] - np.array(ball_pos)[:2]
        target_coord = np.array(self.ee_target_pose)
        return {
            'object_position': object_position,
            'object_orientation': object_orientation,
            'state': np.concatenate((ee_pos, ee_quat, ball_relative_pos, target_coord)),
        }
        
    def _get_target_pose(self) -> np.ndarray:
        workspace_pose = bullet.get_random_workspace_pose(self.ee_pos_low, self.ee_pos_high, self.arm_min_radius)
        # workspace_pose = [sum(coord) for coord in zip(workspace_pose, self.base_position)]
        return workspace_pose

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