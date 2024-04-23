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
from roboverse.envs.widow250_real import Widow250EnvROS
from roboverse.utils.ball_tracking import analyze_frame
from roboverse.utils.camera import Camera

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
        if euclidean_dist_3d <= self.ee_distance_threshold:
            info['ee_pose_success'] = True
            info['target_coord'] = self.ee_target_pose
        else:
            info['ee_pose_success'] = False

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
        if self.reward_type == "balance":
            
            distance_reward = -np.exp(info['distance_from_center']) * self.cfg['distance_center_weight']
            duration_reward = self.duration * self.cfg['duration_weight']
            height_reward = -info['height_distance'] * self.cfg['height_weight']
            tilt_reward = -np.abs(info['plate_angle']) * self.cfg['tilt_weight']
            
            reward += distance_reward + duration_reward + height_reward + tilt_reward
            
            if self.cfg["gcrl"]:
                g_w = self.cfg['goal_reached_weight']
                d_w = self.cfg['distance_goal_weight']

                reward += np.exp(-d_w * info['euclidean_distance'])

                if info['ee_pose_success']:
                    reward = g_w * 1
                    self.done = True
      
            return reward
        else:
            return super().get_reward(info)
        
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
        plate_pos, _ = self.get_plate_pos_quat()
        ball_relative_pos = np.array(plate_pos)[:2] - np.array(ball_pos)[:2]
        return {
            'object_position': object_position,
            'object_orientation': object_orientation,
            'state': np.concatenate((ee_pos, ee_quat, ball_relative_pos)),
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

class Widow250BalanceROS(Widow250EnvROS):
    def __init__(self, camera, cfg=None, **kwargs):
        self.cfg = cfg
        self.camera = camera
        super(Widow250BalanceROS, self).__init__(**kwargs)
        self.ee_distance_threshold = self.cfg['ee_distance_threshold']
        self.ee_target_pose = None
        self.nb_none_ball = 0
        self.action_space = gym.spaces.Box(
            low=self.action_space.low[:-2], high=self.action_space.high[:-2]
        )
    
    # TODO in the server simulation maybe?
    # def render_goal_sphere(self):
    #     if self.debug_sphere_id:
    #         p.removeBody(self.debug_sphere_id)
    #         self.debug_sphere_id = None
            
    #     self.debug_sphere_id = object_utils.create_debug_sphere(self.ee_target_pose, self.ee_distance_threshold)

    #     bullet.step_simulation(self.num_sim_steps)
    def get_plate_ball_pos(self):
        frame = self.camera.get_frame()
        if frame is not None:
            plate_pos, ball_pos, _ = analyze_frame(frame)
        if frame is None or plate_pos is None or ball_pos is None:
            plate_pos, ball_pos = [0, 0], [0, 0]
        return plate_pos, ball_pos

    def get_info(self):
        info = super().get_info()

        # Balance info
        info['ball_pos'], info['plate_pos'] = self.get_plate_ball_pos()
        print(info['ball_pos'], info['plate_pos'])
        info['distance_from_center'] = object_utils.get_distance_from_center(
            info['ball_pos'], info['plate_pos'], self.cfg['center_radius'])
        # info['height_distance'] = np.abs(info['ball_pos'][2] - info['plate_pos'][2])
        # info['plate_angle'] = p.getEulerFromQuaternion(plate_quat)[0] # X angle -> plate tilt angle
        # ----------------
        
        return info
    
    def get_reward(self, info):
        if not info:
            info = self.get_info()
        reward = 0
        if self.reward_type == "balance":
            if info['ball_pos'] == [0, 0]:
                return 0
            distance_reward = -np.exp(info['distance_from_center']) * self.cfg['distance_center_weight']
            duration_reward = self.duration * self.cfg['duration_weight']
            # height_reward = -info['height_distance'] * self.cfg['height_weight']
            # tilt_reward = -np.abs(info['plate_angle']) * self.cfg['tilt_weight']
            
            reward += distance_reward + duration_reward
            
            if self.cfg["gcrl"]:
                g_w = self.cfg['goal_reached_weight']
                d_w = self.cfg['distance_goal_weight']

                reward += np.exp(-d_w * info['euclidean_distance'])

                if info['ee_pose_success']:
                    reward = g_w * 1
                    self.done = True
      
            return reward
        else:
            return super().get_reward(info)
        
    def reset(self, target=None, seed=None, options=None):
        obs, info = super().reset()
        self.is_gripper_open = False
        self.duration = 0
        self.nb_none_ball = 0
        self.done = False
           
        # TODO render in env simulation
        # if self.cfg["gcrl"]:
        #     self.render_goal_sphere()

        return obs, info

    def step(self, action):
        
        obs, reward, done, truncated, info = super().step(action)

        reward = self.get_reward(info)
        if reward == 0:
            self.nb_none_ball += 1
            if self.nb_none_ball > 5:
                truncated = True
        else:
            self.nb_none_ball = 0
            self.duration += 1

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
        self.observation_space = gym.spaces.Dict(spaces)['state']

    def get_observation(self):
        
        # Get Robot Server state
        rs_state = self.robogym.client.get_state_msg().state_dict

        # Check if the length and keys of the Robot Server state received is correct
        self.robogym.check_rs_state_keys(rs_state, self.ee_target_pose)

        # Convert the initial state from Robot Server format to environment format
        self.ee_pos, ee_rot = self.robogym.robot_server_state_to_env_state(rs_state)
        self.ee_quat = bullet.deg_to_quat(ee_rot)
        
        object_position = np.array(self.ee_target_pose)
        object_orientation = np.array([0, 0, 0, 1])
        
        plate_pos, ball_pos = self.get_plate_ball_pos()
        print(plate_pos, ball_pos)
        
        
        ball_relative_pos = (np.array(plate_pos) - np.array(ball_pos)) / 1000 # TODO: check if this is correct + config it
        return np.concatenate((self.ee_pos, self.ee_quat, ball_relative_pos))
        
    def _get_target_pose(self) -> np.ndarray:
        workspace_pose = bullet.get_random_workspace_pose(self.ee_pos_low, self.ee_pos_high, self.arm_min_radius)
        return workspace_pose

if __name__ == "__main__":
    # env = roboverse.make('Widow250BallBalancing-v0',
    #                      gui=True, transpose_image=False)
    # env.reset()

    # for j in range(5):
    #     for i in range(20):
    #         obs, rew, done, info = env.step(
    #             np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0., 0.]))
    #         print("reward", rew, "info", info)
    #         time.sleep(0.1)
    #     env.reset()
    conf = {
        "ee_distance_threshold": 0.1,
        "center_radius": 0.1,
        "distance_center_weight": 1,
        "duration_weight": 1,
        "height_weight": 1,
        "goal_reached_weight": 1,
        "distance_goal_weight": 1,
        "gcrl": False,
        
    }
    import time
    camera = Camera()
    env = Widow250BalanceROS(camera=camera, cfg=conf, rs_address='192.168.1.101:50051')
    STATE = [0., 0., 0., 0, 0., 0., 0., 0., 0.]
    env.reset()
    env.step(np.asarray(STATE))
    time.sleep(0.1)
    env.reset()