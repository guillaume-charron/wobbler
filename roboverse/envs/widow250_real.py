from abc import ABC
from typing import Tuple, Union, Dict, Any

import gym
import numpy as np
import math
import robo_gym_server_modules.robot_server.client as rs_client

from numpy import ndarray, dtype

import roboverse.bullet as bullet
from robo_gym.envs.simulation_wrapper import Simulation

RESET_JOINT_VALUES = [0.0757, -0.0474, -0.0522, -1.55, 0.0058, -0.00076, 0., 0.036, -0.036]
# RESET_JOINT_VALUES = [0.0757, -0.0474, -0.0650, -0.00011, -0.0558, -1.540, 0., 0.036, -0.036]
# RESET_JOINT_VALUES = [1.5691, -1.8039, -0.8089, 0.0055, -1.0515, -1.5852, 2.4914, 0.036, -0.036]
RESET_JOINT_VALUES = [1.5691, -0.8474, 0.19, 0.0055, .5515, -1.54, 2.4914, 0.036, -0.036]


RESET_JOINT_VALUES_GRIPPER_CLOSED = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.015, -0.015]
RESET_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 7, 10, 11]
GUESS = 3.14  # TODO(avi) This is a guess, need to verify what joint this is
JOINT_LIMIT_LOWER = [-3.14, -1.88, -1.60, -3.14, -2.14, -3.14, -GUESS, 0.015,
                     -0.037]
JOINT_LIMIT_UPPER = [3.14, 1.99, 2.14, 3.14, 1.74, 3.14, GUESS, 0.037, -0.015]
JOINT_RANGE = []
for upper, lower in zip(JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER):
    JOINT_RANGE.append(upper - lower)

GRIPPER_LIMITS_LOW = JOINT_LIMIT_LOWER[-2:]
GRIPPER_LIMITS_HIGH = JOINT_LIMIT_UPPER[-2:]
GRIPPER_OPEN_STATE = [0.036, -0.036]
GRIPPER_CLOSED_STATE = [0.015, -0.015]

ACTION_DIM = 8


class Widow250EnvROS(gym.Env):
    real_robot = False

    def __init__(self, observation_mode='state', observation_img_dim=48,
                 reward_type='ee_position', xyz_action_scale=0.2, abc_action_scale=-0.01,
                 ee_pos_low=(0, -0.3, 0.06), ee_pos_high=(0.6, 0.3, 0.50), gui=False, rs_address=None,
                 ee_distance_threshold=0.1, robot_model='wx250s'):

        self.ee_distance_threshold = ee_distance_threshold
        self.observation_mode = observation_mode
        self.observation_img_dim = observation_img_dim

        self.reward_type = reward_type

        self.base_position = [0, 0, 0]

        self.gui = gui
        self.done = False

        self.ee_pos_high = ee_pos_high
        self.ee_pos_low = ee_pos_low

        self.reset_joint_values = RESET_JOINT_VALUES
        self.reset_joint_indices = RESET_JOINT_INDICES

        self.xyz_action_scale = xyz_action_scale
        self.abc_action_scale = abc_action_scale

        self._set_action_space()
        self._set_observation_space()

        self.is_gripper_open = True

        self.rs_state = None
        self.joint_positions = {}
        self.arm_min_radius = 0.100
        self.ee_target_pose = None
        self.ee_pos = np.array([0, 0, 0])
        self.ee_quat = np.array([0, 0, 0, 1])
        self.previous_position = None
        if rs_address:
            self.client = rs_client.Client(rs_address)
        self.robogym = bullet.RoboGymUtils(robot_model, client=self.client)


        self.reset()

    def reset(self, target=None, seed=None, options=None):
        if target:
            assert len(target) == 6
            self.ee_target_pose = target
        else:
            self.ee_target_pose = self.robogym.get_target_pose(self.ee_pos_low, self.ee_pos_high)

        joint_positions = RESET_JOINT_VALUES[:-3]

        # Initialize environment state
        rs_state = dict.fromkeys(self.robogym.get_robot_server_composition(), 0.0)

        # Set initial robot joint positions
        self.joint_positions = self.robogym.set_joint_positions(joint_positions)

        # Update joint positions in rs_state
        rs_state.update(self.joint_positions)

        # Set initial state of the Robot Server
        state_msg = self.robogym.set_initial_robot_server_state(rs_state)

        if not self.robogym.client.set_state_msg(state_msg):
            raise RuntimeError("set_state")

        return self.get_observation(), self.get_info()

    def step(self, action):
        if np.isnan(np.sum(action)):
            print('action', action)
            raise RuntimeError('Action has NaN entries')

        action = np.clip(action, -1, +1)  # TODO Clean this up

        xyz_action = action[:3]  # ee position actions
        abc_action = action[3:6]
        # np.array([0.0003, 0.0003, 0.0003])

        # Get Robot Server state
        rs_state = self.robogym.client.get_state_msg().state_dict

        # Check if the length and keys of the Robot Server state received is correct
        self.robogym.check_rs_state_keys(rs_state, self.ee_target_pose)

        # Convert the initial state from Robot Server format to environment format
        ee_pos, ee_rot = self.robogym.robot_server_state_to_env_state(rs_state)

        target_ee_pos = ee_pos + self.xyz_action_scale * xyz_action
        target_ee_deg = ee_rot + self.abc_action_scale * abc_action

        target_ee_pose = np.append(target_ee_pos, target_ee_deg)

        target_joints, solution_found = self.robogym.interbotix_utils.inverse_kinematics(target_ee_pose, custom_guess=self.robogym.joint_positions)
        
        check_inv_kin = self.robogym.interbotix_utils.forward_kinematics(target_joints)
        check_again = self.robogym.interbotix_utils.forward_kinematics(self.robogym.joint_positions)

        if solution_found:
            # Send action to Robot Server and get state
            rs_state = self.robogym.client.send_action_get_state(target_joints).state_dict
            self.robogym.check_rs_state_keys(rs_state, self.ee_target_pose)

            # Convert the state from Robot Server format to environment format
            ee_pose, ee_rot = self.robogym.robot_server_state_to_env_state(rs_state)

            self.rs_state = rs_state

        info = self.get_info()
        reward = self.get_reward(info)
        done = self.done
        truncated = False
        return self.get_observation(), reward, done, truncated, info

    def get_observation(self):
        gripper_state = GRIPPER_OPEN_STATE
        gripper_binary_state = [float(self.is_gripper_open)]
        # Get Robot Server state
        rs_state = self.robogym.client.get_state_msg().state_dict

        # Check if the length and keys of the Robot Server state received is correct
        self.robogym.check_rs_state_keys(rs_state, self.ee_target_pose)

        # Convert the initial state from Robot Server format to environment format
        self.ee_pos, ee_rot = self.robogym.robot_server_state_to_env_state(rs_state)
        self.ee_quat = bullet.deg_to_quat(ee_rot)

        object_position = np.array(self.ee_target_pose)
        object_orientation = np.array([0, 0, 0, 1])

        if self.observation_mode == 'state':
            observation = {
                'object_position': object_position,
                'object_orientation': object_orientation,
                'state': np.concatenate(
                    (self.ee_pos, self.ee_quat, gripper_state, gripper_binary_state)),
            }
        else:
            observation = {
                'object_position': object_position,
                'object_orientation': object_orientation,
                'state': np.concatenate(
                    (self.ee_pos, self.ee_quat, gripper_state, gripper_binary_state)),
            }

        return observation

    def get_reward(self, info):
        if self.reward_type == 'ee_position':
            reward = 0
            # Reward weight for reaching the goal position
            g_w = 1000
            # Reward weight according to the distance to the goal
            d_w = 10

            # Reward base
            reward += np.exp(-d_w * info['euclidean_distance'])

            if info['ee_pose_success']:
                reward = g_w * 1
                self.done = True
        else:
            raise NotImplementedError

        return reward

    def get_info(self):
        info = {}
        target_coord = np.array(self.ee_target_pose[:3])
        ee_coord = np.array(self.ee_pos)
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)
        if euclidean_dist_3d <= self.ee_distance_threshold:
            info['ee_pose_success'] = True
            info['target_coord'] = self.ee_target_pose
        else:
            info['ee_pose_success'] = False

        info['euclidean_distance'] = euclidean_dist_3d
        info['target_coord'] = self.ee_target_pose

        return info

    def _set_action_space(self):
        self.action_dim = ACTION_DIM
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (self.observation_img_dim ** 2) * 3
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)
            robot_state_dim = 10  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            object_position = gym.spaces.Box(-np.ones(3), np.ones(3))
            object_orientation = gym.spaces.Box(-np.ones(4), np.ones(4))
            spaces = {'image': img_space, 'state': state_space, 'object_position': object_position,
                      'object_orientation': object_orientation}
            self.observation_space = gym.spaces.Dict(spaces)
        elif self.observation_mode == 'state':
            robot_state_dim = 10  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            object_position = gym.spaces.Box(-np.ones(3), np.ones(3))
            object_orientation = gym.spaces.Box(-np.ones(4), np.ones(4))
            spaces = {'state': state_space, 'object_position': object_position,
                      'object_orientation': object_orientation}
            self.observation_space = gym.spaces.Dict(spaces)

        else:
            robot_state_dim = 10  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            object_position = gym.spaces.Box(-np.ones(3), np.ones(3))
            object_orientation = gym.spaces.Box(-np.ones(4), np.ones(4))
            spaces = {'state': state_space, 'object_position': object_position,
                      'object_orientation': object_orientation}
            self.observation_space = gym.spaces.Dict(spaces)


class Widow250EnvRosASim(Widow250EnvROS, Simulation):
    cmd = "roslaunch interbotix_arm_robot_server interbotix_arm_robot_server.launch \
        world_name:=tabletop_sphere50_no_collision.world \
        max_velocity_scale_factor:=0.1 \
        action_cycle_rate:=10 \
        rviz_gui:=false \
        gui:=true \
        gazebo_gui:=true \
        objects_controller:=true \
        rs_mode:=1object \
        n_objects:=1.0 \
        object_0_model_name:=sphere50_no_collision \
        object_0_frame:=target"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, robot_model='rx150', **kwargs):
        self.cmd = self.cmd + ' ' + 'robot_model:=' + robot_model + ' reference_frame:=' + robot_model + "/base_link"
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Widow250EnvROS.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class Widow250EnvROSARob(Widow250EnvROS):
    real_robot = True


if __name__ == "__main__":
    # env = Widow250EnvRosASim(gui=True, ip='127.0.0.1', robot_model='wx250s')
    import time
    env = Widow250EnvROSARob(rs_address='192.168.1.101:50051')
    # STATE = [0., 0., 0., 0, 0., 0., 0., 0., 0.]
    env.reset()
    # env.step(np.asarray(STATE))
    time.sleep(0.1)
    # env.reset()
