from typing import Tuple, Union, Any, List

import numpy as np
from numpy import ndarray, dtype
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.utils import interbotix_utils


class RoboGymUtils:
    def __init__(self, model, client=None):

        assert model in ["rx150", "wx250", "px150", "rx200", "vx250", "vx300", "wx200", "wx250", "px100", "vx300s", "wx250s"]


        self.joint_list = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                           'forearm_roll_joint_position', 'wrist_angle_joint_position',
                           'wrist_rotate_joint_position']
        self.joint_velocity_list = ['base_joint_velocity', 'shoulder_joint_velocity', 'elbow_joint_velocity',
                                    'forearm_roll_joint_velocity', 'wrist_angle_joint_velocity',
                                    'wrist_rotate_joint_velocity']
        self.dof = len(self.joint_list)
        self.interbotix_utils = interbotix_utils.InterbotixArm(model)
        self.joint_positions = []
        self.client = client

    def robot_server_state_to_env_state(self, rs_state) -> Tuple[Any, Any]:
        # Joint positions
        self.joint_positions = []
        joint_positions_keys = self.joint_list

        for position in joint_positions_keys:
            self.joint_positions.append(rs_state[position])
        joint_positions = np.array(self.joint_positions)

        ee_pose, ee_quat = self.interbotix_utils.forward_kinematics(joint_positions)

        return ee_pose, ee_quat

    def check_rs_state_keys(self, rs_state, target_pose) -> None:
        keys = self.get_robot_server_composition()

        if target_pose.any():
            rs_state['object_0_to_ref_translation_x'] = target_pose[0]
            rs_state['object_0_to_ref_translation_y'] = target_pose[0]
            rs_state['object_0_to_ref_translation_z'] = target_pose[0]
            rs_state['object_0_to_ref_rotation_x'] = 0
            rs_state['object_0_to_ref_rotation_y'] = 0
            rs_state['object_0_to_ref_rotation_z'] = 0
            rs_state['object_0_to_ref_rotation_w'] = 1

        # if not len(keys) == len(rs_state.keys()):
        #     raise RuntimeError("Robot Server state keys to not match. Different lengths.")

        for key in keys:
            if key not in rs_state.keys():
                raise RuntimeError("Robot Server state keys to not match")

    def set_initial_robot_server_state(self, rs_state) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}
        state = {}

        state_msg = robot_server_pb2.State(state=state, float_params=float_params,
                                           string_params=string_params, state_dict=rs_state)
        return state_msg

    def set_joint_positions(self, joint_positions) -> dict:
        # Set initial robot joint positions
        joint_positions_dict = {}
        for i in range(len(self.joint_list)):
            joint_positions_dict[self.joint_list[i]] = joint_positions[i]
        return joint_positions_dict

    def get_target_pose(self, min_ee=None, max_ee=None, box_ws=True) -> np.ndarray:
        """Generate target End Effector pose.

        Returns:
            list: [x,y,z] pose.

        """
        if max_ee is None:
            max_ee = []
        if min_ee is None:
            min_ee = []
        return self.interbotix_utils.get_random_workspace_pose(min_ee, max_ee, box_ws)

    @staticmethod
    def get_robot_server_composition() -> list:
        rs_state_keys = [
            'base_joint_position',
            'shoulder_joint_position',
            'elbow_joint_position',
            'forearm_roll_joint_position',
            'wrist_angle_joint_position',
            'wrist_rotate_joint_position',

            'base_joint_velocity',
            'shoulder_joint_velocity',
            'elbow_joint_velocity',
            'forearm_roll_joint_velocity',
            'wrist_angle_joint_velocity',
            'wrist_rotate_joint_velocity',

            'ee_to_ref_translation_x',
            'ee_to_ref_translation_y',
            'ee_to_ref_translation_z',
            'ee_to_ref_rotation_x',
            'ee_to_ref_rotation_y',
            'ee_to_ref_rotation_z',
            'ee_to_ref_rotation_w',

            'in_collision'
        ]

        return rs_state_keys