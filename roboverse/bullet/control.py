import pybullet as p
import numpy as np
import random
import modern_robotics as mr
import math


# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

_EPS = np.finfo(float).eps * 4.0


class uxarm5:
    Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.267, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.5515, 0.0, 0.0535],
                      [0.0, 1.0, 0.0, -0.209, 0.0, 0.131],
                      [0.0, 0.0, -1.0, 0.0, 0.207, 0.0]]).T

    M = np.array([[1.0, 0.0, 0.0, 0.207],
                  [0.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.112],
                  [0.0, 0.0, 0.0, 1.0]])

    Guesses = [[0, 0, -1.5708, 0, 0],
               [2.0944, 0, -1.5708, 0, 0],
               [-2.0944, 0, -1.5708, 0, 0]]


class uxarm6:
    Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.267, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.5515, 0.0, 0.0535],
                      [0.0, 0.0, -1.0, 0.0, 0.131, 0],
                      [0.0, 1.0, 0.0, -0.209, 0.0, 0.131],
                      [0.0, 0.0, -1.0, 0.0, 0.207, 0.0]]).T

    M = np.array([[1.0, 0.0, 0.0, 0.207],
                  [0.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.112],
                  [0.0, 0.0, 0.0, 1.0]])

    Guesses = [[0, 0, -1.5708, 0, 0, 0],
               [2.0944, 0, -1.5708, 0, 0, 0],
               [-2.0944, 0, -1.5708, 0, 0, 0]]


class uxarm7:
    Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.267, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0, 0.5515, 0.0, -0.0535],
                      [0.0, 0.0, -1.0, 0.0, 0.131, 0],
                      [0.0, 1.0, 0.0, -0.209, 0.0, 0.131],
                      [0.0, 0.0, -1.0, 0.0, 0.207, 0.0]]).T

    M = np.array([[1.0, 0.0, 0.0, 0.207],
                  [0.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.112],
                  [0.0, 0.0, 0.0, 1.0]])

    Guesses = [[0, 0, 0, 1.5708, 0, 0, 0],
               [2.0944, 0, 0, 1.5708, 0, 0, 0],
               [-2.0944, 0, 0, 1.5708, 0, 0, 0]]


def get_joint_states(body_id, joint_indices):
    all_joint_states = p.getJointStates(body_id, joint_indices)
    joint_positions, joint_velocities = [], []
    for state in all_joint_states:
        joint_positions.append(state[0])
        joint_velocities.append(state[1])

    return np.asarray(joint_positions), np.asarray(joint_velocities)


def get_movable_joints(body_id):
    num_joints = p.getNumJoints(body_id)
    movable_joints = []
    for i in range(num_joints):
        if p.getJointInfo(body_id, i)[2] != p.JOINT_FIXED:
            movable_joints.append(i)
    return movable_joints


def get_link_state(body_id, link_index):
    position, orientation, _, _, _, _ = p.getLinkState(body_id, link_index)
    return np.asarray(position), np.asarray(orientation)


def get_joint_info(body_id, joint_id, key):
    keys = ["jointIndex", "jointName", "jointType", "qIndex", "uIndex",
            "flags", "jointDamping", "jointFriction", "jointLowerLimit",
            "jointUpperLimit", "jointMaxForce", "jointMaxVelocity", "linkName",
            "jointAxis", "parentFramePos", "parentFrameOrn", "parentIndex"]
    value = p.getJointInfo(body_id, joint_id)[keys.index(key)]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return value


def apply_action_ik(target_ee_pos, target_ee_quat, target_gripper_state,
                    robot_id, end_effector_index, movable_joints,
                    lower_limit, upper_limit, rest_pose, joint_range,
                    num_sim_steps=5):
    joint_poses = p.calculateInverseKinematics(robot_id,
                                               end_effector_index,
                                               target_ee_pos,
                                               target_ee_quat,
                                               lowerLimits=lower_limit,
                                               upperLimits=upper_limit,
                                               jointRanges=joint_range,
                                               restPoses=rest_pose,
                                               jointDamping=[0.001] * len(
                                                   movable_joints),
                                               solver=0,
                                               maxNumIterations=100,
                                               residualThreshold=.01)

    p.setJointMotorControlArray(robot_id,
                                movable_joints,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_poses,
                                # targetVelocity=0,
                                forces=[5] * len(movable_joints),
                                positionGains=[0.03] * len(movable_joints),
                                # velocityGain=1
                                )
    # set gripper action
    p.setJointMotorControl2(robot_id,
                            movable_joints[-2],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[0],
                            force=500,
                            positionGain=0.03)
    p.setJointMotorControl2(robot_id,
                            movable_joints[-1],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[1],
                            force=500,
                            positionGain=0.03)

    for _ in range(num_sim_steps):
        p.stepSimulation()


def reset_robot(robot_id, reset_joint_indices, reset_joint_values):
    assert len(reset_joint_indices) == len(reset_joint_values)
    for i, value in zip(reset_joint_indices, reset_joint_values):
        p.resetJointState(robot_id, i, value)


def move_to_neutral(robot_id, reset_joint_indices, reset_joint_values,
                    num_sim_steps=75):
    assert len(reset_joint_indices) == len(reset_joint_values)
    p.setJointMotorControlArray(robot_id,
                                reset_joint_indices,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=reset_joint_values,
                                forces=[100] * len(reset_joint_indices),
                                positionGains=[0.03] * len(reset_joint_indices),
                                )
    for _ in range(num_sim_steps):
        p.stepSimulation()


def reset_object(body_id, position, orientation):
    p.resetBasePositionAndOrientation(body_id,
                                      position,
                                      orientation)


def get_object_position(body_id):
    object_position, object_orientation = \
        p.getBasePositionAndOrientation(body_id)
    return np.asarray(object_position), np.asarray(object_orientation)


def step_simulation(num_sim_steps):
    for _ in range(num_sim_steps):
        p.stepSimulation()


def compute_ik_pybullet(target_ee_pos, target_ee_quat, robot_id, end_effector_index, movable_joints, lower_limit,
                        upper_limit, rest_pose, joint_range):
    joint_poses = p.calculateInverseKinematics(robot_id,
                                               end_effector_index,
                                               target_ee_pos,
                                               target_ee_quat,
                                               lowerLimits=lower_limit,
                                               upperLimits=upper_limit,
                                               jointRanges=joint_range,
                                               restPoses=rest_pose,
                                               jointDamping=[0.001] * len(
                                                   movable_joints),
                                               solver=0,
                                               maxNumIterations=100,
                                               residualThreshold=.01)
    return joint_poses


def inverse_kinematics(ee_pose, dof, custom_guess=None):
    theta_list = []
    if dof == 4:
        robot_des = uxarm5()
    elif dof == 5:
        robot_des = uxarm6()
    else:
        robot_des = uxarm7()

    ee_transform = pose_to_transformation_matrix(ee_pose)
    if custom_guess is None:
        initial_guesses = robot_des.Guesses
    else:
        initial_guesses = [custom_guess]

    for guess in initial_guesses:
        theta_list, success = mr.IKinSpace(robot_des.Slist, robot_des.M, ee_transform, guess, 0.0001,
                                           0.0001)

        # Check to make sure a solution was found and that no joint limits were violated
        if success:
            solution_found = check_joint_limits(theta_list)
        else:
            solution_found = False

        if solution_found:
            return theta_list, True

    return theta_list, False


def forward_kinematics(positions, dof):
    if dof == 4:
        robot_des = uxarm5()
    elif dof == 5:
        robot_des = uxarm6()
    else:
        robot_des = uxarm7()

    joint_commands = list(positions)
    end_effector_pose = mr.FKinSpace(robot_des.M, robot_des.Slist, joint_commands)
    rpy = rotation_matrix_to_euler_angles(end_effector_pose[:3, :3])
    pose = end_effector_pose[:3, 3]
    return pose, rpy


def check_joint_limits(positions):
    theta_list = [int(elem * 1000) / 1000.0 for elem in positions]
    cntr = 0
    for name in self.joint_names:
        if not (self.limits[name]["lower"] <= theta_list[cntr] <= self.limits[name]["upper"]):
            return False
        cntr += 1
    return True


def pose_to_transformation_matrix(pose):
    mat = np.identity(4)
    mat[:3, :3] = euler_angles_to_rotation_matrix(pose[3:])
    mat[:3, 3] = pose[:3]
    return mat


def rotation_matrix_to_euler_angles(R):
    return list(euler_from_matrix(R, axes="sxyz"))


def euler_from_matrix(matrix, axes='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_angles_to_rotation_matrix(theta):
    return euler_matrix(theta[0], theta[1], theta[2], axes="sxyz")[:3, :3]


def euler_matrix(ai, aj, ak, axes='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def quat_to_deg(quat):
    euler_rad = p.getEulerFromQuaternion(quat)
    euler_deg = rad_to_deg(euler_rad)
    return euler_deg


def deg_to_quat(deg):
    rad = deg_to_rad(deg)
    quat = p.getQuaternionFromEuler(rad)
    return quat


def deg_to_rad(deg):
    return np.array([d * np.pi / 180. for d in deg])


def rad_to_deg(rad):
    return np.array([r * 180. / np.pi for r in rad])


def get_random_workspace_pose(min_pose, max_pose, min_r):
    """Get pose of a random point in the robot workspace.

    Returns:
        np.array: [x,y,z] pose.

    """
    singularity_area = True
    x = y = z = 0

    # check if generated x,y,z are in singularity area
    while singularity_area:

        x = round(random.uniform(min_pose[0], max_pose[0]), 1)
        y = round(random.uniform(min_pose[1], max_pose[1]), 1)
        z = round(random.uniform(min_pose[2], max_pose[2]), 1)

        if (x ** 2 + y ** 2) > min_r ** 2:
            singularity_area = False

    pose = [x, y, z]

    return pose