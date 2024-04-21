import pybullet as p
import roboverse.bullet as bullet
import roboverse.bullet.control as control
import numpy as np


def get_ball_pos(ball_id):
    pos, quat = p.getBasePositionAndOrientation(ball_id)
    return pos

def get_plate_pos_quat(plate_id):
    pos, quat = p.getBasePositionAndOrientation(plate_id)
    return pos, quat

def get_distance_from_center(ball_pos, plate_pos, radius):
    distance = np.sqrt((ball_pos[0] - plate_pos[0])**2 + (ball_pos[1] - plate_pos[1])**2) - radius
    if distance < 0:
        distance = 0
    return distance