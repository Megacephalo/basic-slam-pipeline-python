#!/usr/bin/env python3

import numpy as np
from typing import Tuple, List

def Rot_to_ypr(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to yaw, pitch, roll angles.
    
    :param R: Rotation matrix (3x3)
    :return: Yaw, pitch, roll angles in radians
    """
    assert R.shape == (3, 3), "Rotation matrix must be 3x3"
    
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    return yaw, pitch, roll

def ypr_to_Rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Convert yaw, pitch, roll angles to rotation matrix.
    
    :param yaw: Yaw angle in radians
    :param pitch: Pitch angle in radians
    :param roll: Roll angle in radians
    :return: Rotation matrix (3x3)
    """
    assert isinstance(yaw, float) and isinstance(pitch, float) and isinstance(roll, float), "Angles must be floats"
    
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
    
    return R_yaw @ R_pitch @ R_roll

def transformation_amtrix_to_xyzypr(T: np.ndarray) -> np.ndarray:
    """
    Convert transformation matrix to x, y, z, yaw, pitch, roll.
    
    :param T: Transformation matrix (4x4)
    :return: a 6x1 numpy array containing x, y, z, yaw, pitch, roll
    """
    assert T.shape == (4, 4), "Transformation matrix must be 4x4"
    
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    
    R = T[:3, :3]
    yaw, pitch, roll = Rot_to_ypr(R)
    
    return np.array([x, y, z, yaw, pitch, roll]).reshape(6, 1)

def xyzypr_to_transformation_matrix(x: float, y: float, z: float, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Convert x, y, z, yaw, pitch, roll to transformation matrix.
    
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :param yaw: Yaw angle in radians
    :param pitch: Pitch angle in radians
    :param roll: Roll angle in radians
    :return: Transformation matrix (4x4)
    """
    assert isinstance(x, float) and isinstance(y, float) and isinstance(z, float), "Coordinates must be floats"
    
    R = ypr_to_Rot(yaw, pitch, roll)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    
    return T