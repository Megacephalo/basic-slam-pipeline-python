#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import open3d as o3d


def save_to_pcd(cloud: np.ndarray, file_path: Path)->None:
    '''
    Save the point cloud to the specified PCD file
    Args:
        pcd: np.ndarray: The point cloud to save
        file_path: Path: The file path to save the point cloud
    '''
    if not isinstance(cloud, np.ndarray):
        raise ValueError('Invalid point cloud format')
    if cloud.shape[1] < 3:
        raise ValueError('Invalid point cloud format')
    if not Path(file_path).exists():
        Path(file_path).touch()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    o3d.io.write_point_cloud(str(file_path), pcd)

def save_to_csv(trajectory: np.ndarray, file_path: Path)->None:
    '''
    Save the trajectory to the specified CSV file
    Args:
        trajectory: np.ndarray: The trajectory to save
        file_path: Path: The file path to save the trajectory
    '''
    if not isinstance(trajectory, np.ndarray):
        raise ValueError('Invalid trajectory format')
    if trajectory.shape[1] != 4:
        raise ValueError('Invalid trajectory format')
    if not Path(file_path).exists():
        Path(file_path).touch()

    np.savetxt(file_path, trajectory, delimiter=',')