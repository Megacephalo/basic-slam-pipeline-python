#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import small_gicp

NUM_THREADS = 7
VOXEL_SIZE = 0.5

def voxelize_cloud(pointcloud: np.ndarray, voxel_size: float) -> np.ndarray:
    if pointcloud.shape[1] < 3:
        raise ValueError('The point cloud must have at least 3 columns')
    
    contiguous_pointcloud = np.ascontiguousarray(pointcloud[:, :3])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(contiguous_pointcloud)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return np.asarray(pcd.points)

def transform_cloud (pointcloud: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    '''
    @brief transform the point cloud with the transformation matrix
    @param pointcloud: np.ndarray The point cloud is of N x 3 dimensions whose schema is [x, y, z]
    @param transform_matrix: np.ndarray The transformation matrix is a 4 x 4 matrix
    '''
    points_homogenous = np.hstack([pointcloud, np.ones((pointcloud.shape[0], 1))])
    transformed_pointcloud_homogenous = points_homogenous @ transform_matrix.T
    transformed_pointcloud = transformed_pointcloud_homogenous[:, :3]

    return transformed_pointcloud