#!/usr/bin/env python3

import numpy as np
import open3d as o3d

from scan_matching.scan_matcher_interface import Scan_matcher_interface

class My_GICP(Scan_matcher_interface):
    def __init__(self):
        super().__init__()
        self._source = o3d.geometry.PointCloud()
        self._target = o3d.geometry.PointCloud()
        self._T_last_current = np.identity(4)
        self._T_world_lidar = np.identity(4)
        self._max_correspondence: float = 0.05
        self._max_iteration: int = 100
    
    def set_source(self, source: np.ndarray)->None:
        self._source.points = o3d.utility.Vector3dVector(source[:, :3])
    
    def set_target(self, target: np.ndarray)->None:
        self._target.points = o3d.utility.Vector3dVector(target[:, :3])
    
    def estimate(self)->np.ndarray:
        """
        Apply GICP to align source point cloud to target point cloud.
            
        Returns:
            transformation (np.ndarray): 4x4 transformation matrix
        """
        assert self._source is not None, 'Source point cloud is not set'
        assert self._target is not None, 'Target point cloud is not set'

        # Estimate normals for both point clouds (required for GICP)
        self._source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self._target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Apply GICP
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self._max_iteration)
        reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
            self._source, self._target, 
            self._max_correspondence,
            self._T_last_current,  # Initial transformation
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria
        )

        # Get transformation matrix
        self._T_last_current = reg_p2p.transformation
        # self._T_world_lidar = self._T_world_lidar @ self._T_last_current
        self._T_world_lidar = self._T_last_current
        return self._T_world_lidar
