#!/usr/bin/env python3

import numpy as np
import small_gicp
from typing import Tuple, Any

from .scan_matcher_interface import Scan_matcher_interface

NUM_THREADS = 7
VOXEL_SIZE = 0.25

class Small_gicp(Scan_matcher_interface):
    def __init__(self):
        self._numThreads: int = NUM_THREADS
        self._voxel_size: float = VOXEL_SIZE
        # self._registration_type: str = 'VGICP' # Options: 'GICP', 'VGICP'
        # self._resolution: float = 1.0
        # self._max_correspondence_distance: float = 0.5
        # self._max_iterations: int = 200
        # self._source_cloud: np.ndarray = None
        # self._target_cloud: np.ndarray = None
        # self._T_world_lidar: np.ndarray = np.identity(4)
        # self._is_init: bool = False

        # For the second type of align()
        self._T_last_current: np.ndarray = np.identity(4)
        self._T_world_lidar: np.ndarray = np.identity(4)
        self._target_state: Tuple[np.ndarray, small_gicp.KdTree] = None

    def set_source(self, source: np.ndarray)->None:
        self._source_cloud = source[:, :3]
    
    def set_target(self, target)->None:
        self._target_cloud = target[:, :3]


    # def estimate(self)->np.ndarray:
    #     assert self._source_cloud is not None, 'Source point cloud is not set'
    #     assert self._target_cloud is not None, 'Target point cloud is not set'

    #     if not self._is_init:
    #         self._is_init = True
    #         return self._T_world_lidar

    #     result = small_gicp.align(
    #         target_points=self._target_cloud,
    #         source_points=self._source_cloud,
    #         init_T_target_source=self._T_world_lidar,
    #         registration_type=self._registration_type,
    #         voxel_resolution=self._resolution,
    #         max_correspondence_distance=self._max_correspondence_distance,
    #         max_iterations=self._max_iterations,
    #         num_threads=self._numThreads
    #     )

    #     # DEBUG
    #     # print(f'Result:\n{result}')
    #     print(result.num_inliers)
    #     # DEBUG

    #     self._T_world_lidar = self._T_world_lidar @ result.T_target_source

    #     return self._T_world_lidar
    

    def estimate(self)->np.ndarray:
        downsampled, kd_tree = small_gicp.preprocess_points(self._source_cloud, self._voxel_size, num_threads=self._numThreads)

        if self._target_state is None:
            self._target_state = (downsampled, kd_tree)
            return self._T_world_lidar
        
        result = small_gicp.align(
            self._target_state[0],
            downsampled,
            self._target_state[1],
            init_T_target_source=self._T_last_current,
            num_threads=self._numThreads
        )

        # DEBUG
        # print(f'Inliers: {result.num_inliers}')
        # DEBUG

        self._T_last_current = result.T_target_source
        self._T_world_lidar = self._T_world_lidar @ result.T_target_source
        self._target_state = (downsampled, kd_tree)

        return self._T_world_lidar