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

        self._T_last_current: np.ndarray = np.identity(4)
        self._T_world_lidar: np.ndarray = np.identity(4)
        self._target_state: Tuple[np.ndarray, small_gicp.KdTree] = None

    def set_source(self, source: np.ndarray)->None:
        self._source_cloud = source[:, :3]
    
    def set_target(self, target)->None:
        self._target_cloud = target[:, :3]
    

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

        self._T_last_current = result.T_target_source
        self._T_world_lidar = self._T_world_lidar @ result.T_target_source
        self._target_state = (downsampled, kd_tree)

        return self._T_world_lidar

    def get_final_transformation(self)->np.ndarray:
        return self._T_world_lidar