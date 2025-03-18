#!/usr/bin/env python3

import open3d as o3d
import numpy as np

class Global_map_manager:
    def __init__(self)->None:
        self._global_map = o3d.geometry.PointCloud()
        self._voxel_size = 0.5
    
    def append_frame_cloud(self, frame_cloud: np.ndarray)->None:
        contiguous_frame_cloud = np.ascontiguousarray(frame_cloud[:, :3])
        frame_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(contiguous_frame_cloud))
        self._global_map += frame_cloud
        # downsample the global map
        self._global_map = self._global_map.voxel_down_sample(voxel_size=self._voxel_size)

        return self._global_map
    
    def numpy_global_map(self)->np.ndarray:
        return np.asarray(self._global_map.points)