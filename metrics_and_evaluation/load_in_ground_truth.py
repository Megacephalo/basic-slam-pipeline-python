#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

def is_valid_file(file: str)->bool:
    if file is None or file == '':
        return False
    if not Path(file).exists():
        return False
    if not Path(file).is_file():
        return False
    return True

class Load_Ground_Truth:
    def __init__(self, poses_file: str=None)->None:
        self._poses_file = poses_file
    
    def set_poses_file(self, poses_file: str)->None:
        self._poses_file = poses_file

    def load_ground_truth(self, is_homogenous: bool=False)->List[np.ndarray]:
        '''
        Load the ground truth poses from the given file. The file contains rows of 3x4 matrices and this function will output
        a list of 4x4 transformation matrices (homogeneous coordinates)

        Returns:
            ground_truth_poses: List[np.ndarray]: List of 4x4 transformation matrices
        '''
        if not is_valid_file(self._poses_file):
            print('Invalid file')
            return None
        
        ground_truth_poses = []
        with open(self._poses_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                ground_truth_poses.append(np.array(line).reshape(3, 4).astype(np.float32))
                if is_homogenous:
                    ground_truth_poses[-1] = np.vstack((ground_truth_poses[-1], np.array([0, 0, 0, 1], dtype=np.float32)))
        return ground_truth_poses




# Test
# if __name__ == '__main__':
#     poses_file = Path('/home/charlyhuang/Documents/KITTI_odometry_datasete/data_odometry_poses/dataset/poses/00.txt')
#     if not is_valid_file(poses_file):
#         print('Invalid file')
#         exit()
    
#     gt_loader = Load_Ground_Truth(poses_file=poses_file)
#     ground_truth_poses: List[np.ndarray] = gt_loader.load_ground_truth()
#     print(ground_truth_poses[0])
#     print(ground_truth_poses[0].shape)

#     print('In homogenous form:')
#     ground_truth_poses: List[np.ndarray] = gt_loader.load_ground_truth(is_homogenous=True)
#     print(ground_truth_poses[0])
#     print(ground_truth_poses[0].shape)