#!/usr/bin/env python3

import numpy as np

class Scan_matcher_interface:
    def __init__(self):
        pass

    def set_source(self, source: np.ndarray)->None:
        '''A setter method. Set the source point cloud'''
        pass

    def set_target(self, target: np.ndarray)->None:
        '''A setter method. Set the target point cloud'''
        pass

    def estimate(self)->np.ndarray:
        '''Do the scan matching and return the final transformation'''
        return np.identity(4)

    def get_final_transformation(self)->np.ndarray:
        '''A getter method. Return the final transformation'''
        return np.identity(4)