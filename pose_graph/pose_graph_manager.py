#!/usr/bin/env python3

import numpy as np
import gtsam
from typing import Tuple, List

class Pose_graph_Manager:
    def __init__(self)->None:
        self._params = gtsam.ISAM2Params()
        self._params.setRelinearizeThreshold(0.01)
        self._params.relinearizeSkip = 1

        # Create ISAM2 optimzer
        self._isam = gtsam.ISAM2(self._params)

        # Initialize pose graph
        self._graph = gtsam.NonlinearFactorGraph()

        # Initial estimate
        self._initial_estimate = gtsam.Values()

        # Prior noise model
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

        # odom_optimizeretry noise model
        self._odom_optimizeretry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))

    def add_prior_pose(self, pose: np.ndarray)->None:
        '''
        Add a prior pose to the graph.
        :param pose: The pose to add. The pose is actually a 4x4 transformation matrix.
        '''
        # convert numpy array to Pose3
        gtsa_pose = self._to_gtsam_pose(pose)

        # Add prior factor
        self._graph.add(gtsam.PriorFactorPose3(
            0,                      # Node index
            gtsa_pose,              # Pose3 object
            self._prior_noise       # Noise model
        ))

        # Add initial estimate
        self._initial_estimate.insert(0, gtsa_pose)

    def add_odom_optimizeretry_measurement(self, from_pose: np.ndarray, to_pose: np.ndarray)->None:
        '''
        Add odom_optimizeretry measurement to the graph.

        :param from_pose: Starting pose as 4x4 transformation matrix.
        :param to_pose: Edning pose as 4x4 transformation matrix.
        '''
        assert from_pose.shape == (4, 4), "From pose must be a 4x4 numpy array"
        assert to_pose.shape == (4, 4), "To pose must be a 4x4 numpy array"

        # calculate relative transform
        from_gtsam_pose = self._to_gtsam_pose(from_pose)
        to_gtsam_pose = self._to_gtsam_pose(to_pose)

        # Complete relative odom_optimizeretry
        relative_pose = from_gtsam_pose.between(to_gtsam_pose)

        # Add betwee factor (odom_optimizeretry constraint)
        current_node = self._initial_estimate.size()
        self._graph.add(gtsam.BetweenFactorPose3(
            current_node - 1,    # From node
            current_node,        # To node
            relative_pose,       # Relative pose
            self._odom_optimizeretry_noise  # Noise model
        ))

        # Add initial estimat for the new pose
        self._initial_estimate.insert(current_node, to_gtsam_pose)

    def optimize(self)->List[Tuple[np.ndarray, np.ndarray]]:
        '''
        Optimize pose graph and returned the optimized poses
        :return: List of optimized poses as transformation matrices.
        '''
        # Check if graph is empty
        if self._graph.empty():
            print("Graph is empty. No optimization performed.")
            return []

        # Update ISAM2
        self._isam.update(self._graph, self._initial_estimate)

        # Clear for next iteration
        self._graph.resize(0)
        self._initial_estimate.clear()

        # Get optimized poses
        result = self._isam.calculateEstimate()

        optimized_poses = []
        for i in range(result.size()):
            pose = result.atPose3(i)
            curr_T = self._to_transformation_matrix(pose)

            optimized_poses.append(curr_T)

        return optimized_poses

    def _to_gtsam_pose(self, T: np.ndarray)->gtsam.Pose3:
        '''
        Convert a 4x4 transformation matrix to gtsam Pose3 object.
        :param T: 4x4 transformation matrix
        :return: gtsam Pose3 object
        '''
        assert T.shape == (4, 4), "Transformation matrix must be 4x4"
        
        translation = T[:3, 3]
        rotation = T[:3, :3]

        return gtsam.Pose3(
            gtsam.Rot3(rotation),
            gtsam.Point3(*translation)
        )

    def _to_transformation_matrix(self, gtsam_pose: gtsam.Pose3)->np.ndarray:
        '''
        Convert a GTSAM Pose3 object to a 4x4 transformation matrix.
        :param gtsam_pose: GTSAM Pose3 object
        :return: 4x4 transformation matrix
        '''
        assert isinstance(gtsam_pose, gtsam.Pose3), "Input must be a GTSAM Pose3 object"

        transformation_matrix = np.identity(4)

        rotation_matrix = gtsam_pose.rotation().matrix()
        transformation_matrix[:3, :3] = rotation_matrix

        transformation_matrix[:3, 3] = np.array([gtsam_pose.x(), gtsam_pose.y(), gtsam_pose.z()])

        return transformation_matrix

# Test and saple code
if __name__ == "__main__":
    odom_optimizer = Pose_graph_Manager()

#     # Add initial prior pose
    initial_pose = np.identity(4)
    odom_optimizer.add_prior_pose(initial_pose)

    # Simulate odom_optimizeretry measurements
    initial_guesses = [
        np.array([[1, 0, 0, 1],
                  [0, 1, 0, 2],
                  [0, 0, 1, 3],
                  [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 2],
                  [0, 1, 0, 3],
                  [0, 0, 1, 4],
                  [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 3],
                  [0, 1, 0, 4],
                  [0, 0, 1, 5],
                  [0, 0, 0, 1]])
    ]

    prev_pose = initial_pose
    for pose in initial_guesses:
        odom_optimizer.add_odom_optimizeretry_measurement(prev_pose, pose)
        prev_pose = pose

    # optimize and get results
    optimized_trajectory = odom_optimizer.optimize()
    
    if not optimized_trajectory:
        print("No optimized trajectory found.")
        quit()

    print(f'Optimized trajectory:')
    for i, curr_T in enumerate(optimized_trajectory):
        print(f'Frame {i} T =\n{curr_T}')