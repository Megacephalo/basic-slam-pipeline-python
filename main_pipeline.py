#!/usr/bin/env python3

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# Dataset parsing
from data_processing.data_parser_interface import Data_Parser_Interface
from data_processing.kitti_velodyne_dataset_parser import KITTI_Velodyne_Dataset_Parser

# Scan matching
from scan_matching.scan_matcher_interface import Scan_matcher_interface
from data_processing.coordinate_transformations import voxelize_cloud, transform_cloud
from scan_matching.small_gicp import Small_gicp
from scan_matching.my_gicp import My_GICP
from scan_matching.global_map_manager import Global_map_manager

# Pose graph optimization
from utils.coordinate_transformations import transformation_amtrix_to_xyzypr
from pose_graph.pose_graph_manager import Pose_graph_Manager

# visualizers
from visualization.open3d_visualizer import PointCloudVisualizer
from visualization.pangolin_visualizer import Pangolin_visualizer, RED

# Metriccs and evaluation
from metrics_and_evaluation.load_in_ground_truth import Load_Ground_Truth
from metrics_and_evaluation.save_to_files import save_to_csv, save_to_pcd

def parse_arguments()->argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f'{Path(__file__).stem}', description='A simple tool to showcase SLAM pipeline from input dataset')
    parser.add_argument('-i', '--input_dataset', required=True, type=str, help='The KITTI dataset directory')
    parser.add_argument('-g', '--ground_truth', type=str, help='The ground truth poses file')
    parser.add_argument('-u', '--up_to_frame', type=int, default=-1, help='Specify up to which point cloud frame to run the pipeline')
    parser.add_argument('-v', '--voxel_size', type=float, default=0.5, help='The voxel size for downsampling the point cloud')
    parser.add_argument('-sm', '--save_map_to', type=str, help='Save the generated map to the specified PCD file')
    parser.add_argument('-st', '--save_trajectory_to', type=str, help='Save the generated trajectory to the specified CSV file')
    return parser.parse_args()

if __name__=='__main__':
    print('Welcome to basic LiDAR odometry...')

    args = parse_arguments()

    dataset_dir = Path(args.input_dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f'{dataset_dir} is NOT found')
    
    if not dataset_dir.is_dir():
        raise ValueError(f'{dataset_dir} is NOT a directory')
    
    # Parse the point clouds
    bin_files = [file for file in dataset_dir.iterdir() if file.suffix == '.bin']
    bin_files = sorted(bin_files)

    print(50 * '-')
    print(f'Reading dataset from {args.input_dataset}')
    if args.ground_truth is not None:
        print(f'Ground truth poses are loaded from {args.ground_truth}')

        gt_loader = Load_Ground_Truth(poses_file=args.ground_truth)
        ground_truth_poses = gt_loader.load_ground_truth()
        if ground_truth_poses is None:
            print('Failed to load the ground truth poses')
        else:
            print(f'Loaded {len(ground_truth_poses)} ground truth poses')

    if args.save_map_to is not None:
        print(f'Trajectory will be saved to {args.save_map_to}')
    
    if args.save_trajectory_to is not None:
        print(f'Trajectory will be saved to {args.save_trajectory_to}')

    print(f'There are in total {len(bin_files)} point cloud files')
    if args.up_to_frame > 0:
        print(f'Running the pipeline up to frame {args.up_to_frame}')
    print('Parsing the dataset...')

    dataset_parser: Data_Parser_Interface = KITTI_Velodyne_Dataset_Parser()
    clouds = []
    if args.up_to_frame > 0:
        clouds = dataset_parser.parse_in_batch(bin_files[:args.up_to_frame])
    else:
        clouds = dataset_parser.parse_in_batch(bin_files)

    print(f'Successfully parsed {len(clouds)} frames')
    print(50 * '-')

    print('Run scan matching...')
    scan_matcher: Scan_matcher_interface = Small_gicp()
    # scan_matcher: Scan_matcher_interface = My_GICP()

    scan_matcher._voxel_size = args.voxel_size

    T_world_lidar = np.identity(4)
    trajectory = [T_world_lidar]
    prev_cloud: np.ndarray = None

    map_manager = Global_map_manager()
    pg_manager = Pose_graph_Manager()

    viz = Pangolin_visualizer()

    try:
        for frame_idx, cloud in tqdm( enumerate(clouds), 
                                      desc='Processing frames', 
                                      unit='frame', 
                                      total=len(clouds) ):
            voxelized_cloud = voxelize_cloud(cloud[:, :3], args.voxel_size)

            ground_truth_poses = None if args.ground_truth is None else ground_truth_poses

            if frame_idx == 0:
                scan_matcher.set_target(voxelized_cloud)
                pg_manager.add_prior_pose(T_world_lidar)

                # Scan-to-scan
                prev_cloud = voxelized_cloud

                map_manager.append_frame_cloud(voxelized_cloud)

                viz.draw_frame_cloud(pointcloud=voxelized_cloud, trajectory=trajectory, at_idx=frame_idx, gt_poses=ground_truth_poses)

                map_manager.append_frame_cloud(voxelized_cloud)
                continue

            scan_matcher.set_source(voxelized_cloud)
            scan_matcher.set_target(prev_cloud)
            T_world_lidar = scan_matcher.estimate()

            pg_manager.add_odom_optimizeretry_measurement(trajectory[-1], T_world_lidar)

            trajectory.append(T_world_lidar)

            frame_cloud = transform_cloud(voxelized_cloud[:, :3], T_world_lidar)

            # scan-to-scan
            prev_cloud = voxelized_cloud

            viz.draw_frame_cloud(pointcloud=frame_cloud, trajectory=trajectory, at_idx=frame_idx, gt_poses=ground_truth_poses)

    except Exception as err:
        print(f'Error: {err}')
    

    # Do one global trajectory optimization
    print(50 * '*')
    print('Optimizing the trajectory...')
    print(50 * '*')
    optimized_trajectory = pg_manager.optimize()
    if optimized_trajectory is None:
        print('No optimized trajectory found')
        optimized_trajectory = trajectory
        print('Using the original trajectory')
    else:
        for (curr_T, raw_cloud) in tqdm(list(zip(optimized_trajectory, clouds)), 
                                        desc='Optimizing trajectory', 
                                        unit='frame'):
            # voxelized_cloud = voxelize_cloud(raw_cloud[:, :3], args.voxel_size)
            # frame_cloud = transform_cloud(voxelized_cloud[:, :3], curr_T)
            frame_cloud = transform_cloud(raw_cloud[:, :3], curr_T)
            map_manager.append_frame_cloud(frame_cloud)
        print('Optimized trajectory:')
    print(50 * '-')

    # Visualize the global map
    print('Visualizing the global map...')
    viz.hold_on_one_frame(map_manager.numpy_global_map(), optimized_trajectory, len(optimized_trajectory) - 1, gt_poses=ground_truth_poses)

    if args.save_map_to is not None:
        print('Save to PCD file...')
        save_to_pcd(map_manager.numpy_global_map(), file_path=Path(args.save_map_to))
        print(f'Successfully saved the map to {args.save_map_to}')
    
    if args.save_trajectory_to is not None:
        print('Save trajectory to CSV file...')
        save_to_csv(np.array(optimized_trajectory), file_path=Path(args.save_trajectory_to))
        print(f'Successfully saved the trajectory to {args.save_trajectory_to}')
        
    print('Done')
        



    