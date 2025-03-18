#!/usr/bin/env python3

import numpy as np
import argparse
from pathlib import Path
import time

# Dataset parsing
from data_processing.data_parser_interface import Data_Parser_Interface
from data_processing.kitti_velodyne_dataset_parser import KITTI_Velodyne_Dataset_Parser

# Scan matching
from scan_matching.scan_matcher_interface import Scan_matcher_interface
from data_processing.coordinate_transformations import voxelize_cloud, transform_cloud
from scan_matching.small_gicp import Small_gicp
from scan_matching.my_gicp import My_GICP
from scan_matching.global_map_manager import Global_map_manager

# visualizers
from visualization.open3d_visualizer import PointCloudVisualizer
from visualization.pangolin_visualizer import Pangolin_visualizer

def parse_arguments()->argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f'{Path(__file__).stem}', description='A simple tool to showcase SLAM pipeline from input dataset')
    parser.add_argument('-i', '--input_dataset', required=True, type=str, help='The KITTI dataset directory')
    parser.add_argument('-u', '--up_to_frame', type=int, default=-1, help='Specify up to which point cloud frame to run the pipeline')
    parser.add_argument('-v', '--voxel_size', type=float, default=0.5, help='The voxel size for downsampling the point cloud')
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
    # map_manager._voxel_size = args.voxel_size

    viz = Pangolin_visualizer()

    try:
        for frame_idx, cloud in enumerate(clouds):
            print(f'Processing frame {frame_idx + 1}...')
            voxelized_cloud = voxelize_cloud(cloud[:, :3], args.voxel_size)

            if frame_idx == 0:
                scan_matcher.set_target(voxelized_cloud)

                # Scan-to-scan
                prev_cloud = voxelized_cloud

                map_manager.append_frame_cloud(voxelized_cloud)

                viz.draw_frame_cloud(voxelized_cloud, trajectory, frame_idx)

                map_manager.append_frame_cloud(voxelized_cloud)
                continue

            scan_matcher.set_source(voxelized_cloud)
            scan_matcher.set_target(prev_cloud)
            T_world_lidar = scan_matcher.estimate()

            trajectory.append(T_world_lidar)

            frame_cloud = transform_cloud(voxelized_cloud[:, :3], T_world_lidar)
            
            map_manager.append_frame_cloud(frame_cloud)

            # scan-to-scan
            prev_cloud = voxelized_cloud

            viz.draw_frame_cloud(frame_cloud, trajectory, frame_idx)

    except Exception as err:
        print(f'Error: {err}')
    
    # Visualize the global map
    print('Visualizing the global map...')
    viz.hold_on_one_frame(map_manager.numpy_global_map(), trajectory, len(trajectory) - 1)
    # print('Done')
        



    