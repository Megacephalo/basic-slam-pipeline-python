import numpy as np
import open3d as o3d
import time
import copy

COLOR_BLUE = [0, 0, 1]
COLOR_RED = [1, 0, 0]
COLOR_GREEN = [0, 1, 0]
COLOR_YELLOW = [1, 1, 0]
COLOR_WHITE = [1, 1, 1]
COLOR_BLACK = [0, 0, 0]

class PointCloudVisualizer:
    def __init__(self, window_name="Point Cloud Visualization", width=1280, height=720, 
                 background_color=[0.1, 0.1, 0.1]):
        """
        Initialize a visualizer for point clouds with trajectory tracking
        
        Parameters:
        -----------
        window_name : str
            Title of the visualization window
        width : int
            Width of the visualization window
        height : int
            Height of the visualization window
        background_color : list
            Background color as [R, G, B] with values between 0 and 1
        """
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)

        # Set background color
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt.point_size = 0.5

        # For trajectory tracking
        self.trajectory_points = []
        self.trajectory_lines = []
        self.trajectory_line_set = None
    
    def draw_cloud(self, cloud: np.ndarray)->None:
        """
        Draw a point cloud in the visualization window
        
        Parameters:
        -----------
        cloud : np.ndarray
            Point cloud data as Nx3 array
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
        self.vis.add_geometry(pcd)
        # self.vis.run()
    
    def hold(self)->None:
        """
        Run the visualization window in blocking mode
        """
        self.vis.run()
    
    def update_cloud(self, cloud: np.ndarray)->None:
        """
        Update the point cloud in the visualization window
        
        Parameters:
        -----------
        cloud : np.ndarray
            Point cloud data as Nx3 array
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
        self.vis.clear_geometries()
        self.vis.add_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def __del__(self):
        self.vis.destroy_window()

    def update_trajectory(self, latest_pose: np.ndarray)->None:
        '''Update the visualization of the ego vehicle's trajectory'''
        assert latest_pose.shape == (3, 1), 'Transformation matrix must be 3x1'
        self.trajectory_points.append(latest_pose[:3, 0])

        if len(self.trajectory_points) == 0:
            return
        
        if self.trajectory_line_set is None:
            # Create the line set
            self.trajectory_line_set = o3d.geometry.LineSet()
            self.vis.add_geometry(self.trajectory_line_set)
        
        # Update points
        self.trajectory_line_set.points = o3d.utility.Vector3dVector(self.trajectory_points)

        # Update lines (connect consecutive points)
        lines = [(i, i+1) for i in range(len(self.trajectory_points) - 1)]
        self.trajectory_line_set.lines = o3d.utility.Vector2iVector(lines)

        # Set line color green for trajectory)
        colors = [COLOR_GREEN for i in range(len(lines))]
        self.trajectory_line_set.colors = o3d.utility.Vector3dVector(colors)

        # update the geometry
        self.vis.update_geometry(self.trajectory_line_set)
        self.vis.poll_events()
        self.vis.update_renderer()


if __name__=='__main__':
    viz = PointCloudVisualizer()
    # viz.draw_cloud(np.random.rand(1000, 3))
    for i in range(100):
        viz.update_cloud(np.random.rand(1000, 3))
        time.sleep(0.1)