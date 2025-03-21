#!/usr/bin/env python3

from typing import List
import numpy as np
import OpenGL.GL as gl
import pypangolin as pangolin

from time import sleep
import math

##
# @brief Reference: https://gitlab.com/LinasKo/visual-slam-practice/-/blob/master/src/pangolin_plotter.py?ref_type=heads

WINDOW_WIDTH = 2560
WINDOW_HEIGHT = 1440
UI_WIDTH = 180
LINE_WIDTH = 3

# Color
WHITE = (1.0, 1.0, 1.0)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
YELLOW = (1.0, 1.0, 0.0)
CYAN = (0.0, 1.0, 1.0)
MAGENTA = (1.0, 0.0, 1.0)
BLACK = (0.0, 0.0, 0.0)


class Pangolin_visualizer:
    def __init__(self)->None:
        # Create a window
        pangolin.CreateWindowAndBind('Visualizer', WINDOW_WIDTH, WINDOW_HEIGHT)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(
                WINDOW_WIDTH, WINDOW_HEIGHT, 
                WINDOW_WIDTH/2, WINDOW_HEIGHT/2,  # Principal point at center of window
                WINDOW_WIDTH/2, WINDOW_HEIGHT/2,  # Center of the window
                0.1, 1000
            ),
            pangolin.ModelViewLookAt(0, 0, 10, 0, 0, 0, pangolin.AxisDirection.AxisY)
        )
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = (
            pangolin.CreateDisplay()
            .SetBounds(
                pangolin.Attach(0.0),
                pangolin.Attach(1.0),
                pangolin.Attach.Pix(UI_WIDTH),
                pangolin.Attach(1.0),
                -WINDOW_WIDTH / WINDOW_HEIGHT,
            )
            .SetHandler(self.handler)
        )

        # Store the last user-adjusted view
        self._user_view_matrix: np.ndarray = None
    
    def _update_camera_position(self, current_frame_pose:np.ndarray, follow_mode: bool=True)->None:
        """
        Update the camera position to follow the current frame's pose.
        
        Args:
            current_frame_pose: The current frame's pose matrix (4x4 SE3 transformation)
            follow_mode: If True, follow the camera pose; if False, keep user view
        """
        # If follow_mode is disabled or user has adjusted the view, preserve their view
        if not follow_mode and self._user_view_matrix is not None:
            self._scam.SetModelViewMatrix(self._user_view_matrix)
            return
    
        # Extract camera position from the pose
        cam_pos = current_frame_pose[:3, 3]  # Translation component
        
        # Extract camera orientation
        cam_forward = -current_frame_pose[:3, 2]  # Negative Z-axis of the camera
        cam_up = -current_frame_pose[:3, 1]       # Negative Y-axis of the camera
        
        # Calculate the look-at point (5 units in front of the camera)
        look_at_point = cam_pos + 5 * cam_forward
        
        # Position the visualization camera slightly above the actual camera
        view_pos = cam_pos + np.array([0, 0, 100])  # 100 units above
        
        # Update the OpenGL camera state
        self.scam.SetModelViewMatrix(
            pangolin.ModelViewLookAt(
                view_pos[0], view_pos[1], view_pos[2],      # Camera position
                look_at_point[0], look_at_point[1], look_at_point[2],  # Look-at point
                cam_up[0], cam_up[1], cam_up[2]             # Up vector
            )
        )
    
    # Add a method to capture user interactions
    def _check_user_interaction(self):
        """
        Check if user has manually adjusted the view, and if so, store their view.
        Call this method before updating with new frame data.
        """
        # Get the current model view matrix
        current_view = self.scam.GetModelViewMatrix()
        
        # Check if view has been changed by user input
        if self.handler.HasBeenDisplaced():
            self.user_view_matrix = current_view
            return True
        return False
    
    def _add_origin_axes(self)->None:
        gl.glLineWidth(1)
        gl.glBegin(gl.GL_LINES)
        # X axis in red
        gl.glColor3f(*RED)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(1.0, 0.0, 0.0)
        # Y axis in green
        gl.glColor3f(*GREEN)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 1.0, 0.0)
        # Z axis in blue
        gl.glColor3f(*BLUE)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 1.0)
        gl.glEnd()
    
    def _draw_camera_pose(self, pose:np.ndarray)->None:
        '''
        @brief Draw the camera pose as a set of axes
        @param pose: 4x4 transformation matrix representing the camera pose
        '''
        # Define the axis points in the camera's coordinate frame
        origin = np.array([0.0, 0.0, 0.0, 1.0])
        x_axis = np.array([1.0, 0.0, 0.0, 1.0])
        y_axis = np.array([0.0, 1.0, 0.0, 1.0])
        z_axis = np.array([0.0, 0.0, 1.0, 1.0])
        
        # Transform the points using the camera pose
        transformed_origin = pose @ origin
        transformed_x_axis = pose @ x_axis 
        transformed_y_axis = pose @ y_axis 
        transformed_z_axis = pose @ z_axis

        # Draw the axes
        gl.glLineWidth(2)
        gl.glBegin(gl.GL_LINES)
        # X axis in red
        gl.glColor3f(*RED)
        gl.glVertex3f(*transformed_origin[:3])
        gl.glVertex3f(*transformed_x_axis[:3])
        # Y axis in green
        gl.glColor3f(*GREEN)
        gl.glVertex3f(*transformed_origin[:3])
        gl.glVertex3f(*transformed_y_axis[:3])
        # Z axis in blue
        gl.glColor3f(*BLUE)
        gl.glVertex3f(*transformed_origin[:3])
        gl.glVertex3f(*transformed_z_axis[:3])
        gl.glEnd()
    
    def _draw_pointcloud(self, pointcloud:np.ndarray)->None:
        gl.glPointSize(1)
        gl.glColor3f(*WHITE)
        gl.glBegin(gl.GL_POINTS)
        for point in pointcloud:
            gl.glVertex3f(point[0], point[1], point[2])
        gl.glEnd()

    def _draw_poses(self, poses:np.ndarray, color: List[float]=GREEN)->None:
        gl.glLineWidth(2)
        gl.glColor3f(*color)
        gl.glBegin(gl.GL_LINE_STRIP)
        for pose in poses:
            # Extract the translation part of the pose
            p = pose[:3, 3]
            gl.glVertex3f(p[0], p[1], p[2])
        gl.glEnd()

    def draw_frame_cloud(self, pointcloud: np.ndarray, trajectory:list[np.ndarray], at_idx: int=-1, gt_poses: List[np.ndarray]=None)->None:
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Check if user has interacted with the view
        # user_interacted = self._check_user_interaction()
        # Update camera position, respecting user interactions if follow_mode is False
        self._update_camera_position(trajectory[at_idx])
        self.dcam.Activate(self.scam)

        # Draw the origin axes
        self._add_origin_axes()

        # Draw the pointcloud
        self._draw_pointcloud(pointcloud)
        
        if len(trajectory) > 0:
            self._draw_camera_pose(trajectory[at_idx])
            self._draw_poses(trajectory[:at_idx])
        
        # Draw ground truth poses
        if gt_poses is not None and len(gt_poses) > 0:
            self._draw_poses(gt_poses[:at_idx], color=RED)

        pangolin.FinishFrame()
        # sleep(0.1)

    def hold_on_one_frame(self, pointcloud: np.ndarray, trajectory: list[np.ndarray], at_idx: int, gt_poses: List[np.ndarray]=None)->None:
        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self.dcam.Activate(self.scam)

            # Draw the origin axes
            self._add_origin_axes()

            # Draw the pointcloud
            self._draw_pointcloud(pointcloud)

            # Draw the trajectory
            if len(trajectory) > 0:
                self._draw_camera_pose(trajectory[at_idx])
                self._draw_poses(trajectory[:at_idx])
            
            # Draw ground truth poses
            if gt_poses is not None and len(gt_poses) > 0:
                self._draw_poses(gt_poses[:at_idx], color=RED)

            # Finish the frame
            pangolin.FinishFrame()
            sleep(1)


    def draw_trajectory(self, trajectory: list[np.ndarray], up_to_idx:int = None)->None:
        if up_to_idx is None:
            up_to_idx = len(trajectory) - 1

        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            # Check if user has interacted with the view
            # user_interacted = self._check_user_interaction()
            self._update_camera_position(trajectory[up_to_idx])
            self.dcam.Activate(self.scam)

            # Draw the origin axes
            self._add_origin_axes()

            # Draw the trajectory
            gl.glLineWidth(2)
            gl.glColor3f(0.0, 1.0, 0.0)
            gl.glBegin(gl.GL_LINE_STRIP)
            for pose in trajectory:
                # Extract the translation part of the pose
                p = pose[:3, 3]
                gl.glVertex3f(p[0], p[1], p[2])
            gl.glEnd()

            # Finish the frame
            pangolin.FinishFrame()
            sleep(0.1)
    