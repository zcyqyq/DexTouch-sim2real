#!/usr/bin/env python3
"""
Intel RealSense Camera Headless Capture Script
Captures and saves color and depth frames without GUI
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import argparse
from datetime import datetime
import time
import os
import pickle
from termcolor import cprint
import torch

raw_extr_data = {
    "D455_dexhand": {
        "rvec": [ 0.02322675, -0.00136824,  0.0387247 ],
        "tvec": [-0.18066608, -0.18902881,  0.45980965],
        "aruco_2_world": {
            "rot_mat": [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, -1]
            ],
            "trans_vec": [0.0, 0.0, -47.0]
        },
        "world_2_franka_root":{
            "rot_mat":[
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ],
            "trans_vec": [-52.85, 25.55, 0.0]
        }
    }
}

def raw_data_to_extrinsics(cam_raw_data):
    rvec = np.array(cam_raw_data["rvec"], dtype=np.float32)
    tvec = np.array(cam_raw_data["tvec"], dtype=np.float32)
    rmat, jcob = cv2.Rodrigues(rvec)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rmat
    c2w[:3, 3] = tvec
    w2c = np.linalg.inv(c2w)
    return torch.tensor(w2c, dtype=torch.float32)


def compose_extrinsics(T_cam_A: np.ndarray,
                       R_A2B: np.ndarray,
                       t_A2B: np.ndarray) -> np.ndarray:
    if not isinstance(T_cam_A, np.ndarray):
        T_cam_A = np.array(T_cam_A, dtype=np.float32)
    if not isinstance(R_A2B, np.ndarray):
        R_A2B = np.array(R_A2B, dtype=np.float32)
    if not isinstance(t_A2B, np.ndarray):
        t_A2B = np.array(t_A2B, dtype=np.float32)
    assert T_cam_A.shape == (4,4)
    assert R_A2B.shape == (3,3) and t_A2B.shape == (3,)
    R_cam_A = T_cam_A[:3, :3]
    t_cam_A = T_cam_A[:3,  3]

    R_cam_B = R_A2B @ R_cam_A
    t_cam_B = R_A2B @ t_cam_A + t_A2B

    # 组装成 4x4 同态矩阵
    T_cam_B = np.eye(4, dtype=T_cam_A.dtype)
    T_cam_B[:3, :3] = R_cam_B
    T_cam_B[:3,  3] = t_cam_B

    return T_cam_B


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, name="D455_dexhand"):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.name = name
        cprint(f"Using camera: {name}, please check if it's the case!", 'red')
        
    def initialize(self):
        """Initialize the RealSense camera with specified settings"""
        try:
            # Configure streams
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start streaming
            self.profile = self.pipeline.start(self.config)
            
            # Create alignment object to align depth to color
            self.align = rs.align(rs.stream.color)
            
            # Get device information
            device = self.profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            print(f"Connected to: {device_name}")
            
            # Get depth scale
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth Scale: {self.depth_scale}")
            
            self.extrinsics = self.get_extrinsics()

            # Allow auto-exposure to stabilize
            print("Allowing auto-exposure to stabilize...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
                
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def capture_frame(self):
        """Capture and return aligned color and depth frames"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
            
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Create colorized depth image for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        return color_image, depth_image, depth_colormap

    def save_dexgraspnet2_meta(self, profile, depth_scale, output_dir, frame_id=0,
                                camera_pose=None, cam0_wrt_table=None):
        """
        Save RealSense intrinsics and camera pose in DexGraspNet2-compatible format.

        Args:
            profile: rs.pipeline_profile from RealSense pipeline
            depth_scale: float, depth scale from RealSense
            output_dir: str, where to save the files
            frame_id: int, current frame index (used in camera_poses.npy)
            camera_pose: np.ndarray shape (4, 4), optional camera pose matrix
            cam0_wrt_table: np.ndarray shape (4, 4), optional camera-to-table transform
        """
        os.makedirs(output_dir, exist_ok=True)

        # Extract intrinsics
        color_stream_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_stream_profile.as_video_stream_profile().get_intrinsics()

        K = [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ]

        meta = {
            'factor_depth': 1.0 / depth_scale,
            'intrinsic_matrix': K,
            'extrinsic_matrix': self.extrinsics,
        }

        # Save meta.pkl
        with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)

        print(f"Saved meta.pkl to {output_dir}")


        # Save camera_poses.npy (if provided)
        if camera_pose is not None:
            cam_poses_path = os.path.join(output_dir, 'camera_poses.npy')
            if os.path.exists(cam_poses_path):
                all_poses = np.load(cam_poses_path)
                all_poses = np.concatenate([all_poses, camera_pose[None, ...]], axis=0)
            else:
                all_poses = np.zeros((frame_id + 1, 4, 4))
                all_poses[frame_id] = camera_pose
            np.save(cam_poses_path, all_poses)
            print(f"Saved camera_poses.npy with {frame_id + 1} poses")

        # Save cam0_wrt_table.npy (optional)
        if cam0_wrt_table is not None:
            np.save(os.path.join(output_dir, 'cam0_wrt_table.npy'), cam0_wrt_table)
            print(f"Saved cam0_wrt_table.npy")

    def get_extrinsics(self):
        data = raw_extr_data[self.name]
        extrinsics = raw_data_to_extrinsics(data)
        aruco_2_world = data["aruco_2_world"]
        world_frame_extr = compose_extrinsics(extrinsics, aruco_2_world["rot_mat"], aruco_2_world["trans_vec"])
        return world_frame_extr
    

    def save_frames(self, color_image, depth_image, depth_colormap, output_dir=".", prefix="frame"):
        """Save color, depth data, and colorized depth images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Save color image
        color_filename = f"{output_dir}/{prefix}_color_{timestamp}.png"
        cv2.imwrite(color_filename, color_image)
        
        # Save raw depth data as 16-bit PNG (preserves actual depth values)
        depth_raw_filename = f"{output_dir}/{prefix}_depth_raw_{timestamp}.png"
        cv2.imwrite(depth_raw_filename, depth_image)
        
        # Save colorized depth for visualization
        depth_color_filename = f"{output_dir}/{prefix}_depth_color_{timestamp}.png"
        cv2.imwrite(depth_color_filename, depth_colormap)
        
        # Save depth statistics
        stats_filename = f"{output_dir}/{prefix}_stats_{timestamp}.txt"
        with open(stats_filename, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Resolution: {self.width}x{self.height}\n")
            f.write(f"Depth scale: {self.depth_scale}\n")
            f.write(f"Min depth: {np.min(depth_image[depth_image > 0])}mm\n")
            f.write(f"Max depth: {np.max(depth_image)}mm\n")
            f.write(f"Mean depth (non-zero): {np.mean(depth_image[depth_image > 0]):.2f}mm\n")
            
        return color_filename, depth_raw_filename, depth_color_filename, stats_filename
    
    def stop(self):
        """Stop the camera pipeline"""
        self.pipeline.stop()

def main():
    parser = argparse.ArgumentParser(description='Intel RealSense Headless Capture')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--count', type=int, default=1, help='Number of frames to capture')
    parser.add_argument('--interval', type=float, default=1.0, help='Interval between captures (seconds)')
    parser.add_argument('--output', type=str, default='.', help='Output directory')
    parser.add_argument('--prefix', type=str, default='frame', help='Filename prefix')
    parser.add_argument('--continuous', action='store_true', help='Run continuously until interrupted')
    
    args = parser.parse_args()
    
    # Initialize camera
    camera = RealSenseCamera(args.width, args.height, args.fps)
    save_dir = "./data/scenes/real_world"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    if not camera.initialize():
        print("Failed to initialize camera. Make sure Intel RealSense camera is connected.")
        sys.exit(1)

    print(f"\nCapturing frames to: {save_dir}")
    
    camera.save_dexgraspnet2_meta(
    profile=camera.profile,
    depth_scale=camera.depth_scale,
    output_dir=save_dir,
    frame_id=0
    )
    if args.continuous:
        print("Running in continuous mode. Press Ctrl+C to stop.")
    else:
        print(f"Will capture {args.count} frame(s) with {args.interval}s interval")
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            color_image, depth_image, depth_colormap = camera.capture_frame()
            
            if color_image is None or depth_image is None:
                print("Failed to capture frame")
                continue
            
            # Save frames
            color_file, depth_file, depth_color_file, stats_file = camera.save_frames(
                color_image, depth_image, depth_colormap, save_dir, args.prefix
            )
            
            frame_count += 1
            print(f"\nFrame {frame_count} saved:")
            print(f"  Color: {color_file}")
            print(f"  Depth (raw): {depth_file}")
            print(f"  Depth (color): {depth_color_file}")
            print(f"  Stats: {stats_file}")
            
            # Check if we should continue
            if not args.continuous and frame_count >= args.count:
                break
                
            # Wait for interval
            if args.continuous or frame_count < args.count:
                time.sleep(args.interval)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        camera.stop()
        print(f"\nTotal frames captured: {frame_count}")

if __name__ == "__main__":
    main()
