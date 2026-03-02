#!/usr/bin/env python3

"""
This script shows RGB and depth feeds from the head camera of the Stretch robot.

Usage:
    python show_head_camera.py

Press 'q' to quit, 's' to save a frame.
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    print(f"Using device: {device.get_info(rs.camera_info.name)}")
    
    # Enable streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    print("Starting camera stream...")
    pipeline.start(config)
    
    # Create alignment object to align depth to color
    align = rs.align(rs.stream.color)
    
    try:
        print("Press 'q' to quit, 's' to save a frame")
        frame_count = 0
        
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Rotate both images 90 degrees clockwise
            color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
            depth_colormap = cv2.rotate(depth_colormap, cv2.ROTATE_90_CLOCKWISE)
            
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Show images
            cv2.imshow('RealSense - Color (Left) | Depth (Right)', images)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Press 'q' to quit
            if key == ord('q'):
                print("Quitting...")
                break
            
            # Press 's' to save the current frame
            elif key == ord('s'):
                cv2.imwrite(f'color_frame_{frame_count}.png', color_image)
                cv2.imwrite(f'depth_frame_{frame_count}.png', depth_colormap)
                print(f"Saved frame {frame_count}")
                frame_count += 1
    
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")

if __name__ == "__main__":
    main()