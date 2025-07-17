import cv2
from pathlib import Path
import os  
import argparse
import numpy as np

from utils.trans_disparity_2_depth import disparity_to_depth, transform_depth_to_frame, load_calibration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert disparity to depth")
    parser.add_argument("--disparity_dir", type=str, required=True, help="Root directory containing disparity images")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to save depth images")
    args = parser.parse_args()

    # Recursively process all subdirectories
    for root, dirs, files in os.walk(args.disparity_dir):
        # Check if the current directory contains a calibration file
        calibration_path = os.path.join(root, "calibration", "cam_to_cam.yaml")
        if not os.path.exists(calibration_path):
            print(f"Calibration file not found in {root}, skipping...")
            continue

        # Load calibration parameters
        f_event, B_event, T_10 = load_calibration(calibration_path)

        print(f"Processing directory: {root}")
        print(f"Focal length (f_event): {f_event}, Baseline (B_event): {B_event}")
        if f_event <= 0 or B_event <= 0:
            print(f"Invalid calibration parameters in {calibration_path}, skipping...")
            continue
        print(f"T_10 matrix:\n{T_10}")
        if T_10.shape != (4, 4):
            print(f"Invalid T_10 matrix in {calibration_path}, skipping...")
            continue

        # Prepare output directory for depth maps
        relative_path = os.path.relpath(root, args.disparity_dir)
        output_subdir = os.path.join(args.output_dir, relative_path, "depth")
        os.makedirs(output_subdir, exist_ok=True)

        # Process all disparity images in the current directory
        disparity_dir = os.path.join(root, "disparity", "event")
        if not os.path.exists(disparity_dir):
            print(f"Disparity directory not found in {root}, skipping...")
            continue

        for file_name in os.listdir(disparity_dir):
            disparity_path = os.path.join(disparity_dir, file_name)
            if not disparity_path.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # Filter non-image files
                continue

            # Load disparity map
            disparity_map = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0
            if disparity_map is None:
                print(f"Failed to load disparity map from {disparity_path}, skipping...")
                continue

            print(f"Processing {disparity_path}...")
            print(f"Disparity map min: {np.min(disparity_map)}, max: {np.max(disparity_map)}")
            if np.min(disparity_map) < 0:
                print(f"Disparity map contains negative values, skipping {disparity_path}...")
                continue

            # Convert disparity to depth
            depth_map_event = disparity_to_depth(disparity_map, f_event, B_event)
            if np.min(depth_map_event) < 0:
                print(f"Depth map contains negative values, skipping {disparity_path}...")
                continue

            # Transform depth to frame camera
            depth_map_frame = transform_depth_to_frame(depth_map_event, T_10)

            # Save depth map
            output_path = os.path.join(output_subdir, file_name)
            cv2.imwrite(output_path, (depth_map_frame * 256).astype(np.uint16))
            print(f"Saved depth map to {output_path}")