import cv2
from pathlib import Path
import os  
import argparse
import numpy as np

from utils.trans_disparity_2_depth import disparity_to_depth, transform_depth_to_frame, load_calibration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert disparity to depth")
    parser.add_argument("--disparity_dir", type=str, required=True, help="Directory containing disparity images")
    parser.add_argument("--calibration_path", type=str, required=True, help="Path to calibration YAML file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save depth images")
    args = parser.parse_args()

    # Load calibration parameters
    f_event, B_event, T_10 = load_calibration(args.calibration_path)

    print(f"Focal length (f_event): {f_event}, Baseline (B_event): {B_event}")
    if f_event <= 0 or B_event <= 0:
        raise ValueError("Invalid calibration parameters: focal length or baseline is non-positive.")
    print(f"T_10 matrix:\n{T_10}")
    if T_10.shape != (4, 4):
        raise ValueError("Invalid T_10 matrix: expected shape (4, 4).")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all disparity images in the directory
    for file_name in os.listdir(args.disparity_dir):
        disparity_path = os.path.join(args.disparity_dir, file_name)
        if not disparity_path.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  
            continue

        # Load disparity map
        disparity_map = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0
        if disparity_map is None:
            print(f"Failed to load disparity map from {disparity_path}, skipping...")
            continue

        print(f"Processing {file_name}...")
        print(f"Disparity map min: {np.min(disparity_map)}, max: {np.max(disparity_map)}")
        if np.min(disparity_map) < 0:
            print(f"Disparity map contains negative values, skipping {file_name}...")
            continue

        # Convert disparity to depth
        depth_map_event = disparity_to_depth(disparity_map, f_event, B_event)
        if np.min(depth_map_event) < 0:
            print(f"Depth map contains negative values, skipping {file_name}...")
            continue

        # Transform depth to frame camera
        depth_map_frame = transform_depth_to_frame(depth_map_event, T_10)

        # Save depth map
        output_path = os.path.join(args.output_dir, file_name)
        cv2.imwrite(output_path, (depth_map_frame * 256).astype(np.uint16))
        print(f"Saved depth map to {output_path}")