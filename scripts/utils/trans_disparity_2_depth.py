
import yaml
import numpy as np


def load_calibration(calibration_path):
    """Load calibration parameters from YAML file."""
    with open(calibration_path, 'r') as f:
        calibration = yaml.safe_load(f)
    f_event = calibration['intrinsics']['camRect0']['camera_matrix'][0]
    B_event = calibration['disparity_to_depth']['cams_03'][3][2]
    T_10 = np.array(calibration['extrinsics']['T_10'])
    return f_event, B_event, T_10

def disparity_to_depth(disparity_map, f, B):
    """Convert disparity map to depth map."""
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    valid_mask = disparity_map > 0
    depth_map[valid_mask] = f * B / disparity_map[valid_mask]
    return depth_map

def transform_depth_to_frame(depth_map, T_10):
    """Transform depth map from event camera to frame camera."""
    h, w = depth_map.shape
    depth_map_frame = np.zeros_like(depth_map, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            Z = depth_map[y, x]
            if Z > 0:
                point_event = np.array([x, y, Z, 1])
                point_frame = T_10 @ point_event
                depth_map_frame[y, x] = point_frame[2]  # Z in frame camera
    return depth_map_frame
