import cv2
import numpy as np
from pathlib import Path

def check_saved_depth_map(depth_map_path):
    """Perform checks on the saved depth map."""
    # 检查文件是否存在
    if not Path(depth_map_path).exists():
        raise FileNotFoundError(f"Depth map file not found: {depth_map_path}")

    # 读取深度图
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise ValueError(f"Failed to load depth map from {depth_map_path}")

    # 检查数据类型
    print(f"Depth map dtype: {depth_map.dtype}")
    if depth_map.dtype == np.uint16:
        print("Depth map is stored as 16-bit unsigned integers.")
    elif depth_map.dtype == np.float32:
        print("Depth map is stored as 32-bit floating point values.")
    else:
        raise ValueError("Unexpected depth map data type.")

    # 检查深度图的形状
    print(f"Depth map shape: {depth_map.shape}")
    if len(depth_map.shape) != 2:
        raise ValueError("Depth map is not a single-channel image.")

    # 检查深度值范围
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    print(f"Depth map min value: {depth_min}, max value: {depth_max}")
    if depth_min < 0:
        raise ValueError("Depth map contains negative values, which are invalid.")
    if depth_max == 0:
        raise ValueError("Depth map contains only zero values, which is invalid.")

    # 如果深度图是 uint16 类型，恢复真实深度值（假设存储时乘以 256）
    if depth_map.dtype == np.uint16:
        depth_map = depth_map.astype(np.float32) / 256.0
        print(f"Converted depth map min value: {np.min(depth_map)}, max value: {np.max(depth_map)}")

    print("Depth map checks completed successfully.")

# 示例调用
depth_map_path = "/home/zzt/Project/EVGGT/interlaken_00_c/frame_depth.png"
check_saved_depth_map(depth_map_path)