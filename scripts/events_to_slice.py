import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from visualization.eventreader import EventReader

def render_to_png(x: np.ndarray, y: np.ndarray, pol: np.ndarray, 
                 H: int, W: int, output_path: Path):
    """Render events to PNG image"""
    assert x.size == y.size == pol.size
    assert H > 0 and W > 0
    
    # Create RGB image (white background)
    img = np.full((H, W, 3), fill_value=255, dtype='uint8')
    
    # Convert polarity (0 = -1, 1 = 1)
    pol = pol.astype('int')
    pol[pol == 0] = -1
    
    # Filter valid events
    mask = (x >= 0) & (y >= 0) & (W > x) & (H > y)
    x_valid = x[mask]
    y_valid = y[mask]
    pol_valid = pol[mask]
    
    # Set colors based on polarity
    img[y_valid[pol_valid == -1], x_valid[pol_valid == -1]] = [255, 0, 0]  # 负事件 = 红色
    img[y_valid[pol_valid == 1], x_valid[pol_valid == 1]] = [0, 0, 255]    # 正事件 = 蓝色
    
    # Save as PNG
    Image.fromarray(img).save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Event Slices to PNG')
    parser.add_argument('event_file', type=str, help='Path to events.h5 file')
    parser.add_argument('output_dir', help='Directory to save PNG slices')
    parser.add_argument('--delta_time_ms', '-dt_ms', type=float, default=50.0,
                       help='Time window (in milliseconds) for each slice')
    parser.add_argument('--height', type=int, default=480, help='Image height')
    parser.add_argument('--width', type=int, default=640, help='Image width')
    args = parser.parse_args()

    event_filepath = Path(args.event_file)
    output_dir = Path(args.output_dir)
    dt = args.delta_time_ms
    H = args.height
    W = args.width

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_idx = 0
    for events in tqdm(EventReader(event_filepath, dt)):
        p = events['p']
        x = events['x']
        y = events['y']
        
        output_path = output_dir / f"{slice_idx:06d}.png"
        if(slice_idx % 2):
            pass
        else:
            render_to_png(x, y, p, H, W, output_path)
        slice_idx += 1

    print(f"Saved {slice_idx} PNG slices to {output_dir}")