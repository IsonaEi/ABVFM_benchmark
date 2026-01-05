import argparse
import sys
import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add local Depth-Anything-3 to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DA3_PATH = os.path.join(SCRIPT_DIR, "Depth-Anything-3", "src")
sys.path.append(DA3_PATH)

# try:
from depth_anything_3.api import DepthAnything3
# except ImportError:
#     print(f"Error: Could not import 'depth_anything_3' from {DA3_PATH}.")
#     print("Ensure the repository is cloned correctly and dependencies are installed.")
#     sys.exit(1)

MODEL_ALIASES = {
    "DA3-Small": "depth-anything/DA3-SMALL",
    "DA3-Base": "depth-anything/DA3-BASE",
    "DA3-Large": "depth-anything/DA3-LARGE-1.1",
    "DA3-Giant": "depth-anything/DA3-GIANT-1.1",
    "DA3Metric-Large": "depth-anything/DA3METRIC-LARGE",
    "DA3Mono-Large": "depth-anything/DA3MONO-LARGE"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Mouse Height using Depth Anything 3")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--mask", type=str, required=False, help="Path to mask file (.npy or .h5 format).")
    parser.add_argument("--output", type=str, default="mouse_height_da3.csv", help="Output CSV path")
    parser.add_argument("--model", type=str, default="DA3Metric-Large", help="HuggingFace model key or path.") 
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smooth_window", type=int, default=5, help="Window size for median smoothing (0 to disable)")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum frames to process (-1 for all)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Model
    model_key = MODEL_ALIASES.get(args.model, args.model)
    print(f"Initializing Depth Anything 3 model: {model_key} on {args.device}...")
    try:
        model = DepthAnything3.from_pretrained(model_key).to(args.device)
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        sys.exit(1)

    # 2. Open Video
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        sys.exit(1)
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {total_frames} frames, {fps} fps, {width}x{height}")
    
    # 3. Load Mask
    mask_data = None
    h5_file = None
    
    if args.mask:
        print(f"Loading mask from: {args.mask}")
        try:
            if args.mask.endswith('.h5'):
                import h5py
                h5_file = h5py.File(args.mask, 'r')
                print(f"Opened H5 mask file. Keys sample: {list(h5_file.keys())[:5]}")
            else:
                mask_data = np.load(args.mask)
                print(f"Mask shape: {mask_data.shape}")
        except Exception as e:
            print(f"Error loading mask: {e}")
            sys.exit(1)
            
    # 4. Inference Loop
    results = []
    
    pbar = tqdm(total=total_frames, desc="Processing")
    print(f"Processing loop started. Target: {args.max_frames if args.max_frames > 0 else total_frames} frames.")
    
    frame_idx = 0
    while True:
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            print(f"Reached max_frames: {args.max_frames}")
            break
            
        ret, frame = cap.read()
        if not ret:
            print(f"Video ended at frame {frame_idx}")
            break
            
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        try:
            with torch.no_grad():
                prediction = model.inference([frame_rgb])
        except Exception as e:
            print(f"Inference error at frame {frame_idx}: {e}")
            break
            
        # Extract depth map
        depth_map = prediction.depth[0]
        
        # Determine Mask for current frame
        current_mask = None
        
        if h5_file is not None:
            key = str(frame_idx)
            if key in h5_file:
                current_mask = h5_file[key][:]
            else:
                if frame_idx == 0:
                    print(f"Warning: frame 0 key missing in H5. Keys: {list(h5_file.keys())[:5]}")
        elif mask_data is not None:
            if mask_data.ndim == 3:
                if frame_idx < mask_data.shape[0]:
                    current_mask = mask_data[frame_idx]
            elif mask_data.ndim == 2:
                current_mask = mask_data
                
        # Apply Mask
        roi_values = depth_map
        if current_mask is not None:
            if current_mask.shape != depth_map.shape:
                 current_mask = cv2.resize(current_mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            roi_values = depth_map[current_mask > 0]
        
        # Extract Metric
        if roi_values.size > 0:
            max_val = np.nanmax(roi_values)
        else:
            max_val = np.nan
            
        results.append({
            "frame_index": frame_idx,
            "max_height_raw": max_val,
            "timestamp_ms": timestamp_ms
        })
        
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}...")
            
        # Incremental save every 100 frames
        if frame_idx > 0 and frame_idx % 100 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(args.output + ".tmp", index=False)
            
        frame_idx += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    if h5_file:
        h5_file.close()
    
    # 5. Save and Smooth
    print(f"Processing complete. Collected {len(results)} results.")
    df = pd.DataFrame(results)
    
    # Final save
    df.to_csv(args.output, index=False)
    
    if args.smooth_window > 0:
        print(f"Applying median smoothing (window={args.smooth_window})...")
        df['max_height_smoothed'] = df['max_height_raw'].rolling(window=args.smooth_window, center=True).median()
        df['max_height_smoothed'] = df['max_height_smoothed'].fillna(method='bfill').fillna(method='ffill')
    else:
        df['max_height_smoothed'] = df['max_height_raw']
        
    print(f"Saving results to: {args.output}")
    df.to_csv(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
