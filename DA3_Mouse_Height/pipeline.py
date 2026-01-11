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

def get_background_map(model, cap, h5_file, mask_data, num_frames=50, device='cuda'):
    print(f"Constructing background map (sampling {num_frames} frames)...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    all_depths = []
    
    for idx in tqdm(indices, desc="Sampling Background"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            prediction = model.inference([frame_rgb])
        depth_map = prediction.depth[0]
        
        # Get mask for this frame
        current_mask = None
        if h5_file is not None:
            if str(idx) in h5_file: current_mask = h5_file[str(idx)][:]
        elif mask_data is not None:
            if mask_data.ndim == 3: current_mask = mask_data[idx]
            else: current_mask = mask_data
            
        if current_mask is not None:
            if current_mask.shape != depth_map.shape:
                current_mask = cv2.resize(current_mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Only use pixels NOT in mask for floor estimation
            depth_map[current_mask > 0] = np.nan
            
        all_depths.append(depth_map)
        
    # Reconstruct background using median
    all_depths = np.array(all_depths)
    bg_depth = np.nanmedian(all_depths, axis=0)
    
    # Fill remaining NaNs if any (e.g. pixels that always had a mouse)
    # Simple nearest neighbor filling for background gaps
    mask = np.isnan(bg_depth)
    if np.any(mask):
        print("Filling gaps in background map...")
        bg_depth_filled = bg_depth.copy()
        # Find nearest non-nan value for each nan
        from scipy import ndimage as nd
        indices = nd.distance_transform_edt(mask, return_distances=False, return_indices=True)
        bg_depth = bg_depth[tuple(indices)]
        
    print("Background map construction complete.")
    return bg_depth

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Mouse Height using Depth Anything 3")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--mask", type=str, required=False, help="Path to mask file (.npy or .h5 format).")
    parser.add_argument("--output", type=str, default="mouse_height_da3.csv", help="Output CSV path")
    parser.add_argument("--model", type=str, default="DA3Metric-Large", help="HuggingFace model key or path.") 
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smooth_window", type=int, default=5, help="Window size for median smoothing (0 to disable)")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum frames to process (-1 for all)")
    parser.add_argument("--calibrate", action="store_true", help="Perform floor background calibration")
    parser.add_argument("--num_calib", type=int, default=50, help="Number of frames for calibration")
    parser.add_argument("--save_raw", action="store_true", help="Save raw depth maps to HDF5")
    parser.add_argument("--resolution", type=int, default=504, help="Input resolution for the model (default: 504)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Model
    model_key = MODEL_ALIASES.get(args.model, args.model)
    print(f"Initializing Depth Anything 3 model: {model_key} on {args.device}...")
    model = DepthAnything3.from_pretrained(model_key).to(args.device)

    # 2. Open Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 3. Load Mask (Pre-load helper)
    h5_file = None
    mask_data = None
    if args.mask:
        if args.mask.endswith('.h5'):
            import h5py
            try:
                h5_file = h5py.File(args.mask, 'r')
            except Exception as e:
                print(f"H5 Load Error: {e}")
        else:
            mask_data = np.load(args.mask)

    # 4. Calibration
    bg_map = None
    if args.calibrate:
        bg_map = get_background_map(model, cap, h5_file, mask_data, num_frames=args.num_calib, device=args.device)
        # Reset capture after sampling
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 5. Output Setup
    output_h5_path = args.output.replace(".csv", ".h5")
    if args.save_raw:
        print(f"Raw depth maps will be saved to: {output_h5_path}")
        raw_h5 = h5py.File(output_h5_path, 'w')
        # Create datasets - chunked for efficient writing
        # We don't know exact H/W until first frame, but we can guess or wait.
        # Let's wait for first frame to init datasets.
        dset_depth = None
        dset_ts = raw_h5.create_dataset("timestamp_ms", (0,), maxshape=(None,), dtype='f8')
    
    # 6. Inference Loop
    results = []
    pbar = tqdm(total=total_frames if args.max_frames < 0 else args.max_frames, desc="Processing")
    frame_idx = 0
    
    while True:
        if args.max_frames > 0 and frame_idx >= args.max_frames: break
        ret, frame = cap.read()
        if not ret: break
        
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            prediction = model.inference([frame_rgb], process_res=args.resolution)
        depth_map = prediction.depth[0] # HxW numpy array
        
        # Save Raw Data
        if args.save_raw:
            if dset_depth is None:
                H, W = depth_map.shape
                dset_depth = raw_h5.create_dataset("depth_map", (0, H, W), maxshape=(None, H, W), dtype='f2', compression="lzf") # float16
            
            dset_depth.resize((dset_depth.shape[0] + 1), axis=0)
            dset_depth[-1] = depth_map.astype(np.float16)
            
            dset_ts.resize((dset_ts.shape[0] + 1), axis=0)
            dset_ts[-1] = timestamp_ms

        # Get mask
        current_mask = None
        if h5_file is not None:
            key = str(frame_idx)
            if key in h5_file: current_mask = h5_file[key][:]
        elif mask_data is not None:
            if mask_data.ndim == 3: current_mask = mask_data[frame_idx]
            else: current_mask = mask_data
            
        if current_mask is not None:
            if current_mask.shape != depth_map.shape:
                current_mask = cv2.resize(current_mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Calculate Reference Floor (Dynamic Baseline)
        if current_mask is not None:
            mouse_mask = current_mask > 0
            # Dynamic Floor: 95th percentile of background
            bg_pixels = depth_map[~mouse_mask]
            
            if bg_pixels.size > 0:
                floor_ref = np.nanpercentile(bg_pixels, 95)
            else:
                floor_ref = np.nanmax(depth_map)
                
            mouse_pixels = depth_map[mouse_mask]
            if mouse_pixels.size > 0:
                # Top 25% Height Strategy
                height_per_pixel = floor_ref - mouse_pixels
                threshold = np.nanpercentile(height_per_pixel, 75)
                top_25_mask = height_per_pixel >= threshold
                est_height = np.nanmean(height_per_pixel[top_25_mask])
            else:
                est_height = np.nan
        else:
            est_height = np.nan

        results.append({
            "frame_index": frame_idx,
            "timestamp_ms": timestamp_ms,
            "estimated_height": est_height,
        })
        
        if frame_idx % 100 == 0:
            pd.DataFrame(results).to_csv(args.output + ".tmp", index=False)
            
        frame_idx += 1
        pbar.update(1)

    cap.release()
    if h5_file: h5_file.close()
    if args.save_raw: raw_h5.close()
    
    # Save & Smooth
    df = pd.DataFrame(results)
    if args.smooth_window > 0:
        if 'estimated_height' in df.columns:
            df['estimated_height_smoothed'] = df['estimated_height'].rolling(window=args.smooth_window, center=True).median().fillna(method='bfill').fillna(method='ffill')
    
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    if args.save_raw:
        print(f"Raw depth maps saved to {output_h5_path}")

if __name__ == "__main__":
    main()
