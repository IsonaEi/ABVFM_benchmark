import h5py
import numpy as np
import cv2
import argparse
from tqdm import tqdm

def normalize_to_heatmap(data, min_val, max_val):
    # Normalize to 0 (Blue/Far) - 1 (Red/Close)
    # DA3: Large=Far, Small=Close.
    # So we want (Max - x) / (Max - Min)
    norm = (max_val - data) / (max_val - min_val + 1e-6)
    norm = np.clip(norm, 0, 1)
    heatmap = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_frames", type=int, default=-1)
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading Data...")
    f = h5py.File(args.h5, 'r')
    ds = f['depth_map']
    mask_h5 = h5py.File(args.mask, 'r')
    
    total_frames = ds.shape[0]
    if args.max_frames > 0: total_frames = min(total_frames, args.max_frames)
    
    H, W = ds[0].shape
    dilate_kernel = np.ones((15, 15), np.uint8)

    # 2. Establish Reference (Frame 0)
    # We simply use Frame 0 as the "Anchor" for alignment. 
    # (Assuming Frame 0 has a valid floor).
    print("Establishing Reference (Frame 0)...")
    ref_depth = ds[0].astype(np.float32)
    ref_mask = None
    if '0' in mask_h5:
        m = mask_h5['0'][:]
        if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        ref_mask = cv2.dilate(m, dilate_kernel, iterations=1)
    else:
        ref_mask = np.zeros((H,W), dtype=bool)

    # 3. Pre-pass: Compute Alignment Stats (Global Min/Max of Stabilized Data)
    # We need to know the range of aligned values to set a stable color map.
    print("Computing Global Stabilized Scales...")
    
    global_min = float('inf')
    global_max = float('-inf')
    
    # Sample every Nth frame for speed
    step = max(1, total_frames // 100)
    indices = range(0, total_frames, step)
    
    for idx in tqdm(indices, desc="Scanning Scale"):
        depth = ds[idx].astype(np.float32)
        
        # Get Mask
        key = str(idx)
        curr_mask = np.zeros((H,W), dtype=np.uint8)
        if key in mask_h5:
            m = mask_h5[key][:]
            if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            curr_mask = cv2.dilate(m, dilate_kernel, iterations=1)
            
        # Align
        # valid = Floor in both
        valid = (curr_mask.flatten() == 0) & (ref_mask.flatten() == 0)
        valid &= (~np.isnan(depth.flatten())) & (~np.isnan(ref_depth.flatten()))
        
        x = depth.flatten()[valid]
        y = ref_depth.flatten()[valid]
        
        if len(x) > 100:
            A = np.vstack([x, np.ones(len(x))]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            
            stabilized = a * depth + b
            
            # Update Globals - Percentiles are safer but Min/Max gives full range
            # Use percentiles of this frame to update global estimate
            # Ignore NaNs
            g_min = np.nanpercentile(stabilized, 1)
            g_max = np.nanpercentile(stabilized, 99)
            
            if g_min < global_min: global_min = g_min
            if g_max > global_max: global_max = g_max
            
    print(f"Global Stabilized Scale: Min={global_min:.4f}, Max={global_max:.4f}")
    
    # 4. Render Video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (orig_W, orig_H))
    
    print("Rendering Stabilized Video...")
    for i in tqdm(range(total_frames)):
        depth = ds[i].astype(np.float32)
        
        # Get Mask & Align (Same logic)
        key = str(i)
        curr_mask = np.zeros((H,W), dtype=np.uint8)
        if key in mask_h5:
            m = mask_h5[key][:]
            if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            curr_mask = cv2.dilate(m, dilate_kernel, iterations=1)
            
        valid = (curr_mask.flatten() == 0) & (ref_mask.flatten() == 0)
        valid &= (~np.isnan(depth.flatten())) & (~np.isnan(ref_depth.flatten()))
        
        x = depth.flatten()[valid]
        y = ref_depth.flatten()[valid]
        
        stabilized = depth # Default fallback
        if len(x) > 100:
            A = np.vstack([x, np.ones(len(x))]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            stabilized = a * depth + b
        
        # Visualize
        heatmap = normalize_to_heatmap(stabilized, global_min, global_max)
        
        # Resize
        if heatmap.shape[:2] != (orig_H, orig_W):
            heatmap = cv2.resize(heatmap, (orig_W, orig_H))
            
        out.write(heatmap)
    
    out.release()
    f.close()
    mask_h5.close()
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
