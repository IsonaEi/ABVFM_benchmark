import h5py
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def normalize_to_heatmap(data, cmap=cv2.COLORMAP_JET, vmin=None, vmax=None):
    non_nan = data[~np.isnan(data)]
    if non_nan.size == 0: return np.zeros_like(data, dtype=np.uint8)
    
    if vmin is None: vmin = np.nanmin(non_nan)
    if vmax is None: vmax = np.nanmax(non_nan)
    
    norm = (data - vmin) / (vmax - vmin + 1e-6)
    norm = np.clip(norm, 0, 1)
    heatmap = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cmap)
    heatmap[np.isnan(data)] = 0
    return heatmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="Path to raw HDF5 file")
    parser.add_argument("--mask", required=True, help="Path to mask H5")
    parser.add_argument("--video", required=True, help="Path to original video")
    parser.add_argument("--output_base", required=True, help="Base path for output (video and csv)")
    parser.add_argument("--max_frames", type=int, default=-1)
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading data...")
    raw_h5 = h5py.File(args.h5, 'r')
    dset_depth = raw_h5['depth_map']
    mask_h5 = h5py.File(args.mask, 'r')
    
    total_frames = dset_depth.shape[0]
    if args.max_frames > 0: total_frames = min(total_frames, args.max_frames)
    
    # 2. Build Canonical Background via Iterative Alignment
    print("Building Canonical Background (Aligning all frames to Reference)...")
    indices = range(total_frames)
    indices = indices[::2] if total_frames > 2000 else indices # Sample if too many
    
    H, W = dset_depth[0].shape
    dilate_kernel = np.ones((15, 15), np.uint8) 

    # 2.1 Find Reference Frame (First valid frame with good coverage)
    ref_depth = None
    ref_mask = None
    ref_idx = 0
    
    # helper for alignment
    def align_depth(source, target, mask_source, mask_target):
        # Align source to target considering only valid pixels in both
        # valid = (~mask_source) & (~mask_target) & (~NaNs)
        source_flat = source.flatten()
        target_flat = target.flatten()
        
        # Mask: True=Mouse/Invalid. We want False (Floor)
        valid = (mask_source.flatten() == 0) & (mask_target.flatten() == 0)
        valid &= (~np.isnan(source_flat)) & (~np.isnan(target_flat))
        
        x = source_flat[valid]
        y = target_flat[valid]
        
        if len(x) < 100: return source, 1.0, 0.0 # Fail safe
        
        A = np.vstack([x, np.ones(len(x))]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return a * source + b, a, b

    # Load Ref
    for idx in indices:
        d = dset_depth[idx].astype(np.float32)
        key = str(idx)
        m = None
        if key in mask_h5:
             m = mask_h5[key][:]
             if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
             m = cv2.dilate(m, dilate_kernel, iterations=1)
        else:
             m = np.zeros((H,W), dtype=bool) # No mask? assume all valid? Or all invalid?
             
        # Check if frame has enough floor
        if np.sum(m==0) > (H*W*0.1):
            ref_depth = d
            ref_mask = m
            ref_idx = idx
            break
            
    if ref_depth is None:
        print("Error: Could not find suitable reference frame!")
        return

    print(f"Reference Frame Selected: {ref_idx}")
    
    # 2.2 Align All to Reference and Accumulate
    aligned_samples = []
    
    for idx in tqdm(indices, desc="Aligning Samples"):
        curr_depth = dset_depth[idx].astype(np.float32)
        
        key = str(idx)
        curr_mask = np.zeros((H,W), dtype=np.uint8)
        if key in mask_h5:
            m = mask_h5[key][:]
            if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            curr_mask = cv2.dilate(m, dilate_kernel, iterations=1)
            
        aligned_d, _, _ = align_depth(curr_depth, ref_depth, curr_mask, ref_mask)
        
        # Apply mask to aligned sample (set mouse to NaN)
        aligned_d[curr_mask > 0] = np.nan
        aligned_samples.append(aligned_d)

    # 2.3 Compute Median
    aligned_samples = np.array(aligned_samples)
    canonical_bg = np.nanmedian(aligned_samples, axis=0)
    
    # Fill gaps
    mask_nan = np.isnan(canonical_bg)
    if np.any(mask_nan):
        print(f"Filling {np.sum(mask_nan)} NaN pixels...")
        from scipy import ndimage as nd
        ind = nd.distance_transform_edt(mask_nan, return_distances=False, return_indices=True)
        canonical_bg = canonical_bg[tuple(ind)]

    # Debug Save
    debug_bg = normalize_to_heatmap(canonical_bg, cmap=cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(args.output_base + "_canonical_bg_debug.png", debug_bg)
    print(f"Refined Background Saved.")

    # 3. Process Frames: Align & Subtract
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video Writer
    out_video_path = args.output_base + "_align_vis.mp4"
    out_W = orig_W * 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (out_W, orig_H))
    
    results = []
    
    # Fixed heatmap scale for consistency
    # Estimate typical rearing height range.
    # Height = Canon - Aligned. 
    # If mouse is 0.05 units "closer", Height should be +0.05.
    # Let's dynamic range it per frame or fixed? Fixed is better for video stability.
    # Let's auto-range based on first few frames? No, let's just dynamic per frame for visualization 
    # but print text for absolute values.
    
    for i in tqdm(range(total_frames), desc="Align & Subtract"):
        ret, frame = cap.read()
        if not ret: break
        
        # Load Depth & Mask
        raw_depth = dset_depth[i].astype(np.float32)
        
        key = str(i)
        current_mask = None
        if key in mask_h5: 
            current_mask = mask_h5[key][:]
            if current_mask.shape != (H, W):
                 current_mask = cv2.resize(current_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        if current_mask is None:
             out.write(np.hstack([frame, np.zeros_like(frame), np.zeros_like(frame)]))
             continue
             
        mouse_mask = current_mask > 0
        floor_mask = ~mouse_mask
        
        # Linear Alignment: Fit Raw_Floor to Canonical_Floor
        # Model: Canon = a * Raw + b
        # We want: Aligned_Raw = a * Raw + b
        
        raw_floor_vals = raw_depth[floor_mask]
        canon_floor_vals = canonical_bg[floor_mask]
        
        # Filter valid
        valid = (~np.isnan(raw_floor_vals)) & (~np.isnan(canon_floor_vals))
        y = canon_floor_vals[valid]
        x = raw_floor_vals[valid]
        
        if len(x) > 100:
            # Linear Fit: y = ax + b
            A = np.vstack([x, np.ones(len(x))]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            a, b = 1.0, 0.0
            
        aligned_depth = a * raw_depth + b
        
        # Subtract: Height = Canonical - Aligned
        # (Assuming Metric Depth: Far=Big, Close=Small. So Ground > Mouse. Height = Ground - Mouse)
        height_map = canonical_bg - aligned_depth
        
        # Stats
        mouse_pixels = height_map[mouse_mask]
        if mouse_pixels.size > 0:
            # Top 25% height
            threshold = np.nanpercentile(mouse_pixels, 75)
            est_height = np.nanmean(mouse_pixels[mouse_pixels >= threshold])
        else:
            est_height = 0.0
            
        results.append({
            "frame": i,
            "height": est_height,
            "scale_a": a,
            "offset_b": b
        })
        
        # Visualization
        # 1. Canonical (Resized)
        vis_canon = normalize_to_heatmap(canonical_bg)
        if (H,W) != (orig_H, orig_W): vis_canon = cv2.resize(vis_canon, (orig_W, orig_H))
        
        # 2. Height Map (Resized)
        # Height map only valid on mask? Or generally?
        # Generally valid, but noise on floor.
        # Mask out floor for clear visualization
        vis_height_map = height_map.copy()
        vis_height_map[floor_mask] = np.nan
        vis_height = normalize_to_heatmap(vis_height_map, cmap=cv2.COLORMAP_JET) # Auto-scale
        if (H,W) != (orig_H, orig_W): vis_height = cv2.resize(vis_height, (orig_W, orig_H))
        
        # Composite
        combined = np.hstack([frame, vis_canon, vis_height])
        
        # Text
        cv2.putText(combined, f"Frame {i} | H={est_height:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Canonical BG", (orig_W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Aligned Height (Subtracted)", (orig_W*2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(combined)
        
    out.release()
    cap.release()
    raw_h5.close()
    mask_h5.close()
    
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_base + ".csv", index=False)
    print(f"Alignment Done. Saved to {args.output_base}.csv/.mp4")

if __name__ == "__main__":
    main()
