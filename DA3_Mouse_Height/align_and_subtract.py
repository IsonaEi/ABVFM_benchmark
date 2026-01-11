import h5py
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import pandas as pd
import utils

def main():
    parser = argparse.ArgumentParser(description="Calculate Mouse Height from Depth Maps")
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
    
    H, W = dset_depth[0].shape

    # 2. Build Canonical Background via Iterative Alignment
    print("Building Canonical Background (Aligning all frames to Reference)...")
    indices = range(total_frames)
    # Sample if too many frames to speed up background build
    sample_indices = indices[::2] if total_frames > 2000 else indices 
    
    # 2.1 Find Reference Frame (First valid frame with good coverage)
    ref_depth = None
    ref_mask = None
    ref_idx = 0
    
    for idx in sample_indices:
        d = dset_depth[idx].astype(np.float32)
        m = utils.load_mask(mask_h5, idx, (H, W))
        
        # Check if frame has enough floor (Mask=0 is floor)
        # Floor pixel count > 10%
        if np.sum(m == 0) > (H * W * 0.1):
            ref_depth = d
            ref_mask = m
            ref_idx = idx
            break
            
    if ref_depth is None:
        print("Error: Could not find suitable reference frame with enough floor!")
        return

    print(f"Reference Frame Selected: {ref_idx}")
    
    # 2.2 Align All to Reference and Accumulate
    aligned_samples = []
    
    for idx in tqdm(sample_indices, desc="Aligning Samples"):
        curr_depth = dset_depth[idx].astype(np.float32)
        curr_mask = utils.load_mask(mask_h5, idx, (H, W))
            
        aligned_d, _, _ = utils.align_depth(curr_depth, ref_depth, curr_mask, ref_mask)
        
        # Apply mask to aligned sample (set mouse to NaN so it doesn't contribute to background)
        aligned_d[curr_mask > 0] = np.nan
        aligned_samples.append(aligned_d)

    # 2.3 Compute Median to create Canonical Background
    aligned_samples = np.array(aligned_samples)
    canonical_bg = np.nanmedian(aligned_samples, axis=0)
    
    # Fill gaps in canonical background
    mask_nan = np.isnan(canonical_bg)
    if np.any(mask_nan):
        print(f"Filling {np.sum(mask_nan)} NaN pixels in Canonical BG...")
        from scipy import ndimage as nd
        ind = nd.distance_transform_edt(mask_nan, return_distances=False, return_indices=True)
        canonical_bg = canonical_bg[tuple(ind)]

    # Debug Save
    debug_bg = utils.normalize_to_heatmap(canonical_bg, cmap=cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(args.output_base + "_canonical_bg_debug.png", debug_bg)
    print(f"Refined Background Saved.")

    # 3. Process Frames: Align & Subtract
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video Writer
    out_video_path = args.output_base + "_process_vis.mp4"
    out_W = orig_W * 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (out_W, orig_H))
    
    results = []
    
    for i in tqdm(range(total_frames), desc="Align & Subtract"):
        ret, frame = cap.read()
        if not ret: break
        
        # Load Depth & Mask
        raw_depth = dset_depth[i].astype(np.float32)
        current_mask = utils.load_mask(mask_h5, i, (H, W))
        
        if current_mask is None: # Should typically not be None due to utils
             current_mask = np.zeros_like(raw_depth, dtype=np.uint8)

        mouse_mask = current_mask > 0
        floor_mask = ~mouse_mask
        
        # Linear Alignment: Align Raw to Canonical
        # Note: We align Raw -> Canonical (Source=Raw, Target=Canonical)
        # Using Floor pixels only
        aligned_depth, a, b = utils.align_depth(raw_depth, canonical_bg, current_mask, None) # Canonical has no mask (all valid)
        
        # Subtract: Height = Canonical - Aligned
        # (Assuming Metric Depth: Far=Big, Close=Small. Ground (Big) - Mouse (Small) = Height (Pos))
        height_map = canonical_bg - aligned_depth
        
        # Stats
        mouse_pixels = height_map[mouse_mask]
        if mouse_pixels.size > 0:
            # Top 25% height strategy
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
        vis_canon = utils.resize_to_match(utils.normalize_to_heatmap(canonical_bg), orig_H, orig_W)
        
        # 2. Height Map (Resized)
        # Mask out floor for clear visualization of mouse height
        vis_height_map = height_map.copy()
        vis_height_map[floor_mask] = np.nan
        vis_height = utils.normalize_to_heatmap(vis_height_map, cmap=cv2.COLORMAP_JET)
        vis_height = utils.resize_to_match(vis_height, orig_H, orig_W)
        
        # Composite
        combined = np.hstack([frame, vis_canon, vis_height])
        
        # Text
        cv2.putText(combined, f"Frame {i} | H={est_height:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Canonical BG", (orig_W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Height (Canon - Aligned)", (orig_W*2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
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
