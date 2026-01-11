import h5py
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import os
import utils

def main():
    parser = argparse.ArgumentParser(description="Visualize DA3 Deep Maps (Raw or Stabilized)")
    parser.add_argument("--h5", required=True, help="Path to depth H5 file")
    parser.add_argument("--video", required=True, help="Path to original video (for dimensions/FPS)")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--mask", required=False, help="Path to mask H5 file (required for stabilization)")
    parser.add_argument("--mode", choices=["raw", "stabilized"], default="raw", help="Visualization mode")
    parser.add_argument("--max_frames", type=int, default=-1, help="Limit number of frames")
    parser.add_argument("--ref_frame", type=int, default=0, help="Frame index to use as reference for stabilization")
    args = parser.parse_args()

    if args.mode == "stabilized" and not args.mask:
        print("Error: --mask is required for stabilized mode.")
        return

    # 1. Load Data
    print(f"Loading H5: {args.h5}")
    f = h5py.File(args.h5, 'r')
    ds = f['depth_map']
    
    mask_h5 = None
    if args.mask:
        mask_h5 = h5py.File(args.mask, 'r')

    total_frames = ds.shape[0]
    if args.max_frames > 0: total_frames = min(total_frames, args.max_frames)

    # 2. Get Video Info
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    H, W = ds[0].shape

    # 3. Compute Global Scale
    print("Computing Global Depth Statistics...")
    # For stabilized mode, we should ideally compute stats on stabilized data.
    # But approximate stats from raw data is often 'good enough' or we do a pre-pass.
    # Let's do a quick pre-pass if stabilized to be accurate.
    
    g_min, g_max = 0, 1
    
    if args.mode == "raw":
        g_min, g_max = utils.calculate_global_scale(ds, total_frames)
        print(f"Global Scale (Raw): {g_min:.4f} - {g_max:.4f}")
    else:
        # Stabilized Pre-pass
        print("Pre-scanning for stabilized scale...")
        ref_depth = ds[args.ref_frame].astype(np.float32)
        ref_mask = utils.load_mask(mask_h5, args.ref_frame, (H, W))
        
        sample_indices = np.linspace(0, total_frames - 1, num=min(100, total_frames), dtype=int)
        stab_samples = []
        
        for idx in sample_indices:
            d = ds[idx].astype(np.float32)
            if args.mode == "stabilized":
                m = utils.load_mask(mask_h5, idx, (H, W))
                aligned_d, _, _ = utils.align_depth(d, ref_depth, m, ref_mask)
                stab_samples.append(aligned_d)
        
        stab_samples = np.array(stab_samples)
        g_min = np.nanpercentile(stab_samples, 1)
        g_max = np.nanpercentile(stab_samples, 99)
        print(f"Global Scale (Stabilized): {g_min:.4f} - {g_max:.4f}")

    # 4. Process Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (orig_W, orig_H))
    
    # Reference data for loop (only needed for stabilization)
    ref_depth = None
    ref_mask = None
    if args.mode == "stabilized":
        ref_depth = ds[args.ref_frame].astype(np.float32)
        ref_mask = utils.load_mask(mask_h5, args.ref_frame, (H, W))

    print(f"Rendering {args.mode} video...")
    for i in tqdm(range(total_frames)):
        depth = ds[i].astype(np.float32)
        
        vis_depth = depth
        
        if args.mode == "stabilized":
            curr_mask = utils.load_mask(mask_h5, i, (H, W))
            vis_depth, _, _ = utils.align_depth(depth, ref_depth, curr_mask, ref_mask)
        
        # Visualize
        heatmap = utils.normalize_to_heatmap(vis_depth, g_min, g_max)
        
        # Resize to video dimensions
        heatmap = utils.resize_to_match(heatmap, orig_H, orig_W)
        
        # Add Text
        label = f"Frame {i} ({args.mode})"
        cv2.putText(heatmap, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        out.write(heatmap)

    out.release()
    f.close()
    if mask_h5: mask_h5.close()
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
