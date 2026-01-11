import h5py
import numpy as np
import cv2
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_frames", type=int, default=-1)
    args = parser.parse_args()

    print(f"Loading H5: {args.h5}")
    f = h5py.File(args.h5, 'r')
    ds = f['depth_map']
    
    total_frames = ds.shape[0]
    if args.max_frames > 0: total_frames = min(total_frames, args.max_frames)
    
    # 1. Determine Global Scale (Min/Max)
    print("Computing Global Depth Statistics (Sampling)...")
    indices = np.linspace(0, total_frames-1, num=min(100, total_frames), dtype=int)
    samples = []
    for idx in indices:
        samples.append(ds[idx])
    samples = np.array(samples)
    
    # Robust Min/Max
    # DA3 Metric Depth: Large Value = Far (Floor), Small Value = Close (Mouse)
    # We want: Far -> Blue, Close -> Red.
    # Jet Colormap: 0 -> Blue, 1 -> Red.
    # So we need to map: Far (Max) -> 0, Close (Min) -> 1
    # Norm = (Global_Max - Depth) / (Global_Max - Global_Min)
    
    # Use percentiles to avoid outliers
    g_min = np.nanpercentile(samples, 1)
    g_max = np.nanpercentile(samples, 99)
    print(f"Global Scale: Min={g_min:.4f} (Mouse-ish), Max={g_max:.4f} (Floor-ish)")
    
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (orig_W, orig_H))
    
    print("Generating Global Heatmap Video...")
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break
        
        depth = ds[i].astype(np.float32)
        
        # Normalize: (Max - D) / Range -> 0 (Far) to 1 (Close)
        norm = (g_max - depth) / (g_max - g_min + 1e-6)
        norm = np.clip(norm, 0, 1)
        
        heatmap = (norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Resize to original video size if needed
        if heatmap.shape[:2] != (orig_H, orig_W):
            heatmap = cv2.resize(heatmap, (orig_W, orig_H))
            
        # Optional: Blend with original? User asked for "raw heatmap containing background and mouse".
        # Pure heatmap is best for value comparison.
        # But adding a text overlay of min/max frame value helps.
        
        # Add Text
        # cv2.putText(heatmap, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        out.write(heatmap)
        
    out.release()
    cap.release()
    f.close()
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
