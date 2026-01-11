
import os
import argparse
import numpy as np
import cv2
import sys
import h5py

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.engine import OpticalFlowEngine

def main():
    parser = argparse.ArgumentParser(description="ABVFM Optical Flow Runner (GMFlow)")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--mask', type=str, help='Path to mask file (.h5 or .npy)')
    parser.add_argument('--output', type=str, required=True, help='Path to save results (.npy)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for GPU inference')
    parser.add_argument('--resize', type=int, nargs=2, default=[720, 720], help='Resize dimensions (W H)')
    args = parser.parse_args()

    engine = OpticalFlowEngine(device='cuda')

    mask = None
    if args.mask:
        if args.mask.endswith('.h5'):
             with h5py.File(args.mask, 'r') as f:
                 # Standard ABVFM mask key
                 if 'mask' in f: mask = f['mask'][:]
                 elif 'mask_list' in f: mask = f['mask_list'][0]
        elif args.mask.endswith('.npy'):
             mask = np.load(args.mask)

    flow_magnitude = engine.compute_optical_flow_gpu(
        args.video, 
        mask=mask, 
        resize_dim=tuple(args.resize), 
        batch_size=args.batch_size, 
        save_path=args.output
    )

    print(f"Processing complete. Saved to {args.output}")

if __name__ == "__main__":
    main()
