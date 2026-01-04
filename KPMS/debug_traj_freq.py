
import keypoint_moseq as kpms
import h5py
import numpy as np
import os
from automated_pipeline.analysis import extract_results_safe

# Config params
project_dir = "/home/isonaei/ABVFM/KPMS/results/20260104-0810_ctrl_30fps"
model_name = "20260104-0844-0"
results_path = os.path.join(project_dir, model_name, "results.h5")

print(f"Loading results from {results_path}")
with h5py.File(results_path, 'r') as f:
    # Manual load to check structure
    key = list(f.keys())[0]
    z = f[key]['syllable'][:]
    print(f"Total Frames: {len(z)}")
    
    unique, counts = np.unique(z, return_counts=True)
    freqs = counts / len(z)
    
    print("\nSyllable Frequencies:")
    for s, freq, count in zip(unique, freqs, counts):
        print(f"Syllable {s}: {freq:.6f} ({count} frames)")
        
    print(f"\nMin Frequency found: {min(freqs)}")
