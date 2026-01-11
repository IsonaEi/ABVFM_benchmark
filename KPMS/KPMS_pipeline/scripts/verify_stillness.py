
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_v_vs_s(path):
    if not Path(path).exists(): return
    with h5py.File(path, 'r') as f:
        session = list(f.keys())[0]
        syllables = f[session]['syllable'][()]
        centroid = f[session]['centroid'][()] # usually [v, 2]
        
        # Calculate velocity
        vel = np.sqrt(np.sum(np.diff(centroid, axis=0)**2, axis=1))
        vel = np.concatenate(([0], vel)) # match length
        
        unique_s = np.unique(syllables)
        s_vels = {s: [] for s in unique_s}
        for s, v in zip(syllables, vel):
            s_vels[s].append(v)
            
        # Top 3 syllables
        u, c = np.unique(syllables, return_counts=True)
        idx = np.argsort(c)[::-1]
        
        print(f"File: {path}")
        print(f"{'Rank':<5} {'Syllable':<10} {'Usage (%)':<12} {'Med Vel':<10} {'Mean Vel':<10}")
        for i in range(min(5, len(idx))):
            s = u[idx[i]]
            usage = (c[idx[i]] / len(syllables)) * 100
            v_data = s_vels[s]
            print(f"{i+1:<5} {s:<10} {usage:<12.2f} {np.median(v_data):<10.2f} {np.mean(v_data):<10.2f}")

# Check Exp 2
analyze_v_vs_s("/home/isonaei/ABVFM_benchmark/KPMS/results/exp_20260111_1800_with_calibration/20260111-1827-exp2_ar2e06_full3e05/results.h5")
# Check Exp 5
analyze_v_vs_s("/home/isonaei/ABVFM_benchmark/KPMS/results/exp_20260111_1800_with_calibration/20260111-1906-exp5_ar1e05_full5e03/results.h5")
