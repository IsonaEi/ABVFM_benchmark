
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_usage(results_path):
    print(f"Analyzing: {results_path}")
    with h5py.File(results_path, 'r') as f:
        # Syllables are usually in 'syllables' group or directly accessible
        # In keypoint-moseq, they are stored per session
        # We need to aggregate across all sessions
        
        all_labels = []
        for session in f:
            if 'syllable' in f[session]:
                all_labels.extend(f[session]['syllable'][()])
        
        if not all_labels:
            print("No syllables found.")
            return

        all_labels = np.array(all_labels)
        
        # Calculate usage
        unique, counts = np.unique(all_labels, return_counts=True)
        usage = counts / len(all_labels)
        
        # Sort by usage
        idx = np.argsort(usage)[::-1]
        sorted_unique = unique[idx]
        sorted_usage = usage[idx]
        
        # Calculate durations per syllable
        durations = {s: [] for s in unique}
        # Run-length encoding
        for session in f:
            if 'syllable' not in f[session]: continue
            labels = f[session]['syllable'][()]
            if len(labels) == 0: continue
            
            # Simple RLE
            curr_label = labels[0]
            curr_dur = 1
            for i in range(1, len(labels)):
                if labels[i] == curr_label:
                    curr_dur += 1
                else:
                    if curr_label in durations:
                        durations[curr_label].append(curr_dur)
                    curr_label = labels[i]
                    curr_dur = 1
            if curr_label in durations:
                durations[curr_label].append(curr_dur)
        
        median_durs = {s: np.median(durations[s]) if durations[s] else 0 for s in unique}
        
        # Print top 10
        print(f"{'Rank':<5} {'Syllable':<10} {'Usage (%)':<12} {'Med Dur (ms)':<15}")
        print("-" * 45)
        for i in range(min(20, len(sorted_unique))):
            s = sorted_unique[i]
            u = sorted_usage[i] * 100
            d = median_durs[s] * (1000/30) # Assuming 30fps
            print(f"{i+1:<5} {s:<10} {u:<12.2f} {d:<15.2f}")

if __name__ == "__main__":
    # Check Exp 2 Merged
    merged_path = "/home/isonaei/ABVFM_benchmark/KPMS/results/exp_20260111_1800_with_calibration/20260111-1827-exp2_ar2e06_full3e05_merged_0pt05/results.h5"
    if Path(merged_path).exists():
        analyze_usage(merged_path)
    
    # Check Exp 2 Unmerged
    unmerged_path = "/home/isonaei/ABVFM_benchmark/KPMS/results/exp_20260111_1800_with_calibration/20260111-1827-exp2_ar2e06_full3e05/results.h5"
    if Path(unmerged_path).exists():
        print("\n--- UNMERGED ---")
        analyze_usage(unmerged_path)
