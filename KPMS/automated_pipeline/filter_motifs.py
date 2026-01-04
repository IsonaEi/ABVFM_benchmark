
import os
import sys
import numpy as np
import h5py
import yaml
from pathlib import Path

def get_run_lengths(arr):
    """
    Computes run lengths for a 1D array.
    Returns: (values, lengths)
    """
    if len(arr) == 0:
        return np.array([]), np.array([])
        
    # Find points where values change
    # Append -1 at the end to ensure the last run is counted if we diff
    # Or standard itertools.groupby approach, or numpy diff
    
    # Numpy way:
    # 1. Identify indices where value changes
    mismatch = arr[1:] != arr[:-1]
    i = np.append(np.where(mismatch), len(arr) - 1)
    
    # 2. Compute lengths
    run_lengths = np.diff(np.append(-1, i))
    
    # 3. Get values at those positions
    run_values = arr[i]
    
    return run_values, run_lengths

def main():
    # Configuration
    project_dir = "/home/isonaei/ABVFM/KPMS/results/20260104_ctrl_30fps"
    model_name = "20260104-053710-3"
    results_path = Path(project_dir) / model_name / "results.h5"
    threshold_frames = 10
    
    print(f"Loading results from: {results_path}")
    
    if not results_path.exists():
        print(f"Error: File not found: {results_path}")
        return

    # Load Results
    # Structure of results.h5 typically has keys for each recording
    # Each recording group has 'syllable' or 'z' dataset
    
    syllable_max_durations = {} # {syllable_id: max_duration}
    
    with h5py.File(results_path, 'r') as f:
        # Iterate over all sessions/recordings
        for key in f.keys():
            # Skip non-recording top level keys if any (metadata usually separate or attributes)
            # In kpms results.h5, top level keys are usually session names
            
            group = f[key]
            
            # Identify syllable dataset
            if 'syllable' in group:
                dset = group['syllable']
            elif 'z' in group:
                dset = group['z']
            else:
                continue # Skip if no syllables found
                
            z = np.array(dset)
            
            if z.ndim > 1:
                z = z.flatten()
                
            # Compute runs for this session
            vals, lengths = get_run_lengths(z)
            
            # Update max durations
            for v, l in zip(vals, lengths):
                current_max = syllable_max_durations.get(v, 0)
                if l > current_max:
                    syllable_max_durations[v] = l
                    
    # Sorting and Analysis
    all_syllables = sorted(syllable_max_durations.keys())
    
    compliant_motifs = [] # Max duration < 10
    non_compliant_motifs = [] # Max duration >= 10
    
    for s in all_syllables:
        max_dur = syllable_max_durations[s]
        if max_dur < threshold_frames:
            compliant_motifs.append(s)
        else:
            non_compliant_motifs.append(s)
            
    print("="*60)
    print(f"Analysis of Model: {model_name}")
    print(f"Total Syllables Found: {len(all_syllables)}")
    print(f"Condition: Max continuous segment < {threshold_frames} frames")
    print("="*60)
    
    print(f"\n[Compliant] Motifs with ALL segments < {threshold_frames} frames (Flickers/Noise):")
    print(f"Count: {len(compliant_motifs)}")
    print(f"IDs: {compliant_motifs}")
    
    print("\n" + "-"*60 + "\n")
    
    print(f"[Non-Compliant] Motifs with AT LEAST ONE segment >= {threshold_frames} frames (Valid/Stable):")
    print(f"Count: {len(non_compliant_motifs)}")
    print(f"IDs: {non_compliant_motifs}")
    print("="*60)

if __name__ == "__main__":
    main()
