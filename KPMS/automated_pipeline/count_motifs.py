
import h5py
import numpy as np
import os

results_path = "/home/isonaei/ABVFM/KPMS/results/20260104_ctrl_30fps/20260104-053710-3/results.h5"

if not os.path.exists(results_path):
    print(f"Error: File not found at {results_path}")
    exit(1)

try:
    with h5py.File(results_path, 'r') as f:
        # Keypoint-MoSeq results usually group by recording name
        # Inside each recording group, there is a 'syllable' dataset (or 'z')
        
        all_syllables = []
        video_count = 0
        
        # Iterate over top-level keys (recording names)
        for key in f.keys():
            # Check if it's a group or dataset (results.h5 usually has groups per video)
            if isinstance(f[key], h5py.Group):
                if 'syllable' in f[key]:
                    data = f[key]['syllable'][:]
                    all_syllables.append(data)
                    video_count += 1
                elif 'z' in f[key]: # Legacy or raw model format
                    data = f[key]['z'][:]
                    all_syllables.append(data)
                    video_count += 1
        
        if all_syllables:
            merged = np.concatenate([s.flatten() for s in all_syllables])
            unique_syllables = np.unique(merged)
            
            # Filter out NaN or -1 if any (though typically they are 0-indexed integers)
            valid_syllables = unique_syllables[unique_syllables >= 0]
            
            print(f"Analysis of {results_path}")
            print(f"Processed {video_count} recordings.")
            print(f"Total Usages: {len(merged)}")
            print(f"Unique Motifs Found: {len(valid_syllables)}")
            print(f"Indices: {valid_syllables}")
            
            # Simple usage stats
            counts = dict(zip(*np.unique(merged, return_counts=True)))
            top_5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Top 5 Most Frequent Motifs: {top_5}")
            
            # Filter for rare motifs (< 30 occurrences)
            rare_motifs = {k: v for k, v in counts.items() if v < 30}
            print(f"\nMotifs with < 30 occurrences (Total: {len(rare_motifs)}):")
            if rare_motifs:
                sorted_rare = sorted(rare_motifs.items(), key=lambda x: x[1])
                for motif, count in sorted_rare:
                    print(f"  Motif {int(motif)}: {count}")
            else:
                print("  None found.")
            
        else:
            print("No syllable data found in the file structure.")
            
except Exception as e:
    print(f"An error occurred: {e}")
