
import os
import sys
import numpy as np
import keypoint_moseq as kpms
from pathlib import Path

def main():
    # --- Configuration ---
    project_dir = "/home/isonaei/ABVFM/KPMS/results/20260104_ctrl_30fps"
    model_name = "20260104-053710-3"
    
    # ID Lists from previous analysis
    short_motifs = [7, 8, 9, 11, 15, 16, 17, 19, 20, 22, 23, 24, 25, 27, 28, 30, 33, 35, 37, 38, 39, 40, 41, 44, 45, 47, 48, 51, 53, 58, 59, 62, 63, 64, 66, 68, 69, 71, 73, 74, 75, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]
    stable_motifs = [0, 1, 2, 3, 4, 5, 6, 10, 12, 13, 14, 18, 21, 26, 29, 31, 32, 34, 36, 42, 43, 46, 49, 50, 52, 54, 55, 56, 57, 60, 61, 65, 67, 70, 72, 76, 77, 81]

    # --- Load Model ---
    print(f"Loading checkpoint: {model_name}")
    model, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
    
    x_all = model['states']['x'] # (n_sessions, n_frames_x, latent_dim)
    z_all = model['states']['z'] # (n_sessions, n_frames_z)
    
    # --- Flatten and Align ---
    # We need to handle the frame offset. 
    # Usually len(x) = len(z) + lag.
    # We will align by taking the *last* N frames of x, where N = len(z).
    # This assumes the 'z' labels correspond to the later frames in 'x'.
    
    x_flat_list = []
    z_flat_list = []
    
    for i in range(x_all.shape[0]):
        xi = x_all[i]
        zi = z_all[i]
        
        len_x = xi.shape[0]
        len_z = zi.shape[0]
        
        if len_z > len_x:
            # Should not happen based on previous inspection
            print(f"Warning: Z longer than X for session {i}")
            continue
            
        # Align ends
        # z[0] usually corresponds to x[lag]
        xi_aligned = xi[-len_z:]
        
        x_flat_list.append(xi_aligned)
        z_flat_list.append(zi)
        
    x_flat = np.concatenate(x_flat_list, axis=0) # (Total_Frames, Latent_Dim)
    z_flat = np.concatenate(z_flat_list, axis=0) # (Total_Frames,)

    print(f"Total aligned frames: {x_flat.shape[0]}")
    
    # --- Compute Centroids ---
    print("Computing motif centroids...")
    centroids = {}
    
    all_indices = set(short_motifs + stable_motifs)
    
    for mid in all_indices:
        # Find frames for this motif
        mask = (z_flat == mid)
        if np.sum(mask) == 0:
            print(f"Warning: Motif {mid} has no frames in loaded model state!")
            centroids[mid] = np.zeros(x_flat.shape[1])
            continue
            
        # Compute mean of latent vectors
        centroid = np.mean(x_flat[mask], axis=0)
        centroids[mid] = centroid
        
    # --- Find Nearest Stable Neighbors ---
    print("\n" + "="*80)
    print("SUGGESTED MERGES (Short Motif -> Nearest Stable Motif)")
    print("="*80)
    print(f"{'Short Motif':<12} | {'Count':<8} | {'Suggested Target':<16} | {'Distance':<10}")
    print("-" * 60)
    
    suggestions = {} # short -> stable
    
    for short_id in short_motifs:
        frame_count = np.sum(z_flat == short_id)
        
        c_short = centroids[short_id]
        
        best_target = None
        min_dist = float('inf')
        
        for stable_id in stable_motifs:
            c_stable = centroids[stable_id]
            
            # Euclidean distance
            dist = np.linalg.norm(c_short - c_stable)
            
            if dist < min_dist:
                min_dist = dist
                best_target = stable_id
                
        suggestions[short_id] = best_target
        print(f"{short_id:<12} | {frame_count:<8} | {best_target:<16} | {min_dist:.4f}")
        
    # --- Generate Copy-Paste List for User ---
    print("\n" + "="*80)
    print("Format for 'syllables_to_merge' list:")
    print("="*80)
    
    # Group by target
    merge_groups = {} # target -> [short_pd, short_id, ...]
    
    for short_id, target_id in suggestions.items():
        if target_id not in merge_groups:
            merge_groups[target_id] = []
        merge_groups[target_id].append(short_id)
        
    print("syllables_to_merge = [")
    for target in sorted(merge_groups.keys()):
        # List contains the target and all its short mergers
        group = [target] + sorted(merge_groups[target])
        print(f"    {group},  # Main: {target}, Merging: {merge_groups[target]}")
    print("]")
    
if __name__ == "__main__":
    main()
