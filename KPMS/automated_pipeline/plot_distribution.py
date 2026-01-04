
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.switch_backend('Agg')

def main():
    # --- Configuration ---
    project_dir = "/home/isonaei/ABVFM/KPMS/results/20260104_ctrl_30fps"
    models_dir = os.path.join(project_dir, "models")
    model_name = "20260104-053710-3"
    results_path = Path(models_dir) / model_name / "results_merged.h5"
    output_dir = Path(models_dir) / model_name / "merged_analysis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_path}...")
    
    # Load Results manually to avoid dependency issues
    all_syllables = []
    
    try:
        with h5py.File(results_path, 'r') as f:
            for session_key in f.keys():
                group = f[session_key]
                # Look for 'syllable' or 'z'
                if 'syllable' in group:
                    dset = group['syllable']
                elif 'z' in group:
                    dset = group['z']
                else:
                    continue
                
                vals = np.array(dset)
                if vals.ndim > 1:
                    vals = vals.flatten()
                
                all_syllables.append(vals)
    except Exception as e:
        print(f"Error loading H5: {e}")
        return

    if not all_syllables:
        print("No syllable data found.")
        return
        
    # Concatenate all sessions
    all_syllables_flat = np.concatenate(all_syllables)
    
    # --- Analysis ---
    unique_ids, counts = np.unique(all_syllables_flat, return_counts=True)
    total_frames = len(all_syllables_flat)
    
    # Calculate percentages
    percentages = (counts / total_frames) * 100
    
    # Sort by ID
    sorted_indices = np.argsort(unique_ids)
    unique_ids = unique_ids[sorted_indices]
    counts = counts[sorted_indices]
    percentages = percentages[sorted_indices]
    
    print(f"Total Frames: {total_frames}")
    print(f"Unique Syllables: {len(unique_ids)}")
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Bar plot
    bars = ax.bar(unique_ids.astype(str), percentages, color='skyblue', edgecolor='navy')
    
    # Formatting
    ax.set_xlabel("Syllable ID", fontsize=12)
    ax.set_ylabel("Usage Frequency (%)", fontsize=12)
    ax.set_title("Merged Syllable Distribution (Total 38 Motifs)", fontsize=14, fontweight='bold')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    
    # Save
    save_path = output_dir / "syllable_distribution.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved distribution plot to: {save_path}")
    
    # Also save a rank-ordered version (Pareto)
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    
    # Sort by count desc
    sort_idx = np.argsort(counts)[::-1]
    sorted_ids = unique_ids[sort_idx]
    sorted_pcts = percentages[sort_idx]
    
    bars2 = ax2.bar(range(len(sorted_ids)), sorted_pcts, color='salmon', edgecolor='maroon')
    ax2.set_xticks(range(len(sorted_ids)))
    ax2.set_xticklabels(sorted_ids.astype(str))
    
    ax2.set_xlabel("Syllable ID (Ranked)", fontsize=12)
    ax2.set_ylabel("Usage Frequency (%)", fontsize=12)
    ax2.set_title("Merged Syllable Distribution (Ranked by Usage)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path2 = output_dir / "syllable_distribution_ranked.png"
    plt.savefig(save_path2, dpi=300)
    print(f"Saved ranked distribution plot to: {save_path2}")

if __name__ == "__main__":
    main()
