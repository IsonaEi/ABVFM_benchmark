
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keypoint_moseq as kpms
from pathlib import Path
from tqdm import tqdm

# Fix for non-interactive plotting
plt.switch_backend('Agg')

def main():
    # --- Configuration ---
    project_dir = "/home/isonaei/ABVFM/KPMS/results/20260104_ctrl_30fps"
    models_dir = os.path.join(project_dir, "models")
    model_name = "20260104-053710-3"
    
    config_path = "/home/isonaei/ABVFM/KPMS/automated_pipeline/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    merged_results_path = Path(models_dir) / model_name / "results_merged.h5"
    output_base = Path(models_dir) / model_name / "merged_analysis"
    
    print(f"Loading merged results from {merged_results_path}...")
    if hasattr(kpms, 'load_hdf5'):
        new_results = kpms.load_hdf5(str(merged_results_path))
    else:
        # Fallback manual load
        import h5py
        new_results = {}
        with h5py.File(merged_results_path, 'r') as f:
            for k in f.keys():
                grp = f[k]
                new_results[k] = {dk: grp[dk][()] for dk in grp.keys()}
                
    # Load coordinates (needed for labeled videos)
    print("Loading keypoints...")
    data_dir = config.get("data_dir")
    search_dir = Path(data_dir)
    coordinates, confidences, _ = kpms.load_keypoints(search_dir, 'deeplabcut', extension='h5')
    
    # ---------------------------------------------------------
    # 1. Ethograms (Behavioral Timelines)
    # ---------------------------------------------------------
    print("\n1. Generating Ethograms...")
    etho_dir = output_base / "ethograms"
    etho_dir.mkdir(exist_ok=True)
    
    # Create a consistent color map for 38 syllables
    unique_syllables = np.sort(np.unique(np.concatenate([v['syllable'] for v in new_results.values()])))
    n_syllables = len(unique_syllables)
    
    # Use 'tab20' or 'husl' for distinctness
    # 'tab20' only has 20 colors, so we cycle or use 'husl'
    palette = sns.color_palette("husl", n_syllables)
    color_map = {s: palette[i] for i, s in enumerate(unique_syllables)}
    
    for session_name, data in tqdm(new_results.items(), desc="Plotting Ethograms"):
        syllables = data['syllable']
        
        # Plot
        fig, ax = plt.subplots(figsize=(15, 2))
        
        # Make an image (1, n_frames)
        # Map syllables to indices in usage map for color consistency? 
        # Or just use imshow with a ListedColormap?
        
        # Simpler approach: Create an image array of RGB values
        # This gives us exact control over colors
        img_arr = np.zeros((1, len(syllables), 3))
        for t, s in enumerate(syllables):
            img_arr[0, t, :] = color_map.get(s, (0,0,0))
            
        ax.imshow(img_arr, aspect='auto', interpolation='nearest')
        
        ax.set_yticks([])
        ax.set_xlabel("Time (Frames)")
        ax.set_title(f"Ethogram: {session_name}")
        
        # Add legend? Too many for 38. Maybe just main bars.
        
        plt.tight_layout()
        plt.savefig(etho_dir / f"{session_name}_ethogram.png", dpi=150)
        plt.close(fig)

    # ---------------------------------------------------------
    # 2. Transition Matrix (Heatmap)
    # ---------------------------------------------------------
    print("\n2. Generating Transition Matrix Heatmap...")
    
    # Calculate bigram counts
    msg = np.zeros((n_syllables, n_syllables))
    # Map syllable ID to matrix index
    syl_to_idx = {s: i for i, s in enumerate(unique_syllables)}
    
    for data in new_results.values():
        s = data['syllable']
        # Pairs: (t, t+1)
        # Only count transitions (where s[t] != s[t+1])? 
        # Standard transition matrix includes self-transitions or usually normalized row-wise.
        # Let's count all (t, t+1)
        for i in range(len(s) - 1):
            src = s[i]
            dst = s[i+1]
            if src in syl_to_idx and dst in syl_to_idx:
                msg[syl_to_idx[src], syl_to_idx[dst]] += 1
                
    # Normalize (Row-wise probability)
    row_sums = msg.sum(axis=1, keepdims=True)
    P = np.divide(msg, row_sums, out=np.zeros_like(msg), where=row_sums!=0)
    
    # Plot Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(P, xticklabels=unique_syllables, yticklabels=unique_syllables, 
                cmap="viridis", vmax=0.3, # Cap max color to see detail
                ax=ax)
    
    ax.set_xlabel("Next Syllable")
    ax.set_ylabel("Current Syllable")
    ax.set_title("Transition Probability Matrix")
    
    plt.tight_layout()
    plt.savefig(output_base / "figures" / "transition_matrix_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"Saved transition matrix to {output_base}/figures/transition_matrix_heatmap.png")

    # ---------------------------------------------------------
    # 3. Labeled Videos (Grid Movies)
    # ---------------------------------------------------------
    print("\n3. Generating Grid Movies (Labeled Videos)...")
    if 'video_dir' not in config:
        print("Warning: 'video_dir' not in config. Skipping videos.")
    else:
        video_dir = config['video_dir']
        print(f"Using videos from: {video_dir}")
        
        lbl_vid_dir = output_base / "grid_movies"
        lbl_vid_dir.mkdir(exist_ok=True)
        
        # Use generate_grid_movies
        try:
             # Based on viz.py signature:
             # generate_grid_movies(results, coordinates, video_dir, output_dir=None, ...)
             kpms.generate_grid_movies(
                 results=new_results,
                 coordinates=coordinates,
                 video_dir=video_dir,
                 output_dir=str(lbl_vid_dir),
                 plot_options=config.get('grid_movies', {}).get('plot_options', {}),
                 fps=config.get('fps', 30),
                 keypoints_only=False, 
                 **config.get('grid_movies', {}) 
             )
        except Exception as e:
            print(f"Error generating grid movies: {e}")
            import traceback
            traceback.print_exc()

    print("\nDONE.")

if __name__ == "__main__":
    main()
