
import os
import sys
import yaml
import numpy as np
import keypoint_moseq as kpms
import matplotlib.pyplot as plt
from pathlib import Path

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

    # --- Load Data ---
    print(f"Loading merged results from {models_dir}...")
    # Load the results we just created
    merged_results_path = Path(models_dir) / model_name / "results_merged.h5"
    if hasattr(kpms, 'load_hdf5'):
        new_results = kpms.load_hdf5(str(merged_results_path))
    else:
        # Fallback to load_results if needed, but we saved it manually/standard
        # kpms.load_results expects standard structure.
        # Let's try simple load first
        try:
             import h5py
             def load_dict_from_hdf5(filename):
                with h5py.File(filename, 'r') as h5file:
                    def recursively_load_dict_contents_from_group(h5file, path):
                        ans = {}
                        for key, item in h5file[path].items():
                            if isinstance(item, h5py._hl.dataset.Dataset):
                                ans[key] = item[()]
                            elif isinstance(item, h5py._hl.group.Group):
                                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
                        return ans
                    return recursively_load_dict_contents_from_group(h5file, '/')
             new_results = load_dict_from_hdf5(str(merged_results_path))
        except Exception as e:
            print(f"Error loading h5 manually: {e}")
            return

    print("Loading keypoints...")
    data_dir = config.get("data_dir")
    search_dir = Path(data_dir)
    coordinates, confidences, _ = kpms.load_keypoints(search_dir, 'deeplabcut', extension='h5')
    
    # --- Debugging ---
    print("\n--- DEBUG INFO ---")
    print(f"Results keys (first 5): {list(new_results.keys())[:5]}")
    print(f"Coordinates keys (first 5): {list(coordinates.keys())[:5]}")
    
    # Check interaction
    common_keys = set(new_results.keys()) & set(coordinates.keys())
    print(f"Number of common sessions: {len(common_keys)}")
    
    if len(common_keys) == 0:
        print("CRITICAL ERROR: No matching sessions between Results and Coordinates!")
        return

    # Check unique syllables in Results
    all_syls = []
    for k in common_keys:
        if 'syllable' in new_results[k]:
            all_syls.append(new_results[k]['syllable'])
        elif 'z' in new_results[k]: # handle different naming
             all_syls.append(new_results[k]['z'])
             
    if all_syls:
        all_syls_flat = np.concatenate(all_syls)
        unique_syls = np.unique(all_syls_flat)
        print(f"Unique syllables in merged results: {len(unique_syls)}")
        print(f"Syllable IDs: {unique_syls}")
        
        # Check frequency of Syllable 0 vs others
        from collections import Counter
        counts = Counter(all_syls_flat)
        print("Top 10 most frequent syllables:", counts.most_common(10))
    else:
        print("No syllable data found in results for common keys.")
        
    print("--- END DEBUG ---\n")
    
    # --- Re-run Trajectory Plots ---
    output_base = Path(models_dir) / model_name / "merged_analysis"
    traj_dir = output_base / "trajectory_plots"
    
    # Ensure dir exists
    traj_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Regenerating trajectory plots in {traj_dir}...")
    print("Setting min_frequency=0 and min_duration=0 to show ALL syllables.")
    
    # We explicitly pass config but OVERRIDE the filters
    kpms.generate_trajectory_plots(
        coordinates, 
        new_results, 
        output_dir=str(traj_dir), 
        fps=config.get('fps', 30),
        min_frequency=0, 
        min_duration=0,
        density_sample=False, # Essential to include rare syllables with < 50 instances
        **config
    )
    
    print("Done.")

if __name__ == "__main__":
    main()
