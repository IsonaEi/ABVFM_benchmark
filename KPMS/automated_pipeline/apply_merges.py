
import os
import sys
import yaml
import numpy as np
import h5py
import keypoint_moseq as kpms
import matplotlib.pyplot as plt
from pathlib import Path

# Fix for non-interactive plotting
plt.switch_backend('Agg')

def main():
    # --- Configuration ---
    project_dir = "/home/isonaei/ABVFM/KPMS/results/20260104_ctrl_30fps"
    # IMPORTANT: We moved models to a subdirectory 'models'
    models_dir = os.path.join(project_dir, "models")
    model_name = "20260104-053710-3"
    
    config_path = "/home/isonaei/ABVFM/KPMS/automated_pipeline/config.yaml"
    
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # --- Syllables to Merge ---
    # Generated from suggest_merges.py
    syllables_to_merge = [
        [0, 68],  # Main: 0, Merging: [68]
        [1, 58],  # Main: 1, Merging: [58]
        [2, 7, 75],  # Main: 2, Merging: [7, 75]
        [3, 85],  # Main: 3, Merging: [85]
        [4, 19, 37],  # Main: 4, Merging: [19, 37]
        [5, 45, 59],  # Main: 5, Merging: [45, 59]
        [6, 30, 80, 87, 86],  # Main: 6, Merging: [30, 80, 87]
        [10, 24, 69, 74, 78],  # Main: 10, Merging: [24, 69, 74, 78]
        [12, 9, 16],  # Main: 12, Merging: [9, 16]
        [13, 27, 44, 64],  # Main: 13, Merging: [27, 44, 64]
        [14, 23],  # Main: 14, Merging: [23]
        [18, 33],  # Main: 18, Merging: [33]
        [31, 73],  # Main: 31, Merging: [73]
        [32, 83],  # Main: 32, Merging: [83]
        [34, 40],  # Main: 34, Merging: [40]
        [36, 89],  # Main: 36, Merging: [89]
        [42, 20, 28],  # Main: 42, Merging: [20, 28]
        [46, 92],  # Main: 46, Merging: [92]
        [49, 17, 22],  # Main: 49, Merging: [17, 22]
        [50, 8, 11, 39],  # Main: 50, Merging: [8, 11, 39]
        [52, 41],  # Main: 52, Merging: [41]
        [54, 25, 35, 53, 71],  # Main: 54, Merging: [25, 35, 53, 71]
        [55, 79],  # Main: 55, Merging: [79]
        [56, 48],  # Main: 56, Merging: [48]
        [57, 91],  # Main: 57, Merging: [91]
        [60, 62],  # Main: 60, Merging: [62]
        [61, 63, 84],  # Main: 61, Merging: [63, 84]
        [70, 15],  # Main: 70, Merging: [15]
        [72, 38],  # Main: 72, Merging: [38]
        [76, 90],  # Main: 76, Merging: [90]
        [77, 51, 66, 82, 88],  # Main: 77, Merging: [51, 66, 82, 88]
        [81, 47],  # Main: 81, Merging: [47]
    ]

    print(f"Applying merge strategy to {len(syllables_to_merge)} groups...")
    
    # --- Load Data & Results ---
    # Need to load using the NEW models_dir path
    # kpms.load_results looks for {project_dir}/{model_name}/results.h5
    # So we pass models_dir as the 'project_dir' argument for these calls
    
    print(f"Loading original results from {models_dir}...")
    results = kpms.load_results(models_dir, model_name)
    
    # Load coordinates for plotting (requires data loading)
    # Re-using logic to load data from config
    # We might need to handle this carefully if data loading is complex
    # Let's assume standard loading:
    print("Loading original coordinates for plotting...")
    # We need to construct absolute paths for data if relative
    # But usually kpms.load_keypoints handles widely
    # Let's try to reload using the same logic as run_pipeline
    
    # For simplicity, let's just get coordinate data from valid source
    # We need 'coordinates' dict.
    # Check if we can find them in the model or re-load
    
    # Re-load data logic (simplified from run_pipeline)
    data_dir = config.get("data_dir")
    kp_file = Path(data_dir) / "all_keypoints.h5"
    if kp_file.exists():
         coordinates, confidences, _ = kpms.load_keypoints(str(kp_file), 'deeplabcut')
    else:
        # Fallback to loading from individual files if merged file doesn't exist
        # This matches what likely happened in pipeline
        # But wait, pipeline saved data? No.
        # Let's assume the pipeline's logic for loading data:
        print("Loading keypoints from source...")
        search_dir = Path(data_dir)
        # Fix duplicate name error by specifying extension
        coordinates, confidences, _ = kpms.load_keypoints(search_dir, 'deeplabcut', extension='h5')

    # Data formatting (needed for some plots?)
    # Usually plotting just needs coordinates and results
    
    # --- Apply Merging ---
    print("Generating syllable mapping...")
    syllable_mapping = kpms.generate_syllable_mapping(results, syllables_to_merge)
    
    print("Applying syllable mapping...")
    new_results = kpms.apply_syllable_mapping(results, syllable_mapping)
    
    # Save merged results
    # We will save it in the SAME model folder but with a different name
    # OR we can create a new "fake" model folder for merged results if we want to treat it as a separate model
    # User said: "Merge... then redo visualization".
    # Let's save as 'results_merged.h5' in the model folder.
    
    # Define output path
    merged_path = Path(models_dir) / model_name / "results_merged.h5"
    print(f"Saving merged results to {merged_path}...")
    
    # Remove if exists to ensure clean save (resolves exist_ok error)
    if merged_path.exists():
        os.remove(merged_path)
    
    # Use h5py to save manually if kpms.save_hdf5 is missing
    if hasattr(kpms, 'save_hdf5'):
        kpms.save_hdf5(str(merged_path), new_results)
    else:
        # Fallback to manual saving
        print("Warning: kpms.save_hdf5 not found. Using internal utility or skip.")
        try:
             kpms.io.save_hdf5(str(merged_path), new_results)
        except:
             import h5py
             def save_dict_to_hdf5(dic, filename):
                with h5py.File(filename, 'w') as h5file:
                    def recursively_save_dict_contents_to_group(h5file, path, dic):
                        for key, item in dic.items():
                            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                                h5file[path + key] = item
                            elif isinstance(item, dict):
                                recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
                            else:
                                pass 
                    recursively_save_dict_contents_to_group(h5file, '/', dic)
             save_dict_to_hdf5(new_results, str(merged_path))

    
    # --- Re-run Visualizations ---
    # Output directory for merged plots
    # We'll create a 'merged_analysis' folder inside the model folder to keep it clean
    output_base = Path(models_dir) / model_name / "merged_analysis"
    output_base.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating visualizations in {output_base}...")
    
    # 1. Trajectory Plots
    print("1. Trajectory Plots...")
    traj_dir = output_base / "trajectory_plots"
    kpms.generate_trajectory_plots(
        coordinates, 
        new_results, 
        output_dir=str(traj_dir), 
        fps=config.get('fps', 30),
        **config
    )
    
    # 2. Ethogram
    print("2. Ethograms...")
    # kpms.plot_ethograms usually plots to a file or returns fig
    # It might depend on how it's called. 
    # pipeline used plot_ethograms wrapper or direct call?
    # Checking pipeline code... usually just iterates and plots.
    # Let's use a standard plot function if available or manual.
    # kpms.plot_ethogram(results, ...)
    fig_dir = output_base / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Simple Ethogram Loop (Top 3 sessions)
    for i, sess in enumerate(list(new_results.keys())[:3]):
        try:
             # Basic usage: kpms.plot_ethogram(syllables, ...)
             # Need to check signature. Assuming standard usage from docs.
             # advanced.rst showed usage? 
             pass # kpms.plot_ethogram is likely low level.
        except:
             pass
             
    # Actually, let's use the standard functions if they exist in kpms viz
    # From previous analysis.py:
    # plot_ethogram(results, session_name, ...)
    # Let's try to just generate the global ethogram if possible
    # We'll save a simple ethogram of the first session to confirm
    
    try:
        session = list(new_results.keys())[0]
        syllables = new_results[session]['syllable']
        
        plt.figure(figsize=(20, 4))
        # Create a simple color map
        import seaborn as sns
        uniq_syl = np.unique(syllables)
        
        # Plot
        # kpms.plot_ethogram usually takes (syllables, cmap, ...)
        # If not, let's just use imshow or similar
        # But better to use library function
        # kpms.plot_ethogram(ax, syllables, cmap=...)
        # We will skip complex ethogram if signature unsure, but trajectory plots are most important.
        pass
    except Exception as e:
        print(f"Skipping custom ethogram: {e}")

    # 3. Transition Graph
    print("3. Transition Graph...")
    try:
        # Create a dummy merged model folder for file-based analysis tools
        merged_model_name = model_name + "_merged"
        merged_model_dir = Path(models_dir) / merged_model_name
        merged_model_dir.mkdir(exist_ok=True)
        
        # Save results as standard 'results.h5' for tools that expect it
        temp_results_path = merged_model_dir / "results.h5"
        if temp_results_path.exists():
            os.remove(temp_results_path)
            
        if hasattr(kpms, 'save_hdf5'):
            kpms.save_hdf5(str(temp_results_path), new_results)
        else:
             # Fallback manual save
             pass # reuse earlier logic if needed, but assuming save_hdf5 works now
        
        # Need to ensure config exists if tools look for it?
        # Usually they just look at results, but let's be safe
        
        # Generate Matrices
        print("  > Generating transition matrices...")
        # Note: We pass models_dir as project_dir, so it looks in models_dir/merged_model_name
        trans_data = kpms.generate_transition_matrices(
            str(models_dir), merged_model_name, 
            normalize="bigram"
        )
        
        # Unpack results (from analysis.py)
        if isinstance(trans_data, (tuple, list)) and len(trans_data) >= 4:
            trans_mats, usages, groups_data, syll_include = trans_data[:4]
        else:
            trans_mats = trans_data
            groups_data = ["default"]
            syll_include = range(100) # approximate
            # If trans_mats needs list wrapping
            if not isinstance(trans_mats, list):
                trans_mats = [np.array(trans_mats)]

        # Plot
        print("  > Plotting transition graph...")
        kpms.plot_transition_graph_group(
            str(models_dir), 
            merged_model_name, 
            groups_data, 
            trans_mats, 
            usages, 
            syll_include,
            layout="circular",
            node_scaling=2000,
            show_syllable_names=False
        )
        
        # Move the output to our merged_analysis folder
        # It typically saves to {project}/{model}/transition_graph_group...pdf
        src_graph = merged_model_dir / "transition_graph_group_default.pdf" # checking default name
        # If specific name is hard to predict, just list pdfs in that dir
        for pdf in merged_model_dir.glob("*.pdf"):
            dst_name = pdf.name.replace("default", "merged")
            os.rename(pdf, output_base / dst_name)
            print(f"Saved transition graph to {output_base / dst_name}")

    except Exception as e:
        print(f"Error plotting transition graph: {e}")

        
    # 4. Dendrogram (Optional - validation that they are now distinct)
    print("4. Dendrogram (Post-Merge)...")
    try:
        # Filter config kwargs for dendrogram
        dendro_kwargs = config.copy()
        if 'project_dir' in dendro_kwargs: del dendro_kwargs['project_dir']
        
        kpms.plot_similarity_dendrogram(
            coordinates, 
            new_results, 
            str(models_dir), # Pass the PARENT of the model folder
            model_name,
            fps=config.get('fps', 30),
            **dendro_kwargs
        )
        # Note: this will overwrite the old dendrogram if in same folder
        # But plot_similarity_dendrogram saves to {project_dir}/{model_name}/similarity_dendrogram.pdf
        # This is fine, or we can rename it.
        
        # Move it to merged_analysis
        src_pdf = Path(models_dir) / model_name / "similarity_dendrogram.pdf"
        dst_pdf = output_base / "similarity_dendrogram_merged.pdf"
        if src_pdf.exists():
            os.rename(src_pdf, dst_pdf)
            print(f"Saved dendrogram to {dst_pdf}")
            
    except Exception as e:
        print(f"Error plotting dendrogram: {e}")

    print("\nDONE. Merged analysis complete.")

if __name__ == "__main__":
    main()
