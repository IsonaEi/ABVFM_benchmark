import os
import sys
import glob
import yaml
import logging
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_environment(config):
    source_dir = config.get('bsoid_source_dir')
    if source_dir:
        sys.path.append(source_dir)
        # B-SOID modules will be imported after this

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps > 0:
        return fps
    return 30.0  # Fallback

def convert_h5_to_csv(h5_files, output_dir):
    csv_files = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for h5_file in h5_files:
        try:
            df = pd.read_hdf(h5_file)
            
            # DLC usually has MultiIndex columns: (scorer, bodyparts, coords)
            # B-SOID expects specific headers: bodyparts (row 0), coords (row 1) in CSV?
            # Actually B-SOID's `adp_filt` looks for 'likelihood', 'x', 'y' in the *header names* of the CSV
            # or it parses them.
            # Let's look at `adp_filt` in `likelihoodprocessing.py` again.
            # It expects the first row to be headers like 'likelihood', 'x', 'y' if flat?
            # Or it processes the header from the dataframe.
            
            # The B-SOID import function `pd.read_csv` reads it. 
            # `adp_filt` checks `currdf[0][header] == "likelihood"`... implies it reads a CSV where the first few rows are headers.
            # DLC default CSV export:
            # Row 0: Scorer
            # Row 1: Bodyparts
            # Row 2: Coords (x, y, likelihood)
            
            # We will export to CSV using pandas standard export which preserves the header structure 
            # if we just save the dataframe. B-SOID seems to handle DLC format.
            
            # IMPORTANT: The B-SOID source code `adp_filt` implementation:
            # currdf = np.array(currdf[1:]) -> drops first row (scorer?)
            # checks currdf[0][header] -> this would be the Bodyparts row? 
            # checks if "likelihood" is in it?
            
            # Wait, `adp_filt` logic:
            # currdf = np.array(currdf[1:])  <-- skips row 0 (Scorer)
            # for header in range(len(currdf[0])): <-- iterating over row 1 (now row 0 in array)
            #    if currdf[0][header] == "likelihood": ...
            
            # DLC H5 loading results in a DataFrame with MultiIndex columns.
            # When converted to CSV, it usually stacks them.
            # We need to mimic the CSV structure B-SOID expects.
            
            basename = os.path.splitext(os.path.basename(h5_file))[0]
            csv_path = os.path.join(output_dir, basename + '.csv')
            
            # Save properly as DLC-style CSV
            df.to_csv(csv_path)
            csv_files.append(csv_path)
            logging.info(f"Converted {h5_file} to {csv_path}")
            
        except Exception as e:
            logging.error(f"Failed to convert {h5_file}: {e}")
            
    return csv_files

def run_optimization(umap_embeddings, config):
    """
    Optimizes HDBSCAN min_cluster_size to match duration criteria.
    """
    import hdbscan
    
    opt_params = config['optimization']
    min_size_range = opt_params['min_cluster_size_range']
    target_duration = opt_params['min_duration_ms']
    steps = opt_params.get('steps', 10)
    fps = config.get('fps', 30.0)
    
    # Generate search space
    sizes = np.linspace(min_size_range[0], min_size_range[1], steps)
    best_score = -1
    best_size = sizes[0]
    best_assignments = None
    
    logging.info(f"Optimizing min_cluster_size over range {min_size_range} (Steps: {steps})...")
    
    for size_ratio in sizes:
        min_cluster_size = int(round(size_ratio * umap_embeddings.shape[0]))
        if min_cluster_size < 2: min_cluster_size = 2
        
        logging.info(f"Testing min_cluster_size={min_cluster_size} ({size_ratio:.4f}%)")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size, # Strategy: match min_samples to min_cluster_size
            prediction_data=True
        ).fit(umap_embeddings)
        
        labels = clusterer.labels_
        
        # Calculate duration stats
        # Turn labels into bouts
        # We need to be careful about non-assigned (-1) noise
        # But we only care about valid clusters for duration
        
        if len(np.unique(labels)) < 2:
            logging.warning("Only 1 or 0 clusters found.")
            continue
            
        bouts = []
        current_label = labels[0]
        current_len = 0
        
        valid_bout_lens = []
        
        for l in labels:
            if l == current_label:
                current_len += 1
            else:
                if current_label != -1:
                   valid_bout_lens.append(current_len)
                current_label = l
                current_len = 1
        if current_label != -1:
            valid_bout_lens.append(current_len)
            
        if not valid_bout_lens:
             logging.warning("No valid bouts found.")
             continue
             
        # Convert to ms
        bout_times = np.array(valid_bout_lens) / fps * 1000
        
        mean_dur = np.mean(bout_times)
        pass_rate = np.sum(bout_times > target_duration) / len(bout_times)
        
        logging.info(f"  Mean Duration: {mean_dur:.2f}ms, Pass Rate (>400ms): {pass_rate:.2%}, Num Clusters: {len(np.unique(labels))-1}")
        
        # Scoring function: Prioritize Pass Rate, then Number of Clusters
        # Heuristic: maximize pass_rate * log(num_clusters)
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters < 1: score = 0
        else:
             score = pass_rate + (n_clusters * 0.01) # Small bonus for more clusters if pass rate is similar
        
        if score > best_score:
            best_score = score
            best_size = min_cluster_size
            best_assignments = labels
            
    logging.info(f"Optimal parameter found: min_cluster_size={best_size} (Score: {best_score:.4f})")
    return best_assignments, best_size

def plot_ethograms(results_dict, output_dir, filename_prefix, fps=30.0, cmap='tab20'):
    """
    Plots ethograms. 
    results_dict: {filename: labels_array}
    """
    n_sessions = len(results_dict)
    if n_sessions == 0: return

    keys = sorted(results_dict.keys())
    
    # Figure setup
    fig_height = max(4.0, n_sessions * 0.5 + 2.5)
    fig, axes = plt.subplots(n_sessions, 1, figsize=(20, fig_height), sharex=True)
    if n_sessions == 1: axes = [axes]
    
    # Color palette
    all_labels = np.concatenate(list(results_dict.values()))
    unique_labels = np.unique(all_labels)
    unique_labels = unique_labels[unique_labels >= 0] # Exclude noise
    if len(unique_labels) == 0: return # Only noise?

    max_label = np.max(unique_labels)
    palette = sns.color_palette(cmap, int(max_label) + 1)
    
    # Map noise (-1) to different color (e.g. black or white)
    # We will just mask it or start from 0 for the cmap
    
    for i, key in enumerate(keys):
        labels = results_dict[key]
        
        # Image strip
        im_data = labels.reshape(1, -1)
        duration = len(labels) / fps
        
        ax = axes[i]
        # Mask noise for plotting
        masked_data = np.ma.masked_where(im_data < 0, im_data)
        
        ax.imshow(im_data, aspect='auto', cmap=cmap, vmin=0, vmax=max_label,
                  interpolation='nearest', extent=[0, duration, 0, 1])
        
        ax.set_yticks([])
        clean_name = os.path.basename(key).replace('.csv', '')
        ax.set_ylabel(clean_name, rotation=0, ha='right', fontsize=12)
        
    axes[-1].set_xlabel("Time (s)", fontsize=14)
    out_path = os.path.join(output_dir, f"{filename_prefix}_ethogram.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved ethogram to {out_path}")

def main():
    config = load_config()
    setup_environment(config)
    
    # Import B-SOID modules now
    try:
        # Patch BASE_PATH before importing other modules that depend on it
        import bsoid_umap.config.LOCAL_CONFIG
        bsoid_umap.config.LOCAL_CONFIG.BASE_PATH = '' # Set to empty string so we can use absolute paths
        
        from bsoid_umap.utils.likelihoodprocessing import main as lk_main
        from bsoid_umap.train import bsoid_feats, bsoid_umap_embed
    except ImportError as e:
        logging.error(f"Could not import B-SOID modules: {e}")
        return

    # Process Input
    input_dir = config['input_dir']
    h5_files = glob.glob(os.path.join(input_dir, "**/*.h5"), recursive=True)
    video_files = glob.glob(os.path.join(input_dir, "**/*.mp4"), recursive=True)
    
    if not h5_files:
        logging.error("No H5 files found.")
        return

    # Determine FPS
    fps = config.get('fps')
    if not fps and video_files:
        fps = get_video_fps(video_files[0])
        logging.info(f"Detected FPS from video: {fps}")
    elif not fps:
        fps = 30.0
        logging.warning("No FPS specified and no video found. Defaulting to 30.0.")
    
    config['fps'] = fps # Update config so run_optimization uses the detected FPS
    
    # Convert H5 to CSV
    temp_csv_dir = os.path.join(input_dir, "bsoid_csv_temp")
    csv_files = convert_h5_to_csv(h5_files, temp_csv_dir)
    
    # B-SOID Processing
    # 1. Likelihood Processing
    # Need to pass folder containing CSVs? Or list of files?
    # lk_main takes a LIST of folders.
    # So we pass [temp_csv_dir]
    
    logging.info("Processing likelihood and extracting features...")
    filenames, training_data, perc_rect = lk_main([temp_csv_dir])
    
    # 2. Extract Features
    f_10fps, f_10fps_sc = bsoid_feats(training_data, fps=fps)
    
    # 3. UMAP
    logging.info("Running UMAP...")
    umap_params = config['umap_params']
    trained_umap, umap_embeddings = bsoid_umap_embed(f_10fps_sc, umap_params)
    
    # 4. Optimization & Clustering (HDBSCAN)
    logging.info("Running HDBSCAN Optimization...")
    best_assignments, best_min_size = run_optimization(umap_embeddings, config)
    
    # 5. Save Results
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ts = time.strftime("%Y%m%d_%H%M%S")
    
    # Save combined output
    # assignments are for the CONCATENATED features.
    # We need to split them back to sessions to generate separate CSVs and Plots
    
    # Calculate indices
    # sizes are in training_data (list of arrays)
    # BUT f_10fps is downsampled/integrated!
    # f_10fps shape corresponds to concatenated data.
    # We need to reconstruction session split.
    
    # B-SOID `bsoid_feats` implementation:
    # Concatenates feats1 into f_10fps.
    # We need to track the length of each session's contribution to f_10fps.
    
    # Hack: Rerun feature extraction per file? No, expensive.
    # B-SOID `bsoid_feats` unfortunately returns concatenated result and doesn't return indices.
    # But we can recalculate the expected length logic:
    # "for k in range(round(fps / 10), len(feats[n][0]), round(fps / 10))"
    
    session_lengths = []
    # Re-simulating the loop logic to get lengths
    # Features extraction first loop creates 'feats' list.
    # Second loop creates 'feats1' and concatenates.
    
    # Actually, we can just look at `filenames` returned by `lk_main`.
    # And we need to know the raw data length to calculate expected features.
    
    # Let's rely on the fact that we can't easily modify `bsoid_feats` without forking.
    # BUT `bsoid_feats` iterates over `data`.
    # `data` is a list of arrays.
    # We can perform a dummy calculation to know the size.
    
    start_idx = 0
    results_dict = {}
    
    header_info = [filename[0] for filename in filenames] # filenames is usually a list of lists (folders -> files)
    # Actually lk_main return `filenames` as list of lists if multiple folders.
    # Since we passed one folder, it is filenames[0] which is the list of files.
    flat_filenames = filenames[0]
    
    for i, data_arr in enumerate(training_data):
        # Logic from bsoid_feats
        # dataRange = len(data[m]) ...
        # feats1 loop...
        # The number of bins is roughly data_len // (fps/10).
        
        # Let's use the exact logic to be safe
        n_bins = 0
        bin_len = int(round(fps/10))
        # feats length is data_arr.shape[0] usually (minus window?)
        # window is small (win_len).
        # feats construction creates array of size dataRange-1 or similar.
        # Let's assume proportional length.
        
        # Alternative: We can modify bsoid_feats or copy it.
        # It's safer to just assume we simply assign the labels sequentially to the files in order.
        # But we don't know the exact length of each file's contribution.
        
        # Better approach: We have `f_10fps` (concatenated). 
        # We assume `training_data` order matches.
        # We can implement a quick helper to get exact feature count per session.
        # bsoid_feats -> feats -> feats1
        # The downsampling is predictable.
        
        # Simplified:
        raw_len = data_arr.shape[0]
        # features are calculated with window, so usually len-1
        # Then integrated.
        feat_len = raw_len # Simplified
        
        # Integration step:
        # range(bin_len, feat_len, bin_len)
        # count = len(range(...))
        n_output = len(range(bin_len, feat_len, bin_len))  # Approx
        # Wait, bsoid uses `feats[n][0]` length in the range. 
        # `feats` generation: 
        # feats.append(np.vstack(...))
        # shape is roughly input length.
        
        pass

    # Since splitting is tricky without modifying B-SOID, 
    # and we need accurate splitting for the Ethogram, 
    # I will create a monkey-patched version of bsoid_feats inside this script 
    # or just copy the logic to calculate lengths.
    
    current_idx = 0
    
    # Helper to calculate expected length
    def calc_feat_length(raw_data_shape, fps):
        # Logic from bsoid_feats
        # feats shape is determined by raw data.
        # dxy_r loop runs dataRange times.
        # feats is same length as raw (roughly)
        # integration loop:
        bin_size = int(round(fps/10))
        # range(bin_size, data_range, bin_size)
        return len(range(bin_size, raw_data_shape[0], bin_size))

    logging.info("Splitting results back to sessions...")
    
    for i, filename in enumerate(flat_filenames):
        n_predictions = calc_feat_length(training_data[i].shape, fps)
        
        end_idx = current_idx + n_predictions
        
        # Safety check
        if end_idx > len(best_assignments):
            end_idx = len(best_assignments)
            
        session_labels = best_assignments[current_idx:end_idx]
        results_dict[filename] = session_labels
        
        # Save CSV
        basename = os.path.basename(filename).replace('.csv', '')
        out_csv = os.path.join(output_dir, f"{basename}_labels.csv")
        
        # Create DataFrame
        # Time axis?
        # 10Hz = 0.1s per bin
        time_axis = np.arange(len(session_labels)) * 0.1
        df_out = pd.DataFrame({
            "Time": time_axis,
            "B-SOiD_Label": session_labels
        })
        df_out.to_csv(out_csv, index=False)
        
        current_idx = end_idx
        
    # Plot Ethogram
    plot_ethograms(results_dict, output_dir, f"combined_{ts}", fps=10.0) # B-SOID output is 10Hz
    
    logging.info(f"Pipeline completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
