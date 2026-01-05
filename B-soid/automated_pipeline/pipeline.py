
import os
import datetime
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


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
    
    for i, key in enumerate(keys):
        labels = results_dict[key]
        
        # Image strip
        im_data = labels.reshape(1, -1)
        duration = len(labels) / fps
        
        ax = axes[i]
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

def plot_combined_ethograms(all_results_by_win, output_dir, filename_prefix, base_fps=30.0, cmap='tab20'):
    """
    Plots a consolidated ethogram showing multiple window sizes for the same session.
    all_results_by_win: {win_ms: {filename: labels_array}}
    """
    if not all_results_by_win: return
    
    # We assume we want to plot for each file, but if data is limited, we might just have one.
    # Let's find all unique filenames across all window sizes.
    all_filenames = set()
    for win_ms in all_results_by_win:
        all_filenames.update(all_results_by_win[win_ms].keys())
    
    for filename in sorted(all_filenames):
        win_sizes = sorted(all_results_by_win.keys())
        n_rows = len(win_sizes)
        
        fig, axes = plt.subplots(n_rows, 1, figsize=(15, n_rows * 0.8 + 2), sharex=True)
        if n_rows == 1: axes = [axes]
        
        # Find global max label for consistent colors if possible
        global_max = 0
        for win_ms in win_sizes:
            if filename in all_results_by_win[win_ms]:
                labels = all_results_by_win[win_ms][filename]
                if len(labels) > 0:
                    global_max = max(global_max, np.max(labels))
        
        for i, win_ms in enumerate(win_sizes):
            ax = axes[i]
            if filename in all_results_by_win[win_ms]:
                labels = all_results_by_win[win_ms][filename]
                im_data = labels.reshape(1, -1)
                
                # Use same extent for all to align them in time
                # B-SOiD integrations cover roughly the same total time
                total_duration = len(labels) * (win_ms / 1000.0) # Approx?
                # Actually it's better to use the video duration if we had it, 
                # but since we are comparing integration bins, let's just stretch them to [0, 1] or a fixed time.
                # Let's assume the first window size sets the reference duration if they differ slightly.
                
                ax.imshow(im_data, aspect='auto', cmap=cmap, vmin=0, vmax=global_max,
                          interpolation='nearest', extent=[0, 100, 0, 1]) # Percent scale
                
            ax.set_yticks([])
            ax.set_ylabel(f"{win_ms}ms", rotation=0, ha='right', fontsize=10)
            
        axes[0].set_title(f"Window Size Comparison - {os.path.basename(filename)}", fontsize=14)
        axes[-1].set_xlabel("Session Progress (%)", fontsize=12)
        
        clean_name = os.path.basename(filename).replace('.csv', '')
        out_path = os.path.join(output_dir, f"combined_ethogram_{clean_name}_{filename_prefix}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved combined ethogram to {out_path}")

def plot_parameter_summary(summary_df, output_dir, ts):
    """
    Plots the summary of parameter search: Duration vs Window Size.
    """
    if summary_df.empty:
        logging.warning("No summary data to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    # Scatter plot: Window Size vs Mean Duration
    # Color by Number of Clusters?
    sns.scatterplot(data=summary_df, x='Window_Size_ms', y='Mean_Duration_ms', 
                    hue='Num_Clusters', palette='viridis', s=100)
    
    # Add line to show trend
    sns.lineplot(data=summary_df, x='Window_Size_ms', y='Mean_Duration_ms', 
                 color='gray', alpha=0.5, sort=True)
    
    plt.title("B-SOiD Parameter Search: Duration vs Window Size")
    plt.xlabel("Window Size (ms)")
    plt.ylabel("Mean Duration (ms)")
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(output_dir, f"parameter_summary_{ts}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"Saved parameter summary plot to {out_path}")

def train_classifier(feats, labels, config):
    """
    Trains MLP classifier on stable clusters to predict all points.
    Fills in noise (-1) and smooths assignments.
    """
    mlp_params = config.get('mlp_params', {})
    
    # Transpose feats to (n_samples, n_features) for sklearn
    feats = feats.T
    
    # Filter out noise for training
    valid_mask = labels >= 0
    feats_train_subset = feats[valid_mask]
    labels_train_subset = labels[valid_mask]
    
    unique_labels = np.unique(labels_train_subset)
    if len(unique_labels) < 2:
        logging.warning("Not enough classes for MLP training.")
        return labels, None
        
    logging.info(f"Training MLP on {len(feats_train_subset)} instances (excluding {np.sum(~valid_mask)} noise points)...")
    
    # Initialize Classifier
    # Default B-SOiD params if not in config
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(mlp_params.get('hidden_layer_sizes', (100, 10))),
        activation=mlp_params.get('activation', 'logistic'),
        solver=mlp_params.get('solver', 'adam'),
        alpha=mlp_params.get('alpha', 0.0001),
        learning_rate_init=mlp_params.get('learning_rate_init', 0.001),
        max_iter=mlp_params.get('max_iter', 1000),
        random_state=42 # Fixed seed for reproducibility of classifier
    )
    
    # Fit
    clf.fit(feats_train_subset, labels_train_subset)
    
    # Validate (internal 5-fold CV score for logging)
    scores = cross_val_score(clf, feats_train_subset, labels_train_subset, cv=5, n_jobs=1)
    logging.info(f"MLP Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Predict ALL data (including previously noisy points)
    final_labels = clf.predict(feats)
    
    return final_labels, clf

def main():
    config = load_config()
    setup_environment(config)
    
    # Import B-SOiD modules now
    try:
        # Patch BASE_PATH before importing other modules that depend on it
        import bsoid_umap.config.LOCAL_CONFIG
        bsoid_umap.config.LOCAL_CONFIG.BASE_PATH = '' # Set to empty string so we can use absolute paths
        
        from bsoid_umap.utils.likelihoodprocessing import main as lk_main
        from bsoid_umap.train import bsoid_feats, bsoid_umap_embed
    except ImportError as e:
        logging.error(f"Could not import B-SOiD modules: {e}")
        return

    # Process Input
    input_dir = config['input_dir']
    h5_files = glob.glob(os.path.join(input_dir, "**/*.h5"), recursive=True)
    video_files = glob.glob(os.path.join(input_dir, "**/*.mp4"), recursive=True)
    
    if not h5_files:
        logging.error("No H5 files found.")
        return

    # Determine FPS
    base_fps = config.get('fps')
    if not base_fps and video_files:
        base_fps = get_video_fps(video_files[0])
        logging.info(f"Detected FPS from video: {base_fps}")
    elif not base_fps:
        base_fps = 30.0
        logging.warning("No FPS specified and no video found. Defaulting to 30.0.")
    
    config['fps'] = base_fps 
    
    # Convert H5 to CSV
    temp_csv_dir = os.path.join(input_dir, "bsoid_csv_temp")
    csv_files = convert_h5_to_csv(h5_files, temp_csv_dir)
    
    # B-SOiD Processing - Likelihood (Common for all runs? No, maybe not)
    # Actually Likelihood processing is independent of window size. It just reads CSV.
    # But bsoid_feats depends on window size.
    
    logging.info("Processing likelihood and extracting features...")
    filenames, training_data, perc_rect = lk_main([temp_csv_dir])
    
    # Helper to calculate expected length
    def calc_feat_length(raw_data_shape, current_fps):
        # Logic from bsoid_feats
        # feats shape is determined by raw data.
        # dxy_r loop runs dataRange times.
        # feats is same length as raw (roughly)
        # integration loop:
        bin_size = int(round(current_fps/10))
        # range(bin_size, data_range, bin_size)
        return len(range(bin_size, raw_data_shape[0], bin_size))

    # Parameter Search Logic
    param_search = config.get('param_search', {})
    if param_search.get('enabled', False):
        window_sizes = param_search.get('window_sizes_ms', [100])
        logging.info(f"Starting Parameter Search. Window Sizes: {window_sizes}")
    else:
        window_sizes = [100] # Default 100ms if search disabled
        logging.info("Parameter Search Disabled. Running single pass with 100ms window.")

    # Prepare main output directory
    # Create a unique run folder with timestamp (to Minute)
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    main_output_dir = os.path.join(config['output_dir'], f"run_{run_timestamp}")
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    logging.info(f"Summary results will be saved in: {main_output_dir}")
        
    ts_run = time.strftime("%Y%m%d_%H%M%S")
    summary_results = []
    all_win_results = {} # {win_ms: {filename: labels}}
    
    for win_ms in window_sizes:
        logging.info(f"--- Running for Window Size: {win_ms}ms ---")
        
        # Calculate Fake FPS
        # B-SOID window = int(fps * 0.1) = 1/10th of a second = 100ms default.
        # If we want Win_ms, we need: Win_ms = (1/10 second) ... wait.
        # B-SOiD hardcodes window = round(fps/10).
        # Window Duration (s) = Window_Frames / FPS
        # Window_Frames = FPS / 10
        # Window Duration = (FPS/10) / FPS = 0.1 sec = 100 ms.
        # Wait, if `window` is frame count, it is always 100ms if it's FPS/10.
        # So B-SOiD ALWAYS uses a 100ms window regardless of FPS?
        # NO. Let's check `bsoid_feats` source code logic again (mental check).
        # `window = int(round(fps/10))`
        # If FPS=30, window=3 frames. 3 frames @ 30fps = 0.1s.
        # If FPS=60, window=6 frames. 6 frames @ 60fps = 0.1s.
        # So B-SOiD is DESIGNED to validly lock to 100ms.
        
        # KEY INSIGHT from Paper: "we reran the pipeline with falsely inflated frame rates"
        # If we tell B-SOiD the video is 300 FPS (but it's actually 30 FPS).
        # Then B-SOiD sets window = 300/10 = 30 frames.
        # But since the REAL data is 30 FPS.
        # 30 frames covers 1.0 second!!
        # So "Inflating FPS" -> "Larger Window".
        
        # Relation:
        # B-SOiD calculates `win_len = fps_param / 10`.
        # Real Duration (s) = win_len / REAL_FPS.
        # Real Duration = (fps_param / 10) / REAL_FPS.
        
        # We want Target Duration (T_target).
        # T_target = (fps_param / 10) / REAL_FPS
        # => fps_param = T_target * REAL_FPS * 10
        
        # Example: Target 200ms = 0.2s. Real FPS = 30.
        # fps_param = 0.2 * 30 * 10 = 60.
        # Check: fps_param=60 -> win_len = 6. 
        # Real duration of 6 frames @ 30fps = 6/30 = 0.2s. Correct.
        
        target_s = win_ms / 1000.0
        fake_fps = target_s * base_fps * 10.0
        
        logging.info(f"Target: {win_ms}ms. Real FPS: {base_fps}. Fake FPS param: {fake_fps}")
        
        # 2. Extract Features (with fake_fps)
        f_10fps, f_10fps_sc = bsoid_feats(training_data, fps=fake_fps)
        
        # 3. UMAP
        logging.info("Running UMAP...")
        umap_params = config['umap_params']
        trained_umap, umap_embeddings = bsoid_umap_embed(f_10fps_sc, umap_params)
        
        # 4. Optimization & Clustering (HDBSCAN)
        logging.info("Running HDBSCAN Optimization...")
        # Note: Optimization also uses FPS to calculate duration. 
        # Should we use Real FPS or Fake FPS for duration calc?
        # `run_optimization` calculates `bout_times`.
        # `bout_times` are in indices. 
        # The indices step is dependent on the `bsoid_feats` integration step.
        # `bsoid_feats` integration step: `range(0, len, round(fps/10))` or similar?
        # Actually `bsoid_feats` outputs features at what rate?
        # It's called "10fps" usually?
        # B-SOiD paper says it extracts features then integrates/bins them.
        # Usually it tries to output features at a standardized rate?
        # Let's assume standard B-SOiD workflow outputs features that represent *epochs*.
        # The epoch length is `round(fps/10)` frames.
        # So each data point in `f_10fps` represents `round(fake_fps/10)` frames of the INPUT data?
        # Wait.
        # If `fake_fps` causes `win_len` to change.
        # And the "integration" step likely uses `win_len` as the step size (stride).
        # `range(0, N, win_len)`
        # If win_len is larger (30 frames instead of 3), we have FEWER feature points.
        # But each point covers more time (1.0s instead of 0.1s).
        
        # So when calculating duration in `run_optimization`:
        # We have N bouts. Each bout is X points.
        # Duration = X * (Time per point).
        # What is Time Per Point?
        # Time Per Point = win_len / REAL_FPS.
        # Time Per Point = (fake_fps/10) / base_fps.
        
        # So we must pass the CORRECT time-per-point info to `run_optimization`.
        # `run_optimization` currently uses `config['fps']`.
        # If it does `frames / fps`, that assumes 1 point = 1 frame? 
        # NO. `run_optimization` logic:
        # `bout_times = np.array(valid_bout_lens) / fps * 1000`
        # This assumes `valid_bout_lens` is in FRAMES?
        # NO, `valid_bout_lens` is in count of LABELS.
        # B-SOiD output labels are at 10Hz typically (if native).
        # If we mess with it, the output rate changes.
        
        # Let's adjust `run_optimization` or pass the effective FPS for duration calc.
        # Effective FPS of the OUTPUT (labels) = 1 / (Time per point).
        # Time per point = (fake_fps/10) / base_fps.
        # Effective FPS = base_fps / (fake_fps/10) = 10 * base_fps / fake_fps.
        
        # Let's calculate effective_output_fps.
        effective_output_fps = (10.0 * base_fps) / fake_fps
        
        # We need to temporarily update config['fps'] just for `run_optimization` 
        # so it calculates duration ms correctly.
        config_copy = config.copy()
        config_copy['fps'] = effective_output_fps 
        
        best_assignments, best_min_size = run_optimization(umap_embeddings, config_copy)
        
        # --- MLP CLASSIFIER STEP ---
        # Post-process with Neural Network if clusters exist
        num_clusters_hdb = len(np.unique(best_assignments[best_assignments >= 0]))
        
        if num_clusters_hdb >= 2:
            logging.info("Training MLP Classifier to refine clusters...")
            final_labels, clf = train_classifier(f_10fps_sc, best_assignments, config)
            # Use final_labels for output
            session_labels_source = final_labels
        else:
            logging.info("Skipping MLP training (not enough clusters). Using HDBSCAN labels.")
            session_labels_source = best_assignments
            
        # 5. Save Results for this run
        run_output_dir = os.path.join(main_output_dir, f"win_{win_ms}ms")
        if not os.path.exists(run_output_dir):
            os.makedirs(run_output_dir)
            
        current_idx = 0
        results_dict = {}
        flat_filenames = filenames[0]
        
        logging.info(f"Splitting results back to sessions (Effective Output FPS: {effective_output_fps:.2f})...")
        
        for i, filename in enumerate(flat_filenames):
            # Recalculate feature length using FAKE FPS (since that determined the stride)
            n_predictions = calc_feat_length(training_data[i].shape, fake_fps)
            
            end_idx = current_idx + n_predictions
            if end_idx > len(session_labels_source): end_idx = len(session_labels_source)
                
            session_labels = session_labels_source[current_idx:end_idx]
            results_dict[filename] = session_labels
            
            # Save CSV
            basename = os.path.basename(filename).replace('.csv', '')
            out_csv = os.path.join(run_output_dir, f"{basename}_labels.csv")
            
            # Time axis based on EFFECTIVE output FPS
            # dt = 1 / effective_output_fps
            time_axis = np.arange(len(session_labels)) * (1.0 / effective_output_fps)
            
            pd.DataFrame({
                "Time": time_axis,
                "B-SOiD_Label": session_labels
            }).to_csv(out_csv, index=False)
            
            current_idx = end_idx
            
        # Plot Ethogram
        plot_ethograms(results_dict, run_output_dir, f"ethogram_win{win_ms}", fps=effective_output_fps)
        all_win_results[win_ms] = results_dict
        
        # Collect Summary Stats
        # Recalculate mean duration from optimization result?
        # Or just re-calc here to comprise all files?
        # Let's iterate assignments again.
        
        # Collect Summary Stats
        labels = session_labels_source # Use the refined labels
        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            # We can re-use the logic or just trust the optimization log?
            # Better to calc from final assignment to be sure.
            
            # Simple bout duration logic
            # This logic is repeating `run_optimization` part, could be refactored, but fine inline.
            bouts = []
            curr = labels[0]
            count = 0
            bout_lens = []
            
            for l in labels:
                if l == curr:
                    count += 1
                else:
                    if curr != -1:
                        bout_lens.append(count)
                    curr = l
                    count = 1
            if curr != -1:
                bout_lens.append(count)
            
            if bout_lens:
                bout_times_ms = (np.array(bout_lens) / effective_output_fps) * 1000
                mean_dur = np.mean(bout_times_ms)
                num_clusters = len(np.unique(valid_labels))
            else:
                mean_dur = 0
                num_clusters = 0
        else:
            mean_dur = 0
            num_clusters = 0
            
        summary_results.append({
            "Window_Size_ms": win_ms,
            "Fake_FPS": fake_fps,
            "Num_Clusters": num_clusters,
            "Mean_Duration_ms": mean_dur
        })
        
        logging.info(f"Run {win_ms}ms complete. Clusters: {num_clusters}, Mean Duration: {mean_dur:.2f}ms")
        
    # Generate Summary CSV and Plot
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv = os.path.join(main_output_dir, f"parameter_summary_{ts_run}.csv")
        summary_df.to_csv(summary_csv, index=False)
        logging.info(f"Saved parameter summary CSV to {summary_csv}")
        
        try:
            plot_parameter_summary(summary_df, main_output_dir, ts_run)
            plot_combined_ethograms(all_win_results, main_output_dir, f"all_windows_{run_timestamp}", base_fps=base_fps)
        except Exception as e:
            logging.error(f"Failed to plot summaries: {e}")
            
    logging.info(f"Pipeline completed. All results saved to {main_output_dir}")

if __name__ == "__main__":
    main()
