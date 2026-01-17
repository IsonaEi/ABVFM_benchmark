
import os
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
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
            
            # DLC SuperAnimal/Multi-animal has MultiIndex levels: (scorer, individuals, bodyparts, coords)
            # B-SOID expects 3 levels: (scorer, bodyparts, coords)
            if 'individuals' in df.columns.names:
                df.columns = df.columns.droplevel('individuals')
            
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
        window_sizes_frames = param_search.get('window_sizes_frames', [3])
        # Support for UMAP parameter search
        umap_search_params = param_search.get('umap_prams_search', {})
        n_neighbors_list = umap_search_params.get('n_neighbors', [config['umap_params']['n_neighbors']])
        min_dist_list = umap_search_params.get('min_dist', [config['umap_params']['min_dist']])
        
        logging.info(f"Starting Grid Search.")
        logging.info(f"Windows (Frames): {window_sizes_frames}")
        logging.info(f"n_neighbors: {n_neighbors_list}")
        logging.info(f"min_dist: {min_dist_list}")
    else:
        # Default single run
        window_sizes_frames = [3] # Default ~100ms at 30fps
        n_neighbors_list = [config['umap_params']['n_neighbors']]
        min_dist_list = [config['umap_params']['min_dist']]
        logging.info("Parameter Search Disabled. Running single pass.")

    # Prepare main output directory
    # Create a unique run folder with timestamp (to Minute)
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    main_output_dir = os.path.join(config['output_dir'], f"run_{run_timestamp}")
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    logging.info(f"Summary results will be saved in: {main_output_dir}")
        
    ts_run = time.strftime("%Y%m%d_%H%M%S")
    summary_results = []
    
    # --- OUTER LOOP: Window Size (Requires B-SOiD Feats Re-calc) ---
    for win_frames in window_sizes_frames:
        # win_frames: desired window size in frames
        # B-SOiD hardcoded logic: window_frames = round(fps/10)
        # Therefore: fps_param = win_frames * 10
        fake_fps = win_frames * 10.0
        
        # Calculate Effective FPS for duration calculations
        effective_output_fps = (10.0 * base_fps) / fake_fps
        win_ms = (win_frames / base_fps) * 1000.0
        
        logging.info(f"=== Window Size: {win_frames} frames ({win_ms:.2f} ms) ===")
        logging.info(f"    Fake FPS Param: {fake_fps}")
        logging.info(f"    Effective Output FPS: {effective_output_fps:.2f} Hz")
        
        # 1. Extract Features (B-SOiD Feats)
        # This is the heavy step for Window Size changes
        logging.info("Extracting features with new window size...")
        try:
            f_10fps, f_10fps_sc = bsoid_feats(training_data, fps=fake_fps)
        except Exception as e:
            logging.error(f"Feature extraction failed for win_frames={win_frames}: {e}")
            continue

        # --- INNER LOOP: UMAP Parameters (Requires Embedding Re-calc) ---
        import itertools
        umap_grid = list(itertools.product(n_neighbors_list, min_dist_list))
        
        for n_neighbors, min_dist in umap_grid:
            combo_name = f"Win{win_frames}f_N{n_neighbors}_D{min_dist}"
            logging.info(f"--- Testing Combo: {combo_name} (n_neighbors={n_neighbors}, min_dist={min_dist}) ---")
            
            # 2. UMAP
            logging.info("Running UMAP...")
            # Create temp params dict for this run
            current_umap_params = config['umap_params'].copy()
            current_umap_params['n_neighbors'] = n_neighbors
            current_umap_params['min_dist'] = min_dist
            
            try:
                trained_umap, umap_embeddings = bsoid_umap_embed(f_10fps_sc, current_umap_params)
            except Exception as e:
                logging.error(f"UMAP failed for {combo_name}: {e}")
                continue
            
            # 3. Optimization & Clustering (HDBSCAN)
            logging.info("Running HDBSCAN Optimization...")
            
            # Update config for optimization duration calcs with effective FPS
            config_copy = config.copy()
            config_copy['fps'] = effective_output_fps 
            
            best_assignments, best_min_size = run_optimization(umap_embeddings, config_copy)
            
            # Calculate Clustering Quality Metrics
            import hdbscan
            try:
                final_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=best_min_size,
                    min_samples=best_min_size,
                    prediction_data=True
                ).fit(umap_embeddings)
                dbcv_score = final_clusterer.relative_validity_
            except Exception:
                dbcv_score = 0.0
                
            # --- MLP & Finalizing ---
            num_clusters_hdb = len(np.unique(best_assignments[best_assignments >= 0]))
            
            if num_clusters_hdb >= 2:
                final_labels, clf = train_classifier(f_10fps_sc, best_assignments, config)
                session_labels_source = final_labels
            else:
                session_labels_source = best_assignments
            
            # --- Saving Results ---
            run_output_dir = os.path.join(main_output_dir, combo_name)
            if not os.path.exists(run_output_dir):
                os.makedirs(run_output_dir)
                
            # Split back to files
            current_idx = 0
            results_dict = {}
            # Check safely if filenames is a nested list or flat
            if isinstance(filenames, list) and len(filenames) > 0 and isinstance(filenames[0], list):
                 flat_filenames = filenames[0]
            else:
                 flat_filenames = filenames

            for i, filename in enumerate(flat_filenames):
                # Recalculate feature length using FAKE FPS logic
                n_predictions = calc_feat_length(training_data[i].shape, fake_fps)
                
                # Check for bounds
                end_idx = current_idx + n_predictions
                if end_idx > len(session_labels_source): 
                     # This happens if calc is slightly off due to rounding
                     end_idx = len(session_labels_source)
                
                # Safe slice
                if current_idx < len(session_labels_source):
                     session_labels = session_labels_source[current_idx:end_idx]
                else:
                     session_labels = np.array([])

                results_dict[filename] = session_labels
                
                # Save Label CSV
                basename = os.path.basename(filename).replace('.csv', '')
                out_csv = os.path.join(run_output_dir, f"{basename}_labels.csv")
                
                time_axis = np.arange(len(session_labels)) * (1.0 / effective_output_fps)
                pd.DataFrame({
                    "Time": time_axis,
                    "B-SOiD_Label": session_labels
                }).to_csv(out_csv, index=False)
                
                current_idx = end_idx

            # Plot Ethogram
            plot_ethograms(results_dict, run_output_dir, f"ethogram_{combo_name}", fps=effective_output_fps)
            
            # --- Collect Statistics for Summary ---
            valid_labels = session_labels_source[session_labels_source >= 0]
            if len(valid_labels) > 0:
                num_clusters = len(np.unique(valid_labels))
                # Calculate mean duration from source (across all files)
                bouts = []
                curr = session_labels_source[0]
                count = 0
                bout_lens = []
                for l in session_labels_source:
                    if l == curr: count += 1
                    else:
                        if curr != -1: bout_lens.append(count)
                        curr = l
                        count = 1
                if curr != -1: bout_lens.append(count)
                
                if bout_lens:
                    bout_times_ms = (np.array(bout_lens) / effective_output_fps) * 1000
                    mean_dur = np.mean(bout_times_ms)
                    # Duration Violation rate (<100ms)
                    violation_rate = np.mean(bout_times_ms < 100)
                else:
                    mean_dur = 0
                    violation_rate = 0
            else:
                num_clusters = 0
                mean_dur = 0
                violation_rate = 0
                
            noise_ratio = np.sum(session_labels_source == -1) / len(session_labels_source)
            
            summary_results.append({
                "Run_ID": combo_name,
                "Win_Frames": win_frames,
                "Win_MS": win_ms,
                "Fake_FPS": fake_fps,
                "N_Neighbors": n_neighbors,
                "Min_Dist": min_dist,
                "Num_Clusters": num_clusters,
                "Mean_Duration_ms": mean_dur,
                "DBCV_Score": dbcv_score,
                "Noise_Ratio": noise_ratio,
                "Duration_Violation_Rate": violation_rate
            })
            
            logging.info(f"Run {combo_name} done. Clusters: {num_clusters}, DBCV: {dbcv_score:.3f}, Dur: {mean_dur:.1f}ms")

    # Generate Summary CSV
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv = os.path.join(main_output_dir, f"parameter_summary_{ts_run}.csv")
        summary_df.to_csv(summary_csv, index=False)
        logging.info(f"Saved parameter summary CSV to {summary_csv}")
        
    logging.info(f"Pipeline completed. All results saved to {main_output_dir}")

if __name__ == "__main__":
    main()
