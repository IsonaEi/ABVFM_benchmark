
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
import logging
import re
import shutil

# Setup detailed logging
logging.basicConfig(level=logging.INFO)

# --- B-SOiD Imports ---
BSOID_SOURCE_DIR = "/home/isonaei/ABVFM_benchmark/venvs/b-soid-official/B-SOID_source"
sys.path.append(BSOID_SOURCE_DIR)

try:
    from bsoid_umap.utils.likelihoodprocessing import main as lk_main
    from bsoid_umap.train import bsoid_feats
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Error: Could not import B-SOiD modules. Make sure the path is correct.")
    sys.exit(1)

# --- Utility Functions ---

def convert_h5_to_csv(h5_files, output_dir):
    """Converts DLC H5 files to CSV format expected by B-SOiD."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_files = []
    print(f"Converting {len(h5_files)} H5 files to CSV...")
    
    for h5_file in h5_files:
        try:
            df = pd.read_hdf(h5_file)
            
            # Handle MultiIndex: (scorer, individuals, bodyparts, coords) -> (scorer, bodyparts, coords)
            if 'individuals' in df.columns.names:
                df.columns = df.columns.droplevel('individuals')
            
            basename = os.path.splitext(os.path.basename(h5_file))[0]
            csv_path = os.path.join(output_dir, basename + '.csv')
            
            df.to_csv(csv_path)
            csv_files.append(csv_path)
            
        except Exception as e:
            print(f"Failed to convert {h5_file}: {e}")
            
    return csv_files

def load_latents(latent_path):
    """Loads DINO latent features for SSI calculation."""
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent file not found: {latent_path}")
    
    data = np.load(latent_path)
    keys = list(data.keys())
    for k in ['features', 'latents', 'embeddings']:
        if k in keys: return data[k]
    return data[keys[0]]

def calculate_ssi_local(features, labels, window=15):
    """
    Calculates State Stability Index (SSI) using Euclidean Distance ratio.
    SSI = Inter-state Distance / Intra-state Variance
    Matches Benchmark/src/metrics.py implementation.
    """
    labels = np.array(labels)
    features = np.array(features)
    
    # Identify transitions
    y = labels
    # Use valid data only? metrics.py assumes matched lengths.
    # But here we filter noise (-1) first
    valid_mask = y >= 0
    f_valid = features[valid_mask]
    l_valid = y[valid_mask]
    
    transitions = np.where(l_valid[1:] != l_valid[:-1])[0] + 1
    
    ssi_scores = []
    
    for t in transitions:
        # Define Pre and Post windows
        pre_idx = max(0, t - window)
        post_idx = min(len(f_valid), t + window)
        
        # Check if we have enough data
        if t - pre_idx < 2 or post_idx - t < 2: continue
        
        pre_feats = f_valid[pre_idx:t]
        post_feats = f_valid[t:post_idx]
        
        # Centroids
        pre_mean = np.mean(pre_feats, axis=0)
        post_mean = np.mean(post_feats, axis=0)
        
        # Inter-state Distance (Euclidean between centroids)
        inter_dist = np.linalg.norm(pre_mean - post_mean)
        
        # Intra-state Variance (Mean distance to own centroid)
        pre_var = np.mean(np.linalg.norm(pre_feats - pre_mean, axis=1))
        post_var = np.mean(np.linalg.norm(post_feats - post_mean, axis=1))
        intra_var = (pre_var + post_var) / 2 + 1e-6 # Avoid div by zero
        
        ssi = inter_dist / intra_var
        ssi_scores.append(ssi)
        
    return ssi_scores

def generate_gamma_baseline_labels(n_frames, n_classes=40, shape=1.5, scale=10.0):
    """
    Generates dummy labels using a Gamma distribution for bout duration.
    Matches KPMS/Benchmark logic.
    """
    print(f"Generating Gamma-distributed baseline ({n_frames} frames)...")
    labels = np.zeros(n_frames, dtype=int)
    idx = 0
    min_len = 1
    max_len = 150 # Cap at 5s (30fps)
    
    segment_counts = 0
    while idx < n_frames:
        dur = int(np.random.gamma(shape, scale))
        dur = max(min_len, min(max_len, dur))
        label = np.random.randint(0, n_classes)
        end_idx = min(idx + dur, n_frames)
        labels[idx:end_idx] = label
        idx = end_idx
        segment_counts += 1
    
    print(f"  Generated {segment_counts} segments.")
    return labels

def merge_rare_clusters_features(base_features, labels, min_freq=0.005):
    """
    Merges rare clusters (< min_freq) into nearest stable clusters
    based on the provided BASE FEATURES (Keypoints/B-SOiD feats).
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0] 
    
    total_frames = len(labels)
    threshold = total_frames * min_freq
    
    counts = {}
    centroids = {}
    
    # Check input shape
    if base_features.shape[0] > base_features.shape[1] and base_features.shape[1] < 5000:
        feats_T = base_features
    else:
        feats_T = base_features.T # (Time, Feat)
        
    min_len = min(len(labels), len(feats_T))
    labels_aligned = labels[:min_len]
    feats_aligned = feats_T[:min_len]
    
    for l in unique_labels:
        mask = (labels_aligned == l)
        counts[l] = np.sum(mask)
        if counts[l] > 0:
            centroids[l] = np.mean(feats_aligned[mask], axis=0)
        else:
            centroids[l] = np.zeros(feats_aligned.shape[1])
            
    stable_labels = [l for l in unique_labels if counts[l] >= threshold]
    rare_labels = [l for l in unique_labels if counts[l] < threshold]
    
    if not rare_labels or not stable_labels:
        return labels_aligned, 0
        
    mapping = {}
    for rare in rare_labels:
        c_rare = centroids[rare]
        best_stable = None
        min_dist = float('inf')
        for stable in stable_labels:
            dist = np.linalg.norm(c_rare - centroids[stable])
            if dist < min_dist:
                min_dist = dist
                best_stable = stable
        mapping[rare] = best_stable
        
    new_labels = labels_aligned.copy()
    for i, l in enumerate(labels_aligned):
        if l in mapping:
            new_labels[i] = mapping[l]
            
    return new_labels, len(rare_labels)


def calculate_duration_metric(labels, fps=30.0):
    if len(labels) == 0: return 0, 0
    changes = np.where(labels[:-1] != labels[1:])[0]
    starts = np.concatenate([[0], changes+1])
    ends = np.concatenate([changes, [len(labels)-1]])
    lengths = ends - starts + 1
    durations_ms = lengths / fps * 1000
    return len(lengths), np.median(durations_ms)

# --- Main Analysis Flow ---

def main():
    BASE_DIR = "/home/isonaei/ABVFM_benchmark/B-soid"
    DATA_DIR = os.path.join(BASE_DIR, "input_data_full")
    CSV_INPUT_DIR = os.path.join(DATA_DIR, "bsoid_csv_temp") 
    
    RESULTS_DIR = sorted(glob.glob(os.path.join(BASE_DIR, "results", "run_*")))[-1]
    print(f"Target Results: {RESULTS_DIR}")
    
    DINO_LATENT_PATH = "/home/isonaei/ABVFM_benchmark/castle-ai-feature-support-dinov3-model/projects/2026-01-05-17-00-40-Project_ctrl_30fps.mp4/latent/dinov3_vitb16/ctrl_30fps_ROI_1_dinov3_vitb16_ctr_rmbg.npz"
    dino_latents = load_latents(DINO_LATENT_PATH)
    
    # 1. Prepare CSV Data
    print("Preparing CSV data for feature extraction...")
    if not os.path.exists(CSV_INPUT_DIR) or not glob.glob(os.path.join(CSV_INPUT_DIR, "*.csv")):
        h5_files = glob.glob(os.path.join(DATA_DIR, "**/*.h5"), recursive=True)
        if not h5_files:
            print("Error: No H5 files found in data dir to regenerate CSVs.")
            return
        convert_h5_to_csv(h5_files, CSV_INPUT_DIR)
        
    # 2. Extract Raw Likelihood/Data
    _, training_data, _ = lk_main([CSV_INPUT_DIR])
    print(f"Loaded {len(training_data)} training files.")
    
    feature_cache = {}

    labels_files = glob.glob(os.path.join(RESULTS_DIR, "**/*_labels.csv"), recursive=True)
    all_ssi_data = []
    summary_stats = []
    
    print(f"Processing {len(labels_files)} result files...")
    
    for l_file in tqdm(labels_files):
        folder_name = os.path.basename(os.path.dirname(l_file))
        
        # Parse Window Size
        match = re.search(r"Win(\d+)f", folder_name)
        if not match: continue
        win_size = int(match.group(1))
        
        # --- Feature Recalc ---
        if win_size not in feature_cache:
            fake_fps = win_size * 10.0
            print(f"Recalculating B-SOiD features for Window={win_size}...")
            f_10fps, f_10fps_sc = bsoid_feats(training_data, fps=fake_fps)
            feature_cache[win_size] = f_10fps_sc
            
        bsoid_feats_curr = feature_cache[win_size]
        
        # Load Labels
        df = pd.read_csv(l_file)
        if 'B-SOiD_Label' not in df.columns: continue
        labels = df['B-SOiD_Label'].values
        
        # --- Keypoint Merge ---
        merged_labels, n_merged = merge_rare_clusters_features(
            bsoid_feats_curr, 
            labels, 
            min_freq=0.005
        )
        
        # --- SSI (DINO space) ---
        min_len = min(len(merged_labels), len(dino_latents))
        ssi_scores = calculate_ssi_local(dino_latents[:min_len], merged_labels[:min_len])
        n_seg, med_dur = calculate_duration_metric(merged_labels)
        
        if ssi_scores:
            for s in ssi_scores:
                all_ssi_data.append({
                    'Method': folder_name,
                    'SSI': s,
                    'Type': 'BSOiD'
                })
            
            summary_stats.append({
                "Method": folder_name,
                "Median_SSI": np.median(ssi_scores),
                "Mean_SSI": np.mean(ssi_scores),
                "N_Clusters": len(np.unique(merged_labels[merged_labels>=0])),
                "N_Segments": n_seg,
                "Median_Duration_ms": med_dur,
                "Merged_Clusters": n_merged
            })
        
    # --- Gamma (KPMS) Baseline ---
    print("Generating Gamma Baseline...")
    gamma_labels = generate_gamma_baseline_labels(len(dino_latents), n_classes=40)
    gamma_ssi = calculate_ssi_local(dino_latents, gamma_labels)
    n_seg_g, med_dur_g = calculate_duration_metric(gamma_labels)
    
    for s in gamma_ssi:
        all_ssi_data.append({'Method': 'Gamma Baseline', 'SSI': s, 'Type': 'Baseline'})
        
    summary_stats.append({
        "Method": 'Gamma Baseline',
        "Median_SSI": np.median(gamma_ssi),
        "Mean_SSI": np.mean(gamma_ssi),
        "N_Clusters": 40,
        "N_Segments": n_seg_g,
        "Median_Duration_ms": med_dur_g,
        "Merged_Clusters": 0
    })
    
    # --- Output ---
    ANALYSIS_DIR = os.path.join(RESULTS_DIR, "ssi_analysis_kpms_style")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    df_summary = pd.DataFrame(summary_stats).sort_values("Median_SSI", ascending=False)
    cols = ["Method", "Median_SSI", "Mean_SSI", "N_Clusters", "N_Segments", "Median_Duration_ms", "Merged_Clusters"]
    for c in cols: 
        if c not in df_summary.columns: df_summary[c] = 0
        
    csv_path = os.path.join(ANALYSIS_DIR, "all_bsoid_ssi_stats.csv")
    df_summary[cols].to_csv(csv_path, index=False)
    print(f"Saved stats to {csv_path}")
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=pd.DataFrame(all_ssi_data), x='Method', y='SSI', order=df_summary['Method'], palette='viridis', inner='quartile')
    plt.xticks(rotation=45, ha='right')
    plt.title("B-SOiD (Keypoint-Merged) vs Gamma Baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "all_bsoid_ssi_violin.png"), dpi=300)
    
    print(df_summary[cols].head())

if __name__ == "__main__":
    main()
