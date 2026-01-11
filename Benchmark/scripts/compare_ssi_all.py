
import os
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import re

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.loader import DataLoader
from src.metrics import MetricsEngine

def parse_params_from_path(path):
    # Extract Win, N, D from path like ".../Win7f_N60_D0.1/..."
    match = re.search(r"Win(\d+)f_N(\d+)_D([\d\.]+)", path)
    if match:
        return int(match.group(1)), int(match.group(2)), float(match.group(3))
    return None, None, None

def main():
    # 1. Setup & Load Embeddings
    config_path = "config.yaml"
    loader = DataLoader(config_path=config_path)
    metrics = MetricsEngine()
    
    # Load DINO Embeddings
    dino_path = "/home/isonaei/ABVFM_benchmark/castle-ai-feature-support-dinov3-model/projects/2026-01-05-17-00-40-Project_ctrl_30fps.mp4/latent/dinov3_vitb16/ctrl_30fps_ROI_1_dinov3_vitb16_ctr_rmbg.npz"
    features = loader.load_embeddings(dino_path)
    if features is None:
        print("Failed to load embeddings.")
        return

    # 2. Find All B-SOiD Result Files
    search_dirs = [
        "/home/isonaei/ABVFM_benchmark/B-soid/results/run_20260110_2112"
    ]
    
    label_files = []
    for d in search_dirs:
        # Find all *_labels.csv recursively
        files = glob.glob(os.path.join(d, "**/*_labels.csv"), recursive=True)
        label_files.extend(files)
        
    print(f"Found {len(label_files)} label files.")
    
    all_ssi_data = [] # List of dicts {Method, SSI} for seaborn
    summary_stats = []
    
    # Target Length (Frame Count)
    target_len = len(features)
    print(f"Target Length (30 FPS): {target_len}")
    
    # 3. Process Each File
    for fpath in label_files:
        # Parse Params
        win, n, dist = parse_params_from_path(fpath)
        if win is None: continue
        
        method_name = f"W{win}_N{n}_D{dist}"
        
        # Load Labels
        df = pd.read_csv(fpath)
        if 'B-SOiD_Label' not in df.columns: continue
        raw_labels = df['B-SOiD_Label'].values
        
        # Upsample Labels to match 30 FPS Video/DINO
        # This restores the original sampling rate for correct duration calc
        labels = loader.resample_labels(raw_labels, target_len)
        
        # Calculate Stats (Classes, Mean Dur) using 30 FPS
        stats = metrics.compute_label_stats(labels, fps=30.0)
        
        # Calculate SSI (Aligned)
        ssi_scores = metrics.compute_ssi(features, labels, window=15)
        
        # Store for Plot
        for score in ssi_scores:
            all_ssi_data.append({
                'Method': method_name,
                'Window': win,
                'SSI': score,
                'Type': 'Real'
            })
            
        # Store Summary
        summary_stats.append({
            'Method': method_name,
            'Window': win,
            'N_Neighbors': n,
            'Min_Dist': dist,
            'Median_SSI': np.median(ssi_scores),
            'Mean_SSI': np.mean(ssi_scores),
            'N_Clusters': stats['n_classes'],
            'Median_Duration_ms': stats.get('median_duration_ms', 0),
            'N_Events': len(ssi_scores)
        })
        print(f"  {method_name}: SSI={np.median(ssi_scores):.3f}, Clusters={stats['n_classes']}, Dur={stats.get('median_duration_ms', 0):.1f}ms")

    # 4. Generate Random Baseline
    print("Generating Random Baseline...")
    # Generate labels resembling natural stats (e.g. 30 classes, ~500ms duration)
    rand_labels = loader.generate_dummy_labels(target_len, n_classes=30)
    rand_ssi = metrics.compute_ssi(features, rand_labels, window=15)
    
    for score in rand_ssi:
        all_ssi_data.append({
            'Method': 'Random',
            'Window': 0, # Dummy
            'SSI': score,
            'Type': 'Baseline'
        })
        
    stats_rand = metrics.compute_label_stats(rand_labels, fps=30.0)
    summary_stats.append({
        'Method': 'Random',
        'Window': 0,
        'Median_SSI': np.median(rand_ssi),
        'Mean_SSI': np.mean(rand_ssi),
        'N_Clusters': 30,
        'Median_Duration_ms': stats_rand.get('median_duration_ms', 0),
        'N_Events': len(rand_ssi)
    })

    # 5. Create DataFrame & Sort
    df_plot = pd.DataFrame(all_ssi_data)
    df_summary = pd.DataFrame(summary_stats)
    
    # Sort by Median SSI
    sort_order = df_summary.sort_values('Median_SSI', ascending=False)['Method'].tolist()
    
    # 6. Plotting (Violin)
    plt.figure(figsize=(40, 10))
    sns.violinplot(data=df_plot, x='Method', y='SSI', order=sort_order, density_norm='width', palette="viridis")
    plt.xticks(rotation=90, fontsize=8)
    plt.title("Comprehensive SSI Analysis: Refined B-SOiD Parameters vs Random")
    plt.tight_layout()
    
    out_dir = "results/ssi_comprehensive"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "all_bsoid_ssi_violin.png"), dpi=300)
    df_summary.to_csv(os.path.join(out_dir, "all_bsoid_ssi_stats.csv"), index=False)
    
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
