
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keypoint_moseq as kpms
import h5py
from pathlib import Path

# Add project root to path for Benchmark imports
sys.path.append("/home/isonaei/ABVFM_benchmark")
sys.path.append("/home/isonaei/ABVFM_benchmark/KPMS/KPMS_pipeline/src")

from Benchmark.src.loader import DataLoader
from Benchmark.src.metrics import MetricsEngine
from kpms_custom.utils.config import load_config
from kpms_custom.core.runner import _load_and_prep

def main():
    print("=== KPMS SSI Analysis (Camellia Edition) ===")
    
    # 1. Setup & Load Embeddings
    print("Loading Reference DINO Embeddings...")
    bench_config_path = "/home/isonaei/ABVFM_benchmark/Benchmark/config.yaml"
    loader = DataLoader(config_path=bench_config_path)
    metrics = MetricsEngine()
    
    # Path to DINO features (Standard used in B-SOiD comparison)
    dino_path = "/home/isonaei/ABVFM_benchmark/castle-ai-feature-support-dinov3-model/projects/2026-01-05-17-00-40-Project_ctrl_30fps.mp4/latent/dinov3_vitb16/ctrl_30fps_ROI_1_dinov3_vitb16_ctr_rmbg.npz"
    features = loader.load_embeddings(dino_path)
    
    if features is None:
        print("Error: Could not load DINO features. Aborting.")
        return

    target_len = len(features)
    print(f"Target Length: {target_len}")

    # 2. KPMS Setup
    config_path = "config/default_config.yaml"
    config = load_config(config_path)
    base_results_dir = Path(config['project_dir'])
    
    # 3. Find Experiments
    # Look for folders starting with "2026"
    exp_dirs = sorted([d for d in base_results_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])
    
    all_ssi_data = [] # List of dicts
    summary_stats = []
    results_dict_for_plotting = {} # Store labels for histogram
    
    # Need to load data structure once to help with extraction if needed, 
    # but kpms.extract_results typically needs 'pca' and 'metadata'.
    # We'll load them per loop or globally if identical.
    # Since all used the same data, we can load once.
    print("Loading KPMS Data structures (PCA, Metadata)...")
    project_dir, data, metadata, pca = _load_and_prep(config)
    
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        print(f"Processing Experiment: {exp_name}")
        
        # Check for checkpoint
        ckpt_path = exp_dir / "checkpoint.h5"
        if not ckpt_path.exists():
            print(f"  No checkpoint found in {exp_name}, skipping.")
            continue
            
        try:
            # Load Model
            # kpms.load_checkpoint returns (model, metadata, pca, latent_dim) usually, or list
            # Actually checking source/docs: load_checkpoint(project_dir, name)
            # But we have absolute path. 
            # kpms.load_checkpoint uses project_dir + name.
            # So:
            model_name = exp_name
            model, _, _, _ = kpms.load_checkpoint(str(base_results_dir), model_name)
            
            # Check if results.h5 exists and has data
            results_path = exp_dir / "results.h5"
            results = None
            
            if results_path.exists():
                try:
                    # Try loading existing results
                    print(f"  Found existing results at {results_path}, loading...")
                    results = kpms.load_results(str(base_results_dir), model_name)
                except Exception as e:
                    print(f"  Could not load existing results ({e}), re-extracting...")
            
            if results is None:
                # Extract Results if not loaded
                results = kpms.extract_results(model, metadata=metadata, project_dir=str(base_results_dir), model_name=model_name)
            
            # Get Labels for the target video
            # results is usually {key: {'syllable': ..., 'latent_state': ...}}
            # We need to find the key for 'ctrl_30fps'
            # Let's inspect keys
            keys = list(results.keys())
            target_key = next((k for k in keys if "ctrl_30fps" in k), None)
            
            if target_key:
                labels = results[target_key]['syllable']
                print(f"  Found labels for {target_key}: Length={len(labels)}")
                
                # Resample / Pad / Crop if length mismatch
                # DINO features length vs KPMS length might differ slightly due to preprocessing
                if len(labels) != target_len:
                    print(f"  Length Mismatch! DINO={target_len}, KPMS={len(labels)}")
                    labels = loader.resample_labels(labels, target_len)
                    print(f"  Resampled to {len(labels)}")
                
                # Store for histogram
                results_dict_for_plotting[exp_name] = labels
                
                # Compute SSI
                ssi_scores = metrics.compute_ssi(features, labels, window=15)
                
                # Stats
                stats = metrics.compute_label_stats(labels, fps=30.0)
                
                # Determine readable name from folder
                base_name, iters_label, merge_label = parse_exp_name_for_plot(exp_name)
                
                # Store
                for score in ssi_scores:
                    all_ssi_data.append({
                        'Method': base_name,
                        'Iterations': iters_label,
                        'Merge': merge_label,
                        'SSI': score,
                        'Type': 'KPMS'
                    })
                    
                summary_stats.append({
                    'Method': base_name,
                    'Iterations': iters_label,
                    'Merge': merge_label,
                    'Median_SSI': np.median(ssi_scores),
                    'Mean_SSI': np.mean(ssi_scores),
                    'N_Clusters': stats['n_classes'],
                    'Median_Duration_ms': stats['median_duration_ms']
                })
                print(f"  > SSI Median: {np.median(ssi_scores):.3f} ({iters_label}, {merge_label})")
                
            else:
                print("  Target video 'ctrl_30fps' not found in results keys.")
                
        except Exception as e:
            print(f"  Error processing {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Random Baseline
    print("Generating Random Baseline...")
    rand_labels = loader.generate_dummy_labels(target_len, n_classes=40) 
    rand_ssi = metrics.compute_ssi(features, rand_labels, window=15)
    
    for score in rand_ssi:
        all_ssi_data.append({
            'Method': 'Random Baseline',
            'Iterations': 'N/A',
            'Merge': 'Baseline',
            'SSI': score,
            'Type': 'Baseline'
        })
        
    stats_rand = metrics.compute_label_stats(rand_labels, fps=30.0)
    summary_stats.append({
        'Method': 'Random Baseline',
        'Iterations': 'N/A',
        'Merge': 'Baseline',
        'Median_SSI': np.median(rand_ssi),
        'Mean_SSI': np.mean(rand_ssi),
        'N_Clusters': 40,
        'Median_Duration_ms': stats_rand['median_duration_ms']
    })

    # 5. Plotting (Filter for 0.5% Merged results + Baseline)
    df_plot = pd.DataFrame(all_ssi_data)
    df_summary = pd.DataFrame(summary_stats)
    
    plt.figure(figsize=(16, 8))
    
    # Filter: Only Show 200 iterations to simplify the 3-threshold comparison
    # We want to compare 0.05 vs 0.1 vs 0.5
    df_plot_filtered = df_plot[df_plot['Iterations'].isin(['200 iters', 'N/A'])]
    
    # Sort order by median SSI of the 0.5% version (likely best)
    # Pivot to rank
    rank_df = df_summary[df_summary['Iterations'].isin(['200 iters', 'N/A'])].sort_values('Median_SSI', ascending=False)
    # Dedup methods
    order = []
    for m in rank_df['Method']:
        if m not in order: order.append(m)
    
    # Violin plot with hue for Merge Threshold
    sns.violinplot(data=df_plot_filtered, x='Method', y='SSI', hue='Merge', order=order, 
                   split=False, gap=0.1, inner='quartile', density_norm='width',
                   palette={"0.05% (Low)": "#95a5a6", "0.1% (Med)": "#3498db", "0.5% (High)": "#e74c3c", "Baseline": "grey"},
                   hue_order=["0.05% (Low)", "0.1% (Med)", "0.5% (High)", "Baseline"])
                   
    plt.xticks(rotation=45, ha='right')
    plt.title("SSI Comparison: Effect of Merge Threshold (200 Iterations)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    out_dir = base_results_dir / "analysis_summary"
    out_dir.mkdir(exist_ok=True)
    
    plt.savefig(out_dir / "kpms_ssi_violin_merged_0.5.png", dpi=300)
    df_summary.to_csv(out_dir / "kpms_ssi_stats_all.csv", index=False)
    
    print(f"Analysis Complete. Saved to {out_dir}")

    # 6. Syllable Distribution Plot (Histogram/Ranked Prob)
    plot_syllable_usage(all_ssi_data, out_dir, results_dict_for_plotting)

    print(f"Analysis Complete. Saved to {out_dir}")

def plot_syllable_usage(ssi_data, out_dir, results_dict):
    """
    Plots the syllable usage probability distribution (Ranked).
    Comparing different merge thresholds for the BEST performing method (Default or Exp4).
    """
    # Filter for a representative method to avoid clutter. 
    # Let's use 'Default (AR=1e6)' and 'Exp 4 (AR=1e9)' as they are top performers.
    target_methods = ["Default (AR=1e6)", "Exp 4 (AR=1e9, Full=2e6)"]
    
    # helper to compute distribution
    def get_distribution(labels):
        counts = pd.Series(labels).value_counts(normalize=True)
        # Sort by rank (highest freq first)
        return counts.values # Array of probabilities
        
    plot_data = []
    
    # We need to access the actual labels. 
    # results_dict is { 'ExpName': labels_array }
    # ssi_data contains metadata mapping ExpName -> Method/Merge/etc.
    
    # Build a lookup from readable properties to ExpName
    # Actually results_dict keys are exp_names (folder names).
    # We parsed them in the main loop. Let's re-parse or store better.
    # Refactoring main to store labels in a list for plotting might be large.
    # Let's use the passed results_dict.
    
    for exp_name, labels in results_dict.items():
        base, iters, merge = parse_exp_name_for_plot(exp_name)
        
        # Filter: Only plot 200 iters (Original), as 400 iters degrades performace
        if iters != "200 iters": continue
        
        # Filter: Only target methods
        if base not in target_methods: continue
        
        probs = get_distribution(labels)
        
        # Create dataframe for plotting: Rank 1...N
        for rank, p in enumerate(probs):
            plot_data.append({
                'Method': base,
                'Merge': merge,
                'Rank': rank + 1,
                'Probability': p
            })
            
    df_dist = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 6))
    # Line plot with markers
    sns.lineplot(data=df_dist, x='Rank', y='Probability', hue='Merge', style='Method', markers=True, dashes=False)
    plt.title("Syllable Usage Probability Distribution (Sorted)")
    plt.ylabel("Probability (Frequency)")
    plt.xlabel("Syllable Rank (1 = Most Common)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "syllable_distribution_histogram.png", dpi=300)

def parse_exp_name_for_plot(name):
    # Returns (BaseName, IterationsLabel, MergeLabel)
    is_merged_0pt5 = "_merged_0pt5" in name
    is_merged_0pt1 = "_merged" in name and not is_merged_0pt5 # Since I renamed, standard _merged is new 0.1%
    
    is_extended = "_extended" in name
    
    clean_name = name.replace("_extended", "").replace("_merged_0pt5", "").replace("_merged", "")
    
    iter_label = "400 iters" if is_extended else "200 iters"
    
    if is_merged_0pt5:
        merge_label = "0.5% (High)"
    elif is_merged_0pt1:
        merge_label = "0.1% (Med)"
    else:
        merge_label = "0.05% (Low)"
    
    if "exp" not in clean_name and "Default" not in clean_name and "2026" in clean_name:
         if clean_name.endswith("-0"):
             base_name = "Default (AR=1e6)"
         else:
             base_name = clean_name
    elif "exp" in clean_name:
        try:
            parts = clean_name.split('-')
            if "exp1" in clean_name: base_name = "Exp 1 (AR=1e6, Full=7e4)"
            elif "exp2" in clean_name: base_name = "Exp 2 (AR=2e6, Full=3e5)"
            elif "exp3" in clean_name: base_name = "Exp 3 (AR=1e6, Full=3e4)"
            elif "exp4" in clean_name: base_name = "Exp 4 (AR=1e9, Full=2e6)"
            elif "exp5" in clean_name: base_name = "Exp 5 (AR=1e5, Full=5e3)"
            else: base_name = clean_name
        except:
            base_name = clean_name
    else:
        base_name = clean_name
        
    return base_name, iter_label, merge_label

def parse_exp_name(name):
    # Legacy function kept just in case, but unused in main now
    return name


if __name__ == "__main__":
    main()
