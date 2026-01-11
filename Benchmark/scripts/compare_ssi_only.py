
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.loader import DataLoader
from src.metrics import MetricsEngine

def main():
    # Load Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    loader = DataLoader(config_path="config.yaml")
    metrics = MetricsEngine()
    
    # 1. Load DINO Embeddings
    dino_path = config['paths']['dino_embeddings']
    features = loader.load_embeddings(dino_path)
    
    if features is None:
        print("Failed to load DINO embeddings.")
        return

    # 2. Identify B-SOiD methods
    bsoid_methods = {k: v for k, v in config['methods'].items() if k.startswith('BSOID_') and v['enabled']}
    
    if not bsoid_methods:
        print("No enabled BSOID methods found in config.")
        return
        
    all_ssi = {}
    
    # 3. Calculate SSI for each
    for name, info in bsoid_methods.items():
        print(f"Processing {name}...")
        labels = loader.load_labels(info['path'], info['format'])
        
        if labels is None:
            print(f"  Warning: Failed to load labels for {name}")
            continue
            
        # Ensure lengths match
        L = min(len(features), len(labels))
        feats_subset = features[:L]
        labels_subset = labels[:L]
        
        # Compute SSI (Window=15 frames as default)
        ssi_scores = metrics.compute_ssi(feats_subset, labels_subset, window=15)
        all_ssi[name] = ssi_scores
        print(f"  Mean SSI: {np.mean(ssi_scores):.4f} (Events: {len(ssi_scores)})")

    # 4. Plotting
    plt.figure(figsize=(12, 7))
    
    # Prepare data for boxplot
    plot_data = []
    labels_list = []
    for name, scores in all_ssi.items():
        plot_data.append(scores)
        labels_list.append(f"{name}\n({len(np.unique(loader.load_labels(bsoid_methods[name]['path'], 'BSOID')))} Classes)")
    
    sns.boxplot(data=plot_data, showfliers=False, palette="viridis")
    plt.xticks(range(len(labels_list)), labels_list, rotation=45)
    plt.ylabel("State Stability Index (SSI)")
    plt.title("B-SOiD Parameter Comparison: State Stability Index (SSI)\nHigher is Better (More stable states relative to transitions)")
    
    # Save Result
    output_dir = "results/ssi_comparison"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "bsoid_ssi_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\nSUCCESS. Plot saved to {out_path}")

    # 5. Print Summary Table
    print("\nSummary Statistics:")
    summary = []
    for name, scores in all_ssi.items():
        summary.append({
            'Method': name,
            'Median_SSI': np.median(scores),
            'Mean_SSI': np.mean(scores),
            'N_Events': len(scores)
        })
    print(pd.DataFrame(summary).to_markdown(index=False))

if __name__ == "__main__":
    main()
