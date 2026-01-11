
import os
import sys
import glob
from pathlib import Path

# Add src to path
sys.path.append("/home/isonaei/ABVFM_benchmark/KPMS/KPMS_pipeline/src")
from kpms_custom.utils.config import load_config
from kpms_custom.core.runner import run_merging
from kpms_custom.utils.logger_utils import get_logger

logger = get_logger()

def main():
    print("=== KPMS Batch Merging (Camellia Edition) ===")
    
    config_path = "config/default_config.yaml"
    config = load_config(config_path)
    base_results_dir = Path(config['project_dir'])
    
    thresholds = [0.0005, 0.001, 0.005]
    suffixes = ["_merged_0pt05", "_merged_0pt1", "_merged_0pt5"]
    
    all_dirs = sorted([d for d in base_results_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])
    
    targets = []
    for d in all_dirs:
        name = d.name
        if "_merged" in name: continue
        if not (d / "checkpoint.h5").exists(): continue
        targets.append(name)
        
    print(f"Found {len(targets)} base models to merge: {targets}")
    
    for thresh, suffix in zip(thresholds, suffixes):
        print(f"\n--- Processing Threshold: {thresh} ({suffix}) ---")
        for model_name in targets:
            print(f"Merging {model_name} with threshold {thresh}...")
            try:
                run_merging(config_path, model_name, merge_suffix=suffix, override_threshold=thresh)
            except Exception as e:
                print(f"Error merging {model_name}: {e}")

if __name__ == "__main__":
    main()
