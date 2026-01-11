
import os
import sys
import glob
from pathlib import Path

# Add src to path
sys.path.append("/home/isonaei/ABVFM_benchmark/KPMS/KPMS_pipeline/src")
from kpms_custom.utils.config import load_config
from kpms_custom.core.runner import run_merging
from kpms_custom.utils.logging import get_logger

logger = get_logger()

def main():
    print("=== KPMS Batch Merging (Camellia Edition) ===")
    
    config_path = "config/default_config.yaml"
    config = load_config(config_path)
    base_results_dir = Path(config['project_dir'])
    
    # Threshold check
    threshold = config['analysis'].get('merge_threshold')
    print(f"Applying Merge Threshold: {threshold} (expecting 0.005)")
    
    # Find all model directories
    # We want: 
    # 1. Original runs (2026... but not _extended, not _merged)
    # 2. Extended runs (2026..._extended)
    # We DO NOT want existing _merged folders.
    
    all_dirs = sorted([d for d in base_results_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])
    
    targets = []
    for d in all_dirs:
        name = d.name
        if "_merged" in name:
            continue # Skip already merged ones
            
        # Optional: Skip "legacy" runs if we only want the recent 5+5
        # But user said "regenerate results", implying for current set.
        # Check if it has a checkpoint
        if not (d / "checkpoint.h5").exists():
            continue
            
        targets.append(name)
        
    print(f"Found {len(targets)} models to merge: {targets}")
    
    for model_name in targets:
        print(f"\nMerging: {model_name} ...")
        try:
            # run_merging signature: run_merging(config_path, model_name)
            # It handles loading, merging, creating _merged folder, and saving results.
            run_merging(config_path, model_name)
            print(f"Done merging {model_name}")
            
        except Exception as e:
            print(f"Error merging {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
