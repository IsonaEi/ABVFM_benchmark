
import os
import sys
import shutil
import keypoint_moseq as kpms
from pathlib import Path

# Add src to path
sys.path.append("/home/isonaei/ABVFM_benchmark/KPMS/KPMS_pipeline/src")
from kpms_custom.utils.config import load_config
from kpms_custom.core.runner import _load_and_prep
from kpms_custom.utils.logger_utils import get_logger

logger = get_logger()

def main():
    print("=== KPMS Model Extension (Camellia Edition) ===")
    
    # 1. Configuration
    config_path = "config/default_config.yaml"
    config = load_config(config_path)
    base_results_dir = Path(config['project_dir'])
    
    # Files to load data once
    print("Loading Data...")
    project_dir, data, metadata, pca = _load_and_prep(config)
    
    # Models to Extend
    targets = [
        "20260111-0627-exp1_ar1e06_full7e04",
        "20260111-0640-exp2_ar2e06_full3e05",
        "20260111-0652-exp3_ar1e06_full3e04",
        "20260111-0705-exp4_ar1e09_full2e06"
    ]
    
    iters_to_add = 200
    
    for model_name in targets:
        print(f"\nProcessing Model: {model_name}")
        src_dir = base_results_dir / model_name
        
        if not src_dir.exists():
            print(f"  Error: Directory {src_dir} not found. Skipping.")
            continue
            
        # 1. Load Checkpoint
        print(f"  Loading checkpoint from {src_dir}...")
        try:
            # load_checkpoint returns (model, metadata, pca, latent_dim) usually
            # But kpms api might differ slightly.
            # Checking library code: keypoint_moseq/io.py -> load_checkpoint(project_dir, name)
            # Returns [model] or [model, metadata, pca...] depending on saved content
            # Our saved checkpoints likely contain everything.
            
            checkpoint = kpms.load_checkpoint(str(base_results_dir), model_name)
            model = checkpoint[0] # The model dict
            
            # 2. Setup New Directory
            new_model_name = f"{model_name}_extended"
            new_model_dir = base_results_dir / new_model_name
            
            print(f"  Target Directory: {new_model_dir}")
            if new_model_dir.exists():
                print("  Target directory exists. Skipping to avoid overwrite.")
                continue
            
            # 3. Resume Training
            # Need to determine current iteration count to set correct logging/schedule?
            # Or just append.
            # fit_model adds to existing history if model is populated.
            
            # We want to run for 200 MORE iterations.
            # fit_model(..., start_iter=CURRENT, num_iters=(CURRENT+200)) OR
            # fit_model(..., ar_only=False, num_iters=200) -> wait, num_iters is TOTAL typically in library?
            # Let's check kpms.fit_model signature.
            # def fit_model(model, data, metadata, project_dir, model_name, ar_only=False, start_iter=0, num_iters=100, save_every_n_iters=20...):
            # It iterates: for i in range(start_iter, num_iters):
            # So if we want +200, we need to know current.
            
            # Inspect model['saved_iters']? Or look at checkpoint name if it has one?
            # Or just assume 200 (Default) / 250 (Exp3) based on logs.
            # Let's check model history length.
            
            current_iter = 0
            if 'history' in model and 'log_likelihood' in model['history']:
                 current_iter = len(model['history']['log_likelihood'])
                 print(f"  Detected current iterations from history: {current_iter}")
            else:
                 print("  Could not detect current iters, assuming 250.")
                 current_iter = 250
            
            target_iter = current_iter + iters_to_add
            print(f"  Extending from {current_iter} to {target_iter} iters.")
            
            # 4. Run Fit
            new_model, _ = kpms.fit_model(
                model, data, metadata, 
                str(base_results_dir), new_model_name,
                ar_only=False,
                start_iter=current_iter,
                num_iters=target_iter,
                save_every_n_iters=25
            )
            
            print(f"  Extension complete for {model_name}. Saved to {new_model_name}")
            
        except Exception as e:
            print(f"  Error extending {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nAll extensions complete.")

if __name__ == "__main__":
    main()
