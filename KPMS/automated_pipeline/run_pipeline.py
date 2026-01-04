import yaml
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg') # Prevent GUI windows from popping up
import warnings
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
import jax
import numpy as np
jax.config.update("jax_enable_x64", True)
from preprocess import load_data, format_data, preprocess_data, detect_fps
from kpms_runner import setup_project, train_pca, fit_model_wrapper, scan_kappa, get_median_duration
from analysis import run_analysis_module, perform_model_evaluation
import keypoint_moseq as kpms
import glob

def main():
    parser = argparse.ArgumentParser(description="Automated KPMS Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "scan", "scan_ar", "scan_full", "analyze", "evaluate"], help="Mode: train, scan, analyze, evaluate")
    parser.add_argument("--model_name", type=str, default=None, help="Specific model name to analyze (optional, defaults to latest)")
    args = parser.parse_args()
    
    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Loaded config from {args.config}")
    
    # 1. Setup Project
    project_dir = setup_project(config)
    print(f"Project initialized at {project_dir}")
    
    # 2. Data Loading & Preprocessing
    fps = detect_fps(config.get('video_dir', ''))
    if config.get('preprocess') is None:
        config['preprocess'] = {}
    config['preprocess']['fps'] = fps
    
    files = load_data(config)
    coordinates, confidences, bodyparts = format_data(files, config)
    data, metadata, bodyparts = preprocess_data(coordinates, confidences, config, bodyparts=bodyparts)
        
    print("Data loaded and preprocessed.")
    
    # 3. PCA (Needed for everything)
    # If PCA model exists in project, fit_pca usually loads or refits.
    # It's fast enough to train/load.
    print("Fitting/Loading PCA...")
    pca = train_pca(data, config, project_dir, bodyparts=bodyparts)
    print("PCA ready.")
    
    # 4. Modeling / Analysis
    if args.mode == "scan" or args.mode == "scan_ar":
        print("Running AR Kappa Scan...")
        results, best_model = scan_kappa(data, metadata, pca, config, project_dir, fps=fps, scan_type='ar')
        print(f"AR Kappa scan complete. Best model: {best_model}")
        
    elif args.mode == "scan_full":
        print("Running Full Model Kappa Scan (Transitions)...")
        results, best_model = scan_kappa(data, metadata, pca, config, project_dir, fps=fps, scan_type='full')
        print(f"Full Kappa scan complete. Result string: {best_model}")
        
        # Parse result string to update config
        # Format: optimized_full_{full_kappa}_AR{ar_kappa}
        import re
        try:
             # Extract Full Kappa
             full_match = re.search(r"optimized_full_([\d\.e\+\-]+)", best_model)
             if full_match:
                 full_val = float(full_match.group(1))
                 config['tuning']['full_kappa'] = full_val
                 print(f"Updated config full_kappa: {full_val:.2e}")
                 
             # Extract AR Kappa
             ar_match = re.search(r"_AR([\d\.e\+\-]+)", best_model)
             if ar_match:
                 ar_val = float(ar_match.group(1))
                 config['tuning']['ar_kappa'] = ar_val
                 print(f"Updated config ar_kappa: {ar_val:.2e}")
                 
             # Save Config
             with open(args.config, 'w') as f:
                 yaml.dump(config, f)
             print(f"Successfully saved updated parameters to {args.config}")
             
        except Exception as e:
             print(f"Error updating config with scan results: {e}")
        
    elif args.mode == "train":
        print("Training Models (Multiple Fits)...")
        num_restarts = config.get('num_fitting_restarts', 1)
        
        for i in range(num_restarts):
            print(f"\n--- Fitting Run {i+1}/{num_restarts} ---")
            
            # Determine suffix for this run
            suffix = f"-{i}"
            
            # Use jax.random.PRNGKey for seeding if needed by kpms, 
            # though kpms.init_model might handle it via kwargs if we pass 'seed'.
            # We will rely on kpms_runner or config handling to ensure diversity.
            # Ideally, we pass a seed to fit_model_wrapper or update config['seed']
            
            # For now, we assume fit_model_wrapper generates a unique name based on timestamp
            # But we want to group them or name them consistently.
            # Let's see how fit_model_wrapper handles naming.
            # It takes 'project_dir' and generates a name if not provided.
            # We should probably enforce a naming convention if we want to find them easily later.
            
            # To ensure different seeds produce different results, we might need to set the seed in config
            current_seed = config.get('seed', 0) + i
            config_for_run = config.copy()
            config_for_run['seed'] = current_seed
            
            # We can pass a specific name or let it generate one. 
            # If we let it generate, we might want to append the index.
            # But fit_model_wrapper signature is: (data, metadata, pca, config, project_dir, name=None, ar_only=False)
            
            # Let's try to control the name if possible, or just print it.
            # If we pass None, it uses timestamp.
            # If we pass a prefix, we can append -i.
            
            model, name = fit_model_wrapper(data, metadata, pca, config_for_run, project_dir, ar_only=False, name_suffix=suffix)
            
            final_dur = get_median_duration(model, config_for_run, fps=fps)
            print(f"Model {name} trained successfully (Seed: {current_seed}).")
            print(f"Final Median Syllable Duration: {final_dur:.2f} ms")
        
    elif args.mode == "analyze":
        print("Running Downstream Analysis...")
        
        # Determine Model Name
        model_name = args.model_name
        if not model_name:
            # Find latest model directory
            # Exclude standard directories like 'pca', 'results' if present
            # KPMS saves models as folders with timestamp or name.
            # We look for folders containing 'checkpoint' or 'params.json' equivalent.
            # Assuming typical structure: project_dir/model_name/
            candidates = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
            # Filter out known non-model dirs if relevant, but typically timestamps are used.
            # Sort by creation time
            candidates = sorted(candidates, key=lambda x: os.path.getctime(os.path.join(project_dir, x)), reverse=True)
            
            # Simple heuristic: ignore 'pca'
            candidates = [c for c in candidates if c != 'pca']
            
            if candidates:
                model_name = candidates[0]
                print(f"Auto-detected latest model: {model_name}")
            else:
                print("Error: No models found in project directory to analyze.")
                sys.exit(1)
                
        print(f"Analyzing Model: {model_name}")
        
        # Load Model
        try:
            checkpoint = kpms.load_checkpoint(project_dir, model_name)
            model = checkpoint[0]
            # data_loaded = checkpoint[1]
            # metadata_loaded = checkpoint[2]
            # iteration = checkpoint[3]
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            sys.exit(1)
            
        # Run Analysis
        # We need to pass data, metadata (which we have from loading step)
        # Note: 'data' contains the preprocessed coordinates/pca scores
        # 'metadata' describes the sessions.
        run_analysis_module(config, project_dir, model_name, model, data, metadata, coordinates, pca)

    elif args.mode == "evaluate":
        print("Running Model Evaluation & Selection...")
        
        # Pattern from config or args?
        # Let's check config first
        eval_cfg = config.get('model_evaluation', {})
        pattern = eval_cfg.get('model_name_pattern', None)
        
        best_model = perform_model_evaluation(config, project_dir, model_name_pattern=pattern)
        
        if best_model:
            print(f"\nEvaluation Complete. Best Model: {best_model}")
            print("You can now run 'analyze' mode with this model using:")
            print(f"  python run_pipeline.py --mode analyze --model_name {best_model}")
        else:
            print("\nEvaluation Failed or No Models Found.")


if __name__ == "__main__":
    main()
