
import sys
import os
import yaml
import numpy as np
import keypoint_moseq as kpms
import jax
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # No GUI

# Add current directory to path so we can import local modules if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import kpms_runner
from preprocess import load_data, format_data, preprocess_data, detect_fps

def get_duration(model, fps):
    """Calculate median syllable duration (in ms)."""
    if fps is None: fps = 30.0
    z = np.array(model['states']['z']) 
    if z.ndim > 1: z = z.flatten()
    z_padded = np.concatenate(([z[0]-1], z, [z[-1]-1]))
    changes = np.where(z_padded[:-1] != z_padded[1:])[0]
    lengths = np.diff(changes)
    median_frames = np.median(lengths)
    return (median_frames / fps) * 1000

def run_verification(config_path="config.yaml"):
    print("=== Starting Kappa Transition Verification ===")
    
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    project_dir = config['project_dir']
    
    # 2. Setup Project (Ensure it exists)
    kpms_runner.setup_project(config)
    kpms_config = kpms.load_config(project_dir)
    
    # 3. Data Loading & Preprocessing
    fps = detect_fps(config.get('video_dir', ''))
    print(f"Detected FPS: {fps}")
    if config.get('preprocess') is None:
        config['preprocess'] = {}
    config['preprocess']['fps'] = fps

    # Correct Loading Sequence
    files = load_data(config)
    coordinates, confidences, bodyparts = format_data(files, config)
    formatted_data, metadata, bodyparts = preprocess_data(coordinates, confidences, config, bodyparts=bodyparts)
    
    # 4. PCA
    # Pass 'bodyparts' to train_pca so it can verify headings
    pca = kpms_runner.train_pca(formatted_data, config, project_dir, bodyparts)
    
    # Calculate Latent Dim
    latent_dim = kpms_runner.calculate_latent_dim(pca, target_variance=0.9)
    # Ensure config reflects this if needed, or just pass to init_model
    
    # 5. Initialize Model
    print("\n--- Initializing Model ---")
    model = kpms.init_model(
        data=formatted_data,
        pca=pca,
        latent_dim=latent_dim,
        **kpms_config
    )
    
    # 6. AR-Only Fit (High Kappa)
    kappa_ar = 1e6 # Example robust start value
    print(f"\n--- Stage 1: AR-Only Fit (Kappa={kappa_ar:.0e}) ---")
    model = kpms.update_hypparams(model, kappa=kappa_ar)
    
    model, _ = kpms.fit_model(
        model, formatted_data, metadata, project_dir, 
        model_name="verify_ar_stage",
        ar_only=True, 
        num_iters=10, 
        verbose=True
    )
    
    dur_ar = get_duration(model, fps)
    print(f"Stage 1 Result: Median Duration = {dur_ar:.2f} ms")
    
    # 7. Full Model Fit (Reduced Kappa)
    kappa_full = kappa_ar / 10
    print(f"\n--- Stage 2: Full Model Fit (Kappa={kappa_full:.0e} [AR/10]) ---")
    model = kpms.update_hypparams(model, kappa=kappa_full)
    
    model, _ = kpms.fit_model(
        model, formatted_data, metadata, project_dir, 
        model_name="verify_full_stage",
        ar_only=False, 
        start_iter=10,
        num_iters=10, # Run 10 more iterations (total 20)
        verbose=True
    )
    
    dur_full = get_duration(model, fps)
    print(f"Stage 2 Result: Median Duration = {dur_full:.2f} ms")
    
    print("\n=== Verification Complete ===")
    print(f"AR-Only Duration ({kappa_ar:.0e}): {dur_ar:.2f} ms")
    print(f"Full Model Duration ({kappa_full:.0e}): {dur_full:.2f} ms")
    
    if abs(dur_full - dur_ar) > 100:
        print(">> Note: Significant duration shift observed. Using the same kappa as AR might be safer unless aiming for finer segmentation.")
    else:
        print(">> Note: Duration remained relatively stable, suggesting the transition is smooth.")

if __name__ == "__main__":
    run_verification()
