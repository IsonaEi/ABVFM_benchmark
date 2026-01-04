
import os
import sys
import numpy as np
import h5py
import joblib
import yaml
from pathlib import Path
import keypoint_moseq as kpms

def main():
    # Configuration
    project_dir = "/home/isonaei/ABVFM/KPMS/results/20260104_ctrl_30fps"
    model_name = "20260104-053710-3"
    
    # Load Config
    config_path = "/home/isonaei/ABVFM/KPMS/automated_pipeline/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading checkpoint: {model_name}")
    # Load model to get latent states 'x'
    model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
    
    print("\nModel Keys:", model.keys())
    if 'states' in model:
        print("Model['states'] keys:", model['states'].keys())
        if 'x' in model['states']:
            print("Model['states']['x'] shape/type:", type(model['states']['x']))
            # x is usually (n_frames, latent_dim)
            x_data = model['states']['x']
            print(f"Latent state X shape: {x_data.shape}")
            
    # Load Results to get syllables labels 'z' (or use model['states']['z'])
    print("\nExtracting/Loading results...")
    # We can try to load results.h5 directly for speed if we know the structure, 
    # but using extract_results ensures we get a dictionary we know.
    # Actually, let's look at model['states']['z'] first
    
    if 'z' in model['states']:
        z_data = model['states']['z']
        print(f"Syllable sequences Z shape: {z_data.shape}")
        
        unique_syllables = np.unique(z_data)
        print(f"Unique syllables in model['states']['z']: {len(unique_syllables)}")
        
    print("\nDone.")

if __name__ == "__main__":
    main()
