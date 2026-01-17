
import sys
import os
import numpy as np
import yaml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.loader import DataLoader

def debug_pca():
    config_path = 'Benchmark/config.yaml'
    print(f"Loading config: {config_path}")
    
    loader = DataLoader(config_path)
    # The path currently in config should be the correct new 27kp file
    data_path = loader.config['paths']['keypoint_data'] 
    
    print(f"Loading keypoints from: {data_path}")
    kps, bodyparts = loader.load_dlc_keypoints(data_path)
    
    if kps is None:
        print("Failed to load keypoints.")
        return

    T, K, C = kps.shape
    print(f"Keypoints shape: {kps.shape} (T={T}, K={K}, C={C})")
    
    # 1. Raw Absolute Coordinates
    print("\n--- Test 1: Raw Absolute Coordinates ---")
    kps_flat_raw = kps.reshape(T, -1)
    kps_flat_raw_sample = kps_flat_raw[np.random.choice(T, 5000, replace=False)] if T > 5000 else kps_flat_raw
    
    pca_raw = PCA(n_components=10)
    pca_raw.fit(kps_flat_raw_sample)
    print("Explained Variance Ratio (First 10 PCs):")
    print(pca_raw.explained_variance_ratio_)
    print(f"Cumulative Variance at PC2: {np.sum(pca_raw.explained_variance_ratio_[:2]):.4f}")
    
    # 2. Centered (Egocentric Translation Invariant) Coordinates
    print("\n--- Test 2: Centered (Egocentric) Coordinates ---")
    # Compute center of mass per frame
    center_of_mass = np.nanmean(kps, axis=1, keepdims=True) # (T, 1, 2)
    kps_centered = kps - center_of_mass
    
    kps_flat_centered = kps_centered.reshape(T, -1)
    kps_flat_centered_sample = kps_flat_centered[np.random.choice(T, 5000, replace=False)] if T > 5000 else kps_flat_centered
    
    pca_centered = PCA(n_components=10)
    pca_centered.fit(kps_flat_centered_sample)
    print("Explained Variance Ratio (First 10 PCs):")
    print(pca_centered.explained_variance_ratio_)
    print(f"Cumulative Variance at PC2: {np.sum(pca_centered.explained_variance_ratio_[:2]):.4f}")
    print(f"Cumulative Variance at PC10: {np.sum(pca_centered.explained_variance_ratio_[:10]):.4f}")

    # 3. Scaled (Optionally check scale invariance)
    # Not strictly necessary for dimensionality, but good to know
    
if __name__ == "__main__":
    debug_pca()
