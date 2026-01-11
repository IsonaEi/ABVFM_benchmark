
import pandas as pd
import numpy as np
import h5py
import cv2
import yaml
import os

class DataLoader:
    def __init__(self, config_path="Benchmark/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.fps = self.config['params']['fps']

    def load_dlc_keypoints(self, path):
        """Loads DLC keypoints from H5 or CSV."""
        print(f"Loading keypoints from {path}...")
        try:
            if path.endswith('.h5'):
                df = pd.read_hdf(path)
            elif path.endswith('.csv'):
                df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
            else:
                return None, None
        except Exception as e:
            print(f"Error reading Keypoints: {e}")
            return None, None
        
        scorer = df.columns.levels[0][0]
        bodyparts = df.columns.levels[1].tolist()
        kps_list = []
        found_bps = []
        for bp in bodyparts:
            try:
                x = df[scorer][bp]['x'].values
                y = df[scorer][bp]['y'].values
                kps_list.append(np.stack([x, y], axis=1))
                found_bps.append(bp)
            except KeyError:
                continue
        kps = np.stack(kps_list, axis=1) # (T, K, 2)
        print(f"Loaded {kps.shape[0]} frames, {kps.shape[1]} bodyparts.")
        
        # Cleanup Outliers
        kps = self.remove_outliers(kps, scale_factor=6.0)
        
        return kps, found_bps

    def load_labels(self, path, format_type):
        """Loads behavior labels from KPMS (h5), CASTLE (npy/csv), B-SOiD (csv)."""
        print(f"Loading labels from {path} ({format_type})...")
        if format_type == 'KPMS':
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                if not keys: return None
                dataset_key = keys[0]
                if 'syllable' in f[dataset_key]: labels = f[dataset_key]['syllable'][:]
                elif 'latent_state' in f[dataset_key]: labels = f[dataset_key]['latent_state'][:]
                else: return None
        elif format_type == 'CASTLE':
            if path.endswith('.npy'):
                labels = np.load(path)
            elif path.endswith('.csv'):
                df = pd.read_csv(path)
                col = next((c for c in ['behavior_label', 'behavior'] if c in df.columns), None)
                if col: labels = df[col].values
                else: return None
            # Handle object dtype (strings) -> factorize
            if labels.dtype == object: labels, _ = pd.factorize(labels)
        elif format_type == 'BSOID':
            df = pd.read_csv(path)
            if 'B-SOiD_Label' in df.columns: labels = df['B-SOiD_Label'].values
            else: return None
        elif format_type == 'kpms':
            print(f"Loading KPMS labels from {path} (kpms)...")
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                dataset = None
                for k in keys:
                     if 'syllable' in f[k]: 
                         dataset = f[k]['syllable'][:]
                         break
                     elif 'latent_state' in f[k]:
                         dataset = f[k]['latent_state'][:]
                         break
                
                if dataset is None:
                    if 'syllable' in f: dataset = f['syllable'][:]
                    elif 'latent_state' in f: dataset = f['latent_state'][:]
                    
                if dataset is None:
                     def find_dataset(h5_obj):
                         for k in h5_obj.keys():
                             if isinstance(h5_obj[k], h5py.Dataset): return h5_obj[k][:]
                             if isinstance(h5_obj[k], h5py.Group): 
                                 d = find_dataset(h5_obj[k])
                                 if d is not None: return d
                         return None
                     dataset = find_dataset(f)

                if dataset is None:
                    raise ValueError(f"Could not find 'syllable' or 'latent_state' in {path}")
                labels = dataset
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        return labels

    def load_embeddings(self, path):
        """Loads embeddings from .npy or .npz."""
        print(f"Loading embeddings from {path}...")
        try:
            if path.endswith('.npz'):
                data = np.load(path)
                # Look for 'latent' or 'embeddings' key
                key = next((k for k in ['latent', 'embeddings', 'features'] if k in data), list(data.keys())[0])
                embeds = data[key]
            elif path.endswith('.npy'):
                embeds = np.load(path)
            else:
                return None
            print(f"Loaded embeddings shape: {embeds.shape}")
            return embeds
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return None

    def load_video(self, path):
        """Returns a cv2.VideoCapture object."""
        print(f"Opening video from {path}...")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Failed to open video: {path}")
            return None
        return cap


    def load_mask(self, path):
        """Loads masks from H5 file."""
        print(f"Loading masks from {path}...")
        try:
            with h5py.File(path, 'r') as f:
                # Assuming key is 'masks' or similar from previous inspection
                key = list(f.keys())[0] # Usually 'masks'
                masks = f[key][:]
            print(f"Loaded masks shape: {masks.shape}")
            return masks
        except Exception as e:
            print(f"Error loading masks: {e}")
            return None

    def generate_dummy_labels(self, n_frames, n_classes=30):
        """
        Generates random behavior labels with biologically plausible statistics.
        Constraints:
        - Duration: 33ms (1 frame) to 5s (150 frames)
        - Median Duration: ~400ms (~12 frames)
        - Distribution: Heavy-tailed (Gamma/Exponential-like), more short segments.
        - Total Segments: ~3500 for 1800s video (54000 frames)
        """
        print(f"Generating realistic random labels for {n_frames} frames...")
        labels = np.zeros(n_frames, dtype=int)
        
        # Parameters for Gamma Distribution to achieve Median ~12, Mean ~15
        # Mean = k * theta, Mode = (k-1)*theta
        # Trial and error or estimation: 
        # Aiming for a distribution that peaks early but has a tail.
        # k=1.5, theta=10 -> Mean=15, Median~12-13.
        shape = 1.5
        scale = 10.0 
        
        idx = 0
        min_len = 1    # 33ms
        max_len = 150  # 5000ms
        
        segment_counts = 0
        
        while idx < n_frames:
            # Sample duration
            dur = int(np.random.gamma(shape, scale))
            
            # Clip to [1, 150]
            dur = max(min_len, min(max_len, dur))
            
            # Choose random class
            label = np.random.randint(0, n_classes)
            
            end_idx = min(idx + dur, n_frames)
            labels[idx:end_idx] = label
            
            idx = end_idx
            segment_counts += 1
            
        print(f"  Generated {segment_counts} segments. Mean dur: {n_frames/segment_counts:.1f} frames.")
        return labels

    def resample_labels(self, labels, target_len):
        """Resamples labels to match target length."""
        if len(labels) == target_len: return labels
        indices = np.linspace(0, len(labels) - 1, target_len)
        return labels[np.round(indices).astype(int)]

    # --- Outlier Removal Logic (KPMS Style) ---
    def remove_outliers(self, kps, scale_factor=6.0):
        """
        Removes outlier keypoints based on distance to medoid using MAD.
        kps: (T, K, 2)
        """
        print(f"Removing outliers (Scale Factor={scale_factor})...")
        T, K, C = kps.shape
        
        # 1. Compute Distances to Medoid
        medoids = np.nanmedian(kps, axis=0) # (K, 2) - This is per keypoint median? 
        # Wait, KPMS logic is 'medoid of all bodyparts at each frame'.
        # "Euclidean distance from each keypoint to the medoid (median position) of all keypoints at each frame."
        # KPMS: medoids = np.median(coordinates, axis=1) # (T, 2)
        
        medoid_per_frame = np.nanmedian(kps, axis=1) # (T, 2)
        # Distance from each Keypoint to that frame's Medoid
        dists = np.linalg.norm(kps - medoid_per_frame[:, None, :], axis=2) # (T, K)
        
        # 2. Compute Threshold per Bodypart
        # "Median of distances for that bodypart"
        median_dists = np.nanmedian(dists, axis=0) # (K,)
        # "MAD of distances for that bodypart"
        mads = np.nanmedian(np.abs(dists - median_dists[None, :]), axis=0) # (K,)
        
        thresholds = median_dists + mads * scale_factor
        
        # 3. Identify Outliers
        mask = dists > thresholds[None, :] # (T, K)
        
        # 4. Interpolate
        kps_clean = self._interpolate_keypoints(kps, mask)
        
        n_outliers = np.sum(mask)
        print(f"  Interpolated {n_outliers} outlier points ({n_outliers/(T*K)*100:.2f}%).")
        return kps_clean

    def _interpolate_keypoints(self, kps, outliers):
        """Linear interpolation for outliers."""
        kps_interp = kps.copy()
        T, K, C = kps.shape
        for k in range(K):
            # Find indices where it is NOT an outlier
            valid_mask = ~outliers[:, k]
            # Also check for NaNs in original data just in case
            valid_mask &= ~np.isnan(kps[:, k, 0])
            
            valid_idx = np.flatnonzero(valid_mask)
            if len(valid_idx) == 0: continue # No valid data for this bodypart
            
            # Missing indices (outliers or NaNs)
            missing_idx = np.flatnonzero(~valid_mask)
            
            if len(missing_idx) > 0:
                # Interpolate X and Y separately
                kps_interp[missing_idx, k, 0] = np.interp(missing_idx, valid_idx, kps[valid_idx, k, 0])
                kps_interp[missing_idx, k, 1] = np.interp(missing_idx, valid_idx, kps[valid_idx, k, 1])
                
        return kps_interp

