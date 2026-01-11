
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from tqdm import tqdm
import cv2
import sys
import os

class PhysicsEngine:
    def __init__(self):
        pass
        
    def _align_egocentric(self, kps, bodyparts):
        """
        Aligns keypoints to egocentric coordinates (Reference Point at origin).
        Reference point: Centroid (KPMS default) or TailBase.
        """
        # Calculate Centroid per frame
        centroid = np.nanmean(kps, axis=1, keepdims=True) # (T, 1, 2)
        kps_ego = kps - centroid
        return kps_ego

    def compute_keypoint_change_score(self, kps, bodyparts, sigma=1.0):
        """
        Computes Keypoint Change Score matching KPMS exactly:
        1. Egocentric Alignment (Subtract Centroid).
        2. Gaussian Smoothing (sigma=1).
        3. Diff (Velocity).
        4. Norm -> Sum over bodyparts.
        """
        print("Computing KPMS Keypoint Change Score (Aligned)...")
        # 1. Align
        kps_aligned = self._align_egocentric(kps, bodyparts)
        
        # 2. Smooth
        kps_smooth = gaussian_filter1d(kps_aligned, sigma=sigma, axis=0)
        
        # 3. Diff (Velocity of aligned structure = Pose Change)
        diff = np.diff(kps_smooth, axis=0, prepend=kps_smooth[:1])
        
        # 4. Norm & Sum
        vel_norm = np.linalg.norm(diff, axis=2) # (T, K)
        total_vel = np.nansum(vel_norm, axis=1) # (T,)
        
        return total_vel

    def compute_kinematics(self, kps, sigma=1.0):
        """
        Computes absolute kinematics: Velocity, Acceleration, Jerk.
        """
        print("Computing kinematics (Vel, Acc, Jerk)...")
        # Smooth keypoints first
        kps_smooth = gaussian_filter1d(kps, sigma=sigma, axis=0)
        
        # Velocity (1st derivative)
        vel = np.diff(kps_smooth, axis=0, prepend=kps_smooth[:1])
        vel_norm = np.linalg.norm(vel, axis=2) # (T, K)
        mean_vel = np.nanmean(vel_norm, axis=1) # Mean velocity across bodyparts
        
        # Acceleration (2nd derivative)
        acc = np.diff(vel, axis=0, prepend=vel[:1])
        acc_norm = np.linalg.norm(acc, axis=2)
        mean_acc = np.nanmean(acc_norm, axis=1)
        
        # Jerk (3rd derivative)
        jerk = np.diff(acc, axis=0, prepend=acc[:1])
        jerk_norm = np.linalg.norm(jerk, axis=2)
        mean_jerk = np.nanmean(jerk_norm, axis=1)
        
        return {
            'velocity': mean_vel,
            'acceleration': mean_acc,
            'jerk': mean_jerk
        }

    def compute_morphology(self, kps, bodyparts):
        """
        Computes morphological metrics: Snout-Tail Distance and Compactness.
        """
        print("Computing morphology...")
        
        # Find indices
        try:
            snout_idx = next(i for i, b in enumerate(bodyparts) if 'nose' in b or 'snout' in b)
            tail_idx = next(i for i, b in enumerate(bodyparts) if 'tail' in b)
        except StopIteration:
            snout_idx, tail_idx = 0, -1 # Fallback
            
        # 1. Snout-Tail Distance
        st_dist = np.linalg.norm(kps[:, snout_idx] - kps[:, tail_idx], axis=1)
        
        # 2. Compactness (Sum of distances to Centroid)
        centroid = np.nanmean(kps, axis=1, keepdims=True)
        compactness = np.nansum(np.linalg.norm(kps - centroid, axis=2), axis=1)
        
        return {
            'snout_tail_dist': st_dist,
            'compactness': compactness
        }

    def compute_orientation(self, kps, bodyparts, fps=30.0):
        """
        Computes Head Orientation metrics:
        1. Relative: Angle(Body-Pivot) vs Angle(Pivot-Head).
        2. Absolute: Angle(Pivot-Head) vs Global X-axis.
        """
        print("Computing orientation (Relative & Absolute)...")
        
        def find_bp(names):
            for n in names:
                matches = [i for i, b in enumerate(bodyparts) if n in b]
                if matches: return matches[0]
            return None

        # 1. Head (Nose > HeadMid)
        head_idx = find_bp(['nose', 'snout'])
        if head_idx is None: head_idx = find_bp(['head_midpoint', 'head'])
        
        # 2. Pivot (Neck > HeadMid)
        pivot_idx = find_bp(['neck'])
        if pivot_idx is None: pivot_idx = find_bp(['head_midpoint', 'head'])
        
        # 3. Body (MidBackend2 > (TailBase+Centroid)/2)
        body_idx = find_bp(['mid_backend2', 'spine2', 'center'])
        
        # Get Coordinates
        P_head = kps[:, head_idx] if head_idx is not None else kps[:, 0]
        P_pivot = kps[:, pivot_idx] if pivot_idx is not None else kps[:, 0]
        
        if body_idx is not None:
             P_body = kps[:, body_idx]
        else:
             tail_idx = find_bp(['tail', 'tail_base'])
             if tail_idx is not None:
                 P_tail = kps[:, tail_idx]
                 P_centroid = np.nanmean(kps, axis=1)
                 P_body = (P_tail + P_centroid) / 2
             else:
                 P_body = kps[:, 0]
                 
        # Vectors
        V_body_axis = P_pivot - P_body      # Body Axis
        V_head_axis = P_head - P_pivot      # Head Axis
        
        # Angles (atan2)
        ang_body = np.arctan2(V_body_axis[:, 1], V_body_axis[:, 0])
        ang_head = np.arctan2(V_head_axis[:, 1], V_head_axis[:, 0])
        
        # 1. Relative Angle (Head - Body)
        rel_angle = ang_head - ang_body
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi # Wrap [-pi, pi]
        
        # 2. Absolute Angle (Head vs Global X)
        abs_angle = ang_head # Already absolute
        
        # Derivatives (AngVel, AngAcc)
        def compute_derivatives(angle_array):
            unwrapped = np.unwrap(angle_array)
            vel = np.diff(unwrapped, prepend=unwrapped[:1]) * fps
            acc = np.diff(vel, prepend=vel[:1]) * fps
            return vel, acc
            
        rel_vel, rel_acc = compute_derivatives(rel_angle)
        abs_vel, abs_acc = compute_derivatives(abs_angle)
        
        return {
            'relative_angle': rel_angle,
            'relative_ang_vel': rel_vel,
            'relative_ang_acc': rel_acc,
            'absolute_angle': abs_angle,
            'absolute_ang_vel': abs_vel,
            'absolute_ang_acc': abs_acc
        }
    
    def compute_killer_case_residuals(self, z_of, z_sv):
        """
        Computes Killer Case Index (Residuals).
        """
        valid_mask = ~np.isnan(z_of) & ~np.isnan(z_sv)
        if np.sum(valid_mask) < 10:
             return np.zeros_like(z_of), 0, 0
             
        res = linregress(z_sv[valid_mask], z_of[valid_mask])
        slope = res.slope
        intercept = res.intercept
        expected_of = slope * z_sv + intercept
        residuals = z_of - expected_of
        
        return residuals, slope, intercept

    def compute_masked_flow_from_h5(self, h5_path, mask, batch_size=1000):
        """
        Computes Mean Magnitude from pre-computed H5 flow (dense), applying the given mask.
        H5 Shape expected: (T, H, W, 2)
        Kept in Benchmark for light-weight loading of pre-computed results via NumPy.
        """
        print(f"Loading pre-computed Optical Flow from {h5_path}...")
        import h5py
        
        with h5py.File(h5_path, 'r') as f:
            if 'optical_flow' not in f:
                raise ValueError("H5 file must contain 'optical_flow' dataset.")
            ds = f['optical_flow'] 
            n_frames, h, w, c = ds.shape
            print(f"Flow Shape: {ds.shape}")
            
            magnitudes = []
            
            for i in tqdm(range(0, n_frames, batch_size), desc="Loading Flow"):
                batch = ds[i : i+batch_size] # (B, H, W, 2)
                mag = np.sqrt(np.sum(batch**2, axis=3)) # (B, H, W)
                
                if mask is not None:
                    # Resize mask if needed
                    if mask.shape != (h, w):
                         mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                         mask_resized = mask.astype(bool)
                    
                    # Apply mask and mean
                    batch_means = []
                    for b_idx in range(mag.shape[0]):
                         batch_means.append(np.mean(mag[b_idx][mask_resized]))
                    magnitudes.extend(batch_means)
                else:
                    magnitudes.extend(np.mean(mag, axis=(1, 2)))
                    
        return np.array(magnitudes)
