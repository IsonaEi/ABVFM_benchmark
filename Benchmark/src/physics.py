
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
        
    def align_keypoints_egocentric(self, kps, bodyparts):
        """
        Aligns keypoints to egocentric coordinates:
        1. Translation Invariant (Center at origin).
        2. Rotation Invariance (Rotate so Spine is Vertical).
        """
        print("Aligning keypoints (Centering + Rotation)...")
        T, K, C = kps.shape
        
        # 1. Centering (Subtract Mean per frame)
        centroid = np.nanmean(kps, axis=1, keepdims=True)
        kps_centered = kps - centroid
        
        # 2. Rotation Matrix
        # Define Spine Vector: TailBase -> Neck (or similar)
        # Find indices again (helper needed or duplicate logic)
        def find_bp(names):
            for n in names:
                matches = [i for i, b in enumerate(bodyparts) if n in b]
                if matches: return matches[0]
            return None

        neck_idx = find_bp(['neck', 'head', 'snout'])
        tail_idx = find_bp(['tail_base', 'tail', 'mid_backend2'])
        
        if neck_idx is None or tail_idx is None:
            print("Warning: Could not find Neck/Tail for rotation alignment. Skipping rotation.")
            return kps_centered
            
        # Vector: Tail -> Neck
        spine_vec = kps_centered[:, neck_idx] - kps_centered[:, tail_idx] # (T, 2)
        
        # Calculate angle of spine relative to vertical (Y-axis)
        # Target: We want spine to point UP (0, 1) or RIGHT (1, 0). Let's maximize variance on X implies horizontal?
        # Standard: Head Up (90 deg) or Right (0 deg).
        # Let's align to Positive X-axis (0 radians).
        
        angles = np.arctan2(spine_vec[:, 1], spine_vec[:, 0]) # Current angle
        
        # We want to rotate by -angle to align with X-axis (0)
        # Or if we want vertical, rotate by (pi/2 - angle)
        
        # Let's target X-axis alignment
        theta = -angles
        
        c, s = np.cos(theta), np.sin(theta)
        # Rotation Matrix per frame: R = [[c, -s], [s, c]]
        # Apply to all K points: x' = x*c - y*s, y' = x*s + y*c
        
        x = kps_centered[:, :, 0]
        y = kps_centered[:, :, 1]
        
        x_new = x * c[:, None] - y * s[:, None]
        y_new = x * s[:, None] + y * c[:, None]
        
        kps_aligned = np.stack([x_new, y_new], axis=2)
        return kps_aligned

    def _align_egocentric(self, kps, bodyparts):
        # Fallback for simple centering if needed, but we replace usage with above
        return kps - np.nanmean(kps, axis=1, keepdims=True)

    def compute_keypoint_change_score(self, kps, bodyparts):
        """
        Computes Keypoint Change Score matching KPMS exactly:
        1. Egocentric Alignment (Subtract Centroid).
        2. Diff (Velocity). - Input KPS assumed smoothed.
        3. Norm -> Sum over bodyparts.
        """
        print("Computing KPMS Keypoint Change Score (Aligned)...")
        # 1. Align
        kps_aligned = self._align_egocentric(kps, bodyparts)
        
        # 2. Smooth - SKIP (Global Smoothing applied)
        # kps_smooth = gaussian_filter1d(kps_aligned, sigma=sigma, axis=0)
        kps_smooth = kps_aligned
        
        # 3. Diff (Velocity of aligned structure = Pose Change)
        diff = np.diff(kps_smooth, axis=0, prepend=kps_smooth[:1])
        
        # 4. Norm (Strict definition: |yt - yt-1|)
        # Calculate Euclidean distance of the entire pose vector change
        diff_flat = diff.reshape(diff.shape[0], -1)
        total_vel = np.linalg.norm(diff_flat, axis=1) # (T,)
        
        return total_vel

    def compute_kinematics(self, kps):
        """
        Computes absolute kinematics: Velocity, Acceleration, Jerk.
        Input kps assumed smoothed.
        """
        print("Computing kinematics (Vel, Acc, Jerk)...")
        # Smooth keypoints first - SKIP (Global Smoothing applied)
        # kps_smooth = gaussian_filter1d(kps, sigma=sigma, axis=0)
        kps_smooth = kps
        
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
        Mask can be 2D (Static ROI) or 3D (Dynamic per-frame).
        """
        print(f"Loading pre-computed Optical Flow from {h5_path}...")
        import h5py
        
        with h5py.File(h5_path, 'r') as f:
            if 'optical_flow' not in f:
                raise ValueError("H5 file must contain 'optical_flow' dataset.")
            ds = f['optical_flow'] 
            n_frames, h_flow, w_flow, c = ds.shape
            print(f"Flow Shape: {ds.shape}")
            
            magnitudes = []
            
            # Pre-resize mask if it's 2D to save time
            is_3d_mask = (mask is not None and mask.ndim == 3)
            mask_static = None
            if mask is not None and not is_3d_mask:
                if mask.shape != (h_flow, w_flow):
                    mask_static = cv2.resize(mask.astype(np.uint8), (w_flow, h_flow), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask_static = mask.astype(bool)

            for i in tqdm(range(0, n_frames, batch_size), desc="Processing Flow"):
                batch = ds[i : i+batch_size] # (B, H, W, 2)
                mag = np.sqrt(np.sum(batch**2, axis=3)) # (B, H, W)
                
                if mask is not None:
                    batch_means = []
                    for b_idx in range(mag.shape[0]):
                        t_idx = i + b_idx
                        
                        if is_3d_mask:
                            # Dynamic Mask: Get specific frame
                            m_frame = mask[t_idx] if t_idx < len(mask) else mask[-1]
                            if m_frame.shape != (h_flow, w_flow):
                                m_frame = cv2.resize(m_frame.astype(np.uint8), (w_flow, h_flow), interpolation=cv2.INTER_NEAREST).astype(bool)
                            else:
                                m_frame = m_frame.astype(bool)
                        else:
                            # Static Mask: Use pre-resized
                            m_frame = mask_static
                        
                        # Apply mask and mean (handle cases where mask might be empty)
                        if np.any(m_frame):
                            batch_means.append(np.mean(mag[b_idx][m_frame]))
                        else:
                            batch_means.append(0.0)
                    magnitudes.extend(batch_means)
                else:
                    magnitudes.extend(np.mean(mag, axis=(1, 2)))
                    
        return np.array(magnitudes)
