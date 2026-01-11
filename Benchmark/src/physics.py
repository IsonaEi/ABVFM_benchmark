
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_small
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from tqdm import tqdm
import cv2
import sys
import os

# Add GMFlow path for dynamic import
GMFLOW_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party/gmflow')
if GMFLOW_PATH not in sys.path:
    sys.path.append(GMFLOW_PATH)

try:
    from gmflow.gmflow import GMFlow
    HAS_GMFLOW = True
except ImportError:
    HAS_GMFLOW = False
    # User requested strict GMFlow usage. We will raise error in compute_optical_flow_gpu if missing.

class PhysicsEngine:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def _align_egocentric(self, kps, bodyparts):
        """
        Aligns keypoints to egocentric coordinates (Reference Point at origin).
        Reference point: Centroid (KPMS default) or TailBase.
        KPMS Paper: "transforming keypoints into egocentric coordinates" 
        usually means subtracting the centroid (or root) from all points.
        """
        # Calculate Centroid per frame
        centroid = np.nanmean(kps, axis=1, keepdims=True) # (T, 1, 2)
        kps_ego = kps - centroid
        
        # Optional: Rotate to align spine with Y-axis?
        # KPMS "Change Score" definition simply says "transforming...". 
        # Usually for "Change Score" (velocity sum), just translation subtraction is enough 
        # to remove "animal global translation".
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

    def compute_optical_flow_gpu(self, video_path, mask=None, resize_dim=(720, 720), batch_size=4, save_path=None):
        """
        Computes Optical Flow using GMFlow (Strict).
        Raises Error if GMFlow package or weights are missing.
        Default resize_dim is (720, 720).
        """
        print(f"Computing Optical Flow (GPU: {self.device}) on {video_path}...")
        
        # --- Model Initialization ---
        if not HAS_GMFLOW:
            raise ImportError("GMFlow is required but not installed or found in path. Please check 'venvs/benchmark' and 'third_party/gmflow'.")
            
        try:
            print("Using GMFlow (Sintel Weights)...")
            model = GMFlow(feature_channels=128, num_scales=1, upsample_factor=8, 
                           num_head=1, attention_type='swin', ffn_dim_expansion=4, 
                           num_transformer_layers=6).to(self.device)
            
            weights_path = os.path.join(GMFLOW_PATH, 'pretrained/gmflow_sintel-0c07dcb3.pth')
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)
            else:
                raise FileNotFoundError(f"GMFlow weights not found at {weights_path}! Please download them.")
        except Exception as e:
            raise RuntimeError(f"GMFlow initialization failed: {e}")
            
        model.eval()
            
        model.eval()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
            
        # Optional H5
        h5_file = None
        ds_flow = None
        if save_path:
             if save_path.endswith('.h5'):
                 import h5py
                 h5_file = h5py.File(save_path, 'w')
                 n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                 ds_flow = h5_file.create_dataset('optical_flow', shape=(n_frames_total, resize_dim[1], resize_dim[0], 2), dtype='float32', compression="gzip") 
        
        # Mask Prep
        static_mask = None
        if mask is not None:
             print("Mask provided. Optical Flow will be restricted to mask area.")
             if mask.ndim == 2:
                 m = mask.astype(np.uint8)
                 m_resized = cv2.resize(m, resize_dim, interpolation=cv2.INTER_NEAREST)
                 static_mask = torch.from_numpy(m_resized).bool().to(self.device).unsqueeze(0)

        flow_magnitudes = []
        frames_buffer = []
        mask_buffer = [] 
        
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        prev_gm_tensor = None # For GMFlow
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, resize_dim)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = F.to_tensor(frame_rgb) * 255.0 # (C, H, W)
            
            frames_buffer.append(frame_tensor)
            mask_buffer.append(static_mask.squeeze(0) if static_mask is not None else None)
            
            # RAFT needs batch of pairs
            # GMFlow usually does inference one by one or batch.
            # strict run:
            
            if len(frames_buffer) >= batch_size + 1:
                img1 = torch.stack(frames_buffer[:-1]).to(self.device)
                img2 = torch.stack(frames_buffer[1:]).to(self.device)
                
                with torch.no_grad():
                    # GMFlow Forward
                    results = model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=False)
                    flows = results['flow_preds'][-1] # (B, 2, H, W)
                
                # Magnitude
                mag = torch.sum(flows ** 2, dim=1).sqrt() # (B, H, W)
                
                # --- Dynamic Background Compensation (Anti-Flicker) ---
                # Calculate mean flow of background (unmasked area) if mask exists
                # Then subtract it from the whole magnitude map (or just foreground)
                # To be robust, we subtract the MEDIAN of the background flow vector magnitudes?
                # Actually, simpler: Subtract the MEDIAN magnitude of the unmasked area.
                
                if mask is not None or static_mask is not None:
                     mb = torch.stack(mask_buffer[:-1]) # (B, H, W) Boolean
                     
                     # 1. Background Compensation
                     # Invert mask to get background
                     bg_mask = ~mb
                     
                     # We compute median background magnitude per frame
                     # GMFlow flickering is often global, so median is safe.
                     # (B, H, W) -> (B,)
                     bg_vals = []
                     for b_idx in range(mag.shape[0]):
                         m_bg = bg_mask[b_idx]
                         if torch.sum(m_bg) > 0:
                             # Median of background
                             bg_val = torch.median(mag[b_idx][m_bg])
                         else:
                             bg_val = torch.tensor(0.0, device=self.device)
                         bg_vals.append(bg_val)
                     
                     bg_vals = torch.stack(bg_vals).view(-1, 1, 1) # (B, 1, 1)
                     
                     # Subtract background noise floor
                     # Clamp at 0 to avoid negative magnitude
                     mag_corrected = torch.relu(mag - bg_vals)
                     
                     # 2. Foreground Aggregation (95th Percentile)
                     # Instead of mean, we take the top 5% of motion in the mask
                     means = []
                     for b_idx in range(mag_corrected.shape[0]):
                         m_fg = mb[b_idx]
                         if torch.sum(m_fg) > 0:
                             fg_pixels = mag_corrected[b_idx][m_fg]
                             # 95th Percentile
                             # torch.quantile requires float
                             p95 = torch.quantile(fg_pixels, 0.95)
                             means.append(p95)
                         else:
                             means.append(torch.tensor(0.0, device=self.device))
                             
                     mean_mag = torch.stack(means).cpu().numpy()

                else:
                    # No mask - fallback to global mean or 95th??
                    # User comparison context implies Mask is always used for "Mouse Motion".
                    # If no mask, we probably still want 95th percentile to catch "fastest moving object".
                    # But without mask, background is huge.
                    # Let's use 95th percentile of whole image.
                     mean_mag = torch.quantile(mag.view(mag.shape[0], -1), 0.95, dim=1).cpu().numpy()
                    
                flow_magnitudes.extend(mean_mag)
                
                # Save chunks
                if h5_file and ds_flow:
                    # Not implemented fully for chunk writing here, simplified
                    pass

                frames_buffer = [frames_buffer[-1]]
                mask_buffer = [mask_buffer[-1]]
                pbar.update(len(mean_mag))

        # Process remaining
        if len(frames_buffer) > 1:
            img1 = torch.stack(frames_buffer[:-1]).to(self.device)
            img2 = torch.stack(frames_buffer[1:]).to(self.device)
            with torch.no_grad():
                 results = model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=False)
                 flows = results['flow_preds'][-1]
                     
                 mag = torch.sum(flows ** 2, dim=1).sqrt()
                 
                 # Apply Mask & Compensation (Same logic as batch)
                 if mask is not None or static_mask is not None:
                     mb = torch.stack(mask_buffer[:-1])
                     
                     # BG Compensation
                     bg_mask = ~mb
                     bg_vals = []
                     for b_idx in range(mag.shape[0]):
                         m_bg = bg_mask[b_idx]
                         if torch.sum(m_bg) > 0:
                             bg_val = torch.median(mag[b_idx][m_bg])
                         else:
                             bg_val = torch.tensor(0.0, device=self.device)
                         bg_vals.append(bg_val)
                     bg_vals = torch.stack(bg_vals).view(-1, 1, 1)
                     mag_corrected = torch.relu(mag - bg_vals)
                     
                     # FG 95th Percentile
                     means = []
                     for b_idx in range(mag_corrected.shape[0]):
                         m_fg = mb[b_idx]
                         if torch.sum(m_fg) > 0:
                             fg_pixels = mag_corrected[b_idx][m_fg]
                             p95 = torch.quantile(fg_pixels, 0.95)
                             means.append(p95)
                         else:
                             means.append(torch.tensor(0.0, device=self.device))
                     mean_mag = torch.stack(means).cpu().numpy()
                 else:
                     mean_mag = torch.quantile(mag.view(mag.shape[0], -1), 0.95, dim=1).cpu().numpy()
                     
                 flow_magnitudes.extend(mean_mag)
            pbar.update(len(frames_buffer) - 1)
            
        pbar.close()
        cap.release()
        if h5_file: h5_file.close()
        
        result = np.concatenate(([0], np.array(flow_magnitudes)))
        
        if save_path and save_path.endswith('.npy'):
            np.save(save_path, result)
            print(f"Saved raw optical flow magnitude to {save_path}")
            
        return result

    def compute_masked_flow_from_h5(self, h5_path, mask, batch_size=1000):
        """
        Computes Mean Magnitude from pre-computed H5 flow (dense), applying the given mask.
        H5 Shape expected: (T, H, W, 2)
        """
        print(f"Loading pre-computed Optical Flow from {h5_path}...")
        import h5py
        
        with h5py.File(h5_path, 'r') as f:
            if 'optical_flow' not in f:
                raise ValueError("H5 file must contain 'optical_flow' dataset.")
            ds = f['optical_flow'] 
            n_frames, h, w, c = ds.shape
            print(f"Flow Shape: {ds.shape}")
            
            # Prep Mask
            if mask is None:
                print("WARNING: No mask provided for H5 flow! Using global mean.")
                mask_gpu = None
            else:
                # Resize user mask to Flow dims
                m_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_gpu = torch.from_numpy(m_resized).bool().to(self.device) # (H, W)
                mask_count = torch.sum(mask_gpu).float() + 1e-6
            
            magnitudes = []
            
            # Batch Process
            for i in tqdm(range(0, n_frames, batch_size), desc="Processing H5 Flow"):
                batch_flow_np = ds[i : i+batch_size] # (B, H, W, 2)
                batch_flow = torch.from_numpy(batch_flow_np).to(self.device)
                
                # Mag: (B, H, W)
                mag = torch.sum(batch_flow ** 2, dim=3).sqrt()
                
                if mask_gpu is not None:
                    masked_mag = mag * mask_gpu 
                    means = (torch.sum(masked_mag, dim=[1, 2]) / mask_count).cpu().numpy()
                else:
                    means = torch.mean(mag, dim=[1, 2]).cpu().numpy()
                    
                magnitudes.extend(means)
                
        return np.array(magnitudes)
