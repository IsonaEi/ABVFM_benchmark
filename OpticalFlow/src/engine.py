
import numpy as np
import torch
import torchvision.transforms.functional as F
import cv2
import sys
import os
from tqdm import tqdm

# Add GMFlow path for dynamic import
GMFLOW_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gmflow')
if GMFLOW_PATH not in sys.path:
    sys.path.append(GMFLOW_PATH)

try:
    from gmflow.gmflow import GMFlow
    HAS_GMFLOW = True
except ImportError:
    HAS_GMFLOW = False

class OpticalFlowEngine:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def compute_optical_flow_gpu(self, video_path, mask=None, resize_dim=(720, 720), batch_size=4, save_path=None):
        """
        Computes Optical Flow using GMFlow (Strict).
        Raises Error if GMFlow package or weights are missing.
        Default resize_dim is (720, 720).
        """
        print(f"Computing Optical Flow (GPU: {self.device}) on {video_path}...")
        
        # --- Model Initialization ---
        if not HAS_GMFLOW:
            raise ImportError("GMFlow is required but not installed or found in path. Please check 'OpticalFlow/gmflow'.")
            
        try:
            print("Using GMFlow (Sintel Weights)...")
            model = GMFlow(feature_channels=128, num_scales=1, upsample_factor=8, 
                           num_head=1, attention_type='swin', ffn_dim_expansion=4, 
                           num_transformer_layers=6).to(self.device)
            
            # Updated weights path relative to new structure
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(base_path, 'pretrained/gmflow_sintel-0c07dcb3.pth')
            
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)
            else:
                raise FileNotFoundError(f"GMFlow weights not found at {weights_path}! Please download them.")
        except Exception as e:
            raise RuntimeError(f"GMFlow initialization failed: {e}")
            
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
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, resize_dim)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = F.to_tensor(frame_rgb) * 255.0 # (C, H, W)
            
            frames_buffer.append(frame_tensor)
            mask_buffer.append(static_mask.squeeze(0) if static_mask is not None else None)
            
            if len(frames_buffer) >= batch_size + 1:
                img1 = torch.stack(frames_buffer[:-1]).to(self.device)
                img2 = torch.stack(frames_buffer[1:]).to(self.device)
                
                with torch.no_grad():
                    results = model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=False)
                    flows = results['flow_preds'][-1] # (B, 2, H, W)
                
                mag = torch.sum(flows ** 2, dim=1).sqrt() # (B, H, W)
                
                if mask is not None or static_mask is not None:
                     mb = torch.stack(mask_buffer[:-1]) # (B, H, W) Boolean
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
                
                if h5_file and ds_flow:
                    # Optional: Could write dense flow here
                    pass

                frames_buffer = [frames_buffer[-1]]
                mask_buffer = [mask_buffer[-1]]
                pbar.update(len(mean_mag))

        if len(frames_buffer) > 1:
            img1 = torch.stack(frames_buffer[:-1]).to(self.device)
            img2 = torch.stack(frames_buffer[1:]).to(self.device)
            with torch.no_grad():
                 results = model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=False)
                 flows = results['flow_preds'][-1]
                 mag = torch.sum(flows ** 2, dim=1).sqrt()
                 
                 if mask is not None or static_mask is not None:
                     mb = torch.stack(mask_buffer[:-1])
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
            
            if mask is None:
                print("WARNING: No mask provided for H5 flow! Using global mean.")
                mask_gpu = None
            else:
                m_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_gpu = torch.from_numpy(m_resized).bool().to(self.device) # (H, W)
                mask_count = torch.sum(mask_gpu).float() + 1e-6
            
            magnitudes = []
            
            for i in tqdm(range(0, n_frames, batch_size), desc="Processing H5 Flow"):
                batch_flow_np = ds[i : i+batch_size] # (B, H, W, 2)
                batch_flow = torch.from_numpy(batch_flow_np).to(self.device)
                
                mag = torch.sum(batch_flow ** 2, dim=3).sqrt()
                
                if mask_gpu is not None:
                    masked_mag = mag * mask_gpu 
                    means = (torch.sum(masked_mag, dim=[1, 2]) / mask_count).cpu().numpy()
                else:
                    means = torch.mean(mag, dim=[1, 2]).cpu().numpy()
                    
                magnitudes.extend(means)
                
        return np.array(magnitudes)
