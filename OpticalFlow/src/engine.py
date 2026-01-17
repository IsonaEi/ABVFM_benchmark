
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import IterableDataset, DataLoader
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

class VideoIterableDataset(IterableDataset):
    def __init__(self, video_path, resize_dim=(720, 720), mask_path=None, mask_array=None):
        self.video_path = video_path
        self.resize_dim = resize_dim
        self.mask_path = mask_path
        self.mask_array = mask_array
        
    def __iter__(self):
        # Open video inside iterator (worker process safe)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
            
        # Open H5 mask if needed
        mask_h5 = None
        if self.mask_path and self.mask_path.endswith('.h5'):
             import h5py
             mask_h5 = h5py.File(self.mask_path, 'r')
             
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_resized = cv2.resize(frame, self.resize_dim)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_tensor = F.to_tensor(frame_rgb) * 255.0 # (C, H, W)
            
            # Mask handling
            current_mask_np = np.zeros(self.resize_dim, dtype=bool) # Placeholder for "No Mask"
            has_mask = False
            
            if mask_h5:
                key = str(frame_idx)
                if key in mask_h5:
                    m = mask_h5[key][:]
                    m = cv2.resize(m.astype(np.uint8), self.resize_dim, interpolation=cv2.INTER_NEAREST)
                    current_mask_np = m.astype(bool)
                    has_mask = True
            elif self.mask_array is not None:
                # Static mask
                if self.mask_array.ndim == 2:
                    m = cv2.resize(self.mask_array.astype(np.uint8), self.resize_dim, interpolation=cv2.INTER_NEAREST)
                    current_mask_np = m.astype(bool)
                    has_mask = True
            
            # We return: (Tensor, NumpyOriginal, MaskBoolean, HasMaskFlag)
            # DataLoader collate might struggle with varying sizes if not careful.
            # Fixed sizes: Tensor (3, H, W), Numpy (H, W, 3) (uint8), Mask (H, W) (bool)
            yield frame_tensor, frame_resized, current_mask_np, has_mask
            
            frame_idx += 1
            
        cap.release()
        if mask_h5: mask_h5.close()

class OpticalFlowEngine:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def compute_optical_flow_gpu(self, video_path, mask=None, resize_dim=(720, 720), batch_size=6, save_path=None, full_flow_path=None, viz_path=None):
        """
        Computes Optical Flow using GMFlow (Strict) with DataLoader efficiency.
        """
        print(f"Computing Optical Flow (GPU: {self.device}) on {video_path}...")
        
        # --- Model Initialization ---
        if not HAS_GMFLOW:
            raise ImportError("GMFlow is required but not installed. Check 'OpticalFlow/gmflow'.")
            
        try:
            print("Using GMFlow (Sintel Weights)...")
            model = GMFlow(feature_channels=128, num_scales=1, upsample_factor=8, 
                           num_head=1, attention_type='swin', ffn_dim_expansion=4, 
                           num_transformer_layers=6).to(self.device)
            
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(base_path, 'pretrained/gmflow_sintel-0c07dcb3.pth')
            
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)
            else:
                raise FileNotFoundError(f"Weights not found at {weights_path}")
        except Exception as e:
            raise RuntimeError(f"GMFlow initialization failed: {e}")
            
        model.eval()
        
        # --- Output Initialization ---
        # Need total frames for H5 sizing. Open cap briefly.
        cap_temp = cv2.VideoCapture(video_path)
        if not cap_temp.isOpened(): return None
        n_frames_total = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_temp.get(cv2.CAP_PROP_FPS)
        cap_temp.release()
        
        h5_file = None
        ds_flow = None
        if full_flow_path and full_flow_path.endswith('.h5'):
             import h5py
             h5_file = h5py.File(full_flow_path, 'w')
             ds_flow = h5_file.create_dataset('optical_flow', shape=(n_frames_total, resize_dim[1], resize_dim[0], 2), dtype='float32', compression="gzip") 
        
        viz_writer = None
        if viz_path:
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             viz_writer = cv2.VideoWriter(viz_path, fourcc, fps, (resize_dim[0] * 2, resize_dim[1]))

        # --- DataLoader Setup ---
        mask_path_str = None
        mask_array_np = None
        
        if mask is not None:
            if isinstance(mask, str) and mask.endswith('.h5'):
                mask_path_str = mask
                print(f"Using dynamic mask H5: {mask}")
            elif isinstance(mask, np.ndarray):
                mask_array_np = mask
                print(f"Using static mask array.")

        dataset = VideoIterableDataset(video_path, resize_dim, mask_path=mask_path_str, mask_array=mask_array_np)
        
        # Num workers = 1 to put IO on background thread. 
        # pin_memory = True for faster CPU->GPU transfer.
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)
        
        flow_magnitudes = []
        
        # State preservation across batches
        prev_frame_tensor = None
        prev_frame_orig = None
        prev_mask = None
        prev_has_mask = None
        
        current_frame_write_idx = 1 # Start writing flow at index 1 (frame 0 -> 1)
        
        pbar = tqdm(total=n_frames_total)
        
        for batch in dataloader:
            # batch is list of tensors: [frames, origs, masks, has_masks]
            curr_frames, curr_origs, curr_masks, curr_has_masks = batch
            
            # curr_frames: (B, C, H, W)
            # Move to device
            curr_frames = curr_frames.to(self.device, non_blocking=True)
            
            # We need to construct pairs.
            # If we have prev_frame, prepend it.
            if prev_frame_tensor is not None:
                combined_frames = torch.cat([prev_frame_tensor.unsqueeze(0), curr_frames], dim=0)
            else:
                combined_frames = curr_frames
                
            # Now we have N frames. We want flow for (0->1), (1->2)...
            # If batch is first (prev is None), we generate B-1 flows from B frames?
            # Or is B=1 case?
            # Standard: if 6 frames [0..5]. We want 0->1, 1->2... 4->5.
            # Next batch starts with 6. We want 5->6.
            # So if prev is None, we just take flows of combined.
            
            if combined_frames.shape[0] < 2:
                # Should not happen unless batch size 1 and first batch
                prev_frame_tensor = combined_frames[-1]
                # Keep other props
                prev_frame_orig = curr_origs[-1].numpy()
                prev_mask = curr_masks[-1].to(self.device)
                prev_has_mask = curr_has_masks[-1]
                continue
                
            img1 = combined_frames[:-1]
            img2 = combined_frames[1:]
            
            with torch.no_grad():
                results = model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=False)
                flows = results['flow_preds'][-1] # (N_flows, 2, H, W)

            # Prepare Masks for these flows
            # Flow[i] corresponds to Frame[i] -> Frame[i+1]
            # We should use Mask[i].
            # combined_frames[i] is the source.
            # We need to assemble masks similarly.
            curr_masks_gpu = curr_masks.to(self.device) 
            # Note: curr_has_masks is CPU tensor likely
            
            if prev_mask is not None:
                combined_masks = torch.cat([prev_mask.unsqueeze(0), curr_masks_gpu], dim=0)
                # combined_has_mask... let's skip checking flag elementwise and just rely on boolean mask being empty or not?
                # Actually efficient to know if we need to run mask logic.
            else:
                combined_masks = curr_masks_gpu
            
            # Masks for source frames of flow
            source_masks = combined_masks[:-1] 
            
            # --- Magnitude & Local Ring ---
            mag = torch.sum(flows ** 2, dim=1).sqrt()
            
            means = []
            for b_idx in range(mag.shape[0]):
                m_fg = source_masks[b_idx]
                
                # Check if mask is "valid" (has content)
                if torch.sum(m_fg) > 0:
                    m_float = m_fg.float().unsqueeze(0).unsqueeze(0)
                    kernel_size = 51 
                    m_dilated = torch.nn.functional.max_pool2d(m_float, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
                    m_ring = (m_dilated.squeeze() > 0.5) & (~m_fg)
                    
                    if torch.sum(m_ring) > 0:
                        noise_floor = torch.median(mag[b_idx][m_ring])
                    else:
                        noise_floor = torch.tensor(0.0, device=self.device)
                        
                    fg_pixels = mag[b_idx][m_fg]
                    fg_rectified = torch.relu(fg_pixels - noise_floor)
                    if fg_rectified.numel() > 0:
                         val = torch.quantile(fg_rectified, 0.95)
                    else:
                         val = torch.tensor(0.0, device=self.device)
                    means.append(val)
                else:
                    # Global P95 if no mask
                    means.append(torch.quantile(mag[b_idx].view(-1), 0.95))
            
            mean_mag = torch.stack(means).cpu().numpy()
            flow_magnitudes.extend(mean_mag)
            
            # --- Saving ---
            if ds_flow:
                flows_np = flows.permute(0, 2, 3, 1).cpu().numpy()
                idx_end = current_frame_write_idx + flows_np.shape[0]
                if idx_end <= n_frames_total:
                    ds_flow[current_frame_write_idx : idx_end] = flows_np
            
            if viz_writer:
                flows_np = flows.permute(0, 2, 3, 1).cpu().numpy()
                # Need original frames. Assemble similarly.
                # curr_origs is batch numpy
                if prev_frame_orig is not None:
                    # Stack requires consistent dims. curr_origs is (B, H, W, 3). prev is (H, W, 3).
                    # numpy stack
                    curr_origs_np = curr_origs.numpy()
                    combined_origs = np.concatenate([prev_frame_orig[None, ...], curr_origs_np], axis=0)
                else:
                    combined_origs = curr_origs.numpy()
                
                source_origs = combined_origs[:-1]
                
                for b_idx in range(flows_np.shape[0]):
                    fl = flows_np[b_idx]
                    orig = source_origs[b_idx]
                    mag_np, _ = cv2.cartToPolar(fl[..., 0], fl[..., 1])
                    mag_norm = cv2.normalize(mag_np, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap = cv2.applyColorMap(mag_norm.astype(np.uint8), cv2.COLORMAP_JET)
                    combined_viz = np.hstack([orig, heatmap])
                    viz_writer.write(combined_viz)

            # Update State
            prev_frame_tensor = curr_frames[-1]
            prev_frame_orig = curr_origs[-1].numpy()
            prev_mask = curr_masks_gpu[-1]
            
            n_processed = len(mean_mag)
            current_frame_write_idx += n_processed
            pbar.update(n_processed)

        pbar.close()
        if h5_file: h5_file.close()
        if viz_writer: viz_writer.release()
        
        result = np.concatenate(([0], np.array(flow_magnitudes)))
        if save_path:
            np.save(save_path, result)
            print(f"Saved raw optical flow magnitude to {save_path}")
            
        return result
