
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import av
import h5py
import cv2
import sys
import os
import time
import multiprocessing as mp
import threading
import queue
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# Setup GMFlow Path
try:
    GMFLOW_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gmflow')
    if GMFLOW_PATH not in sys.path:
        sys.path.append(GMFLOW_PATH)
    from gmflow.gmflow import GMFlow
    HAS_GMFLOW = True
except ImportError as e:
    print(f"ERROR: GMFlow not found. Details: {e}")
    sys.exit(1)

RES_DIM = (720, 720) 

# --- Worker Function (CPU Consumer) ---
def worker_process(input_queue, output_queue):
    """
    Consumes (index, mag_map_cpu, frame_original_cpu, mask_cpu)
    Produces (index, metric_val, viz_frame)
    """
    while True:
        item = input_queue.get()
        if item is None:
            break
            
        frame_idx, mag_map, frame_orig, mask = item
        
        # 1. Analysis: Local Ring Subtraction
        # mag_map is (H, W) float32
        
        # Mask handling
        if mask is not None:
            mask_u8 = mask.astype(np.uint8)
            # Dilation (approx 25px radius = 51 kernel)
            kernel_size = 51
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            m_dilated = cv2.dilate(mask_u8, kernel, iterations=1)
            m_ring = (m_dilated > 0) & (mask_u8 == 0)
            
            noise_floor = 0.0
            if np.sum(m_ring) > 0:
                noise_floor = np.median(mag_map[m_ring])
            
            fg_pixels = mag_map[mask]
            fg_rectified = np.maximum(fg_pixels - noise_floor, 0)
            
            if fg_rectified.size > 0:
                val = np.percentile(fg_rectified, 95)
            else:
                val = 0.0
        else:
            # Fallback if no mask: simple P95
            val = np.percentile(mag_map, 95)
            
        # 2. Visualization
        # Normalize Magnitude
        mag_norm = cv2.normalize(mag_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(mag_norm.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Resize frame if needed (should already be 720p from main if logic is consistent, but safety check)
        if frame_orig.shape[:2] != RES_DIM:
            frame_resized = cv2.resize(frame_orig, RES_DIM)
        else:
            frame_resized = frame_orig
            
        combined_viz = np.hstack([frame_resized, heatmap])
        
        output_queue.put((frame_idx, val, combined_viz))

# --- Saver Thread ---
class SaverThread(threading.Thread):
    def __init__(self, output_path_npy, output_path_viz, fps, total_frames, result_queue):
        super().__init__()
        self.output_path_npy = output_path_npy
        self.output_path_viz = output_path_viz
        self.fps = fps
        self.total_frames = total_frames
        self.queue = result_queue
        self.stop_event = threading.Event()
        self.metrics = [0.0] * total_frames # Pre-allocate
        # Since we might skip frame 0 flow or have offsets, let's keep it simple: index matched
        
    def run(self):
        # Video Writer
        # Layout: Side-by-Side (720x2, 720)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        viz_writer = cv2.VideoWriter(self.output_path_viz, fourcc, self.fps, (RES_DIM[0] * 2, RES_DIM[1]))
        
        # Re-ordering Buffer
        # We expect frame 0, 1, 2...
        # Flow produces result for frame i (representing i->i+1 motion typically assigned to i or i+1)
        # Let's assume input index is adhered to.
        next_idx = 0
        buffer = {}
        
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1.0) # Wait 1s
            except queue.Empty:
                continue
            
            if item is None:
                # Poison pill from main
                break
                
            idx, val, frame_viz = item
            
            self.metrics[idx] = val
            buffer[idx] = frame_viz
            
            # Write sequential frames
            while next_idx in buffer:
                viz_writer.write(buffer.pop(next_idx))
                next_idx += 1
                
        # Flush remaining
        while next_idx in buffer:
            viz_writer.write(buffer.pop(next_idx))
            next_idx += 1
            
        viz_writer.release()
        
        # Save NPY
        # Note: If we missed frames at end (e.g. n-1 flows for n frames), truncate or pad?
        # Usually n frames video -> n-1 flows using pairs (0,1), (1,2)... 
        # We can pad the last one with 0 or copy previous.
        np.save(self.output_path_npy, np.array(self.metrics))

# --- Dataset ---
class PyAVPairDataset(IterableDataset):
    def __init__(self, video_path):
        self.video_path = video_path
        
    def __iter__(self):
        worker_info = get_worker_info()
        container = av.open(self.video_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        # Seek logic if multi-worker loader (but we will likely use 1 loader worker to keep sequence simple)
        # For simplicity in this complex pipeline, let's stick to num_workers=1 for DataLoader
        # and rely on Multi-Process downstream consumer for speed.
        
        for frame in container.decode(stream):
            if frame.pts is None: continue
            
            img_np = frame.to_ndarray(format='rgb24')
            curr_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            
            yield curr_tensor, img_np # Yield tensor + original (for viz)
            
        container.close()

# --- GMFlow Wrapper ---
class GMFlowExtractor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
    def _load_model(self):
        model = GMFlow(feature_channels=128, num_scales=1, upsample_factor=8, 
                       num_head=1, attention_type='swin', ffn_dim_expansion=4, 
                       num_transformer_layers=6).to(self.device)
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(base_path, 'pretrained/gmflow_sintel-0c07dcb3.pth')
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)
        model.eval()
        return model

def main():
    parser = argparse.ArgumentParser(description="One-Pass Optical Flow Pipeline")
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--mask', type=str, help='Path to mask (.h5 or .npy)')
    parser.add_argument('--output-npy', type=str, required=True, help='Path for result metrics (.npy)')
    parser.add_argument('--output-viz', type=str, required=True, help='Path for visualization video (.mp4)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-workers', type=int, default=4, help='Number of CPU consumer processes')
    
    args = parser.parse_args()
    
    # Init multiprocessing
    mp.set_start_method('spawn', force=True)
    
    print(f"Initializing Pipeline...")
    
    # 1. Video Info
    container = av.open(args.video)
    stream = container.streams.video[0]
    n_frames_total = stream.frames
    fps = float(stream.average_rate)
    container.close()
    
    # 2. Setup Resources (Queue, Workers)
    task_queue = mp.Queue(maxsize=50)
    result_queue = mp.Queue()
    
    consumers = []
    print(f"Spawning {args.n_workers} Consumer Processes...")
    for _ in range(args.n_workers):
        p = mp.Process(target=worker_process, args=(task_queue, result_queue))
        p.start()
        consumers.append(p)
        
    # 3. Setup Saver
    saver = SaverThread(args.output_npy, args.output_viz, fps, n_frames_total, result_queue)
    saver.start()
    
    # 4. Extract & Produce
    # Mask Loading (Main Thread)
    mask_h5 = None
    mask_static = None
    if args.mask:
        if args.mask.endswith('.h5'):
            mask_h5 = h5py.File(args.mask, 'r')
            print(f"Opened Mask H5: {args.mask}")
        else:
            mask_static = np.load(args.mask)
            # Resize static mask once
            mask_static = cv2.resize(mask_static.astype(np.uint8), RES_DIM, interpolation=cv2.INTER_NEAREST).astype(bool)
            print(f"Loaded Static Mask.")
            
    dataset = PyAVPairDataset(args.video)
    # Single worker loader to preserve sequence easily. Bottleneck should be handled by Prefetch of DataLoader if logic is simple.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)
    
    extractor = GMFlowExtractor()
    
    try:
        pbar = tqdm(total=n_frames_total, desc="Processing")
        
        # State
        prev_tensor = None
        prev_orig = None
        current_idx = 0
        
        for batch in dataloader:
            # batch: (tensors, orig_numpy)
            curr_tensors, curr_origs_batch = batch # curr_origs is byte tensor/numpy from collate? 
            # DataLoader collate turns list of numpy into Batch Tensor for current_origs? 
            # Yield was (tensor, np). Collate handles Tensor fine. Numpy array -> Tensor/Stack?
            # Actually PyTorch default collate converts numpy arrays to tensors.
            # So curr_origs_batch is (B, H, W, 3) Byte Tensor.
            
            curr_origs_batch = curr_origs_batch.numpy() # Back to numpy for Viz
            
            # Prepare GPU Input
            curr_tensors_gpu = curr_tensors.to(extractor.device, non_blocking=True)
            
            # Resize Logic
            target_h, target_w = RES_DIM[1], RES_DIM[0]
            if curr_tensors_gpu.shape[-2:] != (target_h, target_w):
                 processing_tensors = F.interpolate(curr_tensors_gpu, size=(target_h, target_w), mode='bilinear', align_corners=False)
            else:
                 processing_tensors = curr_tensors_gpu
                 
            # Sliding Window Logic for Batches
            # We need to pair frame i with i-1.
            # If batch is [0, 1, 2], we need flows (prev,0), (0,1), (1,2).
            # If prev is None (start), we just do (0,1), (1,2) effectively dropping flow for frame 0 (which is 0 anyway).
            
            # Logic: Combine [prev] + [curr]
            if prev_tensor is not None:
                combined = torch.cat([prev_tensor.unsqueeze(0), processing_tensors], dim=0)
                # Origs needed for viz
                combined_origs = np.concatenate([prev_orig[None,...], curr_origs_batch], axis=0)
            else:
                combined = processing_tensors
                combined_origs = curr_origs_batch
                
            if combined.shape[0] < 2:
                # Should only happen if batch=1 and first frame?
                prev_tensor = processing_tensors[-1]
                prev_orig = curr_origs_batch[-1]
                continue
                
            img1 = combined[:-1]
            img2 = combined[1:]
            
            # Inference
            with torch.no_grad():
                results = extractor.model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=False)
                flows = results['flow_preds'][-1] # (B, 2, H, W)
                
            # Compute Magnitude on GPU (B, H, W)
            # mag = sqrt(x^2 + y^2)
            mags = torch.sum(flows ** 2, dim=1).sqrt()
            
            # Move Result to CPU
            mags_cpu = mags.cpu().numpy() # (B, H, W) float32
            
            # Prepare Tasks
            # We have B flows.
            # They correspond to frames [prev_if_exist ... N-1] -> [ ... N]
            # Viz usually shows the *Source* frame overlaid with flow.
            # So we use combined_origs[:-1]
            
            source_frames = combined_origs[:-1]
            
            for i in range(mags_cpu.shape[0]):
                # Frame Index determination
                # If first batch (no prev), 0->1 is flow 0?
                # current_idx counts frames read.
                # If we process flow, we assign it to an index.
                # Let's say flow i is for frame (current_idx_start + i).
                # Wait, complexity with prev.
                # Simply: The flow calculated is for source_frames[i].
                # We need to know the true global index of source_frames[i].
                
                # Careful: current_idx tracks *start of this batch*.
                # If `prev` exists, the first flow is (prev -> batch[0]). prev's index was `current_idx - 1`.
                # If `prev` does NOT exist, first flow is (batch[0] -> batch[1]).
                
                # To simplify: Just track frame index continuously.
                pass
            
            # Correct approach for Indexing:
            # We are iterating batches.
            # If prev exists, combined has N+1 frames.
            # Flow 0 corresponds to prev (index: global_current-1).
            # Flow 1 corresponds to curr[0] (index: global_current).
            # But wait, we already processed prev in previous loop?
            # No, Sliding Window:
            # Loop 1: Frames [0, 1, 2]. Flows (0->1), (1->2). Output indices 0, 1. Prev=2.
            # Loop 2: Frames [3, 4, 5]. Combined [2, 3, 4, 5]. Flows (2->3), (3->4), (4->5). Output indices 2, 3, 4. Prev=5.
            # This covers all flows 0...N-1.
            
            start_offset = -1 if prev_tensor is not None else 0
            
            for i in range(mags_cpu.shape[0]):
                # Determine absolute Frame Index
                frame_abs_idx = current_idx + start_offset + i
                
                # Get Mask
                mask = None
                if mask_static is not None:
                    mask = mask_static
                elif mask_h5:
                    key = str(frame_abs_idx)
                    if key in mask_h5:
                        m = mask_h5[key][:]
                        m = cv2.resize(m.astype(np.uint8), RES_DIM, interpolation=cv2.INTER_NEAREST)
                        mask = m.astype(bool)
                
                # Put in Task Queue
                # (index, mag_map, orig_frame, mask)
                # Clone numpy arrays to ensure they are independent in memory when pickled?
                # Usually fine.
                task_queue.put((frame_abs_idx, mags_cpu[i], source_frames[i], mask))
            
            # Update state
            prev_tensor = processing_tensors[-1]
            prev_orig = curr_origs_batch[-1]
            current_idx +=  len(curr_origs_batch)
            pbar.update(len(curr_origs_batch))
            
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        pbar.close()
        
        # Stop Workers --
        for _ in range(args.n_workers):
            task_queue.put(None)
        
        # Wait for workers
        for p in consumers:
            p.join()
            
        # Stop Saver
        saver.stop_event.set() # Should be mostly empty
        saver.join()
        
        if mask_h5: mask_h5.close()
        
    print("Done.")

if __name__ == "__main__":
    main()
