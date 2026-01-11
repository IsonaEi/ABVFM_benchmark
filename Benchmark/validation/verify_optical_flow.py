
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import functional as TF
from tqdm import tqdm
import h5py

# Add GMFlow path (Benchmark/third_party/gmflow)
# Script is in Benchmark/validation/verify_optical_flow.py
# So we need to go up two levels to get to Benchmark/
sys.path.append(os.path.join(os.path.dirname(__file__), '../third_party/gmflow'))

try:
    from gmflow.gmflow import GMFlow
    HAS_GMFLOW = True
except ImportError as e:
    HAS_GMFLOW = False
    print(f"WARNING: GMFlow import failed: {e}")

def load_gmflow(model, weights_path):
    print(f"Loading GMFlow weights from {weights_path}...")
    checkpoint = torch.load(weights_path)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    return model

def run_all_methods_verification(video_path, output_path, duration_sec=30, device='cuda'):
    global HAS_GMFLOW
    print(f"Video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = min(total_frames, int(duration_sec * fps))
    
    # 1. Init RAFT (Big for better quality per request?) User said "suitable weights".
    # We were using RAFT Small. Let's upgrade to RAFT Large (Big) since user wants "best suitable".
    # raft_big is standard SOTA availability in torchvision.
    print("Loading RAFT (Large)...")
    raft_model = raft_large(weights='DEFAULT', progress=False).to(device)
    raft_model.eval()

    # 2. Init GMFlow
    gm_model = None
    if HAS_GMFLOW:
        try:
            # Init Model Architecture (Standard Sintel Config)
            gm_model = GMFlow(feature_channels=128,
                              num_scales=1,
                              upsample_factor=8,
                              num_head=1,
                              attention_type='swin',
                              ffn_dim_expansion=4,
                              num_transformer_layers=6,
                              ).to(device)
            gm_model.eval()
            
            # Load Weights
            # Weights are in Benchmark/third_party/gmflow/pretrained/...
            weights_path = os.path.join(os.path.dirname(__file__), '../third_party/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth')
            if os.path.exists(weights_path):
                load_gmflow(gm_model, weights_path)
            else:
                print(f"WARNING: Weights not found at {weights_path}")
                HAS_GMFLOW = False
        except Exception as e:
            print(f"GMFlow init failed: {e}")
            HAS_GMFLOW = False

    # Writer
    # 2x2 Grid
    # TL: Orig | TR: RAFT
    # BL: Farneback | BR: GMFlow
    out_w = w * 2
    out_h = h * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    prev_gray = None
    prev_frame_tensor = None      # For GMFlow
    prev_raft_img = None          # For RAFT (expects (1,3,H,W))

    print(f"Processing {target_frames} frames (All Methods)...")
    for i in tqdm(range(target_frames)):
        ret, frame_bgr = cap.read()
        if not ret: break
        
        # Prepare Inputs
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) # (1, 3, H, W)
        raft_input = TF.to_tensor(frame_rgb).unsqueeze(0).to(device) * 255.0
        
        # Canvases
        vis_raft = np.zeros((h, w, 3), dtype=np.uint8)
        vis_fb = np.zeros((h, w, 3), dtype=np.uint8)
        vis_gm = np.zeros((h, w, 3), dtype=np.uint8)
        
        if prev_gray is not None:
            # --- Farneback ---
            flow_fb = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag_fb, _ = cv2.cartToPolar(flow_fb[..., 0], flow_fb[..., 1])
            norm_fb = cv2.normalize(mag_fb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            try: vis_fb = cv2.applyColorMap(norm_fb, cv2.COLORMAP_INFERNO)
            except: vis_fb = cv2.applyColorMap(norm_fb, cv2.COLORMAP_HOT)
            
            # --- RAFT (Big) ---
            with torch.no_grad():
                list_of_flows = raft_model(prev_raft_img, raft_input)
                pred_raft = list_of_flows[-1][0].cpu().numpy() # (2, H, W)
                mag_raft, _ = cv2.cartToPolar(pred_raft[0], pred_raft[1])
                norm_raft = cv2.normalize(mag_raft, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                try: vis_raft = cv2.applyColorMap(norm_raft, cv2.COLORMAP_INFERNO)
                except: vis_raft = cv2.applyColorMap(norm_raft, cv2.COLORMAP_HOT)

            # --- GMFlow ---
            if HAS_GMFLOW and gm_model:
                with torch.no_grad():
                    results = gm_model(prev_frame_tensor, frame_tensor, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=False)
                    pred_gm = results['flow_preds'][-1][0].cpu().numpy()
                    mag_gm, _ = cv2.cartToPolar(pred_gm[0], pred_gm[1])
                    norm_gm = cv2.normalize(mag_gm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    try: vis_gm = cv2.applyColorMap(norm_gm, cv2.COLORMAP_INFERNO)
                    except: vis_gm = cv2.applyColorMap(norm_gm, cv2.COLORMAP_HOT)
            elif not HAS_GMFLOW:
                cv2.putText(vis_gm, "MiSSING", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Assemble Grid
        # Row 1: Orig | RAFT
        row1 = np.hstack([frame_bgr, vis_raft])
        # Row 2: Farneback | GMFlow
        row2 = np.hstack([vis_fb, vis_gm])
        
        combined = np.vstack([row1, row2])
        
        # Labels
        cv2.putText(combined, "Original (720p)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "RAFT (Big)", (w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Farneback", (50, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "GMFlow (Sintel)", (w + 50, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(combined)
        
        prev_gray = frame_gray
        prev_frame_tensor = frame_tensor
        prev_raft_img = raft_input
        
    cap.release()
    out.release()
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    VIDEO_PATH = "data/temp/ctrl_30fps.mp4"
    OUTPUT_PATH = "Benchmark/optical_flow_all_methods.mp4"
    run_all_methods_verification(VIDEO_PATH, OUTPUT_PATH)
