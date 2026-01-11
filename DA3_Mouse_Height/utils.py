import numpy as np
import cv2
import h5py

def normalize_to_heatmap(data, vmin=None, vmax=None, cmap=cv2.COLORMAP_JET):
    """
    Normalizes data to 0-255 using min/max (or vmin/vmax) and applies colormap.
    NaNs are rendered as black (0,0,0).
    """
    if data is None: return None
    
    non_nan = data[~np.isnan(data)]
    if non_nan.size == 0:
        return np.zeros((*data.shape, 3), dtype=np.uint8)
    
    if vmin is None: vmin = np.nanmin(non_nan)
    if vmax is None: vmax = np.nanmax(non_nan)
    
    # Avoid div by zero
    val_range = vmax - vmin
    if val_range == 0: val_range = 1e-6
    
    norm = (data - vmin) / val_range
    norm = np.clip(norm, 0, 1)
    
    heatmap = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cmap)
    
    # Set NaNs to black
    if np.any(np.isnan(data)):
        heatmap[np.isnan(data)] = 0
        
    return heatmap

def load_mask(mask_h5, frame_idx, target_shape, dilate_iter=1):
    """
    Loads mask for a specific frame index from h5py file or dictionary.
    Resizes and dilates if necessary.
    Returns binary mask (0=Floor, 1=Object) or None.
    """
    H, W = target_shape
    key = str(frame_idx)
    
    if mask_h5 is None or key not in mask_h5:
        return np.zeros((H, W), dtype=np.uint8)

    m = mask_h5[key][:]
    
    # Resize if needed
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Binarize if not already
    binary_mask = (m > 0).astype(np.uint8)

    # Dilate
    if dilate_iter > 0:
        kernel = np.ones((15, 15), np.uint8) 
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=dilate_iter)
        
    return binary_mask

def align_depth(source, target, mask_source=None, mask_target=None):
    """
    Aligns source depth map to target depth map using linear regression (y = ax + b).
    Only valid pixels (not masked, not NaN) in both images are used for fitting.
    
    Returns:
        aligned_source: np.array (transformed source)
        a: scale factor
        b: offset
    """
    source_flat = source.flatten()
    target_flat = target.flatten()
    
    valid = (~np.isnan(source_flat)) & (~np.isnan(target_flat))
    
    if mask_source is not None:
        valid &= (mask_source.flatten() == 0)
    if mask_target is not None:
        valid &= (mask_target.flatten() == 0)
        
    x = source_flat[valid]
    y = target_flat[valid]
    
    # If not enough points overlap, return original (identity transform)
    if len(x) < 100:
        return source, 1.0, 0.0 
    
    # Linear Fit: y = ax + b
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    aligned_source = a * source + b
    return aligned_source, a, b

def calculate_global_scale(h5_ds, total_frames, sample_count=100, p_min=1, p_max=99):
    """
    Computes robust global min/max values (percentiles) by sampling frames.
    """
    indices = np.linspace(0, total_frames - 1, num=min(sample_count, total_frames), dtype=int)
    samples = []
    for idx in indices:
        d = h5_ds[idx]
        samples.append(d)
    
    samples = np.array(samples)
    g_min = np.nanpercentile(samples, p_min)
    g_max = np.nanpercentile(samples, p_max)
    return g_min, g_max

def resize_to_match(image, target_h, target_w):
    h, w = image.shape[:2]
    if (h, w) != (target_h, target_w):
        return cv2.resize(image, (target_w, target_h))
    return image
