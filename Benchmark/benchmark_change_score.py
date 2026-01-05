
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, sem
import os

# --- Configuration ---
BODYPARTS_OF_INTEREST = [
    'snout', 'left_ear', 'right_ear', 'shoulder', 
    'spine_1', 'spine_2', 'spine_3', 'spine_4', 'tail_base'
]  # Adjust based on actual file content if needed

def load_dlc_keypoints(path):
    """
    Loads keypoints from DLC H5 output (pandas DataFrame).
    Returns: 
        kps: (T, K, 2) numpy array
        bodyparts: list of bodypart names
    """
    print(f"Loading keypoints from {path}...")
    try:
        df = pd.read_hdf(path)
    except ImportError:
        # Fallback if pytables is not installed, though usually it is with pandas
        print("Error: PyTables not found or HDF5 read error.")
        return None, None
        
    # DLC columns are MultiIndex: (scorer, bodyparts, coords)
    # Get scorer name (usually level 0)
    scorer = df.columns.levels[0][0]
    
    # Get bodyparts
    bodyparts = df.columns.levels[1].tolist()
    
    # Extract X and Y for each bodypart
    # We want a generic way to extract (T, K, 2)
    # Sort bodyparts to ensure consistent order if we rely on indices, 
    # but here we simply iterate the list we found.
    
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
            
    # Stack to (T, K, 2)
    kps = np.stack(kps_list, axis=1) # (T, K, 2)
    
    print(f"Loaded {kps.shape[0]} frames, {kps.shape[1]} bodyparts.")
    return kps, found_bps

def load_kpms_labels(path):
    """
    Loads KPMS syllables from H5.
    Expected path: /<filename>/syllable
    """
    print(f"Loading KPMS labels from {path}...")
    with h5py.File(path, 'r') as f:
        # Assuming only one top-level group exists which matches the video name
        # If multiple, we might need a specific key.
        keys = list(f.keys())
        if len(keys) == 0:
            raise ValueError("KPMS H5 file is empty.")
        
        # Heuristic: use the first key
        dataset_key = keys[0]
        print(f"Accessing KPMS group: {dataset_key}")
        
        if 'syllable' in f[dataset_key]:
            labels = f[dataset_key]['syllable'][:]
        elif 'latent_state' in f[dataset_key]: # Fallback
            print("Warning: 'syllable' dataset not found, using 'latent_state'.")
            labels = f[dataset_key]['latent_state'][:]
        else:
            raise KeyError("'syllable' or 'latent_state' not found in KPMS file.")
            
    return labels

def load_castle_labels(path):
    """
    Loads CASTLE labels from NPY or CSV.
    """
    print(f"Loading CASTLE labels from {path}...")
    if path.endswith('.npy'):
        labels = np.load(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
        labels = df['behavior_label'].values # Adjust column name if needed
        if labels.dtype == object:
             labels, _ = pd.factorize(labels)
    else:
        raise ValueError("Unsupported format for CASTLE labels.")
    
    return labels

def load_bsoid_labels(path):
    """
    Loads B-SOiD labels from CSV.
    Expected columns: 'Time', 'B-SOiD_Label'
    Returns:
        labels: numpy array of labels
        fps: float, calculated from Time column
    """
    print(f"Loading B-SOiD labels from {path}...")
    df = pd.read_csv(path)
    
    if 'Time' not in df.columns or 'B-SOiD_Label' not in df.columns:
        raise ValueError("B-SOiD CSV must have 'Time' and 'B-SOiD_Label' columns.")
        
    labels = df['B-SOiD_Label'].values
    times = df['Time'].values
    
    # Calculate FPS from time diffs
    # Taking median diff to be robust against small jitters
    dt = np.median(np.diff(times))
    if dt <= 0:
        print("Warning: B-SOiD time diff <= 0, defaulting to 10 FPS.")
        fps = 10.0
    else:
        fps = 1.0 / dt
        
    print(f"B-SOiD FPS estimated as: {fps:.2f} Hz")
    
    return labels, fps

def egocentric_alignment(kps, bodyparts):
    """
    Aligns keypoints to egocentric frame (Heading: Tail -> Snout).
    Returns: aligned_kps (T, K, 2)
    """
    print("Performing egocentric alignment...")
    
    # Identify indices
    try:
        # Heuristic matching
        snout_candidates = [i for i, b in enumerate(bodyparts) if 'snout' in b.lower() or 'nose' in b.lower()]
        tail_candidates = [i for i, b in enumerate(bodyparts) if 'tail' in b.lower()]
        
        snout_idx = snout_candidates[0] if snout_candidates else 0
        tail_idx = tail_candidates[-1] if tail_candidates else -1 # Use base of tail usually
        
        print(f"Using '{bodyparts[snout_idx]}' as snout and '{bodyparts[tail_idx]}' as tail/body anchor.")
    except IndexError:
        print("Warning: Could not identify snout/tail automatically. Skipping rotation.")
        snout_idx, tail_idx = None, None

    T, K, C = kps.shape
    aligned_kps = np.zeros_like(kps)
    
    # Vectorized approach when possible, but loop is safer for handling NaNs per frame
    # Let's try a mostly vectorized approach for speed, with nan handling
    
    # 1. Centroid
    # nanmean over keypoints
    centroid = np.nanmean(kps, axis=1, keepdims=True) # (T, 1, 2)
    centered = kps - centroid
    
    if snout_idx is not None and tail_idx is not None:
        # 2. Heading Vector
        snout = centered[:, snout_idx, :]
        tail = centered[:, tail_idx, :]
        
        diff = snout - tail
        angles = np.arctan2(diff[:, 1], diff[:, 0]) # (T,)
        
        # 3. Rotation Matrix
        # We want to rotate by -angle
        c = np.cos(-angles)
        s = np.sin(-angles)
        
        # R shape: (T, 2, 2)
        # [[c, -s],
        #  [s, c]]
        R = np.zeros((T, 2, 2))
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        
        # Apply rotation: (T, K, 2)
        # Einsum: tji, tkj -> tki (Wait, let's correspond dims)
        # R is (T, 2, 2) -> (frame, row, col)
        # centered is (T, K, 2) -> (frame, point, coord)
        # We want for each t: R[t] @ centered[t].T
        # equivalent to (centered[t] @ R[t].T) ??
        # Let's use einsum: 'tij, tkj -> tki' where i=out_coord, j=in_coord
        # Actually standard matmul broadcast:
        # centered vector v.  R.v
        # Let's treat centered as (T, K, 2, 1) and R as (T, 1, 2, 2) -> matmul
        # Easiest: 'txy, tkx -> tky' (cols x, y used as indices)... confusing.
        
        # Simple loop for rotation is robust and standard
        for t in range(T):
            if np.isnan(angles[t]):
                aligned_kps[t] = centered[t]
                continue
                
            R_t = R[t] # (2, 2)
            pts = centered[t] # (K, 2)
            
            # (2,2) @ (2, K) -> (2, K) -> T
            aligned_kps[t] = (R_t @ pts.T).T
            
    else:
        aligned_kps = centered

    return aligned_kps

def compute_change_score(kps, sigma=1.0):
    """
    Computes Z-scored velocity of aligned keypoints.
    """
    print("Computing change score...")
    
    # 1. Handle NaNs: Interpolate
    # Reshape to (T, K*2) for dataframe interpolation
    T, K, C = kps.shape
    flat = kps.reshape(T, -1)
    df = pd.DataFrame(flat)
    df = df.interpolate(method='linear', limit_direction='both')
    kps_clean = df.values.reshape(T, K, C)
    
    # 2. Gaussian Smooth
    kps_smooth = gaussian_filter1d(kps_clean, sigma=sigma, axis=0)
    
    # 3. Velocity (Euclidean dist sum)
    # diff[t] = kps[t] - kps[t-1]
    diff = np.diff(kps_smooth, axis=0) # (T-1, K, 2)
    norms = np.linalg.norm(diff, axis=2) # (T-1, K)
    total_vel = np.sum(norms, axis=1) # (T-1,)
    
    # Pad to T
    total_vel = np.concatenate(([0], total_vel))
    
    # 4. Z-score
    return zscore(total_vel)

def get_event_triggered_traces(score, labels, window=15, downsample_factor=1):
    """
    Extracts traces of change_score around label transitions.
    handles downsampling (e.g., if labels are 6Hz and score is 30Hz, factor=5).
    """
    # Detect transitions in label space
    # labels shape (N,)
    transitions = np.where(labels[1:] != labels[:-1])[0] + 1
    
    traces = []
    T_score = len(score)
    window_score = window # Window is usually given in score frames? 
    # Or window is given in label frames? Usually window is defined in time (e.g. 0.5s)
    # If window=15 frames @ 30Hz = 0.5s.
    
    # If labels are downsampled, 'transition index' t corresponds to t * factor in score
    
    for t_idx in transitions:
        # center in score domain
        center_score = int(t_idx * downsample_factor)
        
        start = center_score - window
        end = center_score + window
        
        if start >= 0 and end < T_score:
            traces.append(score[start:end])
            
    if not traces:
        return np.array([]), np.array([]), 0
        
    stack = np.stack(traces)
    mean_trace = np.mean(stack, axis=0)
    sem_trace = sem(stack, axis=0)
    
    return mean_trace, sem_trace, stack

def compute_state_durations(labels, fps):
    """
    Computes duration of each state segment in seconds.
    """
    # RLE (Run Length Encoding)
    # transitions: indices where value changes
    # Calculate lengths between transitions
    
    # E.g. [0, 0, 1, 1, 1, 0] -> lengths [2, 3, 1]
    
    n = len(labels)
    if n == 0:
        return np.array([])
        
    y = np.array(labels)
    mismatch = np.where(y[1:] != y[:-1])[0]
    
    # End indices of each run
    run_ends = np.append(mismatch, n-1)
    # Start indices of each run (0 + shifted ends)
    run_starts = np.insert(run_ends[:-1] + 1, 0, 0)
    
    run_lengths = run_ends - run_starts + 1
    
    # Convert to seconds
    durations = run_lengths / fps
    return durations

def main():
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KEYPOINT_DIR = os.path.join(BASE_DIR, "keypoint_data")
    LABEL_DIR = os.path.join(BASE_DIR, "lable_data") # Preserving user typo 'lable'
    
    # Find files (robustness)
    try:
        kps_file = [f for f in os.listdir(KEYPOINT_DIR) if f.endswith('.h5')][0]
        kps_path = os.path.join(KEYPOINT_DIR, kps_file)
        
        kpms_path = os.path.join(LABEL_DIR, "KPMS_results.h5")
        
        # Robustly find CASTLE file
        castle_files = [f for f in os.listdir(LABEL_DIR) if 'CASTLE' in f and (f.endswith('.npy') or f.endswith('.csv'))]
        if not castle_files:
             print("Warning: No CASTLE file found.")
             castle_path = None
        else:
             castle_path = os.path.join(LABEL_DIR, castle_files[0])
             
        # Robustly find B-SOiD file
        bsoid_files = [f for f in os.listdir(LABEL_DIR) if 'B-soid' in f and f.endswith('.csv')]
        if not bsoid_files:
             print("Warning: No B-SOiD file found.")
             bsoid_path = None
        else:
             bsoid_path = os.path.join(LABEL_DIR, bsoid_files[0])
             
    except IndexError as e:
        print(f"Error finding files: {e}")
        return

    # 1. Load Data
    kps, bodyparts = load_dlc_keypoints(kps_path)
    kpms_labels = load_kpms_labels(kpms_path)
    
    castle_labels = None
    if castle_path:
        castle_labels = load_castle_labels(castle_path)
        print(f"CASTLE Labels: {castle_labels.shape}")
        
    bsoid_labels = None
    bsoid_fps = None
    if bsoid_path:
        bsoid_labels, bsoid_fps = load_bsoid_labels(bsoid_path)
        print(f"B-SOiD Labels: {bsoid_labels.shape}")
    
    print(f"Keypoints: {kps.shape}")
    print(f"KPMS Labels: {kpms_labels.shape}")
    
    # 2. Compute Change Score on FULL data
    aligned_kps = egocentric_alignment(kps, bodyparts)
    change_score = compute_change_score(aligned_kps, sigma=1.0)
    
    # 3. Analyze
    # Determine downsample factors
    # Assuming KPS is the high-res ground truth (30Hz)
    
    # KPMS
    factor_kpms = 1
    if len(kpms_labels) != len(change_score):
        factor_kpms = len(change_score) / len(kpms_labels)
        print(f"Detected KPMS downsampling factor: {factor_kpms}")
        
    # CASTLE
    factor_castle = 1
    if castle_labels is not None and len(castle_labels) != len(change_score):
        factor_castle = len(change_score) / len(castle_labels)
        print(f"Detected CASTLE downsampling factor: {factor_castle}")
        
    # B-SOiD
    factor_bsoid = 1
    if bsoid_labels is not None and len(bsoid_labels) != len(change_score):
         factor_bsoid = len(change_score) / len(bsoid_labels)
         print(f"Detected B-SOiD downsampling factor: {factor_bsoid}")
        
    WINDOW = 45 # 1.5s @ 30Hz
    
    print("Extracting KPMS traces...")
    kpms_m, kpms_s, kpms_stack = get_event_triggered_traces(change_score, kpms_labels, WINDOW, factor_kpms)
    
    # --- Experiment: Downsampled KPMS ---
    print("Extracting KPMS (Simulated 6Hz) traces...")
    kpms_labels_down = kpms_labels[::5] 
    factor_kpms_down = len(change_score) / len(kpms_labels_down)
    kpms_down_m, kpms_down_s, kpms_down_stack = get_event_triggered_traces(change_score, kpms_labels_down, WINDOW, factor_kpms_down)
    # ------------------------------------
    
    if castle_labels is not None:
        print("Extracting CASTLE traces...")
        castle_m, castle_s, castle_stack = get_event_triggered_traces(change_score, castle_labels, WINDOW, factor_castle)
    else:
        castle_stack = []
        
    if bsoid_labels is not None:
        print("Extracting B-SOiD traces...")
        bsoid_m, bsoid_s, bsoid_stack = get_event_triggered_traces(change_score, bsoid_labels, WINDOW, factor_bsoid)
    else:
        bsoid_stack = []
    
    # 4. Plot (2 Subplots)
    print("Generating plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
    
    # --- Linear Plot (Left) ---
    x = np.arange(-WINDOW, WINDOW)
    # Convert frames to seconds (30Hz)
    x_sec = x / 30.0
    
    # KPMS (Gold)
    if len(kpms_stack) > 0:
        ax1.plot(x_sec, kpms_m, label=f'KPMS (30Hz)', color='#E0C068', linewidth=2, zorder=3)
        ax1.fill_between(x_sec, kpms_m-kpms_s, kpms_m+kpms_s, color='#E0C068', alpha=0.3, zorder=3)

    # KPMS Downsampled (Dashed Gold)
    if len(kpms_down_stack) > 0:
        ax1.plot(x_sec, kpms_down_m, label=f'KPMS (Sim. 6Hz)', color='#E0C068', linewidth=2, linestyle='--', zorder=2)
        
    # CASTLE (Blue)
    if len(castle_stack) > 0:
        ax1.plot(x_sec, castle_m, label=f'CASTLE (6Hz)', color='#4A90E2', linewidth=2, zorder=2)
        ax1.fill_between(x_sec, castle_m-castle_s, castle_m+castle_s, color='#4A90E2', alpha=0.3, zorder=2)
        
    # B-SOiD (Green)
    if len(bsoid_stack) > 0:
        # Round label for legend
        fps_lbl = f"{int(round(bsoid_fps))}Hz" if bsoid_fps else "N/A"
        ax1.plot(x_sec, bsoid_m, label=f'B-SOiD ({fps_lbl})', color='#50C878', linewidth=2, zorder=2)
        ax1.fill_between(x_sec, bsoid_m-bsoid_s, bsoid_m+bsoid_s, color='#50C878', alpha=0.3, zorder=2)
        
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time from transition (s)')
    ax1.set_ylabel('Change score (z)')
    ax1.set_title('Event-Triggered Average')
    ax1.legend()
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Violin Plot (Right) ---
    # Extract values at t=0 (center of window)
    # Window index is exactly WINDOW (since range is start:end, length is 2*window)
    # indices: 0 ... WINDOW ... 2*WINDOW-1
    center_idx = WINDOW
    
    violin_data = []
    violin_labels = []
    violin_colors = []
    
    if len(kpms_stack) > 0:
        violin_data.append(kpms_stack[:, center_idx])
        violin_labels.append('KPMS')
        violin_colors.append('#E0C068')
        
    if len(kpms_down_stack) > 0:
        violin_data.append(kpms_down_stack[:, center_idx])
        violin_labels.append('KPMS\n(6Hz)')
        violin_colors.append('#E0C068') # Same color, maybe lighter?
        
    if len(castle_stack) > 0:
        violin_data.append(castle_stack[:, center_idx])
        violin_labels.append('CASTLE')
        violin_colors.append('#4A90E2')
        
    if len(bsoid_stack) > 0:
        violin_data.append(bsoid_stack[:, center_idx])
        violin_labels.append('B-SOiD')
        violin_colors.append('#50C878')

    if violin_data:
        parts = ax2.violinplot(violin_data, showmeans=True, showextrema=False)
        
        # Color the bodies
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors[i])
            pc.set_alpha(0.7)
            
        # Style the means
        parts['cmeans'].set_edgecolor('black')
        parts['cmeans'].set_linewidth(1.5)
        
        ax2.set_xticks(np.arange(1, len(violin_labels) + 1))
        ax2.set_xticklabels(violin_labels)
        ax2.set_ylabel('Change score (z)')
        ax2.set_title('Distribution at Transition')
        ax2.set_ylim(-2, 2)
        
        # Remove top and right spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False) # Optional for clean look
        ax2.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    
    out_file = os.path.join(BASE_DIR, "benchmark_change_score_combined.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")
    
    # 5. Combined Ethogram & Duration Plot (Fig 3a/b style)
    print("Generating Ethogram & Duration plot...")
    
    # Pack data
    labels_dict = {'KPMS': kpms_labels}
    fps_dict = {'KPMS': 30.0}
    
    if castle_labels is not None:
        labels_dict['CASTLE'] = castle_labels
        fps_dict['CASTLE'] = 6.0 # Hardcoded usually, or infer? let's stick to 6.0 as known
        
    if bsoid_labels is not None:
        labels_dict['B-SOiD'] = bsoid_labels
        fps_dict['B-SOiD'] = bsoid_fps if bsoid_fps else 30.0

    plot_ethogram_and_durations(
        labels_dict,
        fps_dict,
        window_min=(5, 7) # 5th to 7th minute
    )

def plot_ethogram_and_durations(labels_dict, fps_dict, window_min=(5, 7)):
    """
    Generates a figure with Ethograms (Left) and Duration Histograms (Right).
    Matches layout of Fig 3a,b in KPMS paper.
    """
    methods = list(labels_dict.keys())
    n_methods = len(methods)
    
    # Define method-specific cmaps (Qualitative but within a hue range)
    # Actually paper uses distinct colors for states, but user wants "same method same color system"
    # This implies using shades of Gold for KPMS states, and shades of Blue for CASTLE states.
    # We can generate a colormap from a base color.
    
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    import matplotlib.cm as cm
    
    def get_method_cmap(base_color_hex, n_states=20):
        # Create a palette of n_states mixing base_color with white/black or varying saturation
        # Or just use a matplotlib sequential map like 'Oranges' or 'Blues' but shuffled
        # to avoid adjacent states looking too similar?
        
        # Let's use a sequential map but sample it to get discrete colors
        if 'E0C068' in base_color_hex: # Gold -> Oranges/YlOrBr
             base_cmap = cm.get_cmap('YlOrBr', n_states) # Discrete sampling
        elif '50C878' in base_color_hex: # Green -> Greens
             base_cmap = cm.get_cmap('Greens', n_states)
        else: # Blue -> Blues/PuBu
             base_cmap = cm.get_cmap('Blues', n_states)
             
        # Extract colors
        colors = base_cmap(np.linspace(0.2, 1.0, n_states))
        # Shuffle to make adjacent states distinct
        np.random.seed(42) 
        np.random.shuffle(colors)
        
        return ListedColormap(colors)

    fig = plt.figure(figsize=(10, 3 * n_methods))
    gs = fig.add_gridspec(n_methods, 2, width_ratios=[3, 1], hspace=0.4, wspace=0.1)
    
    for i, method in enumerate(methods):
        # Data
        lbls = labels_dict[method]
        fps = fps_dict[method]
        
        # Base color
        if 'KPMS' in method:
            base_color = '#E0C068'
        elif 'B-SOiD' in method:
            base_color = '#50C878'
        else:
            base_color = '#4A90E2'
        
        # --- Left: Ethogram (Barcode) ---
        ax_eth = fig.add_subplot(gs[i, 0])
        
        start_frame = int(window_min[0] * 60 * fps)
        end_frame = int(window_min[1] * 60 * fps)
        start_frame = max(0, start_frame)
        end_frame = min(len(lbls), end_frame)
        
        segment = lbls[start_frame:end_frame]
        
        # Remap labels to 0..N for color mapping
        # We need a consistent mapping if we want specific states to track, 
        # but here just visual density matters.
        unique_labels = np.unique(segment)
        # Create a localized mapping for this segment to maximize color usage? 
        # Or use global max label.
        # Let's map unique labels in segment to random indices in our cmap range
        
        if len(segment) > 0:
            # Create a compact mapping 0..K
            # But to ensure visual distinctness of adjacent different states:
            # Just plotting the raw IDs with a high-res cmap (shuffled) works well.
            
            # Use method specific cmap
            n_states_local = len(np.unique(unique_labels))
            # Heuristic: roughly 30 states max usually
            my_cmap = get_method_cmap(base_color, n_states=max(30, n_states_local))
            
            # Map segment values to random integers in [0, 30) to avoid order bias
            # This ensures adjacent processing-order states don't get adjacent colors
            mapper = {l: np.random.randint(0, 30) for l in unique_labels}
            mapped_segment = np.array([mapper.get(x, 0) for x in segment])
            
            im_data = mapped_segment.reshape(1, -1)
            ax_eth.imshow(im_data, aspect='auto', cmap=my_cmap, interpolation='nearest', vmin=0, vmax=29)
        
        ax_eth.set_title(method, loc='left', fontsize=12, fontweight='bold')
        ax_eth.set_yticks([]) 
        
        duration_sec = (end_frame - start_frame) / fps
        # Ticks every 30s
        xticks = np.arange(0, len(segment), 30 * fps)
        xticklabels = [f"{int(t/fps)}s" for t in xticks]
        
        ax_eth.set_xticks(xticks)
        ax_eth.set_xticklabels(xticklabels)
        if i < n_methods - 1:
            ax_eth.set_xticklabels([])
        
        for spine in ax_eth.spines.values():
            spine.set_visible(False)
            
        # --- Right: Duration Histogram ---
        ax_hist = fig.add_subplot(gs[i, 1])
        durations = compute_state_durations(lbls, fps)
        bins = np.linspace(0, 2, 41) 
        
        ax_hist.hist(durations, bins=bins, color=base_color, alpha=0.8, density=True, edgecolor='white', linewidth=0.5)
        
        ax_hist.set_xlim(0, 2)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.set_yticks([]) 
        
        if i == 0:
            ax_hist.set_title("State Durations", fontsize=10)
        if i == n_methods - 1:
            ax_hist.set_xlabel("Duration (s)")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_fig3_AB_style.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved Fig 3a/b style plot to {out_path}")

if __name__ == "__main__":
    main()
