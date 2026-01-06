
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, sem
import os
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import re

def load_dlc_keypoints(path):
    print(f"Loading keypoints from {path}...")
    try:
        df = pd.read_hdf(path)
    except Exception as e:
        print(f"Error reading HDF5: {e}")
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
    return kps, found_bps

def load_kpms_labels(path):
    print(f"Loading KPMS labels from {path}...")
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        if not keys: raise ValueError("KPMS H5 file is empty.")
        dataset_key = keys[0]
        if 'syllable' in f[dataset_key]: labels = f[dataset_key]['syllable'][:]
        elif 'latent_state' in f[dataset_key]: labels = f[dataset_key]['latent_state'][:]
        else: raise KeyError("'syllable' or 'latent_state' not found in KPMS file.")
    return labels

def load_castle_labels(path):
    print(f"Loading CASTLE labels from {path}...")
    if path.endswith('.npy'):
        labels = np.load(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
        if 'behavior_label' in df.columns:
            labels = df['behavior_label'].values
        elif 'behavior' in df.columns:
            labels = df['behavior'].values
        else:
            raise KeyError(f"No 'behavior_label' or 'behavior' column found in {path}")
        if labels.dtype == object: labels, _ = pd.factorize(labels)
    else: raise ValueError("Unsupported format for CASTLE labels.")
    return labels

def load_bsoid_labels(path):
    print(f"Loading B-SOiD labels from {path}...")
    df = pd.read_csv(path)
    if 'Time' not in df.columns or 'B-SOiD_Label' not in df.columns:
        raise ValueError("B-SOiD CSV must have 'Time' and 'B-SOiD_Label' columns.")
    labels = df['B-SOiD_Label'].values
    return labels

def resample_labels(labels, target_len):
    """
    Resample labels to target length using nearest neighbor interpolation.
    """
    if len(labels) == target_len:
        return labels
    indices = np.linspace(0, len(labels) - 1, target_len)
    return labels[np.round(indices).astype(int)]

def egocentric_alignment(kps, bodyparts):
    print("Performing egocentric alignment...")
    snout_cand = [i for i, b in enumerate(bodyparts) if 'snout' in b.lower() or 'nose' in b.lower()]
    tail_cand = [i for i, b in enumerate(bodyparts) if 'tail' in b.lower()]
    snout_idx = snout_cand[0] if snout_cand else 0
    tail_idx = tail_cand[-1] if tail_cand else -1
    T, K, C = kps.shape
    centroid = np.nanmean(kps, axis=1, keepdims=True)
    centered = kps - centroid
    aligned = np.zeros_like(kps)
    snout = centered[:, snout_idx, :]
    tail = centered[:, tail_idx, :]
    diff = snout - tail
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    for t in range(T):
        if np.isnan(angles[t]):
            aligned[t] = centered[t]
            continue
        c, s = np.cos(-angles[t]), np.sin(-angles[t])
        R_t = np.array([[c, -s], [s, c]])
        aligned[t] = (R_t @ centered[t].T).T
    return aligned

def compute_change_score(kps, sigma=1.0):
    print("Computing change score...")
    T, K, C = kps.shape
    df = pd.DataFrame(kps.reshape(T, -1)).interpolate(method='linear', limit_direction='both')
    kps_clean = df.values.reshape(T, K, C)
    kps_smooth = gaussian_filter1d(kps_clean, sigma=sigma, axis=0)
    diff = np.diff(kps_smooth, axis=0)
    total_vel = np.sum(np.linalg.norm(diff, axis=2), axis=1)
    return zscore(np.concatenate(([0], total_vel)))

def get_event_triggered_traces(score, labels, window=45):
    transitions = np.where(labels[1:] != labels[:-1])[0] + 1
    traces = []
    T_score = len(score)
    for t_idx in transitions:
        start, end = t_idx - window, t_idx + window
        if start >= 0 and end < T_score: traces.append(score[start:end])
    if not traces: return None, None, None
    stack = np.stack(traces)
    return np.mean(stack, axis=0), sem(stack, axis=0), stack

def compute_state_durations(labels, fps):
    n = len(labels)
    if n == 0: return np.array([])
    y = np.array(labels)
    mismatch = np.where(y[1:] != y[:-1])[0]
    run_ends = np.append(mismatch, n-1)
    run_starts = np.insert(run_ends[:-1] + 1, 0, 0)
    return (run_ends - run_starts + 1) / fps

def plot_ethogram_and_durations(methods_data, fps_ground_truth, window_min=(5, 7)):
    n_methods = len(methods_data)
    fig = plt.figure(figsize=(12, 2.5 * n_methods))
    gs = fig.add_gridspec(n_methods, 2, width_ratios=[3, 1], hspace=0.5, wspace=0.15)
    
    for i, m in enumerate(methods_data):
        ax_eth = fig.add_subplot(gs[i, 0])
        labels = m['labels_resampled']
        start, end = int(window_min[0]*60*fps_ground_truth), int(window_min[1]*60*fps_ground_truth)
        segment = labels[start:min(len(labels), end)]
        
        if len(segment) > 0:
            unique_l = np.unique(segment)
            n_l = len(unique_l)
            # Select colormap based on method type
            name = m['type'].lower()
            if 'kpms' in name: cmap_name = 'YlOrBr'
            elif 'castle' in name: cmap_name = 'Blues'
            elif 'bsoid' in name: cmap_name = 'Greens'
            else: cmap_name = 'viridis'

            n_colors = max(20, n_l + 1)
            base_cmap = plt.get_cmap(cmap_name)
            colors = base_cmap(np.linspace(0.3, 1.0, n_colors))
            np.random.seed(42); np.random.shuffle(colors)
            my_cmap = ListedColormap(colors)
            
            mapper = {l: idx % n_colors for idx, l in enumerate(unique_l)}
            mapped = np.array([mapper.get(x, 0) for x in segment])
            ax_eth.imshow(mapped.reshape(1, -1), aspect='auto', cmap=my_cmap, interpolation='nearest')
            
        ax_eth.set_title(m['display_name'], loc='left', fontsize=12, fontweight='bold', color=m['color'])
        ax_eth.set_yticks([]); ax_eth.set_xticks([])
        for s in ax_eth.spines.values(): s.set_visible(False)
        
        # Duration Hist
        ax_hist = fig.add_subplot(gs[i, 1])
        # Note: We compute duration on the RESAMPLED labels using ground truth FPS
        # This will be technically correct as resampling scales both frames and total time.
        durs = compute_state_durations(labels, fps_ground_truth)
        ax_hist.hist(durs, bins=np.linspace(0, 2, 41), color=m['color'], alpha=0.7, density=True, edgecolor='white')
        ax_hist.set_xlim(0, 2); ax_hist.spines['top'].set_visible(False); ax_hist.spines['right'].set_visible(False)
        ax_hist.set_yticks([])
        if i == 0: ax_hist.set_title("State Durations", fontsize=10)
        if i == n_methods - 1: ax_hist.set_xlabel("Duration (s)")

    plt.savefig("Benchmark/benchmark_fig3_AB_style.png", dpi=300, bbox_inches='tight')
    print("Saved Fig 3 style plot.")

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KEYPOINT_DIR = os.path.join(BASE_DIR, "keypoint_data")
    LABEL_DIR = os.path.join(BASE_DIR, "lable_data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M")
    output_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Ground Truth (Keypoints)
    kps_file = [f for f in os.listdir(KEYPOINT_DIR) if f.endswith('.h5')][0]
    kps, bodyparts = load_dlc_keypoints(os.path.join(KEYPOINT_DIR, kps_file))
    total_frames = len(kps)
    fps_ground_truth = 30.0 # Verified via ffprobe previously
    change_score = compute_change_score(egocentric_alignment(kps, bodyparts))
    
    # 2. Collect Methods and Resample Labels
    methods = []
    for f in sorted(os.listdir(LABEL_DIR)):
        path = os.path.join(LABEL_DIR, f)
        version_match = re.search(r'\((.*?)\)', f)
        version = version_match.group(1) if version_match else None
        
        if 'KPMS' in f and f.endswith('.h5'):
            methods.append({'type': 'KPMS', 'name': 'KPMS', 'display_name': 'KPMS', 'path': path, 'color': '#E0C068'})
        elif 'CASTLE' in f and (f.endswith('.npy') or f.endswith('.csv')):
            if not version:
                # Try to extract key info like 'lowpass...' from filenames without parents
                match = re.search(r'30fps_(.*?)\.csv', f)
                version = match.group(1) if match else None
            disp_name = f"CASTLE ({version})" if version else "CASTLE"
            methods.append({'type': 'CASTLE', 'name': f'CASTLE_{version}' if version else 'CASTLE', 'display_name': disp_name, 'path': path, 'color': None})
        elif 'B-soid' in f and f.endswith('.csv'):
            win_match = re.search(r'_(\d+ms)\.csv$', f)
            win = win_match.group(1) if win_match else version
            disp_name = f"B-SOiD ({win})" if win else "B-SOiD"
            methods.append({'type': 'BSOID', 'name': f'B-SOiD_{win}' if win else 'B-SOiD', 'display_name': disp_name, 'path': path, 'color': None})
            
    # Color Assignment
    castle_methods = [m for m in methods if m['type'] == 'CASTLE']
    # Use a high-contrast qualitative-like scale for CASTLE
    blue_shades = [
        '#08306B', # Deep Blue
        '#2171B5', # Steel Blue
        '#6BAED6', # Sky Blue
        '#00CED1', # Dark Cyan/Turquoise
        '#40E0D0', # Turquoise
        '#7FFFD4'  # Aquamarine
    ]
    for i, m in enumerate(castle_methods): m['color'] = blue_shades[i % len(blue_shades)]
    
    bsoid_methods = [m for m in methods if m['type'] == 'BSOID']
    # High contrast greens
    green_shades = [
        '#00441B', # Darkest Green
        '#238B45', # Medium Green
        '#74C476', # Light Green
        '#CCFF00', # Electric Lime
        '#ADFF2F'  # Green Yellow
    ]
    for i, m in enumerate(bsoid_methods): m['color'] = green_shades[i % len(green_shades)]
        
    # Load, Resample and Analyze
    valid_methods = []
    for m in methods:
        try:
            if m['type'] == 'KPMS': raw_labels = load_kpms_labels(m['path'])
            elif m['type'] == 'CASTLE': raw_labels = load_castle_labels(m['path'])
            else: raw_labels = load_bsoid_labels(m['path'])
            
            # RESAMPLE to match ground truth length
            m['labels_resampled'] = resample_labels(raw_labels, total_frames)
            
            # Analyze
            m['mean'], m['sem'], m['stack'] = get_event_triggered_traces(change_score, m['labels_resampled'], 45)
            if m['mean'] is not None: valid_methods.append(m)
        except Exception as e:
            print(f"Error processing {m['name']}: {e}")

    # 3. Plot 1: Combined Change Score
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    x_sec = np.arange(-45, 45) / fps_ground_truth
    violin_data, violin_labels, violin_colors = [], [], []
    
    for m in valid_methods:
        ax1.plot(x_sec, m['mean'], label=m['display_name'], color=m['color'], linewidth=2)
        ax1.fill_between(x_sec, m['mean']-m['sem'], m['mean']+m['sem'], color=m['color'], alpha=0.2)
        violin_data.append(m['stack'][:, 45])
        violin_labels.append(m['display_name'])
        violin_colors.append(m['color'])
        
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time from transition (s)'); ax1.set_ylabel('Change score (z)')
    ax1.legend(loc='upper right', fontsize=9); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    
    parts = ax2.violinplot(violin_data, showmeans=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[i]); pc.set_alpha(0.6)
    parts['cmeans'].set_edgecolor('black')
    ax2.set_xticks(np.arange(1, len(violin_labels)+1))
    ax2.set_xticklabels(violin_labels, rotation=45); ax2.set_ylabel('Change score (z)')
    ax2.set_ylim(-2, 2)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_change_score_combined.png"), dpi=300)
    print(f"Saved Change Score plot to {output_dir}")
    
    # 4. Plot 2: Ethogram & Durations
    plot_ethogram_and_durations(valid_methods, fps_ground_truth, output_dir)

    # 5. Output Statistics
    stats_data = []
    summary_md = ["# Benchmark Run Summary\n", f"**Timestamp:** {timestamp}\n", f"**Ground Truth FPS:** {fps_ground_truth}\n", "| Method | Classes | Transitions | Avg Duration (ms) |", "| :--- | :---: | :---: | :---: |"]
    
    for m in valid_methods:
        durs = compute_state_durations(m['labels_resampled'], fps_ground_truth)
        n_classes = len(np.unique(m['labels_resampled']))
        n_trans = len(np.where(m['labels_resampled'][1:] != m['labels_resampled'][:-1])[0])
        avg_dur = np.mean(durs) * 1000 if len(durs) > 0 else 0
        
        stats_data.append({
            'Method': m['display_name'],
            'Classes': n_classes,
            'Transitions': n_trans,
            'AvgDuration_ms': round(avg_dur, 2)
        })
        summary_md.append(f"| {m['display_name']} | {n_classes} | {n_trans} | {avg_dur:.2f} |")

    # Save CSV
    pd.DataFrame(stats_data).to_csv(os.path.join(output_dir, "statistics.csv"), index=False)
    # Save MS
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("\n".join(summary_md))
    
    print(f"Results reorganized in {output_dir}")

def plot_ethogram_and_durations(methods_data, fps_ground_truth, output_dir, window_min=(5, 7)):
    n_methods = len(methods_data)
    fig = plt.figure(figsize=(12, 2.5 * n_methods))
    gs = fig.add_gridspec(n_methods, 2, width_ratios=[3, 1], hspace=0.5, wspace=0.15)
    
    for i, m in enumerate(methods_data):
        ax_eth = fig.add_subplot(gs[i, 0])
        labels = m['labels_resampled']
        start, end = int(window_min[0]*60*fps_ground_truth), int(window_min[1]*60*fps_ground_truth)
        segment = labels[start:min(len(labels), end)]
        
        if len(segment) > 0:
            unique_l = np.unique(segment)
            n_l = len(unique_l)
            name = m['type'].lower()
            if 'kpms' in name: cmap_name = 'YlOrBr'
            elif 'castle' in name: cmap_name = 'Blues'
            elif 'bsoid' in name: cmap_name = 'Greens'
            else: cmap_name = 'viridis'

            n_colors = max(20, n_l + 1)
            base_cmap = plt.get_cmap(cmap_name)
            colors = base_cmap(np.linspace(0.3, 1.0, n_colors))
            np.random.seed(42); np.random.shuffle(colors)
            my_cmap = ListedColormap(colors)
            
            mapper = {l: idx % n_colors for idx, l in enumerate(unique_l)}
            mapped = np.array([mapper.get(x, 0) for x in segment])
            ax_eth.imshow(mapped.reshape(1, -1), aspect='auto', cmap=my_cmap, interpolation='nearest')
            
        ax_eth.set_title(m['display_name'], loc='left', fontsize=12, fontweight='bold', color=m['color'])
        ax_eth.set_yticks([]); ax_eth.set_xticks([])
        for s in ax_eth.spines.values(): s.set_visible(False)
        
        ax_hist = fig.add_subplot(gs[i, 1])
        durs = compute_state_durations(labels, fps_ground_truth)
        ax_hist.hist(durs, bins=np.linspace(0, 2, 41), color=m['color'], alpha=0.7, density=True, edgecolor='white')
        ax_hist.set_xlim(0, 2); ax_hist.spines['top'].set_visible(False); ax_hist.spines['right'].set_visible(False)
        ax_hist.set_yticks([])
        if i == 0: ax_hist.set_title("State Durations", fontsize=10)
        if i == n_methods - 1: ax_hist.set_xlabel("Duration (s)")

    plt.savefig(os.path.join(output_dir, "benchmark_fig3_AB_style.png"), dpi=300, bbox_inches='tight')
    print("Saved Fig 3 style plot.")

if __name__ == "__main__":
    from datetime import datetime
    main()
