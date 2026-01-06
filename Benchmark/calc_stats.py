
import h5py
import numpy as np
import pandas as pd
import os
import re

def resample_labels(labels, target_len):
    if len(labels) == target_len:
        return labels
    indices = np.linspace(0, len(labels) - 1, target_len)
    return labels[np.round(indices).astype(int)]

def compute_stats(labels, fps, name):
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    n = len(labels)
    if n == 0:
        mean_dur = 0
        num_transitions = 0
    else:
        y = np.array(labels)
        mismatch = np.where(y[1:] != y[:-1])[0]
        run_ends = np.append(mismatch, n-1)
        run_starts = np.insert(run_ends[:-1] + 1, 0, 0)
        run_lengths = run_ends - run_starts + 1
        durations = run_lengths / fps
        mean_dur = np.mean(durations) * 1000 # ms
        num_transitions = len(mismatch)
        
    print(f"--- {name} ---")
    print(f"Number of Classes: {n_classes}")
    print(f"Number of Transitions: {num_transitions}")
    print(f"Average Duration: {mean_dur:.2f} ms")
    print(f"Total Frames: {n}")
    print(f"FPS used: {fps}")
    print("-" * 20)

def main():
    LABEL_DIR = "/home/isonaei/ABVFM_benchmark/Benchmark/lable_data"
    KPS_DIR = "/home/isonaei/ABVFM_benchmark/Benchmark/keypoint_data"
    
    # Get ground truth length
    kps_file = [f for f in os.listdir(KPS_DIR) if f.endswith('.h5')][0]
    with h5py.File(os.path.join(KPS_DIR, kps_file), 'r') as f:
        # Check structure for length
        # Typically DLC H5 has 'df_with_missing' or similar, but simpler to just use pandas or h5py keys
        pass
    
    # Use pandas to get length reliably as in benchmark script
    kps_df = pd.read_hdf(os.path.join(KPS_DIR, kps_file))
    total_frames = len(kps_df)
    fps_ground_truth = 30.0
    
    print(f"Ground Truth Frames: {total_frames}")
    print(f"Ground Truth FPS: {fps_ground_truth}")
    print("=" * 30)

    for f in sorted(os.listdir(LABEL_DIR)):
        path = os.path.join(LABEL_DIR, f)
        
        version_match = re.search(r'\((.*?)\)', f)
        version = version_match.group(1) if version_match else f
        
        try:
            if 'KPMS' in f and f.endswith('.h5'):
                with h5py.File(path, 'r') as hf:
                    dataset = list(hf.keys())[0]
                    raw = hf[dataset]['syllable'][:] if 'syllable' in hf[dataset] else hf[dataset]['latent_state'][:]
                name = "KPMS"
            elif 'CASTLE' in f and (f.endswith('.npy') or f.endswith('.csv')):
                if f.endswith('.npy'):
                    raw = np.load(path)
                else:
                    df = pd.read_csv(path)
                    raw = df['behavior_label'].values if 'behavior_label' in df.columns else df['behavior'].values
                    if raw.dtype == object: raw, _ = pd.factorize(raw)
                name = f"CASTLE ({version})"
            elif 'B-soid' in f and f.endswith('.csv'):
                df = pd.read_csv(path)
                raw = df['B-SOiD_Label'].values
                name = f"B-SOiD ({version})"
            else:
                continue
                
            # Resample
            resampled = resample_labels(raw, total_frames)
            compute_stats(resampled, fps_ground_truth, name)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    main()
