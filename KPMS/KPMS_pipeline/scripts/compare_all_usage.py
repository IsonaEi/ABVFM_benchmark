
import h5py
import numpy as np
import os
from pathlib import Path

def get_usage(path):
    if not os.path.exists(path): return None
    try:
        with h5py.File(path, 'r') as f:
            all_labels = []
            for session in f:
                if 'syllable' in f[session]:
                    all_labels.extend(f[session]['syllable'][()])
            if not all_labels: return None
            all_labels = np.array(all_labels)
            u, c = np.unique(all_labels, return_counts=True)
            usage = np.sort(c / len(all_labels))[::-1]
            return usage
    except:
        return None

import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    
    base_dir = Path(args.dir)

    print(f"{'Experiment':<60} {'Top 1 (%)':<10} {'Top 2 (%)':<10} {'Top 3 (%)':<10}")
    print("-" * 95)

    # Use glob to find all unmerged results
    result_files = sorted(glob.glob(str(base_dir / "20260111-*-exp*" / "results.h5")))
    # Filter out merged ones for this specific check
    unmerged_files = [f for f in result_files if "_merged" not in f]

    for f_path in unmerged_files:
        folder_name = Path(f_path).parent.name
        usage = get_usage(f_path)
        if usage is not None:
            t1 = usage[0]*100
            t2 = usage[1]*100 if len(usage)>1 else 0
            t3 = usage[2]*100 if len(usage)>2 else 0
            print(f"{folder_name:<60} {t1:<10.2f} {t2:<10.2f} {t3:<10.2f}")

if __name__ == "__main__":
    main()
