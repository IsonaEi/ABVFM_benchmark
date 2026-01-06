# ABVFM Benchmark Suite Usage Guide

This guide explains how to use the benchmarking tools to compare different behavioral analysis methods (KPMS, CASTLE, B-SOiD).

## Directory Structure

Ensure your `Benchmark` directory is organized as follows:

```
Benchmark/
├── benchmark_change_score.py   # Main analysis script
├── keypoint_data/              # Input: DLC Keypoint files (.h5)
│   └── *.h5
├── lable_data/                 # Input: Method result labels
│   ├── KPMS_results.h5         # Keypoint-MoSeq output
│   ├── CASTLE_*.npy            # CASTLE output (numpy array)
│   └── B-soid_*.csv            # B-SOiD output (Time, Label CSV)
├── results/                    # Output: Timestamped result folders
│   └── run_YYYYMMDD_HHMM/      # Each run creates a new folder
│       ├── benchmark_change_score_combined.png
│       ├── benchmark_fig3_AB_style.png
│       ├── statistics.csv
│       └── summary.md
└── docs/
    └── usage.md                # This file
```

## Input File Formats

### 1. Keypoint Data (Ground Truth)
- **Format**: DeepLabCut (DLC) H5 file.
- **Location**: `keypoint_data/` directory.
- **Content**: Must contain tracked body parts (`snout`, `tail_base`, etc.) for calculating change scores.

### 2. Behavioral Labels
All label files should be placed in `lable_data/`.

- **Keypoint-MoSeq (KPMS)**
  - **File**: `KPMS_results.h5`
  - **Structure**: H5 file containing a `syllable` or `latent_state` dataset.

- **CASTLE**
  - **File**: `*.npy` or `*.csv`
  - **Structure**: Numpy array or CSV with `behavior_label` or `behavior` column.
  - **Note**: Versioning info in parentheses `(MX)` or `(Raiso)` will be used as labels.

- **B-SOiD**
  - **File**: `*.csv` (e.g., `B-soid_run_..._200ms.csv`)
  - **Structure**: CSV file with at least two columns: `Time` and `B-SOiD_Label`.
  - **Note**: Windows size or version in parentheses will be used as labels.

## Running the Benchmark

1. **Activate the Environment**
   Ensure you have the necessary dependencies installed (pandas, numpy, matplotlib, h5py, scipy).

   ```bash
   # From ABVFM_benchmark directory
   /home/isonaei/ABVFM_benchmark/venvs/keypoint-moseq/bin/python Benchmark/benchmark_change_score.py
   ```

2. **Execute the Script**
   Run the python script from the `Benchmark` directory or the parent directory.

   ```bash
   python Benchmark/benchmark_change_score.py
   ```

## Output

Each run creates a new directory in `Benchmark/results/run_YYYYMMDD_HHMM/` containing:

1.  **`benchmark_change_score_combined.png`**
    -   **Left**: Event-Triggered Average of Change Scores. Shows velocity behavior around transitions.
    -   **Right**: Violin plots of change score distributions at transition (Y-axis fixed to [-2, 2]).

2.  **`benchmark_fig3_AB_style.png`**
    -   **Left**: Ethogram showing behavioral sequences (5-7 minute window).
    -   **Right**: Histogram of state durations for each method.

3.  **`statistics.csv`**
    -   Machine-readable table containing counts of classes, transitions, and average durations.

4.  **`summary.md`**
    -   A Markdown summary of the run results, mirroring the table in `statistics.csv`.

## troubleshooting

-   **"Error: PyTables not found"**: Install pytables via `pip install tables`.
-   **"Error finding files"**: Ensure you have exactly one relevant file for each method in the `lable_data` folder as described above.
