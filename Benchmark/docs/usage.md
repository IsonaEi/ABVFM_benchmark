# ABVFM Benchmark Suite Usage Guide

This guide explains how to use the benchmarking tools to compare different behavioral analysis methods (KPMS, CASTLE, B-SOiD).

## Directory Structure

Ensure your `Benchmark` directory is organized as follows:

```
Benchmark/
├── scripts/
│   └── run_benchmark.py        # Main orchestration script
├── config.yaml                 # Central configuration
├── keypoint_data/              # Input: DLC Keypoint files (.h5)
│   └── *.h5
├── lable_data/                 # Input: Method result labels
│   ├── KPMS.h5                 # Keypoint-MoSeq output
│   ├── CASTLE.csv              # CASTLE output
│   └── B-soid.csv              # B-SOiD output
├── results/                    # Output: Timestamped result folders
│   └── run_YYYYMMDD_HHMM/      # Each run creates a new folder
│       ├── benchmark_report.html
│       ├── statistics.csv
│       ├── trace_feature_ResidualMotion.png
│       └── ...                 # Other plots
└── docs/
    └── usage.md                # This file
```

## Input File Formats

### 1. Keypoint Data (Ground Truth)
- **Format**: DeepLabCut (DLC) H5 file.
- **Location**: Configured in `config.yaml`.

### 2. Behavioral Labels
All label files are configured in the `methods` section of `config.yaml`.

- **Keypoint-MoSeq (KPMS)**: Standard .h5 export.
- **CASTLE**: Standard .csv export.
- **B-SOiD**: Standard .csv export.

## Running the Benchmark

   **Active the Environment**
   Ensure you use the `benchmark` environment (Python 3.10 with GMFlow support).

   ```bash
   # From ABVFM_benchmark directory
   venvs/benchmark/bin/python Benchmark/scripts/run_benchmark.py --config Benchmark/config.yaml [--skip-gpu]
   ```
   
   *   `--skip-gpu`: (Optional) Skip GPU-intensive Optical Flow calculation if you don't have a GPU or want to use pre-computed flow only.

## Output

Each run creates a new directory in `Benchmark/results/run_YYYYMMDD_HHMM/` containing:

### 1. Reports & Data
*   **`benchmark_report.html`**: Comprehensive HTML report.
*   **`statistics.csv`**: Summary metrics per method (Mean SSI, Median SSI, Total Duration, etc.).
*   **`significance_test.csv`**: Mann-Whitney U test results comparing Peak Amplitudes.

### 2. Key Visualizations
*   **`benchmark_fig3_style.png`**: Ethogram & Duration Histogram.
*   **`ssi_comparison.png`**: State Stability Index distribution ([0.01%, 99.9%] range).
*   **`trace_feature_{Metric}.png`**: Event-triggered traces grouped by feature (Velocity, Jerk, Compactness, Optical Flow, etc.).
*   **`trace_feature_ResidualMotion.png`**: High-resolution analysis of micro-movements.
*   **`reconstruction_scores.png`**: Feature Completeness Analysis (DINO vs Keypoints reconstruction R²).
*   **`pca_variance.png`**: Intrinsic Dimensionality Analysis (PCA Cumulative Variance).
*   **`confusion_{M1}_vs_{M2}.png`**: Pairwise comparison matrices (Probability of Method 2 label given Method 1 label).

### 3. Troubleshooting

-   **"Error: PyTables not found"**: Install pytables via `pip install tables`.
-   **"GMFlow is required..."**: Ensure you are running with `venvs/benchmark/bin/python`.
-   **"GMFlow weights not found"**: Download `gmflow_sintel-0c07dcb3.pth` to `Benchmark/third_party/gmflow/pretrained/`.
-   **"Error finding files"**: Ensure you have exactly one relevant file for each method in the `lable_data` folder as described above.
