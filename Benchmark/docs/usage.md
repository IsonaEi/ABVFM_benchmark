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
  - **File**: `*.npy` (e.g., `CASTLE_ctrl_syll_align.npy`)
  - **Structure**: Numpy array of integer labels matching the total number of frames (or a consistent fraction).

- **B-SOiD**
  - **File**: `*.csv` (e.g., `B-soid_300ms.csv`)
  - **Structure**: CSV file with at least two columns: `Time` and `B-SOiD_Label`.
  - **Note**: The script automatically calculates the FPS from the `Time` column.

## Running the Benchmark

1. **Activate the Environment**
   Ensure you have the necessary dependencies installed (pandas, numpy, matplotlib, h5py, scipy).

   ```bash
   # Example (adjust to your specific venv)
   source ../venvs/b-soid-official/bin/activate 
   ```

2. **Execute the Script**
   Run the python script from the `Benchmark` directory or the parent directory.

   ```bash
   python Benchmark/benchmark_change_score.py
   ```

## Output

The script generates the following visualizations in the `Benchmark` directory:

1.  **`benchmark_change_score_combined.png`**
    -   **Left**: Event-Triggered Average of Change Scores. Shows how the velocity (change score) behaves around the transition point of behavioral syllables for each method.
    -   **Right**: Violin plots showing the distribution of change scores at the transition point.

2.  **`benchmark_fig3_AB_style.png`**
    -   **Left**: Ethogram (Barcode plot) showing the sequence of behaviors over a specific time window.
    -   **Right**: Histogram of state durations for each method.

## troubleshooting

-   **"Error: PyTables not found"**: Install pytables via `pip install tables`.
-   **"Error finding files"**: Ensure you have exactly one relevant file for each method in the `lable_data` folder as described above.
