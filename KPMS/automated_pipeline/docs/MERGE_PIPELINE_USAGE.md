
# Keypoint-MoSeq Motif Merging Pipeline Usage
**Version:** 1.0
**Author:** 玉茗 (Camellia)

This document outlines the workflow for applying the data-driven **Motif Merging Strategy** to Keypoint-MoSeq results. The goal is to consolidate fragmented syllables into stable behavioral motifs based on Latent Space Similarity.

---

## 🚀 Workflow Overview

The pipeline consists of the following sequential steps:

1.  **Suggest Merges** (`suggest_merges.py`)
    *   Inspects existing results.
    *   Identify "Stable" vs "Short" motifs.
    *   Calculates centroids and suggests optimal merges.
    *   **Output:** Prints a merge map (copy this to `apply_merges.py` or strategy doc).
2.  **Apply Merges & Re-run Analysis** (`apply_merges.py`)
    *   Applies the merge mapping to `results.h5`.
    *   Saves new merged results as `results_merged.h5`.
    *   Regenerates basic plots (Trajectory, Transition, Dendrogram).
3.  **Regenerate Visualization** (Optional but Recommended)
    *   **Trajectory Plots:** Use `regenerate_plots.py` (Ensures all syllables are plotted by disabling filters).
    *   **Ethograms/Videos:** Use `generate_missing_viz.py` (Creates ethograms, heatmap, and labeled grid movies).
4.  **Distribution Analysis** (`plot_distribution.py`)
    *   Generates histograms of the new motif distribution.
5.  **Export Data** (`save_merged_csv.py`)
    *   Converts the merged H5 results into user-friendly CSV files.

---

## 🛠️ Step-by-Step Guide

### 1. Suggest Merges
Analyze your model to generate a merging strategy.
```bash
cd KPMS/automated_pipeline
python suggest_merges.py
```
*   **Modify:** Check `suggest_merges.py` to adjust `short_motif_threshold` (default: 10 frames).

### 2. Apply Merges
Apply the strategy. **Note:** You must manually update the `syllables_to_merge` list in this script based on Step 1's output if it changes.
```bash
python apply_merges.py
```
*   **Result:** Creates `results_merged.h5` in your model folder.

### 3. Generate Visualizations (The "Good" Stuff)
The default analysis might filter out some rare but stable motifs. Run these for complete plots.

**A. Trajectory Plots (GIFs/PDFs)**
```bash
python regenerate_plots.py
```
*   *Feature:* Forces plotting of ALL merged syllables, even rarer ones.

**B. Advanced Viz (Ethograms, Heatmaps, Grid Movies)**
```bash
python generate_missing_viz.py
```
*   *Feature:* Creates `ethograms/`, `grid_movies/`, and `transition_matrix_heatmap.png`.

### 4. Analyze Distribution
See which behaviors are most common.
```bash
python plot_distribution.py
```
*   **Output:** `syllable_distribution.png` (Ranked and Ordered).

### 5. Export to CSV
Get the data out for other tools (Excel, Python, MATLAB).
```bash
python save_merged_csv.py
```
*   **Output:** `merged_analysis/csv_results/*.csv`

---

## 📂 Directory Structure

After running the full pipeline, your model directory (`models/YOUR_MODEL_NAME/`) will look like this:

```text
├── results.h5              # Original Raw Results
├── results_merged.h5       # ✅ Merged Results
├── merged_analysis/        # ✅ All New Outputs
│   ├── csv_results/        # Merged CSV files
│   ├── ethograms/          # Behavior over time images
│   ├── figures/            # Histograms, Heatmaps, Dendrogram
│   ├── grid_movies/        # Labeled video clips of motifs
│   └── trajectory_plots/   # Trajectory visualizations
└── motif_merging_strategy.md
```

## ⚠️ Notes
*   **Config:** All scripts rely on `config.yaml` for paths (e.g. `video_dir`). Ensure it is correct.
*   **Environment:** Run within the `keypoint-moseq` environment (`conda activate keypoint-moseq` or equivalent).
