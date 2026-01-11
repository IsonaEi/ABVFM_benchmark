# B-SOiD Pipeline (ABVFM Custom Implementation)

This subproject implements an automated pipeline for B-SOiD (Behavioral segmentation of Open-field in DeepLabCut), integrated into the ABVFM benchmark.

## Features
- **Automated Pipeline**: End-to-end processing from DLC .h5 files to ethograms.
- **Parameter Optimization**: Grid search for Window Size and UMAP parameters (`n_neighbors`, `min_dist`).
- **Optimization Strategy**: Optimizes for bout duration stability and reasonable cluster counts.
- **Visualizations**: Auto-generated Ethograms and parameter summary plots.

## Environment Setup
Recommended environment: `b-soid-official` (requires legacy B-SOiD dependencies).

```bash
source ../venvs/b-soid-official/bin/activate
pip install -r requirements.txt
```

**Note**: You must have the `bsoid_umap` package accessible or configured in `config.yaml` under `bsoid_source_dir`.

### ⚠️ Important Installation Steps (Workaround)
Since the official B-SOiD repository has strict dependency pinning and uses deprecated functions:
1.  **Clone the Source Manually**:
    ```bash
    git clone https://github.com/YttriLab/B-SOID.git <path_to_venvs>/b-soid-official/B-SOID_source
    ```
2.  **Patch Source Code**:
    The current codebase of B-SOiD uses `sklearn.metrics.plot_confusion_matrix` which was removed in scikit-learn 1.2+.
    **Fix**: Replace all instances of `plot_confusion_matrix(...)` with `ConfusionMatrixDisplay.from_estimator(...)` in:
    - `bsoid_app/bsoid_utilities/visuals.py`
    - `bsoid_umap/train.py`
    - `bsoid_py/train.py`
    Also update imports to `from sklearn.metrics import ConfusionMatrixDisplay`.

3.  **Install Configured Dependencies**:
    Use the provided `requirements.txt` in this folder (looser versioning) instead of the one in the B-SOiD repo.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Configuration
Modify `automated_pipeline/config.yaml` to set your input/output directories and parameters.

### 2. Run Pipeline
```bash
python automated_pipeline/pipeline.py
```

## Structure
- `automated_pipeline/`: Core scripts.
- `input_data_full/`: Directory for input DLC .h5 files.
- `results/`: Output directory for ethograms, CSV labels, and stats.
