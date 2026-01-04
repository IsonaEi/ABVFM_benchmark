# Model Module Documentation

The `kpms_custom.model` module wraps the core training and tuning logic of Keypoint-MoSeq.

## `kpms_custom.model.trainer`

### `setup_project(config)`
Initializes the KPMS project directory structure if it doesn't exist.
- **Wraps**: `kpms.setup_project`.

### `fit_model(data, metadata, pca, config, project_dir, name_suffix="", ar_only=False, num_iters_override=None)`
Main training routine supporting Two-Stage Training (AR Warmup -> Full Fit).
- **Parameters**:
    - `data`, `metadata`, `pca`: Preprocessed inputs.
    - `config`: Pipeline config.
    - `name_suffix` (str): Suffix for the model directory (e.g., run index).
    - `ar_only` (bool): If True, stops after AR training (used for scans).
    - `num_iters_override` (int): Override config iteration count.
- **Logic**:
    1. **Init**: Calls `kpms.init_model` with config parameters and `float64` data.
    2. **Stage 1 (Warmup)**: Trains with `ar_only=True` for `ar_warmup_iters`. Applies `ar_kappa`.
    3. **Stage 2 (Full)**: transitions to full model training for `num_iters`. Applies `full_kappa`.
- **Returns**: `(model, model_name)`

## `kpms_custom.model.tuning`

### `scan_kappa(data, metadata, pca, config, project_dir, scan_type='ar')`
Performs a bisection search to find the Kappa hyperparameter that yields a target median syllable duration.
- **Parameters**:
    - `scan_type`: `'ar'` (for AR-HMM) or `'full'` (for full model).
- **Process**:
    1. **Bounds**: Reads min/max from config.
    2. **Objective**: Trains a temporary model and measures `get_median_duration`.
    3. **Search**: Adjusts Kappa via bisection until duration is within `tolerance` of `target_motif_duration`.
- **Returns**: `(best_kappa, best_duration)`

### `get_median_duration(model, fps)`
Calculates the median duration (in ms) of syllables in the current model state.
