# Core Module Documentation

The `kpms_custom.core` module serves as the entry point and orchestration layer for the pipeline. It handles CLI argument parsing and high-level execution routines.

## `kpms_custom.core.cli`

Handles command-line interface arguments and dispatches commands to the runner.

### `main()`
Entry point script installed as `kpms` command.
- **Usage**: `kpms <command> [options]`
- **Commands**:
    - `train`: Train models.
    - `scan`: Run hyperparameter scan (AR or Full).
    - `evaluate`: Compare models via EML.
    - `analyze`: Run visualization and statistics.
    - `merge`: Run motif merging analysis.

## `kpms_custom.core.runner`

Contains the high-level logic for each CLI command. It bridges the configuration, data loading, and model execution.

### `run_training(config_path, restarts=None)`
Executes the training pipeline.
- **Parameters**:
    - `config_path` (str): Path to the YAML configuration file.
    - `restarts` (int, optional): Override the number of training restarts defined in config.
- **Workflow**:
    1. Loads config.
    2. Calls `_load_and_prep` to get data and PCA.
    3. Loops `n` times calling `kpms_custom.model.trainer.fit_model`.

### `run_scan(config_path, scan_type='ar')`
Runs a hyperparameter scan to find optimal Kappa values.
- **Parameters**:
    - `config_path` (str): Path to config.
    - `scan_type` (str): `'ar'` (autoregressive) or `'full'` (full model). Defaults to `'ar'`.
- **Workflow**:
    1. Loads data.
    2. Calls `kpms_custom.model.tuning.scan_kappa`.
    3. Updates `config.yaml` with the best found Kappa.

### `run_evaluation(config_path)`
Compares multiple trained models in the project directory using Expected Marginal Likelihood (EML).
- **Parameters**:
    - `config_path` (str): Path to config.
- **Output**: Prints recommended model name and saves comparison plot.

### `run_analysis(config_path, model_name=None, results_path=None, output_dir=None)`
Runs the visualization and analysis suite on a trained model.
- **Parameters**:
    - `config_path` (str): Path to config.
    - `model_name` (str, optional): Name of the model folder (e.g., `20230101-1200`).
    - `results_path` (str, optional): Direct path to `results.h5`.
    - `output_dir` (str, optional): Custom output directory for figures.
- **Features**:
    - Ethograms (`viz.plot_ethograms`)
    - Syllable Distributions (`viz.plot_syllable_distribution`)
    - Trajectory Plots (`viz.plot_trajectories`)
    - Grid Movies (`viz.generate_grid_movie`)
    - Dendrograms (`viz.plot_dendrogram`)
    - Transition Graphs (`viz.plot_transition_graph`)
    - 3D Latent Scatter (`viz.generate_3d_scatter`)
    - Labeled Videos (`viz.generate_labeled_video`)

### `run_merging(config_path, model_name)`
Executes the Motif Merging strategy.
- **Parameters**:
    - `config_path` (str): Path to config.
    - `model_name` (str): Name of the model to merge.
- **Workflow**:
    1. Loads model checkpoint.
    2. Identifies short vs. stable motifs based on `merge_threshold`.
    3. Maps short motifs to nearest stable centroids.
    4. Applies merging and regenerates basic plots.
