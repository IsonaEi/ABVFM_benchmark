# Analysis Module Documentation

The `kpms_custom.analysis` module provides tools for visualization, evaluation, and post-processing (motif merging).

## `kpms_custom.analysis.viz`

Handles all figure and video generation. Global matplotlib settings are enforced here.

### `plot_ethograms(results, output_dir, config, cmap='tab20', filename='ethogram.png')`
Generates a raster plot of syllable usage over time for each session.
- **Output**: `ethogram.png`
- **Wraps**: Custom implementation using `matplotlib.imshow`.

### `plot_syllable_distribution(results, output_dir, config)`
Plots a ranked frequency bar chart of syllables.
- **Output**: `distribution.png`

### `plot_trajectories(...)`
Generates trajectory plots (GIFs) for each syllable.
- **Wraps**: `kpms.generate_trajectory_plots`
- **Key Params**: `sampling_options={'n_neighbors': 1}` to force inclusion of rare syllables.

### `generate_grid_movie(...)`
Creates grid movies showing raw video examples of each syllable.
- **Wraps**: `kpms.generate_grid_movies`

### `generate_3d_scatter(results, output_dir, config)`
Visualizes the latent space.
1. **Static**: `3d_scatter.png` (Matplotlib)
2. **Interactive**: `interactive_3d_scatter.html` (Plotly) - Uses `px.scatter_3d` with `Alphabet` color palette.

### `generate_labeled_video(results, output_dir, video_dir, ...)`
Overlays syllable IDs on raw videos.
- **Output**: `labeled_videos/{video_name}_labeled.mp4`

## `kpms_custom.analysis.evaluation`

### `evaluate_models(project_dir, model_name_pattern=None)`
Computes Expected Marginal Likelihood (EML) for model comparison.
- **Wraps**: `kpms.expected_marginal_likelihoods`
- **Output**: `model_comparison_eml.png`

## `kpms_custom.analysis.merging`

Implements the Motif Merging strategy to consolidate redundant states.

### `class MotifMerger`
- **`__init__(model, threshold_frames=10)`**:
    - `threshold_frames`: Minimum duration to consider a motif "stable".
- **`calculate_centroids()`**: Computes mean latent vector for each syllable.
- **`identify_motif_types()`**: Classifies motifs as **Stable** (duration >= threshold) or **Short** (duration < threshold).
- **`suggest_merges()`**: Maps every Short motif to the nearest Stable motif (Euclidean distance in latent space).
- **`apply_merges(results, ...)`**:
    - Generates a mapping via `kpms.generate_syllable_mapping`.
    - Applies mapping via `kpms.apply_syllable_mapping`.
    - Saves strategy to `motif_merging_strategy.md`.
