# Data Module Documentation

The `kpms_custom.data` module handles loading, preprocessing, and formatting data from DeepLabCut (h5) files for Keypoint-MoSeq.

## `kpms_custom.data.loader`

Responsible for I/O operations and raw data parsing.

### `load_h5_files(config)`
Finds all `.h5` files in the configured `data_dir`.
- **Returns**: List of file paths.

### `detect_bodyparts(file_path, conf_threshold=0.5)`
Automatically detects bodypart names from a DLC h5 file.
- **Parameters**:
    - `file_path` (str): Path to h5 file.
- **Logic**: Checks for MultiIndex columns or flat columns containing 'nose'/'snout' patterns.

### `detect_fps(video_dir, extension='mp4')`
Auto-detects frame rate from the first video found in `video_dir`.
- **Returns**: FPS (float), defaults to 30.0 if failed.

### `parse_dlc_data(files, config)`
Parses a list of H5 files into coordinate and confidence dictionaries.
- **Parameters**:
    - `files` (list): List of file paths.
    - `config` (dict): Config dict containing `project_config` (for bodypart overrides).
- **Returns**: `(coordinates, confidences, bodyparts)`
    - `coordinates`: Dictionary `{filename: (T, K, 2)}`
    - `confidences`: Dictionary `{filename: (T, K)}`

### `filter_bad_bodyparts(coordinates, bodyparts, threshold=0.9)`
Removes bodyparts that are missing (NaN) in more than `threshold` fraction of frames.
- **Background**: High-dropout bodyparts can ruin PCA.

### `interpolate_data(coordinates, noise_scale=1.0)`
Fills missing data via linear interpolation (`limit_direction='both'`) and adds slight noise to prevent singular matrices in PCA.

## `kpms_custom.data.preprocessor`

Prepares data for the Keypoint-MoSeq model (PCA, formatting).

### `prepare_for_kpms(coordinates, confidences, config, bodyparts)`
Full preprocessing pipeline.
1. Filters bad bodyparts.
2. Interpolates missing data.
3. Formats data using `kpms.format_data`.
4. Casts all floating point data to `float64` (Required for JAX).

### `train_pca(data, config, project_dir, bodyparts=None)`
Fits a PCA model on the keypoint data.
- **Features**:
    - Support for **Auto-Heading Inference**: Automatically determines `anterior_idxs` and `posterior_idxs` from bodypart names if not specified in config.
- **Returns**: Fitted PCA object.

### `calculate_latent_dim(pca, target_variance=0.9)`
Determines the optimal number of PCA components.
- **Parameters**:
    - `target_variance` (float or int): If < 1, represents % variance explained. If > 1, represents explicit component count.
