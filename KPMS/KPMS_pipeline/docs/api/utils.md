# Utils Module Documentation

Helper utilities for configuration and logging.

## `kpms_custom.utils.config`

### `load_config(config_path)`
Loads a YAML file into a Python dict.

### `validate_config(config)`
Checks for existence of critical keys (`project_dir`, `video_dir`, `data_dir`, `preprocess`). Raises `ValueError` if missing.

### `save_config(config, config_path)`
Dumps a dictionary to a YAML file (default flow style=False).

## `kpms_custom.utils.logging`

### `setup_logger(log_file='pipeline.log', level=logging.INFO)`
Configures the global logger with file and console handlers.
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

### `get_logger(name='kpms_custom')`
Returns the standardized logger instance.
