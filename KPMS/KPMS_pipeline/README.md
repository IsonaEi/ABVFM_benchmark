# KPMS Pipeline (Custom Wrapper)

This project is a custom wrapper around the **Keypoint-MoSeq** (KPMS) library, designed to streamline the behavior analysis pipeline. It provides a structured CLI, automated hyperparameter tuning (Kappa Scan), and enhanced visualization tools.

## Architecture Overview

The pipeline is organized into modular components that orchestrate the standard KPMS workflow.

```mermaid
graph TD
    CLI[CLI (kpms command)] --> Runner[Core Runner]
    
    subgraph "KPMS Custom Pipeline"
        Runner --> Data[Data Loader & Preprocessor]
        Runner --> Trainer[Model Trainer]
        Runner --> Analysis[Analysis & Viz]
        
        Data -->|Load .h5| DLC[DeepLabCut Data]
        Data -->|Fit PCA| PCA[PCA Model]
        
        Trainer -->|Init & Fit| KPMS_Lib[Keypoint-MoSeq Library]
        Trainer -->|Kappa Scan| Tuning[Tuning Module]
        
        Analysis -->|Plotting| Viz[Visualization]
        Analysis -->|Merging| Merge[Motif Merger]
    end
    
    Config[config.yaml] --> Runner
```

## Installation

1.  **Environment**: Ensure you are in the correct virtual environment (e.g., `keypoint-moseq`).
2.  **Install**:
    ```bash
    pip install -e .
    ```

## Usage

The pipeline is controlled via the `kpms` command line interface. All commands require a configuration file (defaults to `config/default_config.yaml`).

### 1. Training (`train`)
Trains the model. Supports automatic restarts.
```bash
kpms train --config config/my_config.yaml --restarts 3
```

### 2. Hyperparameter Scan (`scan`)
Finds the optimal `kappa` (stiffness) parameter to match a target syllable duration (defined in config).
```bash
# Scan for AR-HMM Kappa (Stage 1)
kpms scan --type ar

# Scan for Full Model Kappa (Stage 2)
kpms scan --type full
```
*The config file will be automatically updated with the best found values.*

### 3. Evaluation (`evaluate`)
Compares multiple trained models in the project directory using Expected Marginal Likelihood (EML).
```bash
kpms evaluate
```

### 4. Analysis (`analyze`)
Generates all visualizations (Ethograms, Trajectories, Grid Movies, 3D Scatter, Labeled Videos).
```bash
kpms analyze --model 20231025-1200
```

### 5. Motif Merging (`merge`)
Consolidates short/redundant motifs into stable ones based on latent space proximity.
```bash
kpms merge --model 20231025-1200
```

## Configuration

The `config.yaml` file controls all aspects of the pipeline. Key sections:

- **project_dir**: Where results are saved.
- **video_dir / data_dir**: Input paths.
- **tuning**: Target duration for kappa scanning.
- **analysis**: Toggle specific plots on/off.

## Module Documentation

For detailed API references, see the `docs/api` folder:

- [Core (CLI & Runner)](docs/api/core.md)
- [Data (Loader & Preprocessor)](docs/api/data.md)
- [Model (Trainer & Tuning)](docs/api/model.md)
- [Analysis (Viz & Merging)](docs/api/analysis.md)
- [Utils](docs/api/utils.md)
