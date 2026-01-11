# ABVFM Benchmark (Behavior Performance Evaluation Suite)

This sub-project provides a standardized framework for evaluating and comparing various automated behavioral analysis methods (e.g., **CASTLE**, **KPMS**, **B-SOiD**). It focuses on statistical analysis, physics-based feature extraction, and report generation.

## 🚀 Quick Start

Ensure you have the `castle` virtual environment activated:

```bash
# From project root
source venvs/castle/bin/activate

# Execute the benchmark
python Benchmark/run_benchmark.py --config Benchmark/config.yaml
```

> [!NOTE]
> **GPU Requirement removed**: As of the recent refactor, this sub-project no longer requires a GPU or `torch`. GPU-intensive optical flow is handled by the standalone [OpticalFlow](../OpticalFlow) sub-project.

## 📊 Key Features

- **Multi-Source Data Support**: Standardized loaders for KPMS (.h5), B-SOiD (.csv), and CASTLE (.csv) outputs.
- **Visual-Physical Alignment**: Compares behavior transitions against high-dimensional physics metrics (Velocity, Jerk, Compactness) and pre-computed visual flow.
- **Residual Motion Analysis**: Quantifies "Micro-movements" by calculating the discrepancy between **Pre-computed Optical Flow** and Skeletal Velocity (from keypoints).
- **State Stability Index (SSI)**: A novel metric for quantifying the boundary clarity of behavioral states.
- **Comprehensive Reporting**: Automatically generates an interactive HTML report with summary statistics and comparative visualizations.

## 📂 Project Structure

- `run_benchmark.py`: Main orchestration script.
- `config.yaml`: Central configuration for paths and parameters.
- `src/`: Core implementation modules.
  - `loader.py`: Universal data loading for various formats.
  - `physics.py`: Feature extraction (Kinematics, Morphology, Orientation).
  - `metrics.py`: Statistical evaluation (SSI, NMI, Mann-Whitney U).
  - `visualizer.py`: Advanced plotting logic.
  - `report_generator.py`: HTML report compilation.

## 📝 Workflow

1.  **Compute Optical Flow**: Use the `OpticalFlow` sub-project to generate `.npy` results.
2.  **Configure Paths**: Update `Benchmark/config.yaml` with the paths to your results (KPMS, B-SOID, Optical Flow).
3.  **Run Benchmark**: Execute `run_benchmark.py` to generate the evaluation report.
