# ABVFM Benchmark (Behavior Performance Evaluation Suite)

This sub-project provides a standardized framework for evaluating and comparing various automated behavioral analysis methods (e.g., **CASTLE**, **KPMS**, **B-SOiD**).

## 🚀 Quick Start

Ensure you have the `castle` virtual environment activated:

```bash
conda activate castle
# or
source venvs/castle/bin/activate

# Execute the benchmark
python Benchmark/run_benchmark.py --config Benchmark/config.yaml
```

### Dependency Note: GMFlow
This project uses **GMFlow** for optical flow estimation. It is not tracked in this repository and must be cloned manually:

```bash
cd Benchmark/third_party
git clone https://github.com/haofeixu/gmflow.git
```

## 📊 Key Features

- **Multi-Source Data Support**: Standardized loaders for KPMS (.h5), B-SOiD (.csv), and CASTLE (.csv) outputs.
- **Visual-Physical Alignment**: Compares behavior transitions against high-dimensional physics metrics (Velocity, Jerk, Compactness) and visual flow.
- **Residual Motion Analysis**: Quantifies "Micro-movements" by calculating the discrepancy between Optical Flow (real motion) and Skeletal Velocity (modeled motion).
- **State Stability Index (SSI)**: A novel metric for quantifying the boundary clarity of behavioral states.
- **Comprehensive Reporting**: Automatically generates an interactive HTML report with summary statistics and Comparative visualizations.

## 📂 Project Structure

- `run_benchmark.py`: Main orchestration script.
- `config.yaml`: Central configuration for paths and parameters.
- `src/`: Core implementation modules.
  - `loader.py`: Universal data loading for various formats.
  - `physics.py`: Feature extraction (Kinematics, Optical Flow, Residuals).
  - `metrics.py`: Statistical evaluation (SSI, NMI, Mann-Whitney U).
  - `visualizer.py`: Advanced plotting logic.
  - `report_generator.py`: HTML report compilation.
- `docs/`:
  - `benchmark_strategy.md`: Theoretical framework and "Figure Plan".
  - `usage.md`: Detailed setup and execution instructions.

## 📝 Documentation
For more details, see:
- [Usage Guide](docs/usage.md)
- [Benchmarking Strategy](docs/benchmark_strategy.md)
