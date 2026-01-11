# ABVFM Benchmark Suite (Automated Behavior Video Feature Mapping)

A comprehensive suite for benchmarking, analysis, and validation of automated animal behavior quantification tools. This repository aggregates multiple pipelines to provide a unified framework for converting raw video data into robust, statistical behavioral insights.

## 🌟 Overview

The **ABVFM Benchmark** is modularized into 5 independent sub-projects, each serving a distinct role in the behavioral analysis pipeline:

| Sub-project | Description | Key Tech | Status |
| :--- | :--- | :--- | :--- |
| **[DA3 (Mouse Height)](DA3_Mouse_Height/README.md)** | Monocular depth estimation for measuring animal height. | `Depth-Anything-3`, `PyTorch` | ✅ Active |
| **[OpticalFlow](OpticalFlow/README.md)** | Standalone GPU-accelerated engine for visual motion estimation. | `GMFlow`, `PyTorch` | ✅ Active |
| **[Benchmark](Benchmark/README.md)** | Core statistical analysis, physics extraction, and comparison suite. | `NumPy`, `SciPy`, `statsmodels` | ✅ Active |
| **[KPMS (Custom)](KPMS/KPMS_pipeline/README.md)** | Optimized wrapper for Keypoint-MoSeq (Unsupervised behavioral segmentation). | `Keypoint-MoSeq`, `JAX` | ✅ Active |
| **[B-SOiD (Custom)](B-soid/README.md)** | Implementation of B-SOiD with modernized dependencies and auto-tuning. | `B-SOiD`, `UMAP`, `HDBSCAN` | ✅ Active |

## 🚀 Getting Started

This repository uses a **multi-venv strategy** to manage conflicting dependencies between sub-projects (e.g., JAX vs PyTorch, Legacy vs Modern Scikit-learn).

### 1. Prerequisites
- Linux Environment
- Python 3.8+
- NVIDIA GPU (Required for `OpticalFlow` and `DA3`)

### 2. Installation
Please refer to the detailed installation guide in the **[Project Details](docs/PROJECT_DETAILS.md)** or the individual README of each sub-project.

**General Setup:**
```bash
git clone https://github.com/IsonaEi/ABVFM_benchmark.git
cd ABVFM_benchmark
mkdir -p venvs
```

### 3. Usage Workflow

1.  **Extract Physical Features**: Run `DA3` for height and `OpticalFlow` for visual motion.
2.  **Generate Behavior Labels**: Run `KPMS` or `B-SOiD` on your keypoint data.
3.  **Evaluate & Compare**: Use the `Benchmark` suite to align these labels with physical features and generate a performance report.

## 📚 Documentation

- **[Project Details & Version History](docs/PROJECT_DETAILS.md)**: Detailed changelog, versioning, and architectural decisions.
- **[Benchmark Strategy](Benchmark/docs/benchmark_strategy.md)**: Theoretical framework behind the evaluation metrics.

## 🛠️ Maintainers
- **System Architect**: IsonaEi
- **Lead Developer**: Meng-Xuan Liu

---
*Version: 0.0.1 (Initial Release)*
