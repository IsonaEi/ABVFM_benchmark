# Project Details: ABVFM Benchmark

This document tracks the version history, architectural decisions, and detailed specifications of the ABVFM Benchmark suite.

## 📦 Version History

### v0.0.1 - Initial Release (2025-01-11)
**Status**: Stable / Deployment Verified

#### Major Features
- **Modular Architecture**: Consolidated 5 sub-projects into a single repository with isolated environments.
- **Optical Flow Decoupling**: Extracted GPU-heavy GMFlow logic into a standalone `OpticalFlow` sub-project to optimize resource usage and reduce dependency conflicts.
- **Benchmark Refactor**: Transformed `Benchmark` into a lightweight, statistics-focused suite (NumPy/SciPy only).
- **Environment Standardization**: Established a clear `venvs/` directory structure for managing conflicting dependencies (JAX vs PyTorch).

#### Component Versions
| Component | Version | Core Strategy |
| :--- | :--- | :--- |
| **DA3_Mouse_Height** | 1.0.0 | Utilizes `Depth-Anything-3` for robust monocular height estimation. |
| **OpticalFlow** | 1.0.0 | Wraps `GMFlow` (Sintel weights) with dynamic background subtraction. |
| **Benchmark** | 1.0.0 | Implements "Killer Case" residual analysis and "State Stability Index (SSI)". |
| **KPMS** | 1.0.0 | Custom wrapper for `Keypoint-MoSeq` with auto-kappa scanning. |
| **B-SOiD** | 1.0.0 | patched `B-SOiD` implementation compatible with modern `scikit-learn`. |

## 🏗️ Architectural Decisions

### 1. Multi-Environment Strategy
Due to conflicting deep learning frameworks (KPMS uses JAX, DA3/OpticalFlow use PyTorch, B-SOiD implies legacy libs), we explicitly decided **NOT** to merge requirements. Each sub-project operates in its own virtual environment under `venvs/`.

### 2. Optical Flow Separation
**Problem**: The original `Benchmark` suite was bloated with PyTorch/GMFlow dependencies, making it heavy just to run simple statistical plots.
**Solution (v0.0.1)**: Extracted all GPU optical flow logic to `OpticalFlow`. The `Benchmark` suite now simply loads the pre-computed `.npy` flow magnitudes, allowing it to run on any CPU-only machine for analysis.

## 🛠️ Maintenance Notes

### Directory Structure
```
ABVFM_benchmark/
├── Benchmark/       # Stats & Analysis (CPU)
├── OpticalFlow/     # Visual Motion Engine (GPU)
├── DA3_Mouse_Height/# Depth Estimation (GPU)
├── KPMS/            # Unsupervised Behavior (JAX)
├── B-soid/          # Supervised/Unsupervised (Scikit-learn)
├── venvs/           # Virtual Environments
├── docs/            # Global Documentation
└── README.md        # Entry Point
```

### Future Roadmap
- [ ] Integration of "SimBA" for supervised classification comparison.
- [ ] Docker containerization for unified deployment.
- [ ] Web-based dashboard for `Benchmark` report visualization.
