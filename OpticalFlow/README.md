# Optical Flow Engine (ABVFM Sub-project)

This sub-project provides a standalone engine for computing optical flow using **GMFlow**, decoupled from the main benchmark for efficient GPU resource management.

## 🚀 Quick Start

Ensure you have the `optical-flow` virtual environment activated:

```bash
# From project root
source venvs/optical-flow/bin/activate

# Run optical flow on a video
python OpticalFlow/scripts/run_optical_flow.py --video path/to/video.mp4 --output results/flow_mag.npy
```

## 📋 Features

- **GMFlow (Sintel)**: High-performance optical flow estimation.
- **Dynamic Background Compensation**: Automatically calculates and subtracts background flicker/noise using median flow logic.
- **Masked Aggregation**: Restricts flow calculation to a provided mask (e.g., foreground mouse) to minimize recording noise.
- **Batch Processing**: Configurable batch size for GPU inference.

## 🛠️ Installation

1.  **Environment**:
    ```bash
    python3 -m venv venvs/optical-flow
    source venvs/optical-flow/bin/activate
    pip install -r OpticalFlow/requirements.txt
    ```

2.  **Weights**:
    Ensure the GMFlow weights (`gmflow_sintel-0c07dcb3.pth`) are placed in `OpticalFlow/pretrained/`.

## 📂 Project Structure

- `scripts/run_optical_flow.py`: CLI wrapper for the engine.
- `src/engine.py`: Core logic for GPU inference and masking.
- `gmflow/`: Submodule/Source for the GMFlow model structure.
- `pretrained/`: Directory for model weight files.

## 📝 Usage Details

### Command Line Arguments
- `--video`: Path to input video file (Required).
- `--output`: Path to save the resulting 1D magnitude array (.npy) (Required).
- `--mask`: Path to a binary mask file (.h5 or .npy).
- `--batch-size`: Number of frame pairs to process in one GPU pass (Default: 4).
- `--resize`: Dimensions to resize video frames before processing (Default: 720 720).

### Output Format
The script saves a `.npy` file containing a 1D array of optical flow magnitudes (one value per frame, frame 0 is always 0).
