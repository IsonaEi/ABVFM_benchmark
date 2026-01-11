# DA3_Mouse_Height

This subproject provides tools for estimating mouse height from video using monocular depth estimation (Depth-Anything-3) and background subtraction.

## Environment Setup

This project requires a Python environment with PyTorch and standard scientific libraries. A dedicated virtual environment `depth-anything-3` is recommended.

1.  **Activate Environment**:
    ```bash
    source ../venvs/depth-anything-3/bin/activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    # Also ensure Depth-Anything-3 is installed or in PYTHONPATH
    pip install -e Depth-Anything-3/
    ```

## Scripts

### 1. `pipeline.py`
The main entry point for running the depth estimation model.
- **Input**: Video file.
- **Output**: Raw Depth H5, CSV with height estimates.
- **Usage**:
  ```bash
  python pipeline.py --video path/to/video.mp4 --output results/output_name.csv --save_raw
  ```

### 2. `align_and_subtract.py`
Calculates accurate height by establishing a canonical background (floor) and subtracting the aligned depth map of the mouse.
- **Input**: Raw Depth H5 (from `pipeline.py`), Mask file, Original Video.
- **Output**: Visualization video (`_process_vis.mp4`), CSV statistics.
- **Usage**:
  ```bash
  python align_and_subtract.py --h5 results/output_name.h5 --mask path/to/mask.h5 --video path/to/video.mp4 --output_base results/height_analysis
  ```

### 3. `visualize.py`
Visualizes the depth maps with a consistent global color scale.
- **Modes**:
    - `raw`: Plain depth visualization (consistent color scale).
    - `stabilized`: Aligns every frame to a reference frame (Frame 0) to remove camera jitter or breathing.
- **Usage**:
  ```bash
  # Standard Visualization
  python visualize.py --h5 results/output_name.h5 --video path/to/video.mp4 --output results/viz_raw.mp4 --mode raw

  # Stabilized Visualization (Requires Mask to align floor)
  python visualize.py --h5 results/output_name.h5 --mask path/to/mask.h5 --video path/to/video.mp4 --output results/viz_stable.mp4 --mode stabilized
  ```

### 4. `utils.py`
Shared utility functions for alignment, mask loading, and visualization.
