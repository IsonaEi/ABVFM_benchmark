# Benchmark Implementation Plan: Keypoint Change Score Alignment

**Goal**: Implement the "Keypoint Change Score Alignment" benchmark (KPMS Fig. 3c) to compare the temporal precision of KPMS, B-SOiD, VAME, and CASTLE.

## 1. Mathematical Logic (Recap from Paper)
The **Keypoint Change Score** quantifies the magnitude of pose dynamics at any given moment. The core hypothesis is that **true behavioral transitions** should align with **peaks in pose change**.

Formula:
$$ \text{ChangeScore}(t) = \text{Z-score} \left( \sum_{k=1}^{K} \| \tilde{y}_{t,k} - \tilde{y}_{t-1,k} \| \right) $$
Where $\tilde{y}_{t,k}$ is the **egocentrically aligned** and **smoothed** coordinate of keypoint $k$ at time $t$.

## 2. Implementation Steps

### Step 1: Data Preparation
- **Input Data**:
    - `keypoints`: Shape `(T, K, 2)` (Raw 2D coordinates)
    - `labels_dict`: Dictionary containing label arrays for each method.
        - `{'KPMS': array(T), 'BSOID': array(T), 'VAME': array(T), 'CASTLE': array(T)}`
- **Output**:
    - `change_score`: Shape `(T,)`
    - `alignment_results`: Dictionary store aligned traces.

### Step 2: Calculate Keypoint Change Score
1.  **Egocentric Alignment**:
    - Calculate **Centroid** ($v_t$) and **Heading** ($h_t$, tail-to-nose angle).
    - Rotate and center keypoints to a canonical frame:
      $$ y'_{t} = R(-h_t) \cdot (y_t - v_t) $$
2.  **Smoothing**:
    - Apply Gaussian filter ($\sigma = 1$ frame) to $y'_t$ to remove high-frequency jitter.
3.  **Velocity Calculation**:
    - Compute Euclidean distance between adjacent frames: $d_t = \sum_k \| y'_{t,k} - y'_{t-1,k} \|$.
4.  **Z-Scoring**:
    - Normalize $d_t$ to get the final Z-score.

### Step 3: Transition Alignment (The "Event-Triggered Average")
1.  **Find Transitions**:
    - For each method, identify indices $t$ where $z_t \neq z_{t-1}$.
2.  **Extract Windows**:
    - Define a window (e.g., $\pm 0.5$ seconds, or $\pm 15$ frames @ 30Hz).
    - For each transition $t_{trans}$, extract `change_score[t_trans - w : t_trans + w]`.
3.  **Aggregate**:
    - Compute the **Mean** and **95% Confidence Interval** (Standard Error) across all transitions for each method.

### Step 4: Visualization
- **Plot**: Line plot with Time (x-axis) vs. Z-scored Change Score (y-axis).
- **Overlay**: 4 lines (KPMS, B-SOiD, VAME, CASTLE) with different colors.
- **Expected Result**: KPMS should show a sharp, symmetric peak centered at 0. Others may be flatter or offset.

## 3. Python Function Signature Draft

```python
def compute_change_score(keypoints, smooth_sigma=1):
    """
    Computes the z-scored keypoint change score from raw keypoints.
    1. Align (center + rotate).
    2. Smooth.
    3. Diff + Norm + Sum.
    4. Z-score.
    """
    pass

def get_transition_alignment(change_score, labels, window_size=15):
    """
     aligns change score to label transitions.
    Returns: mean_trace, sem_trace, time_axis
    """
    pass

def plot_benchmark(alignment_results):
    """
    Generates the comparison plot (Fig 3c style).
    """
    pass
```

## 4. Requirement Checklist
- [ ] Need access to raw `keypoints` (npy/h5).
- [ ] Need loaded `labels` sequences from all 4 methods (aligned to the same frames).
- [ ] Need `matplotlib` and `scipy` (for gaussian_filter1d, zscore).
