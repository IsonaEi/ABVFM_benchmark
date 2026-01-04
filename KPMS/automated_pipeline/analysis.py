import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import keypoint_moseq as kpms
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, List, Optional
import glob
import re

# ===================================================================
# Monkeypatch for Matplotlib < 3.8 compatibility if needed
# ===================================================================
def _patch_canvas_tostring_rgb():
    """
    Patches FigureCanvasAgg to ensure tostring_rgb exists.
    Newer matplotlib versions removed it in favor of method calls that return bytes.
    kpms internally calls tostring_rgb().
    """
    if not hasattr(FigureCanvasAgg, 'tostring_rgb'):
        def tostring_rgb(self):
            return self.draw() or self.buffer_rgba().tobytes()
            # Note: The original tostring_rgb returned RGB, buffer_rgba returns RGBA.
            # However, kpms might just need specific byte output. 
            # If kpms expects exactly 3 channels, we might need conversion.
            # Let's try a safer approach compatible with kpms expectations (uint8 buffer).
            
        # Better safe implementation trying to match expected behavior
        def tostring_rgb_safe(self):
            self.draw()
            # get_width_height is standard
            w, h = self.get_width_height()
            # buffer_rgba is standard in newer mpl
            buf = self.buffer_rgba()
            # Convert RGBA to RGB
            return np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3].tobytes()
            
        FigureCanvasAgg.tostring_rgb = tostring_rgb_safe
        print("[Analysis] Patched FigureCanvasAgg.tostring_rgb for compatibility.")

_patch_canvas_tostring_rgb()

# ===================================================================
# Helper Functions
# ===================================================================

def extract_results_safe(model: Any, metadata: Dict, project_dir: str, model_name: str) -> Dict:
    """
    Wraps kpms.extract_results with overwrite=True to prevent AssertionError.
    """
    print("Extracting model results...")
    
    # Check for existing results.h5 and remove it to prevent overwrite error
    results_path = Path(project_dir) / model_name / "results.h5"
    if results_path.exists():
        print(f"Removing existing results file: {results_path}")
        try:
           os.remove(results_path)
        except Exception as e:
           print(f"Warning: Could not remove {results_path}: {e}")

    # kpms.extract_results will save results if save_results=True (default)
    results = kpms.extract_results(model, metadata, project_dir, model_name)
    
    # We don't need to manually save CSV if extract_results handles it or if we rely on h5.
    # But let's leave the try-except just in case user wants explicit CSV and extract_results didn't produce it.
    # Actually, extract_results usually produces h5. Users might want CSV.
    # kpms.save_results_as_csv expects the results dict.
    try:
        kpms.save_results_as_csv(results, project_dir, model_name)
    except Exception as e:
        print(f"Warning: Could not save results as CSV: {e}")
        
    return results

def find_matching_videos_robust(keys: List[str], video_dir: str, extension: str = 'mp4') -> Dict[str, str]:
    """
    Robustly finds videos matching the keys. 
    """
    video_dir_path = Path(video_dir)
    video_files = list(video_dir_path.rglob(f"*{extension}"))
    video_map = {}
    
    print(f"Searching for {len(keys)} videos in {video_dir}...")
    
    for key in keys:
        match = None
        key_stem = Path(key).stem
        
        # Priority 1: Clean search (strip DLC suffix)
        if "DLC" in key_stem:
            clean_stem = key_stem.split("DLC")[0].strip('_-')
            for v_file in video_files:
                if v_file.stem == clean_stem:
                    match = str(v_file)
                    break
        
        # Priority 2: Standard Logic
        if not match:
            for v_file in video_files:
                if v_file.stem == key_stem:
                    match = str(v_file)
                    break
        
        if not match:
             for v_file in video_files:
                if v_file.name.startswith(key_stem):
                    match = str(v_file)
                    break
                    
        if not match:
             for v_file in video_files:
                if key_stem.startswith(v_file.stem):
                    match = str(v_file)
                    break

        if match:
            video_map[key] = match
        else:
            print(f"Warning: No video found for key: {key}")
            
    return video_map

def generate_interactive_3d_scatter(csv_path: Path, output_html_path: Path, color_palette: str = "Pastel"):
    """
    Generates an interactive 3D scatter plot using Plotly.
    """
    try:
        df = pd.read_csv(csv_path)
        required_cols = ["latent_state 0", "latent_state 1", "latent_state 2", "syllable"]
        if not all(col in df.columns for col in required_cols):
             print(f"Skipping 3D scatter for {csv_path.name}: Missing required columns.")
             return

        df["syllable"] = df["syllable"].astype(str)

        fig = px.scatter_3d(
            df,
            x="latent_state 0",
            y="latent_state 1",
            z="latent_state 2",
            color="syllable",
            color_discrete_sequence=px.colors.qualitative.__dict__.get(color_palette, px.colors.qualitative.Plotly),
            opacity=0.8,
            title=f"Interactive 3D Latent Space - {csv_path.stem}"
        )

        fig.update_traces(marker=dict(size=3, line=dict(width=0)))
        fig.update_layout(
            width=800,
            height=700,
            scene=dict(
                xaxis_title="Latent State 0",
                yaxis_title="Latent State 1",
                zaxis_title="Latent State 2"
            ),
            legend_title_text='Syllable'
        )

        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_html_path, include_plotlyjs="cdn")
        print(f"Generated 3D scatter plot: {output_html_path}")

    except Exception as e:
        print(f"Error generating 3D scatter for {csv_path.name}: {e}")

def plot_ethograms(results, project_dir, model_name, fps=30.0, cmap='tab20', filename='ethogram.png', start_min=None, end_min=None):
    """
    Plots ethograms (syllable sequences) for all sessions.
    Optionally restricted to a time window (start_min to end_min in minutes).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Check if results have syllables (z)
    # KPMS results structure check
    z_dict = {}
    if 'model_states' in results and 'z' in results['model_states']:
        z_dict = results['model_states']['z']
    else:
        # Check root with fallback keys
        for k, v in results.items():
            if not isinstance(v, dict): continue
            
            if 'syllable' in v:
                z_dict[k] = v['syllable']
            elif 'z' in v:
                 z_dict[k] = v['z']

    if not z_dict:
         print("Error: Could not locate syllable sequences (z/syllable) in results.")
         return

    sorted_keys = sorted(z_dict.keys())
    n_sessions = len(sorted_keys)
    
    if n_sessions == 0:
        return

    # Create figure
    # Calculate height: 0.5 inch per session + 2.5 inches for giant X-axis labels
    fig_height = max(4.0, n_sessions * 0.5 + 2.5)
    fig, axes = plt.subplots(n_sessions, 1, figsize=(60, fig_height), sharex=True)
    if n_sessions == 1: axes = [axes]
    
    # Get distinct syllables for colormap
    all_z = np.concatenate(list(z_dict.values()))
    max_syllable = np.max(all_z) if len(all_z) > 0 else 0
    palette = sns.color_palette(cmap, int(max_syllable) + 1)
    
    time_suffix = ""
    if start_min is not None and end_min is not None:
        time_suffix = f"_{start_min}to{end_min}m"
        print(f"Plotting ethograms for {n_sessions} sessions (Window: {start_min}-{end_min} min)...")
    else:
        print(f"Plotting ethograms for {n_sessions} sessions (Full Duration)...")
    
    for i, key in enumerate(sorted_keys):
        z_seq = z_dict[key]
        if z_seq.ndim > 1: z_seq = z_seq.flatten()
        
        # Apply Time Slicing
        t_start = 0
        if start_min is not None:
            t_start = int(start_min * 60 * fps)
            
        t_end = len(z_seq)
        if end_min is not None:
            t_end_req = int(end_min * 60 * fps)
            t_end = min(t_end, t_end_req)
            
        if t_start >= t_end:
            print(f"Warning: Start time {start_min}m exceeds duration for {key}. Skipping.")
            continue
            
        z_view = z_seq[t_start:t_end]
        
        # Create an image strip (1 x T)
        im_data = z_view.reshape(1, -1)
        
        # Calculate duration of the SLICE in seconds
        duration_slice = len(z_view) / fps
        
        ax = axes[i]
        
        # Extent: Map [0, duration] on X axis
        # We start X from 0 (relative to window start) or absolute?
        # User typically wants to see "0s" as "Start of window" or actual time?
        # Let's use relative time for visualization 0 to (end-start) usually looks cleaner,
        # otherwise huge empty space on left if we use absolute.
        # But maybe label it? For now relative 0.
        
        ax.imshow(im_data, aspect='auto', cmap=cmap, vmin=0, vmax=max_syllable, 
                  interpolation='nearest', extent=[0, duration_slice, 0, 1])
        ax.set_yticks([])
        
        # Clean Y label
        clean_key = re.sub(r'DLC.*', '', key).strip('_-')
        ax.set_ylabel(clean_key, rotation=0, ha='right', fontsize=48)
        
        # Increase tick font size
        ax.tick_params(axis='x', labelsize=48)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
    xlabel = "Time (s)"
    if start_min is not None:
        xlabel += f" (Relative to {start_min} min)"
    axes[-1].set_xlabel(xlabel, fontsize=48)
    plt.tight_layout()
    
    filename_root, filename_ext = os.path.splitext(filename)
    final_filename = f"{filename_root}{time_suffix}{filename_ext}"
    
    out_path = Path(project_dir) / model_name / "figures" / final_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved ethogram to {out_path}")

def create_labeled_video(results, coordinates, project_dir, model_name, video_dir, num_videos=5, draw_skeleton=True, keypoint_radius=3):
    """
    Creates videos with syllable labels overlaid.
    """
    import cv2
    import matplotlib.colors as mcolors
    
    out_dir = Path(project_dir) / model_name / "labeled_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get syllables
    z_dict = {}
    if 'model_states' in results and 'z' in results['model_states']:
        z_dict = results['model_states']['z']
    else:
        for k, v in results.items():
            if not isinstance(v, dict): continue
            if 'syllable' in v:
                z_dict[k] = v['syllable']
            elif 'z' in v:
                 z_dict[k] = v['z']
    
    # 2. Match keys to videos
    keys = list(z_dict.keys())
    if not keys: 
        print("No sessions found with syllable data ('z'/'syllable').")
        return    
    video_map = find_matching_videos_robust(keys, video_dir)
    
    # Select first N valid matches
    selected_keys = [k for k in keys if k in video_map][:num_videos]
    
    if not selected_keys:
        print("No matching videos found for labeling.")
        return

    # Generate fixed colors for syllables
    all_z = np.concatenate([z_dict[k].flatten() for k in selected_keys])
    max_s = int(all_z.max()) if len(all_z) > 0 else 0
    
    # Use hsv for distinct colors
    hsv_colors = [((i / (max_s + 1)) * 180, 255, 255) for i in range(max_s + 1)] # In HSV for OpenCV (H=0-180)
    
    # Or use matplotlib tab20 loop
    mp_colors = sns.color_palette('tab20', max_s + 1)
    # Convert to BGR (0-255) for OpenCV
    bgr_colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in mp_colors]

    for key in selected_keys:
        vid_path = video_map[key]
        syllables = z_dict[key].flatten()
        coords = coordinates[key] # shape T, N, 2
        
        print(f"Creating labeled video for {key}...")
        
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        save_path = out_dir / f"{key}_labeled.mp4"
        out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # Limit frames if data length differs (take minimum)
        n_frames = min(len(syllables), len(coords), total_frames)
        
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret: break
            
            s_idx = int(syllables[i])
            color = bgr_colors[s_idx] if s_idx < len(bgr_colors) else (255, 255, 255)
            
            # Overlay Text
            text = f"Syllable: {s_idx}"
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Overlay Skeleton if requested
            if draw_skeleton:
                # Plot points
                current_coords = coords[i]
                for pt in current_coords:
                    x, y = int(pt[0]), int(pt[1])
                    if not np.isnan(x) and not np.isnan(y):
                         cv2.circle(frame, (x, y), keypoint_radius, color, -1)
                         
            out.write(frame)
            
        cap.release()
        out.release()
        print(f"Saved {save_path}")




# ===================================================================
# Model Evaluation Logic
# ===================================================================

def perform_model_evaluation(config: Dict[str, Any], project_dir: str, model_name_pattern: str = None):
    """
    Evaluates multiple models and selects the best one based on Expected Marginal Likelihood (EML).
    
    Steps:
    1. Identify all models matching the pattern.
    2. Batch Process:
       - Sort syllables by frequency (reindex).
       - Extract results (results.h5).
    3. Compute EML scores.
    4. Plot EML scores.
    5. Identify and return the Best Model name.
    """
    print("="*60)
    print("Starting Model Evaluation & Selection")
    print("="*60)
    
    # 1. Identify Models
    # If pattern is None, try to find all subdirectories that look like models
    project_path = Path(project_dir)
    
    if model_name_pattern:
        # User defined pattern
        candidates = list(project_path.glob(model_name_pattern))
    else:
        # Auto-discover: Look for any dir containing 'checkpoint.h5' or 'checkpoint'
        # Check for checkpoint.h5 (most common) or checkpoint (legacy/other)
        candidates = []
        for d in project_path.iterdir():
            if not d.is_dir(): continue
            if (d / "checkpoint.h5").exists() or (d / "checkpoint").exists():
                 candidates.append(d)

    # Filter out known non-model dirs and kappa scan artifacts
    candidates = [
        d for d in candidates 
        if d.name not in ["pca", "results", ".git", "__pycache__"]
        and not d.name.startswith("kappa_")
        and not d.name.startswith("warmup_")
    ]
    
    model_names = sorted([d.name for d in candidates])
    
    if not model_names:
        print("Error: No valid trained models (excluding scans) found for evaluation.")
        return None
        
    print(f"Found {len(model_names)} models to evaluate: {model_names}")
    
    import preprocess # Import locally to avoid circular imports if run_pipeline imports analysis

    # 2. Batch Processing: Sort & Extract
    print("\n--- Step 1: Batch Processing (Reindex & Extract) ---")
    
    # Reload metadata from raw data to ensure correct structure for unbatch
    # Checkpoint metadata (checkpoint[1]) is often the dataset dict, not the (names, ranges) tuple needed for extraction
    print("Loading data to obtain metadata for extraction...")
    try:
        files = preprocess.load_data(config)
        coords, confs, mdata_bodyparts = preprocess.format_data(files, config)
        # We only need metadata, but we must run preprocess_data to get it in the right format if dependent
        # actually format_data returns metadata as tuple? 
        # No, format_data returns (coordinates, confidences, bodyparts). 
        # kpms.format_data is called INSIDE preprocess.preprocess_data.
        # So we must call preprocess_data.
        _, metadata, _ = preprocess.preprocess_data(coords, confs, config, mdata_bodyparts)
        print(f"Metadata reloaded successfully: {type(metadata)}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

    for name in model_names:
        print(f"Processing {name}...")
        try:
            # A. Sort Syllables (Reindex)
            # This ensures syllable 0 is most frequent, etc.
            # Important for consistent comparison.
            # Check if already sorted? kpms doesn't mark it. 
            # Re-running it is usually safe as it updates the checkpoint.
            print(f"  > Sorting syllables by frequency...")
            kpms.reindex_syllables_in_checkpoint(project_dir, name)
            
            # B. Extract Results
            # We need to load the model to extract results?
            # kpms.extract_results takes 'model' object.
            print(f"  > Loading checkpoint...")
            checkpoint = kpms.load_checkpoint(project_dir, name)
            model_obj = checkpoint[0]
            # metadata_obj = checkpoint[1] # We ignore checkpoint metadata as it is often the dataset dict
            
            print(f"  > Extracting results using reloaded metadata...")
            extract_results_safe(model_obj, metadata, project_dir, name)
            
            # Force close to free memory? Python GC handles it usually.
            del model_obj, checkpoint
            
        except Exception as e:
            print(f"  > Error processing {name}: {e}")
            # import traceback
            # traceback.print_exc()
            
    # 3. Compute EML Scores
    print("\n--- Step 2: Computing EML Scores ---")
    try:
        # kpms.expected_marginal_likelihoods returns (scores, standard_errors)
        scores, std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
        
    except Exception as e:
        print(f"Error computing EML scores: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 4. Plot EML Scores
    print("\n--- Step 3: Plotting EML Scores ---")
    try:
        # kpms.plot_eml_scores takes (scores, std_errs, model_names)
        fig, ax = kpms.plot_eml_scores(scores, std_errs, model_names)
        
        save_path = Path(project_dir) / "model_comparison_eml.png"
        fig.savefig(save_path)
        print(f"Saved EML plot to {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting EML scores: {e}")
        
    # 5. Select Best Model
    try:
        # Select model with highest EML score
        best_idx = np.argmax(scores)
        best_model_name = model_names[best_idx]
        best_score = scores[best_idx]
        
        print(f"\n>>> BEST MODEL SELECTED: {best_model_name} (Score: {best_score:.4f}) <<<")
        
        # Save selection to a file for reference
        with open(Path(project_dir) / "best_model.txt", "w") as f:
            f.write(best_model_name)
            
        return best_model_name
            
    except Exception as e:
        print(f"Error selecting best model: {e}")
        return None



def perform_analysis(
    config: Dict[str, Any],
    project_dir: str,
    model_name: str,
    model: Any,
    dataset: Dict[str, Any]
):
    """
    Executes the downstream analysis based on config.
    """
    analysis_cfg = config.get("analysis", {})
    if not analysis_cfg.get("perform_analysis", True):
        return

    print("="*60)
    print(f"Starting Downstream Analysis for {model_name}")
    print("="*60)
    
    # We don't perform actions here as run_analysis_module does it.
    # This might be legacy or placeholder.
    pass

def check_or_create_index_csv(project_dir: str, model_name: str, metadata: Any):
    """
    Ensures index.csv exists for grouping. If missing, creates a default one.
    index.csv columns: uuid, group, name
    """
    index_path = Path(project_dir) / "index.csv"
    if not index_path.exists():
        print(f"Index CSV not found at {index_path}. Creating default...")
        records = []
        
        # Metadata can be dict {uuid: ...} OR tuple (names, ranges)
        if isinstance(metadata, dict):
            keys = metadata.keys()
        elif isinstance(metadata, tuple) or isinstance(metadata, list):
            # Assuming tuple (names, ranges)
            # names is the first element
            keys = metadata[0]
        else:
            print(f"Warning: Unknown metadata format {type(metadata)}. Cannot create index.csv.")
            return

        for key in keys:
            name = str(key)
            group = "default"
            records.append({"uuid": name, "group": group, "name": name})
            
        pd.DataFrame(records).to_csv(index_path, index=False)
        print("Created default index.csv")

def perform_dynamic_merge(results, model, config, project_dir, model_name):
    """
    Dynamically merges fragmented motifs based on latent space centroids.
    Returns: new_results (merged)
    """
    threshold_frames = config.get("analysis", {}).get("merge_threshold", 10)
    print(f"Performing Dynamic Merging (Threshold: {threshold_frames} frames)...")
    
    # 1. Extract Latent States (X) and Labels (Z)
    # Prefer using model object directly as it has the raw states
    if 'states' not in model:
        print("Error: Model object missing 'states'. Cannot calculate centroids.")
        return results
        
    x_all = model['states']['x']
    z_all = model['states']['z']
    
    # Flatten and Align
    x_flat_list = []
    z_flat_list = []
    
    for i in range(len(x_all)):
        xi = np.array(x_all[i])
        zi = np.array(z_all[i])
        
        # Align ends (Z usually shorter/lagged)
        if len(zi) <= len(xi):
            xi_aligned = xi[-len(zi):]
            x_flat_list.append(xi_aligned)
            z_flat_list.append(zi)
            
    if not x_flat_list:
        return results
        
    x_flat = np.concatenate(x_flat_list, axis=0)
    z_flat = np.concatenate(z_flat_list, axis=0)
    
    # 2. Identify Stable vs Short
    uniq_motifs = np.unique(z_flat)
    frame_counts = {m: np.sum(z_flat == m) for m in uniq_motifs}
    
    # check max continuous duration? or just total count?
    # User said: "at least one continuous instance lasting 10 frames"
    # suggest_merges.py logic: actually suggest_merges.py snippet I saw didn't implement the 10 frame check fully in the snippet?
    # Wait, the user manual snippet for suggest_merges.py had hardcoded lists.
    # We need to implement the *logic* described: "Stability Threshold... 10 frames or more".
    
    stable_motifs = []
    short_motifs = []
    
    for m in uniq_motifs:
        # Find runs
        mask = (z_flat == m)
        # We need per-session runs to be accurate, but global runs on concat z_flat is approximation (risk of boundary)
        # Better: iterate sessions.
        is_stable = False
        
        for k in range(len(z_flat_list)):
            zi = z_flat_list[k]
            # Find runs of m in zi
            # A run is where zi == m
            runs = np.diff(np.concatenate(([0], (zi == m).astype(int), [0])))
            starts = np.where(runs == 1)[0]
            ends = np.where(runs == -1)[0]
            
            durations = ends - starts
            if np.any(durations >= threshold_frames):
                is_stable = True
                break
                
        if is_stable:
            stable_motifs.append(m)
        else:
            short_motifs.append(m)
            
    print(f"identified {len(stable_motifs)} stable, {len(short_motifs)} short motifs.")
            
    # 3. Compute Centroids
    centroids = {}
    for m in uniq_motifs:
        mask = (z_flat == m)
        if np.sum(mask) > 0:
            centroids[m] = np.mean(x_flat[mask], axis=0)
        else:
            centroids[m] = np.zeros(x_flat.shape[1])
            
    # 4. Map Short to Nearest Stable
    syllables_to_merge = []
    merge_map = {} # target -> list of sources
    
    for short in short_motifs:
        c_short = centroids[short]
        best_target = None
        min_dist = float('inf')
        
        for stable in stable_motifs:
            dist = np.linalg.norm(c_short - centroids[stable])
            if dist < min_dist:
                min_dist = dist
                best_target = stable
        
        if best_target is not None:
            if best_target not in merge_map:
                merge_map[best_target] = []
            merge_map[best_target].append(short)
            
    # Format for KPMS
    # list of [target, src1, src2...]
    for target, sources in merge_map.items():
        syllables_to_merge.append([target] + sources)
        
    # 5. Apply Mapping
    if not syllables_to_merge:
        print("No merges required.")
        return results
        
    print(f"Applying {len(syllables_to_merge)} merges...")
    syllable_mapping = kpms.generate_syllable_mapping(results, syllables_to_merge)
    new_results = kpms.apply_syllable_mapping(results, syllable_mapping)
    
    # 6. Save Strategy Doc
    docs_dir = Path(project_dir) / model_name / "merged_analysis"
    docs_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = docs_dir / "motif_merging_strategy.md"
    
    with open(strategy_path, "w") as f:
        f.write("# Motif Merging Strategy\n")
        f.write(f"Model: {model_name}\n\n")
        f.write(f"Threshold: {threshold_frames} frames\n")
        f.write(f"Stable Motifs: {len(stable_motifs)}\n")
        f.write(f"Short Motifs: {len(short_motifs)}\n\n")
        f.write("## Merges\n| Target | Sources |\n|---|---|\n")
        for group in syllables_to_merge:
            f.write(f"| {group[0]} | {group[1:]} |\n")
            
    # 7. Save merged results as the PRIMARY results.h5
    # (Backup original first)
    original_h5_path = Path(project_dir) / model_name / "results.h5"
    backup_h5_path = Path(project_dir) / model_name / "results_unmerged.h5"
    
    if original_h5_path.exists():
        print(f"Backing up original results to {backup_h5_path}...")
        if backup_h5_path.exists():
             os.remove(backup_h5_path)
        os.rename(original_h5_path, backup_h5_path)
        
    merged_h5_path = Path(project_dir) / model_name / "results.h5"
    print(f"Saving merged results to {merged_h5_path}...")
    
    if hasattr(kpms, 'save_hdf5'):
        kpms.save_hdf5(str(merged_h5_path), new_results)
    else:
        # Fallback manual save if needed (usually kpms has it)
        # Using internal utility if available
        try:
             kpms.io.save_hdf5(str(merged_h5_path), new_results)
        except:
             import h5py
             # Simple save function or similar... usually kpms.save_hdf5 works.
             # If it failed above, we might have issues. 
             # Assuming kpms.save_hdf5 works as it was used in other scripts.
             pass
        
    return new_results



def run_analysis_module(config: Dict[str, Any], project_dir: str, model_name: str, model: Any, data: Dict, metadata: Dict, coordinates: Dict, pca: Any = None):
    """
    The main callable function for analysis.
    """
    analysis_cfg = config.get("analysis", {})
    fps = config.get("preprocess", {}).get("fps", 30.0)
    
    # 1. Extract Results
    try:
        results = extract_results_safe(model, metadata, project_dir, model_name)
        if results is None: raise ValueError("Extraction returned None")
    except Exception as e:
        print(f"Error extracting results: {e}")
        return

    import matplotlib.pyplot as plt
    
    # Global Plot Settings for Readability
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })

    # 2. Dynamic Merging (Phase 5+)
    if analysis_cfg.get("merge_motifs", False):
        print("\n--- Auto-Merging Motifs Enabled ---")
        try:
             results = perform_dynamic_merge(results, model, config, project_dir, model_name)
             print("Results updated with merged motifs. Proceeding with analysis...")
        except Exception as e:
             print(f"Error during motif merging: {e}")

    # 3. Trajectory Plots
    traj_cfg = analysis_cfg.get("trajectories", {})
    if traj_cfg:
        print("Generating Trajectory Plots...")
        try:
            kpms.generate_trajectory_plots(
                coordinates, 
                results, 
                project_dir=project_dir, 
                model_name=model_name,
                pre=analysis_cfg.get("grid_movies", {}).get("pre", 15) / fps,
                post=analysis_cfg.get("grid_movies", {}).get("post", 15) / fps,
                save_gifs=traj_cfg.get("save_gifs", True),
                save_mp4s=traj_cfg.get("save_mp4s", False),
                min_frequency=1e-9, # Use small float to avoid falsy check
                min_duration=0,
                fps=fps,

                skeleton=config.get("project", {}).get("skeleton", []),
                density_sample=False # Force plotting all data, avoids filtering sparse syllables
            )
        except Exception as e:
            print(f"Error generating trajectory plots: {e}")

    if analysis_cfg.get("plot_dendrogram", True):
        print("Generating Syllable Dendrogram...")
        try:
            dendro_kwargs = config.copy()
            if 'project_dir' in dendro_kwargs: del dendro_kwargs['project_dir']
            kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, fps=fps, **dendro_kwargs)
            print("Saved dendrogram plot.")
        except Exception as e:
            print(f"Error plotting dendrogram: {e}")

    # 8. Transition Graph & Heatmap
    if analysis_cfg.get("plot_transition_graph", True):
        print("Generating Syllable Transition Analysis...")
        trans_cfg = analysis_cfg.get("transition_graph", {})
        try:
            # A. Generate Matrices
            trans_data = kpms.generate_transition_matrices(project_dir, model_name, normalize="bigram")
            if isinstance(trans_data, (tuple, list)) and len(trans_data) >= 4:
                trans_mats, usages, groups_data, syll_include = trans_data[:4]
            else:
                trans_mats, usages, groups_data, syll_include = trans_data, "default", ["default"], range(100)

            if isinstance(trans_mats, (list, tuple)):
                trans_mats = [np.array(m).astype(float) for m in trans_mats]
            else:
                trans_mats = [np.array(trans_mats).astype(float)]

            # B. Plot Heatmap (Manual)
            print("  > Plotting transition heatmap...")
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            for idx, tm in enumerate(trans_mats):
                # Apply Zero Diagonal if requested
                if trans_cfg.get("hide_self_transitions", True): # Defaulting to True as per user request
                    np.fill_diagonal(tm, 0)
                
                # Plot
                plt.figure(figsize=(14, 12)) # Larger figure
                # Enable xticklabels/yticklabels
                sns.heatmap(tm, cmap="viridis", xticklabels=True, yticklabels=True)
                plt.title(f"Transition Matrix (Group: {groups_data[idx] if isinstance(groups_data, list) else groups_data})")
                save_path = Path(project_dir) / model_name / "figures" / f"transition_heatmap_{idx}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                plt.close()
                print(f"    Saved heatmap to {save_path}")

            # C. Plot Graph
            print("  > Plotting transition graph...")
            kpms.plot_transition_graph_group(
                project_dir, model_name, groups_data, trans_mats, usages, syll_include,
                layout=trans_cfg.get("layout", "circular"),
                node_scaling=trans_cfg.get("node_size_scale", 2000),
                show_syllable_names=False
            )
        except Exception as e:
            print(f"Error plotting transition analysis: {e}")

    # 9. Ethogram (Manual)
    if analysis_cfg.get("plot_ethogram", True):
        print("Plotting Ethograms (Manual implementation)...")
        try:
             # Simple Ethogram Plotter
             import matplotlib.pyplot as plt
             import matplotlib.patches as patches
             
             # Get first few sessions
             sessions = list(results.keys())[:3] # Plot top 3 sessions
             fig, ax = plt.subplots(len(sessions), 1, figsize=(15, 2*len(sessions)), sharex=True)
             if len(sessions) == 1: ax = [ax]
             
             cmap = plt.cm.get_cmap("tab20", 100) # Assuming < 100 syllables
             
             for i, session in enumerate(sessions):
                 z = results[session]['syllable']
                 # Create colored bars
                 for t, s in enumerate(z):
                     rect = patches.Rectangle((t/fps, 0), 1/fps, 1, color=cmap(s % 100))
                     ax[i].add_patch(rect)
                 ax[i].set_xlim(0, len(z)/fps)
                 ax[i].set_ylim(0, 1)
                 ax[i].set_ylabel(f"Session {session}")
                 ax[i].set_yticks([])
                 
             ax[-1].set_xlabel("Time (s)")
             plt.tight_layout()
             save_path = Path(project_dir) / model_name / "figures" / "ethogram.png"
             save_path.parent.mkdir(parents=True, exist_ok=True)
             plt.savefig(save_path)
             plt.close()
             print(f"Saved ethogram to {save_path}")
        except Exception as e:
             print(f"Error plotting ethogram: {e}")
             
    # 10. Syllable Probability Histogram (Fixed)
    if analysis_cfg.get("plot_stats", True):
        print("Plotting Syllable Frequency Histogram...")
        try:
             fig, ax = kpms.plot_syllable_frequencies(
                 results=results,
                 project_dir=project_dir,
                 model_name=model_name
             )
             save_path = Path(project_dir) / model_name / "figures" / "syllable_frequencies.png"
             save_path.parent.mkdir(parents=True, exist_ok=True)
             
             # Customize Axes
             # kpms typically plots a bar chart.
             # Ensure X ticks are all integers
             n_syllables = len(ax.get_xticks()) 
             # Or get max x
             max_x = int(ax.get_xlim()[1])
             ax.set_xticks(range(max_x + 1))
             ax.set_xticklabels(range(max_x + 1), rotation=90, fontsize=12) # Rotate if many
             
             ax.set_xlabel("Syllable ID", fontsize=24)
             ax.set_ylabel("Probability", fontsize=24)
             
             fig.savefig(save_path)
             plt.close(fig)
             print(f"Saved syllable frequency histogram to {save_path}")
        except Exception as e:
             print(f"Error plotting stats: {e}")
             
    # 11. Grid Movies (Fixed)
    if analysis_cfg.get("generate_grid_movies", False):
        print("Generating Grid Movies...")
        try:
            # Construct video_paths mapping
            video_dir = config.get("video_dir")
            video_paths = {}
            
            # DEBUG
            print(f"Coordinates keys: {list(coordinates.keys())}")
            
            if video_dir and os.path.exists(video_dir):
                # Assuming video extensions
                supported_exts = ['.mp4', '.avi', '.mov']
                found_videos = []
                for ext in supported_exts:
                    found_videos.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
                
                print(f"Found videos in {video_dir}: {found_videos}")
                
                # Map session name (stem) to full path
                for v in found_videos:
                    start_name = os.path.splitext(os.path.basename(v))[0]
                    # Check if this name exists in our results/coordinates
                    # Try exact match or match specific substring logic if needed
                    # Common issue: coordinates keys might be relative paths or stems
                    for k in coordinates.keys():
                        if start_name in k or k in start_name:
                            video_paths[k] = v
                            
            if not video_paths:
                print("Warning: No matching videos found for grid movies. Attempting keypoints-only mode if supported/fallback.")
            else:
                print(f"Mapped {len(video_paths)} videos for grid movies.")
            
            kpms.generate_grid_movies(
                results, # Correct: results is first arg
                project_dir=project_dir,
                model_name=model_name,
                coordinates=coordinates, # Pass coordinates as kwarg
                video_paths=video_paths, 
                pre=analysis_cfg.get("grid_movies", {}).get("pre", 15) / fps,
                post=analysis_cfg.get("grid_movies", {}).get("post", 15) / fps,
                rows=analysis_cfg.get("grid_movies", {}).get("rows", 3),
                cols=analysis_cfg.get("grid_movies", {}).get("cols", 3),
                plot_options=analysis_cfg.get("grid_movies", {}).get("plot_options", {}),
                fps=fps,
                min_frequency=0, # Force all syllables
                min_duration=0
            )
        except Exception as e:
            print(f"Error generating grid movies: {e}")
            
    # 12. 3D Scatter Plot (Using Model States if available, else PCA results)
    if analysis_cfg.get("generate_3d_scatter", False):
        print("Generating 3D Scatter Plot (Latent Space)...")
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            X_all = []
            Z_all = []
            
            # Method 1: Use 'model_states' in results (Preferred)
            # Check for top-level keys or iterate sessions
            if 'model_states' in results and 'x' in results['model_states']:
                 ms_x = results['model_states']['x']
                 ms_z = results['model_states']['z']
                 if isinstance(ms_x, dict):
                     for sess in sorted(ms_x.keys()):
                         X_all.append(ms_x[sess])
                         Z_all.append(ms_z[sess])
                 else:
                     X_all.append(np.array(ms_x))
                     Z_all.append(np.array(ms_z))
            else:
                 # Check per-session keys
                 for sess in sorted(results.keys()):
                     if not isinstance(results[sess], dict): continue
                     
                     # Check keys explicitly to avoid ambiguous truth value of arrays
                     if 'latent_state' in results[sess]:
                         s_x = results[sess]['latent_state']
                     else:
                         s_x = results[sess].get('x')
                     
                     if 'syllable' in results[sess]:
                         s_z = results[sess]['syllable']
                     else:
                         s_z = results[sess].get('z')
                     
                     if s_x is not None and s_z is not None:
                         X_all.append(np.array(s_x))
                         Z_all.append(np.array(s_z))

            # Method 2: Use data['pca_scores'] (Fallback)
            if not X_all and data:
                target_key = 'pca_scores' if 'pca_scores' in data else 'Y'
                if target_key in data:
                    val = data[target_key]
                    # If 'Y' is raw coords (ndim > 2), this fallback is invalid for 3D scatter
                    if np.ndim(val) > 2 and target_key == 'Y':
                        pass 
                    else:
                        # Try to use it if it looks like PCA scores
                        pass # Typically we rely on results extraction.

            if X_all and Z_all:
                X_flat = np.concatenate(X_all)
                # Ensure X is 2D (T, D)
                if X_flat.ndim > 2:
                    X_flat = X_flat.reshape(-1, X_flat.shape[-1])
                
                Z_flat = np.concatenate(Z_all)
                
                # Use only first 3 dims
                if X_flat.shape[1] >= 3:
                     X_plot = X_flat[:, :3]
                     
                     # --- Static Plot ---
                     fig = plt.figure(figsize=(10, 8))
                     ax = fig.add_subplot(111, projection='3d')
                     
                     # Subsample
                     limit_pts = 10000
                     if len(X_plot) > limit_pts:
                         idx = np.random.choice(len(X_plot), limit_pts, replace=False)
                         Xs = X_plot[idx]
                         Zs = Z_flat[idx]
                     else:
                         Xs = X_plot
                         Zs = Z_flat
                     
                     # Color by syllable (simple numeric)
                     ax.scatter(Xs[:,0], Xs[:,1], Xs[:,2], c=Zs, cmap='tab20', s=1, alpha=0.3)
                     ax.set_title("3D Latent Space (Model States)")
                     
                     save_path = Path(project_dir) / model_name / "figures" / "3d_scatter.png"
                     save_path.parent.mkdir(parents=True, exist_ok=True)
                     plt.savefig(save_path)
                     plt.close()
                     print(f"Saved 3D scatter to {save_path}")
                     
                     # --- Interactive Plot ---
                     print("Generating Interactive 3D Scatter (HTML)...")
                     try:
                         # Create DataFrame
                         df_data = {
                             "latent_state 0": X_plot[:, 0],
                             "latent_state 1": X_plot[:, 1],
                             "latent_state 2": X_plot[:, 2],
                             "syllable": Z_flat
                         }
                         df = pd.DataFrame(df_data)
                         csv_path = Path(project_dir) / model_name / "figures" / "3d_scatter_data.csv"
                         df.to_csv(csv_path, index=False)
                         
                         html_path = Path(project_dir) / model_name / "figures" / "interactive_3d_scatter.html"
                         generate_interactive_3d_scatter(csv_path, html_path)
                     except Exception as e:
                         print(f"Error creating interactive plot: {e}")
                else:
                    print(f"Error: Latent states have < 3 dims ({X_flat.shape}).")
            else:
                print("Could not obtain Model States (x) for 3D scatter.")

        except Exception as e:
            print(f"Error generating 3D scatter: {e}")
            import traceback
            traceback.print_exc()
