
import os
import glob
import yaml
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import shutil
import cv2
import sys

# Add parent directory to path to allow importing src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.loader import DataLoader
from src.physics import PhysicsEngine
from src.metrics import MetricsEngine
from src.visualizer import Visualizer
from src.report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="CASTLE/Benchmark Runner (Optimized)")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--skip-gpu', action='store_true', help='Skip GPU-intensive Optical Flow')
    args = parser.parse_args()

    # 1. Setup
    # Resolve config path relative to script if not absolute
    config_path = args.config
    if not os.path.isabs(config_path):
        # Try finding it in Benchmark root if not consistent
        if not os.path.exists(config_path):
             potential_path = os.path.join(os.path.dirname(__file__), '..', config_path)
             if os.path.exists(potential_path):
                 config_path = potential_path
                 args.config = config_path # Update args so DataLoader gets the resolved path
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create Result Dir
    # base_dir is Benchmark root (parent of scripts/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = os.path.join(base_dir, "results", f"run_{timestamp}")
    clips_dir = os.path.join(result_dir, "clips")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)
    
    with open(os.path.join(result_dir, "config_snapshot.yaml"), 'w') as f:
        yaml.dump(config, f)

    print(f"=== Starting Benchmark Run: {timestamp} ===")
    print(f"Results will be saved to: {result_dir}")
    
    # Initialize Modules
    loader = DataLoader(args.config)
    physics = PhysicsEngine(device='cuda' if not args.skip_gpu else 'cpu')
    metrics = MetricsEngine()
    visualizer = Visualizer(output_dir=result_dir)
    report = ReportGenerator(output_dir=result_dir)
    
    report.add_header("CASTLE Benchmark Analysis Report")

    # 2. Data Loading
    print("\n--- Phase 1: Data Loading ---")
    data_path = config['paths']['keypoint_data']
    kps, bodyparts = loader.load_dlc_keypoints(data_path)
    
    if kps is None:
        print("Critical Error: Failed to load keypoints.")
        return

    fps = config['params']['fps']
    print(f"FPS: {fps}")

    # 3. Physics & Features
    print("\n--- Phase 2: Physics Calculation (Refined) ---")
    
    # A. KPMS Keypoint Change Score (Aligned)
    kp_change_score = physics.compute_keypoint_change_score(kps, bodyparts, sigma=config['params']['sigma'])
    
    # B. Kinematics (Vel, Acc, Jerk)
    kinematics = physics.compute_kinematics(kps, sigma=config['params']['sigma'])
    
    # C. Morphology
    morphology = physics.compute_morphology(kps, bodyparts)
    
    # D. Orientation (Relative & Absolute, + Derivatives)
    orientation = physics.compute_orientation(kps, bodyparts, fps=fps)
    # User Request: Abs of Orientation Features (Magnitude only)
    for k in ['relative_ang_vel', 'relative_ang_acc', 'absolute_ang_vel', 'absolute_ang_acc']:
        if k in orientation:
            orientation[k] = np.abs(orientation[k])
    
    # E. Optical Flow (Killer Case)
    video_path = config['paths']['video_data']
    if os.path.isdir(video_path):
        vids = glob.glob(os.path.join(video_path, "*.mp4")) + glob.glob(os.path.join(video_path, "*.avi"))
        video_path = vids[0] if vids else None
        
    flow_magnitude = None
    residuals = None
    slope, intercept = 0, 0
    
    # Optical Flow Analysis (Pre-computed Only)
    mask_path = config['paths'].get('mask_data')
    mask = None
    if mask_path and os.path.exists(mask_path): 
        mask = loader.load_mask(mask_path)

    flow_magnitude = None
    h5_flow_path = config['paths'].get('optical_flow_file')

    if h5_flow_path and os.path.exists(h5_flow_path):
        print(f"Using pre-computed optical flow from: {h5_flow_path}")
        flow_magnitude = physics.compute_masked_flow_from_h5(h5_flow_path, mask=mask)
    else:
        print("Note: Optical flow file not found or skipped. Residual analysis will be disabled.")
        print("To generate optical flow, use the separate 'OpticalFlow' sub-project.")
        
    # --- Residual Motion Analysis (Motion-Pose Discrepancy) ---
    if flow_magnitude is not None and kinematics.get('velocity') is not None:
        min_len = min(len(flow_magnitude), len(kinematics['velocity']))
        of_slice = flow_magnitude[:min_len]
        vel_slice = kinematics['velocity'][:min_len]
        
        z_of = metrics.compute_zscore(of_slice)
        z_sv = metrics.compute_zscore(vel_slice)
        
        residuals, slope, intercept = physics.compute_killer_case_residuals(z_of, z_sv)
        
        print(f"Residual Motion Analysis: Slope={slope:.2f}, Intercept={intercept:.2f}")
        
        if video_path and os.path.exists(video_path):
             # User Request (Round 2): Remove Video Clips Section
             pass
        # report.add_video_clips_section(clips_dir)

    # 4. Metrics & Evaluation
    print("\n--- Phase 3: Per-Method Evaluation ---")
    
    methods_config = config['methods']
    methods_data = []
    stats_list = []
    ssi_results = {}
    
    # Store Peak Values for Stats Test
    # Structure: {method: {metric: [peaks]}}
    peak_amplitudes = {} 
    
    # Store Residuals at Transitions for Comparison
    residuals_at_transition = {} 
    
    embeddings_path = config['paths'].get('dino_embeddings')
    dino_features = None
    if embeddings_path and os.path.exists(embeddings_path):
        dino_features = loader.load_embeddings(embeddings_path)
        if len(dino_features) != len(kps):
             dino_features = dino_features[:min(len(dino_features), len(kps))]
    
    if dino_features is None: # Fallback
        feats_list = [kinematics['velocity'], kinematics['acceleration'], 
                      morphology['snout_tail_dist'], morphology['compactness'], 
                      orientation['relative_ang_vel']]
        min_l = min([len(f) for f in feats_list])
        dino_features = np.stack([f[:min_l] for f in feats_list], axis=1)

    # Init Grouped Traces
    grouped_traces = {} # {feature: {method: {mean, sem}}}

    for method_name, params in methods_config.items(): # Pre-loop to count methods
        pass
    
    # --- Add Random Baseline (Sanity Check) ---
    print("Generating Random Baseline...")
    n_random_classes = 10 # Default to 10 classes
    total_frames = len(dino_features)
    random_labels = []
    current_frame = 0
    
    # User Spec: Median ~0.4s (12 frames@30fps). Distribution: "Between Linear and Exponential"
    # We use Exponential but clipped to [1, 300] (10s), with mean chosen to get median=12.
    # Median of Exp(scale) = ln(2)*scale => 12 = 0.693*scale => scale ~ 17.3
    target_median_frames = int(0.4 * fps)
    scale = target_median_frames / np.log(2) 
    
    while current_frame < total_frames:
        # Generate length
        run_len = int(np.random.exponential(scale=scale))
        # Clip: Max 10s (300 frames), Min 1 frame
        run_len = max(1, min(run_len, int(10 * fps))) 
        
        # Pick random class (different from previous if possible)
        label = np.random.randint(0, n_random_classes)
        if random_labels and len(random_labels) > 0 and random_labels[-1] == label:
             label = (label + 1) % n_random_classes
             
        # Add to list
        # Don't overflow total
        actual_len = min(run_len, total_frames - current_frame)
        random_labels.extend([label] * actual_len)
        current_frame += actual_len
        
    random_labels = np.array(random_labels)
    
    # Inject Random into methods_config-like structure for the loop
    # We can just iterate over keys and then process Random manually or append it to a list
    # Let's modify the loop to iterate over a list of (name, params, labels)
    
    methods_to_process = []
    # 1. Configured Methods
    for m_name, params in methods_config.items():
        if params.get('enabled', True):
             methods_to_process.append((m_name, params))
             
    # 2. Add Random
    methods_to_process.append(('Random', {'enabled': True}, random_labels))

    for method_info in methods_to_process:
        if len(method_info) == 2:
            method_name, params = method_info
            # Load Labels
            path, fmt = params['path'], params['format']
            labels = loader.load_labels(path, fmt)
            if labels is None: continue
            labels = loader.resample_labels(labels, len(dino_features))
        else:
             # Random Case
             method_name, _, labels = method_info
             
        print(f"Processing Method: {method_name}")
        
        # A. Stats
        stats = metrics.compute_label_stats(labels, fps)
        stats['method'] = method_name
        stats_list.append(stats)
        
        # B. SSI
        ssi_window = int(0.4 * fps)
        ssi = metrics.compute_ssi(dino_features, labels, window=ssi_window)
        ssi_results[method_name] = ssi
        stats['mean_ssi'] = np.mean(ssi) if ssi else 0
        stats['median_ssi'] = np.median(ssi) if ssi else 0
        
        methods_data.append({'name': method_name, 'labels': labels, 'fps': fps})
        
        # Plot Duration Histogram (Sorted by Total Duration)
        hist_filename = f"duration_hist_{method_name}.png"
        visualizer.plot_label_duration_histogram(labels, fps, method_name, hist_filename)
        report.add_image(os.path.join(result_dir, hist_filename), f"{method_name} - Class Total Duration")
        
        # C. Transitions Residuals
        if residuals is not None:
            trans_idxs = np.where(labels[1:] != labels[:-1])[0] + 1
            valid_idxs = trans_idxs[trans_idxs < len(residuals)]
            if len(valid_idxs) > 0:
                residuals_at_transition[method_name] = residuals[valid_idxs]

        # D. Event Triggered Traces (Collection by Feature)
        peak_amplitudes[method_name] = {}
        
        def collect_trace(key, data, filter_percentile=None):
             z_data = metrics.compute_zscore(data)
             m, s, _, peaks = metrics.get_event_triggered_traces(z_data, labels, return_peaks=True, filter_by_percentile=filter_percentile)
             if m is not None: 
                 if key not in grouped_traces: grouped_traces[key] = {}
                 grouped_traces[key][method_name] = {'mean': m, 'sem': s}
                 peak_amplitudes[method_name][key] = peaks 

        # 1. Kinematics
        collect_trace('Velocity', kinematics['velocity'])
        collect_trace('Acceleration', kinematics['acceleration'])
        collect_trace('Jerk', kinematics['jerk'])
        
        # 2. Orientation
        collect_trace('RelAngVel', orientation['relative_ang_vel'])
        collect_trace('RelAngAcc', orientation['relative_ang_acc'])
        collect_trace('AbsAngVel', orientation['absolute_ang_vel'])
        collect_trace('AbsAngAcc', orientation['absolute_ang_acc'])
        
        # 3. Morphology
        collect_trace('Compactness', morphology['compactness'])
        
        # 4. Change Score
        collect_trace('KPChange', kp_change_score)
        
        # 5. Visual Metrics
        if flow_magnitude is not None:
             collect_trace('OpticalFlow', flow_magnitude)
        
        if residuals is not None:
             # User Request: Top 1% Filter, Rename
             collect_trace('Residual Motion', residuals, filter_percentile=1)

    # Plot Grouped Traces
    print("Generating Feature-wise Trace Plots...")
    for feat_name, m_data in grouped_traces.items():
        # Clean up feature name for filename
        fname_safe = feat_name.replace(" ", "")
        
        # User Request: Specific Title/Label for Residual Motion
        if feat_name == 'Residual Motion':
             title = "Event-Triggered: Residual Motion (Flow vs Skeleton Gap - Top 1%)"
             ylabel = "Z-Score (Gap Magnitude)"
        else:
             title = f"Event-Triggered: {feat_name} (Z-Scored) [Abs]" if 'Ang' in feat_name else f"Event-Triggered: {feat_name} (Z-Scored)"
             ylabel = "Z-Score"
             
        visualizer.plot_combined_traces(m_data, fps, title, ylabel, f"trace_feature_{fname_safe}.png")
        report.add_image(os.path.join(result_dir, f"trace_feature_{fname_safe}.png"), f"Trace: {feat_name}")

    # E. Statistical Significance Test (Mann-Whitney U)
    if peak_amplitudes:
        # Default baseline: "CASTLE" (case insensitive check needed?)
        # Let's assume user named it "CASTLE" or "CASTLE_xxx" in config.
        # We try to find a key containing "CASTLE" as comparison target.
        castle_key = next((k for k in peak_amplitudes.keys() if 'CASTLE' in k.upper()), None)
        
        if castle_key:
            print(f"\nRunning Statistical Tests against baseline: {castle_key}...")
            stats_df = metrics.compute_mann_whitney(peak_amplitudes, method_key=castle_key)
            if stats_df is not None and not stats_df.empty:
                stats_csv_path = os.path.join(result_dir, "significance_test.csv")
                stats_df.to_csv(stats_csv_path, index=False)
                report.add_section("Statistical Significance (Mann-Whitney U)", 
                                   f"Comparing Peak Amplitudes of physical features at transitions. Baseline: {castle_key}.")
                report.md_content += stats_df.to_markdown(index=False)
            else:
                print("No significant comparisons possible or insufficient data.")
        else:
            print("Skipping Stats: No 'CASTLE' method found to use as baseline.")

    # Embedding Analysis
    if dino_features is not None:
         print("\n--- Phase 3.5: Embedding Analysis ---")
         T_kps = kps.shape[0]
         kps_flat = kps.reshape(T_kps, -1)
         min_len = min(len(dino_features), len(kps_flat))
         X_dino = dino_features[:min_len]
         X_kps = kps_flat[:min_len]
         
         # Reconstruction
         r2_dino_kps = metrics.compute_reconstruction_score(X_dino, X_kps)
         r2_kps_dino = metrics.compute_reconstruction_score(X_kps, X_dino)
         
         # Random Baselines (Chance Level)
         print("Calculating Reconstruction Chance Levels...")
         X_rand_dino = np.random.randn(*X_dino.shape)
         X_rand_kps = np.random.randn(*X_kps.shape)
         r2_rand_kps = metrics.compute_reconstruction_score(X_rand_dino, X_kps)
         r2_rand_dino = metrics.compute_reconstruction_score(X_rand_kps, X_dino)
         
         visualizer.plot_reconstruction_scores(
             {
                 'DINO->KP': r2_dino_kps, 
                 'KP->DINO': r2_kps_dino,
                 'Rand->KP': r2_rand_kps,
                 'Rand->DINO': r2_rand_dino
             }, "reconstruction_scores.png"
         )
         
         # Calculate Information Gain / Superset Improvement
         gain_dino = r2_dino_kps - r2_rand_kps
         gain_kp = r2_kps_dino - r2_rand_dino
         
         report.add_section("Feature Completeness", 
                            f"Reconstruction $R^2$ (vs Random Baseline):<br>"
                            f"- **DINO -> KP**: {r2_dino_kps:.3f} (Chance: {r2_rand_kps:.3f}, Gain: {gain_dino:.3f})<br>"
                            f"- **KP -> DINO**: {r2_kps_dino:.3f} (Chance: {r2_rand_dino:.3f}, Gain: {gain_kp:.3f})")
         report.add_image(os.path.join(result_dir, "reconstruction_scores.png"), "Reconstruction")
         
         # PCA
         n_d, v_d = metrics.compute_pca_dimensionality(X_dino, 0.90)
         n_k, v_k = metrics.compute_pca_dimensionality(X_kps, 0.90)
         visualizer.plot_pca_variance({'DINO': {'cum_var': v_d, 'n_90': n_d}, 'Keypoints': {'cum_var': v_k, 'n_90': n_k}}, "pca_variance.png")
         report.add_section("Intrinsic Dimensionality", f"PCs for 90% Var: DINO={n_d}, KPs={n_k}")
         report.add_image(os.path.join(result_dir, "pca_variance.png"), "PCA Variance")

    print("\n--- Phase 4: Validations & Reporting ---")
    
    # Ethogram
    visualizer.plot_ethogram_and_durations(methods_data, fps, filename_suffix="fig3_style")
    report.add_image(os.path.join(result_dir, "benchmark_fig3_style.png"), "Ethograms")
    
    # SSI
    random_mean_ssi = None
    if 'Random' in ssi_results:
         random_mean_ssi = np.mean(ssi_results['Random'])
         
    visualizer.plot_violin_comparison(ssi_results, "SSI Distribution", "SSI", "ssi_comparison.png", baseline_mean=random_mean_ssi)
    report.add_image(os.path.join(result_dir, "ssi_comparison.png"), "SSI")
    
    # SSI
    visualizer.plot_violin_comparison(ssi_results, "SSI Distribution", "SSI", "ssi_comparison.png")
    report.add_image(os.path.join(result_dir, "ssi_comparison.png"), "SSI")
    
    # Confusion Matrices (Pairwise)
    print("\n--- Phase 4.5: Confusion Matrices ---")
    from sklearn.metrics import confusion_matrix
    import itertools
    
    if len(methods_data) >= 2:
        report.add_section("Comparison Matrices", "Pairwise comparison of label assignments (Row-Normalized).")
        for m1, m2 in itertools.combinations(methods_data, 2):
            name1, labels1 = m1['name'], m1['labels']
            name2, labels2 = m2['name'], m2['labels']
            
            # Align lengths
            L = min(len(labels1), len(labels2))
            l1 = labels1[:L]
            l2 = labels2[:L]
            
            # Compute Contingency Table (MxN)
            # Use pandas crosstab to avoid union-of-labels square matrix issue
            ct = pd.crosstab(l1, l2)
            cm = ct.values
            classes1 = ct.index.to_numpy()
            classes2 = ct.columns.to_numpy()
            
            filename = f"confusion_{name1}_vs_{name2}.png"
            visualizer.plot_sorted_confusion_matrix(cm, classes1, classes2, name1, name2, filename)
            report.add_image(os.path.join(result_dir, filename), f"P({name2} | {name1})")

    # Stats Table
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(os.path.join(result_dir, "statistics.csv"), index=False)
    report.add_section("Summary Statistics", "Metrics per method.")
    report.md_content += stats_df.to_markdown(index=False)
    
    report_path = report.generate()
    print(f"\nSUCCESS. Report: {report_path}")

if __name__ == "__main__":
    main()
