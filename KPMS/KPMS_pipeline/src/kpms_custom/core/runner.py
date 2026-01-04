import os
import sys
import keypoint_moseq as kpms
from pathlib import Path

from kpms_custom.utils.config import load_config, save_config
from kpms_custom.utils.logging import get_logger
from kpms_custom.data.loader import load_h5_files, parse_dlc_data, detect_fps
from kpms_custom.data.preprocessor import train_pca, prepare_for_kpms
from kpms_custom.model.trainer import setup_project, fit_model
from kpms_custom.model.tuning import scan_kappa
from kpms_custom.analysis.evaluation import evaluate_models
from kpms_custom.analysis.merging import MotifMerger
from kpms_custom.analysis.viz import (
    plot_ethograms, plot_syllable_distribution, 
    generate_labeled_video, plot_trajectories, generate_grid_movie,
    plot_dendrogram, plot_transition_graph, generate_3d_scatter
)

logger = get_logger()

def _load_and_prep(config):
    """Helper to load data and PCA."""
    # 1. Setup Project
    project_dir = setup_project(config)
    
    # 2. Data
    files = load_h5_files(config)
    coords, confs, bodyparts = parse_dlc_data(files, config)
    
    data, metadata, bodyparts = prepare_for_kpms(coords, confs, config, bodyparts)
    
    # 3. PCA
    pca = train_pca(data, config, project_dir, bodyparts)
    
    return project_dir, data, metadata, pca

def run_training(config_path, restarts=None):
    config = load_config(config_path)
    if restarts: config['num_fitting_restarts'] = restarts
    
    project_dir, data, metadata, pca = _load_and_prep(config)
    
    n_runs = config.get('num_fitting_restarts', 1)
    
    for i in range(n_runs):
        logger.info(f"--- Training Run {i+1}/{n_runs} ---")
        fit_model(data, metadata, pca, config, project_dir, name_suffix=f"-{i}")
        
    logger.info("Training complete.")

def run_scan(config_path, scan_type='ar'):
    config = load_config(config_path)
    project_dir, data, metadata, pca = _load_and_prep(config)
    
    best_kappa, best_dur = scan_kappa(data, metadata, pca, config, project_dir, scan_type=scan_type)
    
    logger.info(f"Scan Result: K={best_kappa:.2e}, Dur={best_dur:.1f}ms")
    
    # Update Config
    if scan_type == 'ar':
        config['tuning']['ar_kappa'] = float(best_kappa)
    else:
        config['tuning']['full_kappa'] = float(best_kappa)
        
    save_config(config, config_path)
    logger.info(f"Updated {config_path}")

def run_evaluation(config_path):
    config = load_config(config_path)
    project_dir = config['project_dir']
    
    best = evaluate_models(project_dir)
    if best:
        print(f"Recommended Model: {best}")

def run_analysis(config_path, model_name=None, results_path=None, output_dir=None):
    """
    Run analysis on existing results.
    """
    config = load_config(config_path)
    
    if results_path:
        logger.info(f"Using explicit results file: {results_path}")
        results_path = Path(results_path)
        if not results_path.exists():
            logger.error("Results file not found.")
            return
        # Deduce context
        model_name = results_path.parent.name
        project_dir = str(results_path.parent.parent)
        config['project_dir'] = project_dir
        
    elif model_name:
        # Standard path
        project_dir = config['project_dir']
        results_path = Path(project_dir) / model_name / 'results.h5'
        if not results_path.exists():
            # Check for checkpoint
            checkpoint_path = results_path.parent / 'checkpoint.h5'
            if not checkpoint_path.exists():
                logger.error(f"Results/Checkpoint not found for {model_name}")
                return
            logger.info("Results file missing, will attempt to extract from checkpoint.")
    else:
        logger.error("Must specify --model or --results_path")
        return

    # Determine Output Directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = results_path.parent / "figures"
        
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Load Data (Coords required for viz)
    logger.info("Loading data for analysis...")
    files = load_h5_files(config)
    coords, confs, bodyparts = parse_dlc_data(files, config)
    
    # Preprocess (needed for metadata)
    data, metadata, bodyparts = prepare_for_kpms(coords, confs, config, bodyparts)

    try:
        logger.info(f"Loading results from {results_path}")
        results = kpms.load_results(project_dir, model_name)
    except:
        logger.warning("kpms.load_results failed, using extract_results...")
        checkpoint = kpms.load_checkpoint(project_dir, model_name)
        results = kpms.extract_results(checkpoint[0], metadata=metadata, project_dir=project_dir, model_name=model_name)

    # Visualizations
    try:
        _perform_viz_pipeline(results, out_dir, coords, confs, bodyparts, config, project_dir, model_name)
        logger.info(f"Analysis complete. Figures saved to {out_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
    return results

def _perform_viz_pipeline(results, out_dir, coords, confs, bodyparts, config, project_dir, model_name):
    """Helper to run all visualization functions."""
    plot_ethograms(results, out_dir, config)
    plot_syllable_distribution(results, out_dir, config)
    
    plot_trajectories(results, out_dir, coords, confs, bodyparts, project_dir, model_name)
    
    generate_grid_movie(results, out_dir, project_dir, model_name, coords, confs, config)
    
    plot_dendrogram(results, out_dir, coords, config, project_dir, model_name)
    
    plot_transition_graph(results, out_dir, config, project_dir, model_name)
    
    generate_3d_scatter(results, out_dir, config)
    
    # Labeled Video
    if config['analysis'].get('generate_labeled_videos', False):
         video_dir = config.get("video_dir")
         num_videos = config['analysis'].get('labeled_videos', {}).get('num_videos', 5)
         generate_labeled_video(results, out_dir, video_dir, num_videos=num_videos)

def run_merging(config_path, model_name):
    config = load_config(config_path)
    project_dir = config['project_dir']
    
    if not model_name:
        logger.error("Please specify --model")
        return
        
    # Load Data (required for metadata and viz)
    files = load_h5_files(config)
    coords, confs, bodyparts = parse_dlc_data(files, config)
    data, metadata, bodyparts = prepare_for_kpms(coords, confs, config, bodyparts)

    checkpoint = kpms.load_checkpoint(project_dir, model_name)
    model = checkpoint[0]
    try:
        results = kpms.load_results(project_dir, model_name)
    except:
        results = kpms.extract_results(model, metadata=metadata, project_dir=project_dir, model_name=model_name, save_results=False)
    
    threshold = config['analysis'].get('merge_threshold', 10)
    
    merger = MotifMerger(results, threshold_frames=threshold)
    merger.calculate_centroids()
    merger.identify_motif_types()
    merger.suggest_merges()
    new_results = merger.apply_merges(results, project_dir, model_name)
    
    # Create new folder for merged results
    merged_model_name = f"{model_name}_merged"
    merged_model_dir = Path(project_dir) / merged_model_name
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy essential files for KPMS API functions
    orig_model_dir = Path(project_dir) / model_name
    if (orig_model_dir / "checkpoint.h5").exists():
        import shutil
        shutil.copy(orig_model_dir / "checkpoint.h5", merged_model_dir / "checkpoint.h5")
    
    # Figures directory inside the merged model folder
    out_dir = merged_model_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Merging complete. Running full analysis for {merged_model_name}...")
    
    # Need data for full analysis
    files = load_h5_files(config)
    coords, confs, bodyparts = parse_dlc_data(files, config)
    data, metadata, bodyparts = prepare_for_kpms(coords, confs, config, bodyparts)
    
    # Save merged results to enable transition analysis
    results_path = merged_model_dir / "results.h5"
    kpms.save_hdf5(str(results_path), new_results, exist_ok=True, overwrite=True)
    
    _perform_viz_pipeline(new_results, out_dir, coords, confs, bodyparts, config, project_dir, merged_model_name)
    
    logger.info(f"Merged analysis complete. Results in {merged_model_dir}")
    return new_results
