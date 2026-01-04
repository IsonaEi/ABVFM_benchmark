import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keypoint_moseq as kpms
import plotly.express as px
import pandas as pd
from pathlib import Path
import re
from kpms_custom.utils.logging import get_logger

logger = get_logger()

# Global Style
plt.switch_backend('Agg') # Force non-interactive backend
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

def plot_ethograms(results, output_dir, config, cmap='tab20', filename='ethogram.png'):
    """Plots ethograms."""
    fps = config['preprocess']['fps']
    z_dict = results.get('model_states', {}).get('z', {})
    if not z_dict:
        # Fallback search
        for k, v in results.items():
            if isinstance(v, dict) and 'syllable' in v:
                z_dict[k] = v['syllable']
                
    if not z_dict:
        logger.warning("No syllable data for ethogram.")
        return

    sorted_keys = sorted(z_dict.keys())
    n_sessions = len(sorted_keys)
    if n_sessions == 0: return

    fig_height = max(4.0, n_sessions * 0.5 + 2.5)
    fig, axes = plt.subplots(n_sessions, 1, figsize=(60, fig_height), sharex=True)
    if n_sessions == 1: axes = [axes]
    
    all_z = np.concatenate([v.flatten() for v in z_dict.values()])
    max_syllable = int(np.max(all_z)) if len(all_z) > 0 else 0
    
    for i, key in enumerate(sorted_keys):
        z_seq = z_dict[key].flatten()
        duration = len(z_seq) / fps
        im_data = z_seq.reshape(1, -1)
        
        ax = axes[i]
        ax.imshow(im_data, aspect='auto', cmap=cmap, vmin=0, vmax=max_syllable, 
                  interpolation='nearest', extent=[0, duration, 0, 1])
        ax.set_yticks([])
        clean_key = re.sub(r'DLC.*', '', key).strip('_-')
        ax.set_ylabel(clean_key, rotation=0, ha='right', fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    axes[-1].set_xlabel("Time (s)", fontsize=24)
    plt.tight_layout()
    
    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved ethogram to {out_path}")

def plot_syllable_distribution(results, output_dir, config):
    """Plots ranked distribution."""
    # Gather all syllables
    all_syllables = []
    
    # Try getting from merged results
    # Logic similar to plot_ethograms to extract z
    z_dict = results.get('model_states', {}).get('z', {})
    if not z_dict:
         for k, v in results.items():
            if isinstance(v, dict) and 'syllable' in v:
                all_syllables.append(v['syllable'].flatten())
            elif isinstance(v, dict) and 'z' in v:
                all_syllables.append(v['z'].flatten())
    else:
        all_syllables = [v.flatten() for v in z_dict.values()]
        
    if not all_syllables:
        return

    flat = np.concatenate(all_syllables)
    unique_ids, counts = np.unique(flat, return_counts=True)
    total = len(flat)
    pcts = (counts / total) * 100
    
    # Sort
    idx = np.argsort(counts)[::-1]
    sorted_ids = unique_ids[idx]
    sorted_pcts = pcts[idx]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(range(len(sorted_ids)), sorted_pcts, color='salmon', edgecolor='maroon')
    ax.set_xticks(range(len(sorted_ids)))
    ax.set_xticklabels(sorted_ids.astype(str))
    ax.set_ylabel("Frequency (%)")
    ax.set_xlabel("Syllable (Ranked)")
    ax.set_title("Syllable Distribution")
    
    out_path = Path(output_dir) / "distribution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved distribution to {out_path}")
import cv2
from tqdm import tqdm

def find_matching_videos(keys, video_dir, extensions=['.mp4', '.avi', '.mov']):
    """Matches result keys to video files."""
    video_path = Path(video_dir)
    if not video_path.exists():
        logger.warning(f"Video directory not found: {video_dir}")
        return {}
        
    all_videos = []
    for ext in extensions:
        all_videos.extend(list(video_path.rglob(f"*{ext}")))
        
    video_map = {}
    for key in keys:
        # Key usually has `DLC_...` format. Clean it to find match?
        # Or simple substring match
        clean_key = re.sub(r'DLC.*', '', key).strip('_-')
        
        # Try finding a video that contains this string
        match = None
        for v in all_videos:
            if clean_key in v.name:
                match = v
                break
        
        if match:
            video_map[key] = str(match)
            
    return video_map

def generate_grid_movie(results, output_dir, project_dir, model_name, coords, confs, config):
    """Generates grid movie using KPMS built-in function."""
    if config is None or not config['analysis'].get('generate_grid_movies', False):
        return
        
    logger.info("Generating Grid Movie...")
    
    video_dir = config.get("video_dir")
    video_paths = find_matching_videos(list(coords.keys()), video_dir)
    
    if not video_paths:
        logger.warning("No matching videos found for grid movies.")
        return

    # Grid Movie params
    gm_cfg = config['analysis'].get('grid_movies', {})
    fps = config['preprocess']['fps']
    
    # For small datasets, default rows/cols must be small to avoid filtering
    use_rows = gm_cfg.get('rows', 2)
    use_cols = gm_cfg.get('cols', 3) 
    
    try:
        kpms.generate_grid_movies(
            results,
            project_dir=project_dir,
            model_name=model_name,
            coordinates=coords,
            video_paths=video_paths,
            plot_options=gm_cfg.get('plot_options', {}),
            rows=use_rows, 
            cols=use_cols, 
            pre=float(gm_cfg.get('pre', 15)) / fps, 
            post=float(gm_cfg.get('post', 15)) / fps,
            fps=fps,
            min_frequency=0, # FORCE ALL SYLLABLES
            min_duration=0   # FORCE ALL SYLLABLES
        )
        logger.info(f"Grid movie generation initiated.")
    except Exception as e:
        logger.error(f"Failed to generate grid movie: {e}")

def plot_trajectories(results, output_dir, coords, confs, bodyparts, project_dir, model_name):
    """Plots trajectories."""
    logger.info("Generating Trajectory Plots...")
    
    # Needs FPS.
    fps = 30.0 
    
    try:
        kpms.generate_trajectory_plots(
            coords,
            results,
            project_dir,
            model_name,
            fps=fps,
            save_gifs=True,
            save_mp4s=False,
            min_frequency=0, # FORCE ALL SYLLABLES
            min_duration=0,  # FORCE ALL SYLLABLES
            plot_options={'n_cols': 6}, # Try to force a grid that fits all 26
            sampling_options={'n_neighbors': 1} # CRITICAL: default 50 filters out rare syllables
        )
        logger.info("Saved trajectory plots via KPMS API.")
    except Exception as e:
        logger.error(f"Failed to generate trajectory plots: {e}")


def plot_dendrogram(results, output_dir, coords, config, project_dir, model_name):
    """Plots syllable similarity dendrogram."""
    if not config['analysis'].get('plot_dendrogram', False): return
    
    logger.info("Generating Dendrogram...")
    fps = config['preprocess']['fps']
    
    try:
        save_path = Path(output_dir) / "similarity_dendrogram"
        
        kpms.plot_similarity_dendrogram(
            coords, 
            results,
            project_dir,
            model_name,
            save_path=str(save_path),
            fps=fps
        )
        logger.info(f"Saved dendrogram to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to plot dendrogram: {e}")

def plot_transition_graph(results, output_dir, config, project_dir, model_name):
    """Plots transition graph and heatmap."""
    if not config['analysis'].get('plot_transition_graph', False): return
    
    logger.info("Generating Transition Analysis...")
    
    try:
        # 1. Generate Matrices
        # model_name must be STRING
        trans_data = kpms.generate_transition_matrices(project_dir, model_name, normalize="bigram")
        
        trans_mats = []
        usages = "default"
        groups_data = [model_name] # Default group is model name
        syll_include = range(100) 
        
        # Check return type
        if isinstance(trans_data, (tuple, list)) and len(trans_data) >= 4 and isinstance(trans_data[0], (list, tuple, np.ndarray)):
            # Assume legacy unpacked format: trans_mats, usages, group, syll_include
            trans_mats, usages, groups_data, syll_include = trans_data[:4]
        elif isinstance(trans_data, list):
            # Assume just list of matrices? Access patterns vary.
            # Based on source: returns trans_mats, usages, group, syll_include
            # So it should be a tuple/list len 4.
            trans_mats = trans_data
            
        if not trans_mats:
             logger.warning("No transition matrices generated.")
             return

        # 2. Plot Heatmap (Manual)
        if hasattr(trans_mats, '__iter__') and not isinstance(trans_mats, np.ndarray):
             mat_list = trans_mats
        else:
             mat_list = [trans_mats]
             
        for i, tm in enumerate(mat_list):
            tm = np.array(tm)
            # Hide self transitions?
            if config['analysis'].get('transition_graph', {}).get('hide_self_transitions', True):
                np.fill_diagonal(tm, 0)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(tm, cmap='viridis', ax=ax)
            ax.set_title(f"Transition Matrix {i}", fontsize=24)
            ax.tick_params(axis='both', labelsize=18)
            fig.savefig(Path(output_dir) / f"transition_heatmap_{i}.png")
            plt.close(fig)

        # 3. Plot Graph
        try:
             # Ensure args are correct types
             if isinstance(groups_data, str): groups_data = [groups_data]
             
             kpms.plot_transition_graph_group(
                project_dir, 
                model_name, 
                groups_data, 
                mat_list, 
                usages, 
                syll_include,
                save_dir=str(output_dir), # EXPLICIT SAVE DIR
                layout=config['analysis'].get('transition_graph', {}).get('layout', 'spring'),
                node_scaling=config['analysis'].get('transition_graph', {}).get('node_size_scale', 100)
            )
             logger.info("Saved transition graphs.")
        except Exception as e2:
             logger.warning(f"kpms.plot_transition_graph_group failed: {e2}")
        
    except Exception as e:
        logger.error(f"Failed to plot transition graph: {e}")

def generate_3d_scatter(results, output_dir, config):
    """Plots 3D scatter of latent space (Static & Interactive)."""
    if not config['analysis'].get('generate_3d_scatter', False): return
    
    logger.info("Generating 3D Scatter (Latent Space)...")
    
    # Needs model states x and z
    X_all = []
    Z_all = []
    
    # Extract X (latent) and Z (syllable)
    if 'model_states' in results and 'x' in results['model_states']:
         ms_x = results['model_states']['x']
         ms_z = results['model_states']['z']
         for sess, x in ms_x.items():
            X_all.append(np.array(x))
            Z_all.append(np.array(ms_z[sess]))
    else:
        # Try per session
        for k, v in results.items():
            if isinstance(v, dict) and ('x' in v or 'latent_state' in v) and ('z' in v or 'syllable' in v):
                x = v.get('x', v.get('latent_state'))
                z = v.get('z', v.get('syllable'))
                if x is not None and z is not None:
                    X_all.append(np.array(x))
                    Z_all.append(np.array(z))
                    
    if not X_all:
        logger.warning("No latent states found for 3D scatter.")
        return
        
    try:
        X_flat = np.concatenate(X_all)
        Z_flat = np.concatenate(Z_all)
        
        # Take only first 3 dimensions for 3D plot!
        if X_flat.shape[1] > 3:
            X_flat = X_flat[:, :3]
            
        if X_flat.shape[1] < 3:
            logger.warning("Latent stats < 3 dimensions.")
            return
            
        # 1. Static Plot (Matplotlib)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Subsample for speed
        idx = np.random.choice(len(X_flat), min(len(X_flat), 10000), replace=False)
        Xs = X_flat[idx]
        Zs = Z_flat[idx]
        
        sc = ax.scatter(Xs[:,0], Xs[:,1], Xs[:,2], c=Zs, cmap='tab20', s=2, alpha=0.5)
        ax.set_title("3D Latent Space", fontsize=24)
        ax.tick_params(labelsize=14)
        plt.colorbar(sc, ax=ax)
        
        fig.savefig(Path(output_dir) / "3d_scatter.png")
        plt.close(fig)
        
        # 2. Interactive Plot (Plotly)
        df = pd.DataFrame(Xs, columns=['x', 'y', 'z'])
        df['syllable'] = Zs.astype(str) # Force discrete color mapping
        
        fig_html = px.scatter_3d(
            df, x='x', y='y', z='z', 
            color='syllable', 
            color_discrete_sequence=px.colors.qualitative.Alphabet, # Distinct colors
            size_max=2, opacity=0.7,
            title="3D Latent Space (Interactive)"
        )
        fig_html.update_layout(font=dict(size=18)) # Increase global plotly font
        
        out_html = Path(output_dir) / "interactive_3d_scatter.html"
        fig_html.write_html(str(out_html))
        
        logger.info(f"Saved 3D scatter plots (PNG & HTML).")
        
    except Exception as e:
        logger.error(f"Failed to plot 3D scatter: {e}")

def generate_labeled_video(results, output_dir, video_dir, num_videos=5, draw_skeleton=True, keypoint_radius=3):
    """
    Overlay syllables on raw videos.
    """
    # 1. Extract z_dict
    z_dict = results.get('model_states', {}).get('z', {})
    if not z_dict: # fallback
         for k, v in results.items():
            if isinstance(v, dict) and 'syllable' in v:
                z_dict[k] = v['syllable']
            elif isinstance(v, dict) and 'z' in v:
                z_dict[k] = v['z']
                
    if not z_dict:
        logger.warning("No syllable data for labeled videos.")
        return
        
    # extract coordinates if available in results? 
    # Usually results from extract_results contains 'errors' which are reconstruction check.
    # It DOES NOT contain original coordinates usually unless we pass them.
    # We might need to load coordinates if we want to draw skeleton.
    # For now, let's just draw Syllable ID text if coords missing.
    coords_dict = {} # TODO: Pass coords to this function if needed
    
    keys = list(z_dict.keys())
    video_map = find_matching_videos(keys, video_dir)
    
    selected_keys = [k for k in keys if k in video_map][:num_videos]
    if not selected_keys:
        logger.warning("No matching videos found for labeling.")
        return

    # Colors
    all_z = np.concatenate([z_dict[k].flatten() for k in selected_keys])
    max_s = int(all_z.max()) if len(all_z) > 0 else 0
    
    # HSV or Tab20
    # Use simple Tab20 for now
    try:
        mp_colors = sns.color_palette('tab20', max_s + 1)
        bgr_colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in mp_colors]
    except:
        bgr_colors = [(0, 255, 0)] * (max_s + 1)

    out_path = Path(output_dir) / "labeled_videos"
    out_path.mkdir(parents=True, exist_ok=True)
    
    for key in selected_keys:
        vid_path = video_map[key]
        syllables = z_dict[key].flatten()
        
        logger.info(f"Labeling {key} -> {Path(vid_path).name}...")
        
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        save_file = out_path / f"{key}_labeled.mp4"
        writer = cv2.VideoWriter(str(save_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        n_frames = min(len(syllables), total_frames)
        
        for i in tqdm(range(n_frames), desc=f"Writing {key}", leave=False):
            ret, frame = cap.read()
            if not ret: break
            
            s_idx = int(syllables[i])
            color = bgr_colors[s_idx] if s_idx < len(bgr_colors) else (255, 255, 255)
            
            # Text
            text = f"Syl: {s_idx}"
            cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            writer.write(frame)
            
        cap.release()
        writer.release()
        logger.info(f"Saved {save_file}")


