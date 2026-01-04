import os
import glob
import numpy as np
import pandas as pd
import keypoint_moseq as kpms
import re

def detect_bodyparts(file_path, conf_threshold=0.5):
    """
    Robustly detect bodyparts from the first available H5 file.
    Returns:
        bodyparts (list): List of unique bodypart names.
    """
    try:
        # Load just the header/first few rows to check columns
        df = pd.read_hdf(file_path)
        
        # Check for MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # DLC format: (scorer, individuals, bodyparts, coords) or (scorer, bodyparts, coords)
            
            # Find the level that contains 'x', 'y'
            # Usually the last level is coords ('x', 'y', 'likelihood')
            # The level before that is bodyparts
            
            # Identify 'bodyparts' level index
            # Heuristic: verify if 'nose' or similar distinct names are in a level
            # OR just take the level above 'coords'
            
            # Let's inspect levels
            levels = df.columns.levels
            
            # Common DLC structure check
            # Multi-animal: [scorer, individual, bodypart, coords]
            # Single-animal: [scorer, bodypart, coords]
            
            # We look for the level that has the most unique items usually, 
            # excluding the coords level (x/y/like) and scorer level (usually 1 item)
            
            bp_candidates = []
            
            for lvl_idx, lvl_values in enumerate(levels):
                if 'x' in lvl_values and 'y' in lvl_values:
                    # This is the coords level. The one BEFORE it is likely bodyparts.
                    if lvl_idx > 0:
                        bp_level = levels[lvl_idx - 1]
                        bp_candidates = list(bp_level)
                        break
            
            if not bp_candidates:
                # Fallback: Try to find standard names
                for lvl in levels:
                    if 'nose' in lvl or 'snout' in lvl:
                        bp_candidates = list(lvl)
                        break
                        
            # Filter out empty or odd names if needed
            bodyparts = [bp for bp in bp_candidates if isinstance(bp, str)]
            
            print(f"Auto-detected {len(bodyparts)} bodyparts: {bodyparts}")
            return bodyparts
            
    except Exception as e:
        print(f"Error detecting bodyparts: {e}")
        return []

def load_data(config):
    """Load data from the directory specified in config."""
    data_dir = config['data_dir']
    file_pattern = os.path.join(data_dir, '*.h5')
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")
        
    print(f"Found {len(files)} files in {data_dir}")
    return files

def format_data(files, config):
    """Format data for Keypoint-MoSeq."""
    # This logic matches the notebook's approach of loading and formatting
    # The KPMS library has a format_data utility, we will use it but might need custom logic
    # if we want to filter specific bodyparts as per the notebook's 'include_parts'.
    
    # Load config from the project directory if it exists, to get bodyparts schema
    # But here we assume we are starting fresh or using the config.yaml
    
    coordinates = {}
    confidences = {}
    
    cfg_bodyparts = config.get('project_config', {}).get('bodyparts', 'AUTO')
    
    if isinstance(cfg_bodyparts, list) and len(cfg_bodyparts) > 0:
        bodyparts = cfg_bodyparts
    else:
        # Detect from first file
        if files:
             # conf_threshold is passed but detection logic currently ignores it or uses it?
             # detect_bodyparts signature: (file_path, conf_threshold=0.5)
             params = config.get('preprocess', {})
             # Handle case where preprocess is None or empty dict, though we fixed run_pipeline to ensure it.
             # But here we are in preprocess.py, called from run_pipeline.
             thresh = params.get('conf_threshold', 0.5) if params else 0.5
             bodyparts = detect_bodyparts(files[0], thresh)
        else:
             bodyparts = []
    
    for f in files:
        name = os.path.basename(f).replace('.h5', '')
        try:
            # We use kpms built-in loading if possible, but for custom filtering
            # we might need pandas. Let's strictly follow the notebook's extensive logic
            # later if needed, but for now use kpms.io.load_deeplabcut_results assuming standard format
            
            # Using basic pandas reading to ensure we control the parts
            df = pd.read_hdf(f)
            
            # Extract XY and Conf
            # Assuming standard DLC MultiIndex: scorer -> individuals -> bodyparts -> coords
            # OR scorer -> bodyparts -> coords
            
            if isinstance(df.columns, pd.MultiIndex):
                # Handle Multi-animal DataFrame structure (scorer -> individuals -> bodyparts -> coords)
                # Flatten to: (bodyparts -> coords) for the single individual
                # Check levels
                if df.columns.nlevels >= 4:
                    # Assuming level 1 is 'individuals'
                    individuals = df.columns.levels[1]
                    # For now, just take the first individual found
                    # TODO: Iterate all individuals if needed
                    indiv = individuals[0] 
                    use_df = df.xs(indiv, level=1, axis=1)
                else:
                    use_df = df

                coords_list = []
                confs_list = []
            
                # Iterate over ALL project bodyparts to ensure fixed K
            # But wait, we want to filter out bad bodyparts?
            # No, 'coordinates' structure in KPMS usually requires fixed K across all animals.
            # If we drop a bodypart for one animal, we must drop it for all.
                # Find which level contains the bodyparts
                bp_level_idx = 0
                if bodyparts:
                    test_bp = bodyparts[0]
                    for i in range(use_df.columns.nlevels):
                        if test_bp in use_df.columns.get_level_values(i):
                            bp_level_idx = i
                            break
                
                # For now, let's load EVERYTHING defined in 'bodyparts' (which comes from AUTO or Config).
                for bp in bodyparts:
                    if bp in use_df.columns.get_level_values(bp_level_idx):
                        # Accessing the specific bodypart. 
                        # We need to slice carefully if there are multiple levels.
                        # use_df[bp] works if bp is in top level? 
                        # If bp is in level 1, use_df[bp] fails if level 0 is not dropped?
                        # If use_df is MultiIndex, use_df[bp] attempts to select on level 0.
                        
                        # Use cross-section or generic selector
                        try:
                            bp_data = use_df.xs(bp, level=bp_level_idx, axis=1)
                            # Now bp_data should contain x, y, likelihood (level -1)
                            x = bp_data['x'].values if 'x' in bp_data else bp_data.iloc[:, 0].values
                            y = bp_data['y'].values if 'y' in bp_data else bp_data.iloc[:, 1].values
                            c = bp_data['likelihood'].values if 'likelihood' in bp_data else (bp_data.iloc[:, 2].values if bp_data.shape[1]>2 else np.zeros(len(bp_data)))
                        except:
                            # Fallback if xs fails or weird structure
                            x = np.full(len(use_df), np.nan)
                            y = np.full(len(use_df), np.nan)
                            c = np.zeros(len(use_df))
                    else:
                        # Missing bodypart: fill with NaNs
                        n_frames = len(use_df)
                        x = np.full(n_frames, np.nan)
                        y = np.full(n_frames, np.nan)
                        c = np.zeros(n_frames)
                        
                    coords_list.append(np.stack([x, y], axis=1))
                    confs_list.append(c)

                if coords_list:
                    # Shape: (T, K, 2)
                    coords = np.stack(coords_list, axis=1)
                    # Shape: (T, K)
                    confs = np.stack(confs_list, axis=1)
                    
                    coordinates[name] = coords
                    confidences[name] = confs
                else:
                    print(f"Skipping {f}: No bodyparts found.")
                
            else:
                print(f"Skipping {f}: unexpected dataframe format")
                
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return coordinates, confidences, bodyparts

def filter_bad_bodyparts(coordinates, bodyparts, threshold=0.9):
    """
    Remove bodyparts that are NaN in > threshold fraction of frames across ALL data.
    """
    # coordinates: dict of {name: (T, K, 2)}
    all_data = [] # List of (T, K, 2)
    for v in coordinates.values():
        all_data.append(v)
        
    if not all_data:
        return coordinates, bodyparts
        
    # Concatenate along time: (Total_T, K, 2)
    mega_matrix = np.concatenate(all_data, axis=0)
    
    # Check NaNs per bodypart
    # (Total_T, K, 2) -> (Total_T, K) boolean
    nans = np.isnan(mega_matrix).any(axis=2)
    nan_ratio = nans.mean(axis=0) # Shape (K,)
    
    keep_indices = np.where(nan_ratio < threshold)[0]
    drop_indices = np.where(nan_ratio >= threshold)[0]
    
    if len(drop_indices) > 0:
        dropped_names = [bodyparts[i] for i in drop_indices]
        print(f"Dropping {len(dropped_names)} bodyparts (> {threshold*100}% NaN): {dropped_names}")
        
        new_bodyparts = [bodyparts[i] for i in keep_indices]
        new_coords = {}
        for name, data in coordinates.items():
            new_coords[name] = data[:, keep_indices, :]
            
        return new_coords, new_bodyparts
    
    return coordinates, bodyparts

import cv2
import glob

def detect_fps(video_dir, extension='mp4'):
    """Detect FPS from the first video file found."""
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return 30.0 # Fallback
        
    pattern = os.path.join(video_dir, f"*.{extension}")
    videos = glob.glob(pattern)
    
    if not videos:
        # Try case insensitive
        pattern = os.path.join(video_dir, f"*.{extension.upper()}")
        videos = glob.glob(pattern)
        
    if videos:
        try:
            vid = cv2.VideoCapture(videos[0])
            fps = vid.get(cv2.CAP_PROP_FPS)
            vid.release()
            print(f"Auto-detected FPS: {fps} from {os.path.basename(videos[0])}")
            if fps > 0:
                return fps
        except Exception as e:
            print(f"Error detecting FPS: {e}")
            
    print("Warning: Could not detect FPS, defaulting to 30.0")
    return 30.0

def interpolate_data(coordinates, noise_scale=1.0):
    """
    Linearly interpolate missing data (NaNs) and add noise to prevent singular matrices.
    Using robust filling for empty columns.
    """
    cleaned_coords = {}
    for name, data in coordinates.items():
        # data shape: (T, K, 2)
        T, K, D = data.shape
        # Reshape to (T, K*2)
        flat_data = data.reshape(T, -1)
        df = pd.DataFrame(flat_data)
        
        # Linear Interpolate
        df = df.interpolate(method='linear', limit_direction='both')
        
        # Fill remaining NaNs (edges)
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Check if still NaN (e.g. entire column was NaN)
        if df.isnull().values.any():
            print(f"Warning: {name} has entirely empty bodyparts. Filling with random noise.")
            # Fill NaNs with random noise centered at image center? 
            # Or just user random noise to avoid Singularity.
            # Assuming 0-centered references (KPMS aligns later, but PCA needs data).
            # We use random integers 0-100 to replicate 'movement' noise
            rand_fill = pd.DataFrame(np.random.normal(0, 10, df.shape), index=df.index, columns=df.columns)
            df = df.fillna(rand_fill)
            
        # Enforce minimum variance
        # If a column is constant (e.g. 0), it causes singularity.
        # Add jitter to everything.
        jitter = np.random.normal(0, noise_scale, df.shape)
        vals = df.values + jitter
        
        cleaned_coords[name] = vals.reshape(T, K, D)
        
    return cleaned_coords

def preprocess_data(coordinates, confidences, config, bodyparts=None):
    """
    Format data for KPMS.
    Note: Filtering and Confidence Thresholding have been removed as requested.
    But we DO apply interpolation to prevent cuSolver errors (NaNs).
    """
    # Filter out mostly empty bodyparts/columns BEFORE interpolation
    if bodyparts is not None:
        coordinates, bodyparts = filter_bad_bodyparts(coordinates, bodyparts, threshold=0.99) # Drop if 99% NaN
        print(f"Remaining Bodyparts: {len(bodyparts)}")
    
    # Interpolate to remove NaNs
    print("Applying linear interpolation to fix missing data...")
    coordinates = interpolate_data(coordinates)
    
    # Prepare arguments for kpms.format_data
    kwargs = config['project_config'].copy()
    if bodyparts is not None:
        kwargs['bodyparts'] = bodyparts
        
    formatted_data, metadata = kpms.format_data(
        coordinates,
        confidences=confidences,
        **kwargs
    )
    
    # Explicitly cast to float64 for JAX 64-bit compatibility
    # Handle both nested (multi-recording) and flat (single-recording) formats
    data_to_cast = []
    if all(hasattr(v, 'keys') for v in formatted_data.values()):
        # Nested format: formatted_data[head][key]
        for head in formatted_data:
            for key in formatted_data[head]:
                data_to_cast.append((formatted_data[head], key))
    else:
        # Flat format: formatted_data[key]
        for key in formatted_data:
            data_to_cast.append((formatted_data, key))

    for container, key in data_to_cast:
        val = container[key]
        if hasattr(val, 'dtype') and np.issubdtype(val.dtype, np.floating):
            container[key] = val.astype(np.float64)
    
    return formatted_data, metadata, bodyparts

