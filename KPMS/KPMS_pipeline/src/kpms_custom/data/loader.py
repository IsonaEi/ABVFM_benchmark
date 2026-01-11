import os
import glob
import pandas as pd
import numpy as np
import cv2
from kpms_custom.utils.logger_utils import get_logger

logger = get_logger()

def detect_bodyparts(file_path, conf_threshold=0.5):
    """
    Robustly detect bodyparts from the first available H5 file.
    """
    try:
        df = pd.read_hdf(file_path)
        
        # Check for MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            levels = df.columns.levels
            bp_candidates = []
            
            for lvl_idx, lvl_values in enumerate(levels):
                if 'x' in lvl_values and 'y' in lvl_values:
                    if lvl_idx > 0:
                        bp_level = levels[lvl_idx - 1]
                        bp_candidates = list(bp_level)
                        break
            
            if not bp_candidates:
                for lvl in levels:
                    if 'nose' in lvl or 'snout' in lvl:
                        bp_candidates = list(lvl)
                        break
                        
            bodyparts = [bp for bp in bp_candidates if isinstance(bp, str)]
            logger.info(f"Auto-detected {len(bodyparts)} bodyparts: {bodyparts}")
            return bodyparts
            
    except Exception as e:
        logger.error(f"Error detecting bodyparts: {e}")
        return []

def load_h5_files(config):
    """Load H5 files from data_dir."""
    data_dir = config['data_dir']
    file_pattern = os.path.join(data_dir, '*.h5')
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")
        
    logger.info(f"Found {len(files)} files in {data_dir}")
    return files

def detect_fps(video_dir, extension='mp4'):
    """Detect FPS from the first video file found."""
    if not os.path.exists(video_dir):
        logger.warning(f"Video directory not found: {video_dir}")
        return 30.0
        
    pattern = os.path.join(video_dir, f"*.{extension}")
    videos = glob.glob(pattern)
    
    if not videos:
        pattern = os.path.join(video_dir, f"*.{extension.upper()}")
        videos = glob.glob(pattern)
        
    if videos:
        try:
            vid = cv2.VideoCapture(videos[0])
            fps = vid.get(cv2.CAP_PROP_FPS)
            vid.release()
            logger.info(f"Auto-detected FPS: {fps} from {os.path.basename(videos[0])}")
            if fps > 0:
                return fps
        except Exception as e:
            logger.error(f"Error detecting FPS: {e}")
            
    logger.warning("Could not detect FPS, defaulting to 30.0")
    return 30.0

def interpolate_data(coordinates, noise_scale=1.0):
    """
    Linearly interpolate missing data (NaNs) and add noise.
    """
    cleaned_coords = {}
    for name, data in coordinates.items():
        T, K, D = data.shape
        flat_data = data.reshape(T, -1)
        df = pd.DataFrame(flat_data)
        
        # Linear Interpolate - limit_direction explicitly for edges
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Fallback for empty columns
        if df.isnull().values.any():
            logger.warning(f"{name} contains entirely empty columns. Filling with noise.")
            rand_fill = pd.DataFrame(np.random.normal(0, 10, df.shape), index=df.index, columns=df.columns)
            df = df.fillna(rand_fill)
            
        # Add jitter to avoid singularities in PCA
        jitter = np.random.normal(0, noise_scale, df.shape)
        vals = df.values + jitter
        
        cleaned_coords[name] = vals.reshape(T, K, D)
        
    return cleaned_coords

def filter_bad_bodyparts(coordinates, bodyparts, threshold=0.9):
    """
    Remove bodyparts that are NaN in > threshold fraction of frames.
    """
    all_data = list(coordinates.values())
    if not all_data:
        return coordinates, bodyparts
        
    mega_matrix = np.concatenate(all_data, axis=0) # (Total_T, K, 2)
    nans = np.isnan(mega_matrix).any(axis=2)
    nan_ratio = nans.mean(axis=0) # (K,)
    
    keep_indices = np.where(nan_ratio < threshold)[0]
    drop_indices = np.where(nan_ratio >= threshold)[0]
    
    if len(drop_indices) > 0:
        dropped_names = [bodyparts[i] for i in drop_indices]
        logger.info(f"Dropping bad bodyparts (> {threshold*100}% NaN): {dropped_names}")
        
        new_bodyparts = [bodyparts[i] for i in keep_indices]
        new_coords = {}
        for name, data in coordinates.items():
            new_coords[name] = data[:, keep_indices, :]
            
        return new_coords, new_bodyparts
    
    return coordinates, bodyparts

def parse_dlc_data(files, config):
    """
    Parse H5 files into coordinates/confidences dictionaries.
    """
    coordinates = {}
    confidences = {}
    
    cfg_bodyparts = config.get('project_config', {}).get('bodyparts', 'AUTO')
    
    if isinstance(cfg_bodyparts, list) and len(cfg_bodyparts) > 0:
        bodyparts = cfg_bodyparts
    elif files:
        bodyparts = detect_bodyparts(files[0])
    else:
        bodyparts = []
    
    for f in files:
        name = os.path.basename(f).replace('.h5', '')
        try:
            df = pd.read_hdf(f)
            
            # Robust extraction logic (matches original preprocess.py)
            if isinstance(df.columns, pd.MultiIndex):
                if df.columns.nlevels >= 4:
                    individuals = df.columns.levels[1]
                    indiv = individuals[0] 
                    use_df = df.xs(indiv, level=1, axis=1)
                else:
                    use_df = df

                coords_list = []
                confs_list = []
            
                bp_level_idx = 0
                if bodyparts:
                    test_bp = bodyparts[0]
                    for i in range(use_df.columns.nlevels):
                        if test_bp in use_df.columns.get_level_values(i):
                            bp_level_idx = i
                            break
                
                for bp in bodyparts:
                    if bp in use_df.columns.get_level_values(bp_level_idx):
                        try:
                            bp_data = use_df.xs(bp, level=bp_level_idx, axis=1)
                            x = bp_data['x'].values if 'x' in bp_data else bp_data.iloc[:, 0].values
                            y = bp_data['y'].values if 'y' in bp_data else bp_data.iloc[:, 1].values
                            c = bp_data['likelihood'].values if 'likelihood' in bp_data else (bp_data.iloc[:, 2].values if bp_data.shape[1]>2 else np.zeros(len(bp_data)))
                        except:
                            x = np.full(len(use_df), np.nan)
                            y = np.full(len(use_df), np.nan)
                            c = np.zeros(len(use_df))
                    else:
                        n_frames = len(use_df)
                        x = np.full(n_frames, np.nan)
                        y = np.full(n_frames, np.nan)
                        c = np.zeros(n_frames)
                        
                    coords_list.append(np.stack([x, y], axis=1))
                    confs_list.append(c)

                if coords_list:
                    coordinates[name] = np.stack(coords_list, axis=1) # (T, K, 2)
                    confidences[name] = np.stack(confs_list, axis=1)
                else:
                    logger.warning(f"Skipping {name}: No bodyparts found.")
            else:
                logger.warning(f"Skipping {name}: Unexpected dataframe format")
                
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")
            
    return coordinates, confidences, bodyparts
