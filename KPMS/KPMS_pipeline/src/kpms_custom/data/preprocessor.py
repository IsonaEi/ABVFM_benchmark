import numpy as np
import keypoint_moseq as kpms
from kpms_custom.utils.logger_utils import get_logger
from kpms_custom.data.loader import filter_bad_bodyparts, interpolate_data

logger = get_logger()

def infer_heading_indices(bodyparts):
    """
    Infer anterior_idxs and posterior_idxs from bodypart names.
    """
    anterior_keywords = ['nose', 'snout', 'ear', 'head', 'ant']
    posterior_keywords = ['tail', 'root', 'hip', 'center', 'post', 'base']
    
    ant_idxs = []
    post_idxs = []
    
    for idx, bp in enumerate(bodyparts):
        bp_lower = bp.lower()
        if any(k in bp_lower for k in anterior_keywords):
            ant_idxs.append(idx)
        if any(k in bp_lower for k in posterior_keywords):
            post_idxs.append(idx)
            
    if not ant_idxs:
        logger.warning("Could not infer anterior bodyparts. using index 0.")
        ant_idxs = [0]
    if not post_idxs:
        logger.warning("Could not infer posterior bodyparts. using last index.")
        post_idxs = [len(bodyparts) - 1]
        
    logger.info(f"Auto-inferred Heading: Anterior={ant_idxs}, Posterior={post_idxs}")
    return ant_idxs, post_idxs

def calculate_latent_dim(pca, target_variance=0.9):
    """
    Calculate number of components using target variance or explicit count.
    """
    # 1. Variance-based count
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    if target_variance > 1:
        # User specified explicit integer count
        var_dim = int(target_variance)
    else:
        var_dim = np.argmax(explained_var >= target_variance) + 1
        
    final_dim = var_dim
    
    logger.info(f"Latent Dim Calculation:")
    logger.info(f"  - Target Variance ({target_variance}): {var_dim} components")
    logger.info(f"  > Final Selection: {final_dim}")
    
    return int(final_dim)

def train_pca(data, config, project_dir, bodyparts=None):
    """Fit PCA."""
    model_params = config.get('model_params', {})
    
    ant_idxs = model_params.get('anterior_idxs')
    post_idxs = model_params.get('posterior_idxs')
    
    if (ant_idxs is None or post_idxs is None) and bodyparts:
        inf_ant, inf_post = infer_heading_indices(bodyparts)
        if ant_idxs is None: ant_idxs = inf_ant
        if post_idxs is None: post_idxs = inf_post
    
    logger.info(f"PCA using Anterior: {ant_idxs}, Posterior: {post_idxs}")
    
    pca = kpms.fit_pca(
        data['Y'],
        data['mask'],
        project_dir=project_dir,
        anterior_idxs=ant_idxs,
        posterior_idxs=post_idxs,
        **config.get('pca', {})
    )
    return pca

def prepare_for_kpms(coordinates, confidences, config, bodyparts):
    """
    Interpolate, format, and cast data for KPMS.
    """
    # 1. Filter Bad Bodyparts
    coordinates, bodyparts = filter_bad_bodyparts(coordinates, bodyparts, threshold=0.99)
    
    # 2. Interpolate
    logger.info("Applying linear interpolation...")
    coordinates = interpolate_data(coordinates)
    
    # 3. Format using KPMS utility
    kwargs = config['project_config'].copy()
    if bodyparts is not None:
        kwargs['bodyparts'] = bodyparts
        
    formatted_data, metadata = kpms.format_data(
        coordinates,
        confidences=confidences,
        **kwargs
    )
    
    # 4. Cast to float64 (JAX requirement)
    # Handle flat or nested dicts
    data_to_cast = []
    if all(hasattr(v, 'keys') for v in formatted_data.values()):
         # Nested
        for head in formatted_data:
            for key in formatted_data[head]:
                data_to_cast.append((formatted_data[head], key))
    else:
        # Flat
        for key in formatted_data:
            data_to_cast.append((formatted_data, key))

    for container, key in data_to_cast:
        val = container[key]
        if hasattr(val, 'dtype') and np.issubdtype(val.dtype, np.floating):
            container[key] = val.astype(np.float64)
            
    return formatted_data, metadata, bodyparts
