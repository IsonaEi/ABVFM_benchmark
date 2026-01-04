import os
import shutil
import numpy as np
from datetime import datetime
import keypoint_moseq as kpms
from kpms_custom.utils.logging import get_logger
from kpms_custom.data.preprocessor import calculate_latent_dim

logger = get_logger()

def setup_project(config):
    """Initialize the project directory."""
    project_dir = config['project_dir']
    
    # If config says overwrite? Usually we check if it exists.
    # We rely on KPMS execution.
    if not os.path.exists(os.path.join(project_dir, 'config.yml')):
        logger.info(f"Initializing new KPMS project at {project_dir}")
        kpms.setup_project(
            project_dir, 
            deeplabcut_config=None,
            overwrite=True
        )
    return project_dir

def fit_model(data, metadata, pca, config, project_dir, name_suffix="", ar_only=False, num_iters_override=None):
    """
    Fits the MOSEQ model (AR-HMM + Transitions).
    Supports Two-Stage training (AR Warmup -> Full Fit).
    """
    # Load default config from project
    kpms_config = kpms.load_config(project_dir)
    
    # Clean config (remove JAX objects)
    clean_config = {k: v for k, v in kpms_config.items() 
                    if not str(type(v)).startswith("<class 'jax")}
                    
    # Setup Dimensions
    target_dim = config.get('model_params', {}).get('latent_dim', 0.9)
    latent_dim = calculate_latent_dim(pca, target_dim)
    
    # Config Heading Indices
    ant_idxs = kpms_config.get('anterior_idxs')
    post_idxs = kpms_config.get('posterior_idxs')
    if ant_idxs is not None: ant_idxs = np.array(ant_idxs).astype(np.int64)
    if post_idxs is not None: post_idxs = np.array(post_idxs).astype(np.int64)
    
    # Seed
    seed = config.get('model_params', {}).get('seed', 42)
    # If name_suffix implies index (e.g. -0), add to seed for diversity
    if '-' in name_suffix:
        try:
            # simple heuristic: suffix "-0" -> add 0
            idx = int(name_suffix.strip('-'))
            seed += idx
        except:
            pass

    # Initialize
    model = kpms.init_model(
        data=data,
        pca=pca,
        latent_dim=latent_dim, 
        anterior_idxs=ant_idxs,
        posterior_idxs=post_idxs,
        seed=seed,
        **clean_config
    )
    
    model_name = datetime.now().strftime("%Y%m%d-%H%M") + name_suffix
    
    # Determine Iterations
    default_iters = config.get('model_params', {}).get('num_iters', 100)
    iters = num_iters_override if num_iters_override else default_iters
    
    if ar_only:
        # Single Stage AR-Only
        logger.info(f"Training Model: {model_name} (AR-Only, Iters: {iters})")
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir, model_name,
            ar_only=True, num_iters=iters
        )
    else:
        # Two-Stage Training
        ar_warmup = config.get('model_params', {}).get('ar_warmup_iters', 50)
        
        # Tuning Params
        tuning_cfg = config.get('tuning', {})
        ar_kappa = tuning_cfg.get('ar_kappa')
        full_kappa = tuning_cfg.get('full_kappa')
        
        # Stage 1: AR Warmup
        if ar_kappa:
            logger.info(f"Applying AR Kappa: {ar_kappa:.2e}")
            model = kpms.update_hypparams(model, kappa=float(ar_kappa))
            
        logger.info(f"Stage 1: AR Warmup ({ar_warmup} iters)")
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir, model_name,
            ar_only=True, num_iters=ar_warmup
        )
        
        # Stage 2: Full Fit
        if full_kappa:
            logger.info(f"Applying Full Kappa: {full_kappa:.2e}")
            model = kpms.update_hypparams(model, kappa=float(full_kappa))
            
        total_iters = ar_warmup + iters
        logger.info(f"Stage 2: Full Fit ({iters} iters, Total: {total_iters})")
        
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir, model_name,
            ar_only=False, start_iter=ar_warmup, num_iters=total_iters
        )
        
    return model, model_name
