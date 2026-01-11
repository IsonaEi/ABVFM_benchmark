import numpy as np
import copy
import keypoint_moseq as kpms
from kpms_custom.utils.logger_utils import get_logger
from kpms_custom.model.trainer import fit_model
from kpms_custom.data.preprocessor import calculate_latent_dim

logger = get_logger()

def get_median_duration(model, fps):
    """Calculate median syllable duration in ms."""
    z = np.array(model['states']['z']) 
    if z.ndim > 1:
        z = z.flatten()
        
    # Pad to separate runs
    z_padded = np.concatenate(([z[0]-1], z, [z[-1]-1]))
    changes = np.where(z_padded[:-1] != z_padded[1:])[0]
    lengths = np.diff(changes)
    
    median_frames = np.median(lengths)
    return (median_frames / fps) * 1000

def bisection_search(objective_fn, target, bounds, tolerance=0.1, max_iter=10):
    """
    Log-scale bisection search.
    bounds: (lower, upper)
    """
    lower_bound, upper_bound = bounds
    history = []
    
    # Check Bounds
    dur_low = objective_fn(lower_bound)
    history.append({'kappa': lower_bound, 'duration': dur_low, 'step': 'bound_low'})
    if abs(dur_low - target)/target <= tolerance: return lower_bound, dur_low, history
    
    dur_high = objective_fn(upper_bound)
    history.append({'kappa': upper_bound, 'duration': dur_high, 'step': 'bound_high'})
    if abs(dur_high - target)/target <= tolerance: return upper_bound, dur_high, history
    
    # Search
    low_log, high_log = np.log10(lower_bound), np.log10(upper_bound)
    best_kappa, best_dur, min_err = None, float('inf'), float('inf')
    
    for i in range(max_iter):
        mid_log = (low_log + high_log) / 2
        mid_kappa = 10 ** mid_log
        
        logger.info(f"Bisection {i+1}: Testing Kappa={mid_kappa:.2e}")
        duration = objective_fn(mid_kappa)
        err = abs(duration - target) / target
        
        history.append({'kappa': mid_kappa, 'duration': duration, 'step': f'iter_{i+1}'})
        logger.info(f"  Result: {duration:.1f}ms (Err: {err:.1%})")
        
        if err < min_err:
            min_err, best_kappa, best_dur = err, mid_kappa, duration
            
        if err <= tolerance:
            logger.info("Converged!")
            return mid_kappa, duration, history
            
        if duration < target:
            # Too short -> Need Higher Kappa (usually?)
            # Wait, relationships: High Kappa = High Stiffness = Longer Duration? Yes.
            # Low duration -> need higher kappa -> raise lower bound
            low_log = mid_log
        else:
            high_log = mid_log
            
    return best_kappa, best_dur, history

def scan_kappa(data, metadata, pca, config, project_dir, scan_type='ar'):
    """
    Run Kappa Scan.
    scan_type: 'ar' or 'full'.
    """
    fps = config['preprocess']['fps']
    target_dur = config['tuning']['target_motif_duration']
    tolerance = config['tuning']['tolerance']
    scan_cfg = config['model_params']['kappa_scan']
    
    # 1. Bounds
    k_min = float(scan_cfg['min'])
    k_max = float(scan_cfg['max'])
    
    iters = scan_cfg['ar_iters'] if scan_type == 'ar' else scan_cfg['full_iters']
    
    # 2. Objective Function
    def objective(kappa):
        # We use a temporary simple model fit
        # To avoid overhead, we use fit_model but pass args manually
        # and suppress heavy logging if possible (via logger level?)
        
        # We need to construct a temporary config for this specific run
        temp_config = copy.deepcopy(config)
        
        if scan_type == 'ar':
            temp_config['tuning']['ar_kappa'] = kappa
            model, _ = fit_model(data, metadata, pca, temp_config, project_dir, 
                                 name_suffix=f"_scan_{kappa:.1e}", ar_only=True, num_iters_override=iters)
        else:
             # Full Scan needs AR warmup?
             # Assuming Warmup Model is passed or handled?
             # For simplicity, we run fresh fits but fast.
             # Ideally we clone a warmup model.
             # Implementing basic fresh fit for robustness.
             temp_config['tuning']['full_kappa'] = kappa
             # Ensure we use an existing AR kappa if found?
             # The user usually ran AR scan first.
             model, _ = fit_model(data, metadata, pca, temp_config, project_dir, 
                                  name_suffix=f"_scan_{kappa:.1e}", ar_only=False, num_iters_override=iters)
                                  
        return get_median_duration(model, fps)

    # 3. Adaptive Bounds (Function is monotonic generally)
    # If full scan, we might want to check bounds first.
    # Bisection handles checking bounds.
    
    # 4. Run
    logger.info(f"Starting {scan_type.upper()} Kappa Scan. Target: {target_dur}ms")
    best_kappa, best_dur, history = bisection_search(objective, target_dur, (k_min, k_max), tolerance)
    
    return best_kappa, best_dur
