import keypoint_moseq as kpms
import jax_moseq
import os
import yaml
import numpy as np
from datetime import datetime

def setup_project(config):
    """Initialize the project directory and config."""
    project_dir = config['project_dir']
    if not os.path.exists(os.path.join(project_dir, 'config.yml')):
        # Create new project
        kpms.setup_project(
            project_dir, 
            deeplabcut_config=None, # We are loading data manually
            overwrite=True
        )
        # We need to write our parameters into the generated config.yml
        # OR simply load the generated one and update it with ours.
        # For simplicity, we assume we control the config via arguments to fit_model
    
    return project_dir

def infer_heading_indices(bodyparts):
    """
    Infer anterior_idxs and posterior_idxs from bodypart names.
    Returns:
        anterior_idxs (list of int)
        posterior_idxs (list of int)
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
            
    # Fallback if detection fails
    if not ant_idxs:
        print("Warning: Could not infer anterior bodyparts. using index 0.")
        ant_idxs = [0]
    if not post_idxs:
        print("Warning: Could not infer posterior bodyparts. using last index.")
        post_idxs = [len(bodyparts) - 1]
        
    print(f"Auto-inferred Heading: Anterior={ant_idxs} ({[bodyparts[i] for i in ant_idxs]}), Posterior={post_idxs} ({[bodyparts[i] for i in post_idxs]})")
    return ant_idxs, post_idxs

def train_pca(data, config, project_dir, bodyparts=None):
    """Fit PCA."""
    model_params = config.get('model_params', {})
    
    # Auto-infer indices if not present in config
    ant_idxs = model_params.get('anterior_idxs')
    post_idxs = model_params.get('posterior_idxs')
    
    if (ant_idxs is None or post_idxs is None) and bodyparts:
        inf_ant, inf_post = infer_heading_indices(bodyparts)
        if ant_idxs is None: ant_idxs = inf_ant
        if post_idxs is None: post_idxs = inf_post
    
    print(f"PCA using Anterior: {ant_idxs}, Posterior: {post_idxs}")
    
    pca = kpms.fit_pca(
        data['Y'],
        data['mask'],
        project_dir=project_dir,
        anterior_idxs=ant_idxs,
        posterior_idxs=post_idxs,
        **config.get('pca', {})
    )
    return pca

def calculate_latent_dim(pca, target_variance=0.9):
    """
    Calculate the number of components needed to reach target variance.
    """
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    
    # If target is integer > 1, return it directly (user specified count)
    if target_variance > 1:
        return int(target_variance)
        
    # Find index where cumulative variance >= target
    n_components = np.argmax(explained_var >= target_variance) + 1
    print(f"Dynamic Latent Dim: {n_components} components explain {explained_var[n_components-1]*100:.1f}% variance (Target: {target_variance*100:.0f}%)")
    
    return int(n_components)

def fit_model_wrapper(data, metadata, pca, config, project_dir, ar_only=False, name_suffix=""):
    """Fit the MOSEQ model (AR-HMM + Transitions)."""
    
    # Load default config from project to get a base model dict
    kpms_config = kpms.load_config(project_dir)
    
    # Check for latent_dim override in config
    # Extract heading indices from project config if they exist
    ant_idxs = kpms_config.get('anterior_idxs', None)
    post_idxs = kpms_config.get('posterior_idxs', None)
    
    # Cast to standard numpy ints to avoid JAX recon issues
    if ant_idxs is not None: ant_idxs = np.array(ant_idxs).astype(np.int64)
    if post_idxs is not None: post_idxs = np.array(post_idxs).astype(np.int64)

    # Use kpms_config but remove jax-specific reconstructed items
    clean_config = {curr_k: curr_v for curr_k, curr_v in kpms_config.items() 
                    if not str(type(curr_v)).startswith("<class 'jax")}
    
    # Initialize model
    target_dim = config.get('model_params', {}).get('latent_dim', 10)
    latent_dim = calculate_latent_dim(pca, target_dim)
    
    # Extract seed from config (populated by run_pipeline loop)
    # Default to random if not set, or 0.
    seed = config.get('seed', np.random.randint(0, 10000))

    model = kpms.init_model(
        data=data,
        pca=pca,
        latent_dim=latent_dim, 
        anterior_idxs=ant_idxs,
        posterior_idxs=post_idxs,
        seed=seed,
        **clean_config
    )
    
    # Model name: Updated timestamp format to YYYYMMDD-HHMM
    model_name = datetime.now().strftime("%Y%m%d-%H%M") + name_suffix
    
    num_iters = config.get('model_params', {}).get('num_iters', 100) # Default 100
    
    if ar_only:
        # AR-Only Mode (e.g. for Scan or if requested)
        print(f"Fitting Model: {model_name} (AR-Only: True, Iters: {num_iters})")
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir, model_name,
            ar_only=True, num_iters=num_iters
        )
    else:
        # Train Mode: Two-Stage Training (AR -> Full)
        # 1. Get Hyperparameters
        ar_iters = config.get('model_params', {}).get('ar_warmup_iters', 50)
        full_iters = num_iters
        
        # Check if we should reduce kappa for full fit
        tuning_cfg = config.get('tuning', {})
        explicit_full_kappa = tuning_cfg.get('full_kappa', None)
        explicit_ar_kappa = tuning_cfg.get('ar_kappa', None)
        
        # Fix 1: Apply AR Kappa BEFORE Warmup if available
        if explicit_ar_kappa is not None:
             ar_kappa = float(explicit_ar_kappa)
             print(f"Using Explicit AR Kappa for Warmup: {ar_kappa:.2e}")
             model = kpms.update_hypparams(model, kappa=float(ar_kappa))
        
        full_kappa = None
        
        if explicit_full_kappa is not None:
             full_kappa = float(explicit_full_kappa)
             print(f"Using Explicit Full Kappa: {full_kappa:.2e}")
        else:
            # Fallback to ratio logic
            full_kappa_ratio = tuning_cfg.get('full_kappa_ratio', 1.0)
            
            # Find current kappa in model structure
            current_kappa = 1e6
            if 'hypparams' in model and 'trans_hypparams' in model['hypparams']:
                current_kappa = float(model['hypparams']['trans_hypparams'].get('kappa', 1e6))
            
            full_kappa = current_kappa * full_kappa_ratio
            if full_kappa_ratio != 1.0:
                 print(f"Using calculated Full Kappa (Ratio {full_kappa_ratio}): {full_kappa:.2e}")

        # Find current kappa for display
        current_kappa_disp = 1e6
        if 'hypparams' in model and 'trans_hypparams' in model['hypparams']:
             current_kappa_disp = float(model['hypparams']['trans_hypparams'].get('kappa', 1e6))
        
        print(f"--- Two-Stage Training: {model_name} ---")
        print(f"Stage 1: AR-Only Warmup (Iters: {ar_iters}, Kappa: {current_kappa_disp:.1e})")
        
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir, model_name,
            ar_only=True, num_iters=ar_iters
        )
        
        total_target_iters = ar_iters + full_iters
        print(f"Stage 2: Full Model Fit (Iters: {full_iters}, Total Target: {total_target_iters}, Kappa: {full_kappa:.1e})")
        
        # Apply Kappa (Always if explicit, or if ratio != 1)
        if full_kappa is not None:
            model = kpms.update_hypparams(model, kappa=float(full_kappa))
            
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir, model_name,
            ar_only=False, start_iter=ar_iters, num_iters=total_target_iters
        )
    
    return model, model_name

def get_median_duration(model, config, fps=None):
    """Calculate median syllable duration (in ms) from model states."""
    # Use detected FPS if passed, else config value
    if fps is None:
        fps = config['preprocess'].get('fps', 30.0)
        
    z = np.array(model['states']['z']) 
    
    if z.ndim > 1:
        z = z.flatten()
        
    z_padded = np.concatenate(([z[0]-1], z, [z[-1]-1]))
    changes = np.where(z_padded[:-1] != z_padded[1:])[0]
    lengths = np.diff(changes)
    
    median_frames = np.median(lengths)
    median_ms = (median_frames / fps) * 1000
    return median_ms

def bisection_search(objective_fn, target, lower_bound, upper_bound, tolerance=0.1, max_iter=10):
    """
    Perform Log-Scale Bisection Search for a monotonically increasing function (Kappa vs Duration).
    """
    history = []
    
    # 1. Check Bounds
    print(f"Checking Bounds: [{lower_bound:.2e}, {upper_bound:.2e}]")
    
    # Check Lower Bound
    dur_low = objective_fn(lower_bound)
    diff_low = dur_low - target
    abs_err_low = abs(diff_low) / target
    history.append({'kappa': lower_bound, 'duration': dur_low, 'step': 'bound_low'})
    
    if abs_err_low <= tolerance:
        print(f"Converged at Lower Bound! Error {abs_err_low:.1%} <= Tolerance {tolerance:.1%}")
        return lower_bound, dur_low, history
        
    if dur_low > target * (1 + tolerance):
        print(f"Error: Lower bound duration ({dur_low:.1f}ms) is already too high (Target: {target}ms). Needed lower kappa.")
        return lower_bound, dur_low, history

    # Check Upper Bound
    dur_high = objective_fn(upper_bound)
    diff_high = dur_high - target
    abs_err_high = abs(diff_high) / target
    history.append({'kappa': upper_bound, 'duration': dur_high, 'step': 'bound_high'})
    
    if abs_err_high <= tolerance:
        print(f"Converged at Upper Bound! Error {abs_err_high:.1%} <= Tolerance {tolerance:.1%}")
        return upper_bound, dur_high, history
        
    if dur_high < target * (1 - tolerance):
        print(f"Error: Upper bound duration ({dur_high:.1f}ms) is still too low (Target: {target}ms). Needed higher kappa.")
        return upper_bound, dur_high, history

    low_log = np.log10(lower_bound)
    high_log = np.log10(upper_bound)
    
    best_kappa = None
    best_dur = float('inf')
    min_error = float('inf')
    
    for i in range(max_iter):
        mid_log = (low_log + high_log) / 2
        mid_kappa = 10 ** mid_log
        
        print(f"\n--- Bisection Step {i+1}/{max_iter} ---")
        print(f"Search Range: [{10**low_log:.2e}, {10**high_log:.2e}] -> Testing Mid: {mid_kappa:.2e}")
        
        duration = objective_fn(mid_kappa)
        diff = duration - target
        abs_relative_error = abs(diff) / target
        
        history.append({'kappa': mid_kappa, 'duration': duration, 'step': f'iter_{i+1}'})
        
        # Track Best
        if abs_relative_error < min_error:
            min_error = abs_relative_error
            best_kappa = mid_kappa
            best_dur = duration
        
        print(f"  Result: Kappa={mid_kappa:.2e}, Duration={duration:.1f}ms (Target: {target}ms, Error: {abs_relative_error:.1%})")
        
        # Check Convergence
        if abs_relative_error <= tolerance:
            print(f"Converged! Error {abs_relative_error:.1%} <= Tolerance {tolerance:.1%}")
            return mid_kappa, duration, history
        
        # Update Bounds (Expect Monotonic Increase: Kappa up -> Duration up)
        if diff < 0:
            # Duration too short -> Need higher Kappa -> Move Low up
            low_log = mid_log
            print(f"  Too Short -> Increasing Lower Bound")
        else:
            # Duration too long -> Need lower Kappa -> Move High down
            high_log = mid_log
            print(f"  Too Long -> Decreasing Upper Bound")
            
    print(f"Max iterations reached. Returning best found: {best_kappa:.2e}")
    return best_kappa, best_dur, history

def scan_kappa(data, metadata, pca, config, project_dir, fps=None, scan_type='ar'):
    """
    Perform Optimized Kappa Scan using Log-Scale Bisection.
    scan_type: 'ar' (AR-Only) or 'full' (Transitions)
    """
    import copy
    
    # 1. Setup Base Params
    tuning_cfg = config.get('tuning', {})
    target_duration = tuning_cfg.get('target_motif_duration', 400)
    tolerance = tuning_cfg.get('tolerance', 0.1)
    
    scan_cfg = config.get('model_params', {}).get('kappa_scan', {})
    if scan_type == 'ar':
        num_iters = scan_cfg.get('ar_iters', 20)
    else:
        num_iters = scan_cfg.get('full_iters', 50)

    ar_warmup_iters = config.get('model_params', {}).get('ar_warmup_iters', 10)
    target_dim = config.get('model_params', {}).get('latent_dim', 10)
    latent_dim = calculate_latent_dim(pca, target_dim)
    
    kpms_config = kpms.load_config(project_dir)
    
    # Extract heading indices from project config if they exist
    ant_idxs = kpms_config.get('anterior_idxs', None)
    post_idxs = kpms_config.get('posterior_idxs', None)
    
    if ant_idxs is not None: ant_idxs = np.array(ant_idxs).astype(np.int64)
    if post_idxs is not None: post_idxs = np.array(post_idxs).astype(np.int64)

    clean_config = {curr_k: curr_v for curr_k, curr_v in kpms_config.items() 
                    if not str(type(curr_v)).startswith("<class 'jax")}

    # 2. Define Scan Bounds
    scan_cfg = config.get('model_params', {}).get('kappa_scan', {})
    if scan_type == 'ar':
        k_min = float(scan_cfg.get('min', 1e3))
        k_max = float(scan_cfg.get('max', 1e24))
    else: # scan_type == 'full'
        # 1. Run AR Scan First (Recursive)
        print("\n=== Prerequisite: Running AR-Only Kappa Scan ===")
        _, ar_result = scan_kappa(data, metadata, pca, config, project_dir, fps=fps, scan_type='ar')
        
        ar_kappa = 1e7
        try:
            # Parse result name expected format: optimized_ar_1.23e+05
            ar_kappa = float(ar_result.split('_')[-1])
            print(f"=== Prerequisite Complete. Best AR Kappa: {ar_kappa:.2e} ===")
        except Exception as e:
            print(f"Warning: Could not parse AR scan result '{ar_result}'. Defaulting to 1e7. Error: {e}")
            
        print(f"\n=== Starting Full Model Kappa Scan ===")
        print(f"Base AR Kappa: {ar_kappa:.2e}")
        
        # Adaptive Range Search
        # Fix: Reduce range from 10000x to 100x as requested
        base_kappa = ar_kappa * 0.1
        k_min = base_kappa / 100.0
        k_max = base_kappa * 100.0
        print(f"Initial Scan Range: {k_min:.2e} ~ {k_max:.2e} (Base: {base_kappa:.2e})")
        
    # 3. Pre-Warmup (for Full Scan Only)
    warmup_model = None
    if scan_type == 'full':
        print(f"--- Pre-Fitting AR Warmup Model (Iters: {ar_warmup_iters}) ---")
        warmup_model = kpms.init_model(
            data=data, pca=pca, latent_dim=latent_dim, 
            anterior_idxs=ant_idxs, posterior_idxs=post_idxs, **clean_config
        )
        
        # Verify: Should we apply AR Kappa here?
        # Yes, technically the warmup for the scan should also respect the finding from AR scan if possible
        # But wait, AR scan is finding kappa for Duration, not necessarily likelihood.
        # But 'scan_kappa' function above extracts 'ar_kappa' from recursive call.
        if 'ar_kappa' in locals():
            print(f"Applying AR Kappa to Warmup: {ar_kappa:.2e}")
            warmup_model = kpms.update_hypparams(warmup_model, kappa=ar_kappa)
            
        warmup_model, _ = kpms.fit_model(
            warmup_model, data, metadata, project_dir=project_dir,
            model_name="warmup_ar", ar_only=True, num_iters=ar_warmup_iters, verbose=False
        )
        print("Warmup Complete.")

    # 4. Define Objective Function
    def evaluate_model(kappa):
        # Silence logs, only show result
        model_name = f"kappa_{scan_type}_{kappa:.2e}_{datetime.now().strftime('%Y%m%d-%H%M')}"
        
        if scan_type == 'ar':
            model = kpms.init_model(
                data=data, pca=pca, latent_dim=latent_dim, 
                anterior_idxs=ant_idxs, posterior_idxs=post_idxs, **clean_config
            )
            model = kpms.update_hypparams(model, kappa=kappa)
            model, _ = kpms.fit_model(
                model, data, metadata, project_dir=project_dir,
                model_name=model_name, ar_only=True, num_iters=num_iters, verbose=False
            )
        else:
            # Clone warmup model for independent runs
            model_in = copy.deepcopy(warmup_model)
            model_in = kpms.update_hypparams(model_in, kappa=kappa)
            
            # Train Full (Transitions)
            model, _ = kpms.fit_model(
                model_in, data, metadata, project_dir=project_dir,
                model_name=model_name, ar_only=False, 
                start_iter=ar_warmup_iters, num_iters=ar_warmup_iters+num_iters, 
                verbose=False
            )
            
            
        # Clean up any figures to prevent memory leak
        import matplotlib.pyplot as plt
        plt.close('all')
            
        return get_median_duration(model, config, fps)

    # 5. Adaptive Bound Expansion (Full Scan Only)
    if scan_type == 'full':
        print("\n--- Verifying Bounds ---")
        max_expansions = 5
        
        # Initialize durations
        dur_low = None
        dur_high = None

        for i in range(max_expansions):
            print(f"Checking Bounds [{k_min:.2e}, {k_max:.2e}]...")
            
            # Evaluate bounds (only if not already computed)
            if dur_low is None:
                dur_low = evaluate_model(k_min)
            if dur_high is None:
                dur_high = evaluate_model(k_max)
            
            print(f"  Low (K={k_min:.2e}): {dur_low:.1f}ms")
            print(f"  High (K={k_max:.2e}): {dur_high:.1f}ms")
            
            # Check for convergence at bounds
            diff_low = dur_low - target_duration
            if abs(diff_low)/target_duration <= tolerance:
                print(f"Converged at Lower Bound! Returning.")
                return [], f"optimized_{scan_type}_{k_min:.2e}"
                
            diff_high = dur_high - target_duration
            if abs(diff_high)/target_duration <= tolerance:
                print(f"Converged at Upper Bound! Returning.")
                return [], f"optimized_{scan_type}_{k_max:.2e}"
            
            # Check Enclosure
            # We need Low < Target < High (Duration increases with Kappa)
            
            expand = False
            if dur_high < target_duration:
                # Upper too low, need even higher kappa
                print(f"  Upper bound too low ({dur_high:.1f} < {target_duration}). Increasing Max x10.") # reduced from 100
                k_max *= 10.0
                dur_high = None # Force re-evaluation
                expand = True
            
            if dur_low > target_duration:
                # Lower too high, need even lower kappa
                print(f"  Lower bound too high ({dur_low:.1f} > {target_duration}). Decreasing Min / 10.") # reduced from 100
                k_min /= 10.0
                dur_low = None # Force re-evaluation
                expand = True
                
            if not expand:
                print("Target duration enclosed. Starting Bisection.")
                break
        else:
            print("Warning: Max expansions reached. Proceeding with best effort.")

    # 6. Run Bisection Search
    best_kappa, best_dur, history = bisection_search(
        evaluate_model, target_duration, k_min, k_max, tolerance=tolerance
    )
    
    print("\n--- Final Optimization Results ---")
    for step in history:
        print(f"{step['step']}: K={step['kappa']:.2e}, Dur={step['duration']:.1f}ms")
        
    print(f"\nGlobal Best: Kappa={best_kappa:.2e}, Duration={best_dur:.1f}ms")
    
    # Return formatted string containing best kappa
    # For AR scan, we specifically want to distinguish it so it can be parsed
    result_str = f"optimized_{scan_type}_{best_kappa:.2e}"
    
    # If this is AR scan calling itself recursively, we are done.
    # But if this is the final return of a 'full' scan, we might want to return AR kappa too?
    # No, run_pipeline calls them separately or handles the string.
    # But wait, if scan_type='full', we found AR kappa inside this function. 
    # The return value is just the BEST FULL kappa.
    # To return AR kappa as well, we might need a composite string or struct.
    # Let's append AR kappa to the string if it's a full scan
    if scan_type == 'full' and 'ar_kappa' in locals():
         result_str += f"_AR{ar_kappa:.2e}"
    
    return history, result_str
