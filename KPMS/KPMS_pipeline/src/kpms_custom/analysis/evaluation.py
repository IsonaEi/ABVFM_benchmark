import numpy as np
import keypoint_moseq as kpms
import matplotlib.pyplot as plt
from pathlib import Path
from kpms_custom.utils.logger_utils import get_logger
from kpms_custom.utils.compat import patch_matplotlib_compatibility

logger = get_logger()
patch_matplotlib_compatibility()

def evaluate_models(project_dir, model_name_pattern=None):
    """
    Evaluates models using Expected Marginal Likelihood (EML).
    Returns best model name.
    """
    project_path = Path(project_dir)
    
    if model_name_pattern:
        candidates = list(project_path.glob(model_name_pattern))
    else:
        candidates = [d for d in project_path.iterdir() if d.is_dir() 
                      and (d / "checkpoint.h5").exists()
                      and not d.name.startswith(('pca', 'results', 'kappa', 'warmup'))]
    
    model_names = sorted([d.name for d in candidates])
    if not model_names:
        logger.error("No models found for evaluation.")
        return None
        
    logger.info(f"Evaluating {len(model_names)} models: {model_names}")
    
    try:
        scores, std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
        
        # Plot
        fig, ax = kpms.plot_eml_scores(scores, std_errs, model_names)
        fig.savefig(project_path / "model_comparison_eml.png")
        plt.close(fig)
        
        best_idx = np.argmax(scores)
        best_model = model_names[best_idx]
        logger.info(f"Best Model: {best_model} (Score: {scores[best_idx]:.4f})")
        
        return best_model
        
    except Exception as e:
        logger.error(f"Error checking EML: {e}")
        return None
