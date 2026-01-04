
import os
import numpy as np
import keypoint_moseq as kpms
import matplotlib.pyplot as plt

project_dir = "/home/isonaei/ABVFM/KPMS/results/010226_02_gemini_open_field"

# Find 5 models
all_dirs = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
model_names = [d for d in all_dirs if not d.startswith("kappa_") and not d.startswith("warmup_") and not d.startswith(".")]
model_names = sorted(model_names)

print(f"Evaluating models: {model_names}")

try:
    scores, std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
    print("\n--- EML Scores ---")
    for name, score, err in zip(model_names, scores, std_errs):
        print(f"Model: {name} | Score: {score:.4f} | StdErr: {err:.4f}")
        
    # Check data range
    print(f"\nMax Score: {np.max(scores)}")
    print(f"Min Score: {np.min(scores)}")
    print(f"Range: {np.max(scores) - np.min(scores)}")
    
except Exception as e:
    print(f"Error: {e}")
