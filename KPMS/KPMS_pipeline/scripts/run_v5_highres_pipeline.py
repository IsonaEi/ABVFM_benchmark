
import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append("/home/isonaei/ABVFM_benchmark/KPMS/KPMS_pipeline/src")
from kpms_custom.utils.logger_utils import setup_logger, get_logger

def run_step(name, cmd, cwd="/home/isonaei/ABVFM_benchmark/KPMS/KPMS_pipeline"):
    logger = get_logger()
    logger.info(f"\n>>>> STARTING STEP: {name} <<<<")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Add JAX environment variables to prevent memory issues
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    
    try:
        process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
        if process.returncode != 0:
            logger.error(f"Step {name} failed with exit code {process.returncode}")
            return False
        logger.info(f">>>> COMPLETED STEP: {name} <<<<")
        return True
    except Exception as e:
        logger.error(f"Error running step {name}: {e}")
        return False

def main():
    setup_logger()
    logger = get_logger()
    logger.info("=== KPMS High-Resolution Campaign V5 (Camellia) ===")
    
    python_bin = "/home/isonaei/ABVFM_benchmark/venvs/keypoint-moseq/bin/python"
    
    # 1. Training
    if not run_step("Training", [python_bin, "src/kpms_custom/core/cli.py", "train"]):
        return

    # 2. Merging
    if not run_step("Batch Merging", [python_bin, "src/kpms_custom/core/batch_merge_all.py"]):
        return

    # 3. Analysis
    if not run_step("SSI Analysis", [python_bin, "src/kpms_custom/analysis/ssi_comparison.py"]):
        return

    logger.info("=== FULL CAMPAIGN COMPLETE ===")

if __name__ == "__main__":
    main()
