__version__ = "0.1.0"

import jax
# Critical for Keypoint-MoSeq precision
jax.config.update("jax_enable_x64", True)

# Expose key modules for convenience if needed
from .core import runner
