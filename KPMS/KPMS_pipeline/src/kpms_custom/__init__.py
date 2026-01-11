__version__ = "0.1.0"

import os
# Prevent JAX from grabbing all VRAM by default
if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
# Critical for Keypoint-MoSeq precision
jax.config.update("jax_enable_x64", True)

# Expose key modules for convenience if needed
from .core import runner
