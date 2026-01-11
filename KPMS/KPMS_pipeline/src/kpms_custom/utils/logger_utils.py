import logging
import sys
from pathlib import Path

def setup_logger(name="kpms_pipeline", log_file=None, level=logging.INFO):
    """
    Sets up a logger that writes to console and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if re-initialized
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    if log_file:
        # Ensure directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def get_logger(name="kpms_pipeline"):
    return logging.getLogger(name)
