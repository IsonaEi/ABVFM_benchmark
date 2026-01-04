import yaml
import os
from pathlib import Path

def load_config(config_path):
    """
    Loads YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

def validate_config(config):
    """
    Validates critical configuration keys.
    """
    required_keys = ['project_dir', 'video_dir', 'data_dir', 'preprocess']
    missing = [key for key in required_keys if key not in config]
    
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    # ensure directories exist (strictly speaking, data_dir usually must exist)
    if not os.path.exists(config['data_dir']):
        # Warning only, maybe creating project?
        pass

    return True

def save_config(config, config_path):
    """
    Saves configuration to YAML.
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
