# configs/__init__.py
"""Configuration utilities for the project."""

import yaml
import os
from pathlib import Path


def load_config(config_path="configs/config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config):
    """
    Create necessary directories based on config.
    
    Args:
        config (dict): Configuration dictionary
    """
    paths = config['paths']
    
    directories = [
        paths['results_dir'],
        paths['checkpoint_dir'],
        paths['figures_dir'],
        paths['logs_dir'],
        config['data']['data_dir']
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def print_config(config):
    """
    Pretty print configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    print("\n" + "="*50)
    print("Configuration")
    print("="*50)
    
    for section, params in config.items():
        print(f"\n[{section.upper()}]")
        if isinstance(params, dict):
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {params}")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    # Test loading config
    config = load_config()
    print_config(config)
    create_directories(config)