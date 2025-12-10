# test_config.py
"""Test configuration loading."""

from configs import load_config, print_config, create_directories


if __name__ == "__main__":
    print("Testing configuration loading...\n")
    
    # Load config
    config = load_config("configs/config.yaml")
    
    # Print config
    print_config(config)
    
    # Create directories
    print("Creating directories...")
    create_directories(config)
    
    # Test accessing config
    print("\nTesting config access:")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Latent dimension: {config['model']['latent_dim']}")
    print(f"Pretrain epochs: {config['training']['pretrain_epochs']}")
    print(f"Number of clusters: {config['model']['n_clusters']}")
    
    print("\n✓ Configuration test passed!")