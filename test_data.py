# test_data.py
"""
Data loader test
Check if data loads correctly
"""

import torch
import matplotlib.pyplot as plt
from data import get_fashion_mnist_loaders, get_full_dataset, get_class_names


def test_dataloader():
    """DataLoader test"""
    print("\n" + "="*60)
    print("1️⃣  DataLoader Test")
    print("="*60)
    
    # Load data
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=32)
    
    # Get first batch
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch info:")
    print(f"  Number of images: {images.shape[0]}")
    print(f"  Image dimension: {images.shape[1]} (28x28 = 784)")
    print(f"  Number of labels: {labels.shape[0]}")
    print(f"  Label examples: {labels[:10].tolist()}")
    
    return train_loader, images, labels


def test_full_dataset():
    """Full dataset test"""
    print("\n" + "="*60)
    print("2️⃣  Full Dataset Test")
    print("="*60)
    
    X, y = get_full_dataset()
    
    print(f"\nData statistics:")
    print(f"  Total images: {X.shape[0]:,}")
    print(f"  Image dimension: {X.shape[1]:,}")
    print(f"  Pixel value range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Number of classes: {len(set(y.tolist()))}")
    
    # Count per class
    print(f"\nClass distribution:")
    class_names = get_class_names()
    for i in range(10):
        count = (y == i).sum()
        print(f"  {i} ({class_names[i]:15s}): {count:,}")
    
    return X, y


def visualize_samples(images, labels):
    """Visualize sample images"""
    print("\n" + "="*60)
    print("3️⃣  Sample Image Visualization")
    print("="*60)
    
    class_names = get_class_names()
    
    # Display 8 images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Fashion-MNIST Sample Images', fontsize=16)  # ← 영어로 변경!
    
    for i, ax in enumerate(axes.flat):
        # Reshape image (784 -> 28x28)
        img = images[i].view(28, 28).numpy()
        label = labels[i].item()
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{label}: {class_names[label]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Image saved: data_samples.png")
    plt.show()


def main():
    """Run all tests"""
    print("\n" + "🧪 " + "="*56 + " 🧪")
    print("   Fashion-MNIST Data Loader Test")
    print("🧪 " + "="*56 + " 🧪")
    
    # 1. DataLoader test
    train_loader, images, labels = test_dataloader()
    
    # 2. Full dataset test
    X, y = test_full_dataset()
    
    # 3. Visualization
    visualize_samples(images, labels)
    
    # Summary
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print(f"📦 Training batches: {len(train_loader)}")
    print(f"📊 Total data: {X.shape[0]:,}")
    print(f"🎨 Number of classes: 10")
    print("="*60)


if __name__ == "__main__":
    main()