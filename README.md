# Fashion-MNIST Deep Clustering

🧥 Unsupervised deep clustering on Fashion-MNIST using IDEC (Improved Deep Embedded Clustering)

## 🎯 Overview

This project implements and compares three clustering approaches on Fashion-MNIST:

- **Baseline**: K-Means on raw pixels
- **AE + K-Means**: K-Means on autoencoder latent space
- **IDEC**: Joint optimization of reconstruction and clustering

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/fashion-mnist-deep-clustering.git
cd fashion-mnist-deep-clustering
pip install -r requirements.txt
```

### Run Experiments

```bash
# Train baseline
python scripts/train_baseline.py

# Train autoencoder
python scripts/train_autoencoder.py

# Train IDEC
python scripts/train_idec.py
```

## 📊 Expected Results

| Method        | NMI   | ARI   | ACC   |
| ------------- | ----- | ----- | ----- |
| K-Means (Raw) | ~0.52 | ~0.41 | ~0.53 |
| AE + K-Means  | ~0.68 | ~0.58 | ~0.70 |
| IDEC          | ~0.82 | ~0.76 | ~0.85 |

## 📁 Project Structure

```
fashion-mnist-deep-clustering/
├── configs/          # Configuration files
├── data/             # Data loading utilities
├── models/           # Model definitions
├── scripts/          # Training scripts
├── utils/            # Evaluation & visualization
└── results/          # Output files (not tracked)
```

## 🛠️ Tech Stack

- **Framework**: PyTorch 2.5.1
- **Dataset**: Fashion-MNIST
- **Methods**: Autoencoder, K-Means, IDEC

## 👤 Author

Your Name - [GitHub](https://github.com/yourusername)
