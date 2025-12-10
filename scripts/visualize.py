# scripts\visualize.py
"""
시각화 스크립트
t-SNE, 학습 곡선, 클러스터 샘플 시각화
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch

from configs import load_config
from data import get_full_dataset
from models import create_autoencoder, create_idec


def plot_tsne_comparison(config, y_true, figures_dir):
    """
    3가지 방법의 t-SNE 비교 시각화
    
    Args:
        config: 설정
        y_true: 실제 라벨
        figures_dir: 저장 디렉토리
    """
    print(f"\n{'='*60}")
    print("📊 t-SNE 시각화")
    print(f"{'='*60}")
    
    results_dir = config['paths']['results_dir']
    
    # 데이터 로드
    X, _ = get_full_dataset(config['data']['data_dir'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ============================================================
    # 1. Baseline (Raw Pixels)
    # ============================================================
    print("\n🔄 Baseline t-SNE 계산 중...")
    tsne = TSNE(n_components=2, random_state=config['seed'], perplexity=30)
    X_tsne_baseline = tsne.fit_transform(X)
    
    y_pred_baseline = np.load(os.path.join(results_dir, 'baseline_predictions.npy'))
    
    scatter = axes[0].scatter(
        X_tsne_baseline[:, 0],
        X_tsne_baseline[:, 1],
        c=y_pred_baseline,
        cmap='tab10',
        s=1,
        alpha=0.6
    )
    axes[0].set_title('Baseline K-Means\n(Raw Pixels)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # ============================================================
    # 2. AE + K-Means (Latent Space)
    # ============================================================
    print("🔄 AE + K-Means t-SNE 계산 중...")
    Z_ae = np.load(os.path.join(results_dir, 'ae_latent_features.npy'))
    
    tsne = TSNE(n_components=2, random_state=config['seed'], perplexity=30)
    Z_tsne_ae = tsne.fit_transform(Z_ae)
    
    y_pred_ae = np.load(os.path.join(results_dir, 'ae_kmeans_predictions.npy'))
    
    axes[1].scatter(
        Z_tsne_ae[:, 0],
        Z_tsne_ae[:, 1],
        c=y_pred_ae,
        cmap='tab10',
        s=1,
        alpha=0.6
    )
    axes[1].set_title('AE + K-Means\n(Latent Space)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    # ============================================================
    # 3. IDEC (Latent Space)
    # ============================================================
    print("🔄 IDEC t-SNE 계산 중...")
    
    # IDEC 모델 로드 및 latent features 추출
    device = torch.device('cpu')
    autoencoder = create_autoencoder(config)
    model = create_idec(autoencoder, config)
    
    idec_checkpoint = os.path.join(config['paths']['checkpoint_dir'], 'idec_final.pth')
    if os.path.exists(idec_checkpoint):
        model.load_state_dict(torch.load(idec_checkpoint))
        model = model.to(device)
        model.eval()
        
        # Latent features 추출
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            Z_idec = model.encoder(X_tensor).cpu().numpy()
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=config['seed'], perplexity=30)
        Z_tsne_idec = tsne.fit_transform(Z_idec)
        
        # 예측 라벨
        y_pred_idec = np.load(os.path.join(results_dir, 'idec_predictions_epoch100.npy'))
        
        axes[2].scatter(
            Z_tsne_idec[:, 0],
            Z_tsne_idec[:, 1],
            c=y_pred_idec,
            cmap='tab10',
            s=1,
            alpha=0.6
        )
    else:
        print(f"⚠️  IDEC 체크포인트를 찾을 수 없습니다: {idec_checkpoint}")
    
    axes[2].set_title('IDEC\n(End-to-End)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('t-SNE 1')
    axes[2].set_ylabel('t-SNE 2')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal', pad=0.05, aspect=30)
    cbar.set_label('Cluster', fontsize=10)
    
    plt.suptitle('t-SNE Visualization Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, 'tsne_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ t-SNE 시각화 저장: {save_path}")


def plot_learning_curves(config, figures_dir):
    """
    학습 곡선 시각화
    
    Args:
        config: 설정
        figures_dir: 저장 디렉토리
    """
    print(f"\n{'='*60}")
    print("📈 학습 곡선 시각화")
    print(f"{'='*60}")
    
    results_dir = config['paths']['results_dir']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ============================================================
    # 1. Autoencoder 학습 곡선
    # ============================================================
    ae_history_path = os.path.join(results_dir, 'ae_history.npy')
    if os.path.exists(ae_history_path):
        ae_history = np.load(ae_history_path, allow_pickle=True).item()
        
        epochs = range(1, len(ae_history['train_loss']) + 1)
        axes[0].plot(epochs, ae_history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(epochs, ae_history['test_loss'], label='Test Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('MSE Loss', fontsize=12)
        axes[0].set_title('Autoencoder Training', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        print("✅ Autoencoder 학습 곡선 로드")
    else:
        print(f"⚠️  Autoencoder history를 찾을 수 없습니다: {ae_history_path}")
    
    # ============================================================
    # 2. IDEC 학습 곡선
    # ============================================================
    idec_history_path = os.path.join(results_dir, 'idec_history.npy')
    if os.path.exists(idec_history_path):
        idec_history = np.load(idec_history_path, allow_pickle=True).item()
        
        epochs = range(1, len(idec_history['total_loss']) + 1)
        axes[1].plot(epochs, idec_history['recon_loss'], label='Reconstruction Loss', linewidth=2)
        axes[1].plot(epochs, idec_history['cluster_loss'], label='Clustering Loss', linewidth=2)
        axes[1].plot(epochs, idec_history['total_loss'], label='Total Loss', linewidth=2, linestyle='--')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('IDEC Training', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        print("✅ IDEC 학습 곡선 로드")
    else:
        print(f"⚠️  IDEC history를 찾을 수 없습니다: {idec_history_path}")
    
    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, 'learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 학습 곡선 저장: {save_path}")


def plot_cluster_samples(config, figures_dir):
    """
    각 클러스터의 샘플 이미지 시각화
    
    Args:
        config: 설정
        figures_dir: 저장 디렉토리
    """
    print(f"\n{'='*60}")
    print("🖼️  클러스터 샘플 시각화")
    print(f"{'='*60}")
    
    results_dir = config['paths']['results_dir']
    X, y_true = get_full_dataset(config['data']['data_dir'])
    
    methods = [
        ('Baseline K-Means', 'baseline_predictions.npy'),
        ('AE + K-Means', 'ae_kmeans_predictions.npy'),
        ('IDEC', 'idec_predictions_epoch100.npy')
    ]
    
    for method_name, pred_file in methods:
        pred_path = os.path.join(results_dir, pred_file)
        
        if not os.path.exists(pred_path):
            print(f"⚠️  {pred_file}를 찾을 수 없습니다")
            continue
        
        y_pred = np.load(pred_path)
        
        # 각 클러스터에서 10개씩 샘플 추출
        fig, axes = plt.subplots(10, 10, figsize=(12, 12))
        fig.suptitle(f'Cluster Samples - {method_name}', fontsize=16, fontweight='bold')
        
        for cluster_id in range(10):
            # 해당 클러스터의 인덱스
            cluster_indices = np.where(y_pred == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # 10개 랜덤 샘플
            sample_indices = np.random.choice(
                cluster_indices,
                size=min(10, len(cluster_indices)),
                replace=False
            )
            
            for i, idx in enumerate(sample_indices):
                ax = axes[cluster_id, i]
                img = X[idx].reshape(28, 28)
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
                if i == 0:
                    ax.set_ylabel(f'Cluster {cluster_id}', fontsize=10, rotation=0, labelpad=30)
        
        plt.tight_layout()
        
        save_name = method_name.lower().replace(' ', '_').replace('+', '')
        save_path = os.path.join(figures_dir, f'cluster_samples_{save_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {method_name} 샘플 저장: {save_path}")


def main():
    """메인 시각화 함수"""
    
    print("\n" + "="*60)
    print("시각화 시작")
    print("="*60)
    
    # 설정 로드
    config = load_config('configs/config.yaml')
    figures_dir = config['paths']['figures_dir']
    
    # 실제 라벨 로드
    _, y_true = get_full_dataset(config['data']['data_dir'])
    
    # ============================================================
    # 1. t-SNE 비교
    # ============================================================
    plot_tsne_comparison(config, y_true, figures_dir)
    
    # ============================================================
    # 2. 학습 곡선
    # ============================================================
    plot_learning_curves(config, figures_dir)
    
    # ============================================================
    # 3. 클러스터 샘플
    # ============================================================
    plot_cluster_samples(config, figures_dir)
    
    print("\n" + "="*60)
    print("✅ 시각화 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()