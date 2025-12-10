# models/__init__.py
"""
모델 모듈
Autoencoder, IDEC, Baseline K-Means 포함
"""

from .autoencoder import Autoencoder
from .idec import IDEC

import torch
from sklearn.cluster import KMeans


def train_kmeans_baseline(X, n_clusters=10, random_state=42):
    """
    베이스라인: Raw 데이터에 K-Means 적용
    
    Args:
        X: 데이터 (N, 784) - torch.Tensor 또는 numpy array
        n_clusters: 군집 개수
        random_state: 랜덤 시드
        
    Returns:
        y_pred: 예측된 군집 라벨 (N,)
        kmeans: 학습된 K-Means 모델
    """
    # Tensor면 numpy로 변환
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    print(f"🔄 K-Means 학습 중... (n_clusters={n_clusters})")
    
    # K-Means 학습
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=20,  # 초기화 횟수
        max_iter=300,
        random_state=random_state
    )
    
    y_pred = kmeans.fit_predict(X)
    
    print(f"✅ K-Means 완료!")
    print(f"   Inertia: {kmeans.inertia_:.2f}")
    
    return y_pred, kmeans


def create_autoencoder(config):
    """
    Config 파일로부터 Autoencoder 생성
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        model: Autoencoder 모델
    """
    model = Autoencoder(
        input_dim=config['model']['input_dim'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims']
    )
    return model


def create_idec(autoencoder, config):
    """
    Config 파일로부터 IDEC 생성
    
    Args:
        autoencoder: 사전학습된 Autoencoder
        config: 설정 딕셔너리
        
    Returns:
        model: IDEC 모델
    """
    model = IDEC(
        autoencoder=autoencoder,
        n_clusters=config['model']['n_clusters'],
        alpha=config['model']['alpha']
    )
    return model


# ============================================================
# 모델 정보 출력
# ============================================================
def print_model_info(model):
    """
    모델 정보 출력
    
    Args:
        model: PyTorch 모델
    """
    print("\n" + "="*60)
    print("모델 정보")
    print("="*60)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"총 파라미터: {total_params:,}개")
    print(f"학습 가능 파라미터: {trainable_params:,}개")
    
    # 모델 구조
    print(f"\n모델 구조:")
    print(model)
    
    print("="*60 + "\n")


__all__ = [
    'Autoencoder',
    'IDEC',
    'train_kmeans_baseline',
    'create_autoencoder',
    'create_idec',
    'print_model_info'
]