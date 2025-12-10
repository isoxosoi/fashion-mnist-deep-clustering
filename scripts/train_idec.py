# scripts/train_idec.py
"""
IDEC 학습
Autoencoder + Clustering 동시 최적화
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.cluster import KMeans

from configs import load_config, create_directories
from data import get_fashion_mnist_loaders, get_full_dataset
from models import create_autoencoder, create_idec


def initialize_cluster_centers(model, data_loader, n_clusters, device):
    """
    K-Means로 클러스터 중심 초기화
    
    Args:
        model: IDEC 모델
        data_loader: 데이터 로더
        n_clusters: 클러스터 개수
        device: 디바이스
        
    Returns:
        cluster_centers: 초기화된 중심점 (n_clusters, latent_dim)
    """
    print("\n🔄 클러스터 중심 초기화 중...")
    
    model.eval()
    latents = []
    
    # 모든 데이터의 latent vector 추출
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            z = model.encoder(x)
            latents.append(z.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    
    # K-Means로 초기화
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(latents)
    
    print(f"✅ 클러스터 중심 초기화 완료 (Inertia: {kmeans.inertia_:.2f})")
    
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)


def train_epoch(model, train_loader, optimizer, device, gamma):
    """
    1 에포크 학습
    
    Args:
        model: IDEC 모델
        train_loader: 학습 데이터 로더
        optimizer: 옵티마이저
        device: 디바이스
        gamma: 클러스터링 손실 가중치
        
    Returns:
        avg_total_loss: 평균 전체 손실
        avg_recon_loss: 평균 복원 손실
        avg_cluster_loss: 평균 클러스터링 손실
    """
    model.train()
    
    total_loss_sum = 0
    recon_loss_sum = 0
    cluster_loss_sum = 0
    
    for x, _ in train_loader:
        x = x.to(device)
        
        # Forward
        x_recon, q, z = model(x)
        
        # Target distribution
        p = model.target_distribution(q).detach()
        
        # 손실 계산
        # 1. 복원 손실 (MSE)
        recon_loss = nn.MSELoss()(x_recon, x)
        
        # 2. 클러스터링 손실 (KL divergence)
        cluster_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(q), 
            p
        )
        
        # 3. 전체 손실
        total_loss = recon_loss + gamma * cluster_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 기록
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        cluster_loss_sum += cluster_loss.item()
    
    n_batches = len(train_loader)
    return (
        total_loss_sum / n_batches,
        recon_loss_sum / n_batches,
        cluster_loss_sum / n_batches
    )


def evaluate(model, test_loader, device):
    """
    평가
    
    Args:
        model: IDEC 모델
        test_loader: 테스트 데이터 로더
        device: 디바이스
        
    Returns:
        predictions: 예측 라벨
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            pred = model.predict(x)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    return predictions


def main():
    """IDEC 학습"""
    
    print("\n" + "="*60)
    print("IDEC 학습")
    print("="*60)
    
    # ============================================================
    # 1. 설정 로드
    # ============================================================
    config = load_config('configs/config.yaml')
    create_directories(config)
    
    # 디바이스
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✅ Device: {device}")
    
    # 랜덤 시드
    seed = config['training']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # ============================================================
    # 2. 데이터 로드
    # ============================================================
    print(f"\n📂 데이터 로딩...")
    train_loader, test_loader = get_fashion_mnist_loaders(
        batch_size=config['data']['batch_size'],
        data_dir=config['data']['data_dir']
    )
    
    print(f"✅ 데이터 로드 완료")
    
    # ============================================================
    # 3. Autoencoder 로드
    # ============================================================
    print(f"\n🏗️  사전학습된 Autoencoder 로드...")
    
    autoencoder = create_autoencoder(config)
    
    # 체크포인트 로드
    checkpoint_path = os.path.join(
        config['paths']['checkpoint_dir'],
        'autoencoder_best.pth'
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"사전학습된 Autoencoder를 찾을 수 없습니다: {checkpoint_path}\n"
            f"먼저 'python scripts/train_autoencoder.py'를 실행하세요."
        )
    
    autoencoder.load_state_dict(torch.load(checkpoint_path))
    autoencoder = autoencoder.to(device)
    
    print(f"✅ Autoencoder 로드 완료")
    
    # ============================================================
    # 4. IDEC 모델 생성
    # ============================================================
    print(f"\n🏗️  IDEC 모델 생성...")
    model = create_idec(autoencoder, config)
    model = model.to(device)
    
    print(f"✅ IDEC 생성 완료")
    
    # ============================================================
    # 5. 클러스터 중심 초기화
    # ============================================================
    cluster_centers = initialize_cluster_centers(
        model, train_loader, 
        config['model']['n_clusters'],
        device
    )
    model.cluster_centers.data = cluster_centers.to(device)
    
    # ============================================================
    # 6. 학습 설정
    # ============================================================
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['finetune_lr'],
        momentum=0.9
    )
    
    num_epochs = config['training']['finetune_epochs']
    gamma = config['training']['gamma']
    
    print(f"\n⚙️  학습 설정:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {config['training']['finetune_lr']}")
    print(f"   Gamma: {gamma}")
    
    # ============================================================
    # 7. 학습 루프
    # ============================================================
    print(f"\n🔄 학습 시작...")
    print("-" * 80)
    
    history = {
        'total_loss': [],
        'recon_loss': [],
        'cluster_loss': []
    }
    
    start_time = datetime.now()
    
    for epoch in range(1, num_epochs + 1):
        # 학습
        total_loss, recon_loss, cluster_loss = train_epoch(
            model, train_loader, optimizer, device, gamma
        )
        
        # 기록
        history['total_loss'].append(total_loss)
        history['recon_loss'].append(recon_loss)
        history['cluster_loss'].append(cluster_loss)
        
        # 출력
        print(f"Epoch [{epoch:3d}/{num_epochs}] "
              f"Total: {total_loss:.6f} | "
              f"Recon: {recon_loss:.6f} | "
              f"Cluster: {cluster_loss:.6f}")
        
        # 주기적으로 예측 저장
        if epoch % 10 == 0 or epoch == num_epochs:
            predictions = evaluate(model, test_loader, device)
            np.save(
                os.path.join(
                    config['paths']['results_dir'],
                    f'idec_predictions_epoch{epoch}.npy'
                ),
                predictions
            )
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print("-" * 80)
    print(f"✅ 학습 완료!")
    print(f"   소요 시간: {elapsed_time/60:.2f}분")
    
    # ============================================================
    # 8. 모델 저장
    # ============================================================
    model_path = os.path.join(
        config['paths']['checkpoint_dir'],
        'idec_final.pth'
    )
    torch.save(model.state_dict(), model_path)
    
    # History 저장
    np.save(
        os.path.join(config['paths']['results_dir'], 'idec_history.npy'),
        history
    )
    
    print(f"\n💾 모델 저장 완료:")
    print(f"   {model_path}")
    
    print("\n" + "="*60)
    print("✅ IDEC 학습 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()