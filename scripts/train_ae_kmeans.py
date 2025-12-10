# scripts/train_ae_kmeans.py
"""
AE + K-Means
사전학습된 Autoencoder의 latent space에 K-Means 적용
"""

import os
import numpy as np
import torch
from datetime import datetime

from configs import load_config, create_directories
from data import get_full_dataset, get_fashion_mnist_loaders
from models import create_autoencoder, train_kmeans_baseline


def extract_latent_features(model, X, device, batch_size=256):
    """
    Autoencoder로 latent features 추출
    
    Args:
        model: 학습된 Autoencoder
        X: 입력 데이터 (N, 784)
        device: 디바이스
        batch_size: 배치 크기
        
    Returns:
        Z: latent features (N, latent_dim)
    """
    model.eval()
    Z_list = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            if isinstance(batch, np.ndarray):
                batch = torch.FloatTensor(batch)
            batch = batch.to(device)
            
            z = model.encode(batch)
            Z_list.append(z.cpu().numpy())
    
    Z = np.concatenate(Z_list, axis=0)
    return Z


def main():
    """AE + K-Means 학습"""
    
    print("\n" + "="*60)
    print("AE + K-Means")
    print("="*60)
    
    # ============================================================
    # 1. 설정 로드
    # ============================================================
    config = load_config('configs/config.yaml')
    create_directories(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✅ Device: {device}")
    
    # 랜덤 시드
    seed = config['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # ============================================================
    # 2. 데이터 로드
    # ============================================================
    print(f"\n📂 데이터 로딩...")
    X, y_true = get_full_dataset(config['data']['data_dir'])
    
    print(f"✅ 데이터 로드 완료")
    print(f"   Shape: {X.shape}")
    
    # ============================================================
    # 3. 사전학습된 Autoencoder 로드
    # ============================================================
    print(f"\n🏗️  Autoencoder 로드...")
    
    autoencoder = create_autoencoder(config)
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
    # 4. Latent features 추출
    # ============================================================
    print(f"\n🔄 Latent features 추출 중...")
    start_time = datetime.now()
    
    Z = extract_latent_features(autoencoder, X, device)
    
    elapsed_extract = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ 추출 완료!")
    print(f"   Original shape: {X.shape}")
    print(f"   Latent shape: {Z.shape}")
    print(f"   소요 시간: {elapsed_extract:.2f}초")
    
    # ============================================================
    # 5. K-Means 학습 (latent space)
    # ============================================================
    print(f"\n🔄 K-Means 학습 (latent space)...")
    start_time = datetime.now()
    
    y_pred, kmeans = train_kmeans_baseline(
        Z,
        n_clusters=config['model']['n_clusters'],
        random_state=seed
    )
    
    elapsed_kmeans = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ 학습 완료!")
    print(f"   소요 시간: {elapsed_kmeans:.2f}초")
    print(f"   Total 소요 시간: {elapsed_extract + elapsed_kmeans:.2f}초")
    
    # ============================================================
    # 6. 결과 저장
    # ============================================================
    results_dir = config['paths']['results_dir']
    
    # 예측 결과
    np.save(
        os.path.join(results_dir, 'ae_kmeans_predictions.npy'),
        y_pred
    )
    
    # Latent features
    np.save(
        os.path.join(results_dir, 'ae_latent_features.npy'),
        Z
    )
    
    # 클러스터 중심점
    np.save(
        os.path.join(results_dir, 'ae_kmeans_centers.npy'),
        kmeans.cluster_centers_
    )
    
    # 메타 정보
    import json
    meta_info = {
        'n_clusters': config['model']['n_clusters'],
        'latent_dim': Z.shape[1],
        'inertia': float(kmeans.inertia_),
        'n_samples': len(y_pred),
        'extract_time': elapsed_extract,
        'kmeans_time': elapsed_kmeans,
        'total_time': elapsed_extract + elapsed_kmeans,
        'seed': seed
    }
    
    with open(os.path.join(results_dir, 'ae_kmeans_meta.json'), 'w') as f:
        json.dump(meta_info, f, indent=2)
    
    print(f"\n💾 결과 저장 완료:")
    print(f"   {results_dir}/ae_kmeans_predictions.npy")
    print(f"   {results_dir}/ae_latent_features.npy")
    print(f"   {results_dir}/ae_kmeans_centers.npy")
    
    # ============================================================
    # 7. 통계
    # ============================================================
    print(f"\n📊 클러스터 분포:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id}: {count:5d}개 ({count/len(y_pred)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ AE + K-Means 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()