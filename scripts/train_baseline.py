# scripts/train_baseline.py
"""
K-Means 베이스라인
Raw pixel 데이터에 직접 K-Means 적용
"""

import os
import numpy as np
import torch
from datetime import datetime

from configs import load_config, create_directories
from data import get_full_dataset
from models import train_kmeans_baseline


def main():
    """K-Means 베이스라인 학습 및 저장"""
    
    print("\n" + "="*60)
    print("K-Means 베이스라인")
    print("="*60)
    
    # ============================================================
    # 1. 설정 로드
    # ============================================================
    config = load_config('configs/config.yaml')
    create_directories(config)
    
    # 랜덤 시드 설정
    seed = config['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n✅ Config 로드 완료")
    print(f"   Seed: {seed}")
    print(f"   Clusters: {config['model']['n_clusters']}")
    
    # ============================================================
    # 2. 데이터 로드
    # ============================================================
    print(f"\n📂 데이터 로딩...")
    X, y_true = get_full_dataset(config['data']['data_dir'])
    
    print(f"✅ 데이터 로드 완료")
    print(f"   Shape: {X.shape}")
    print(f"   Labels: {y_true.shape}")
    
    # ============================================================
    # 3. K-Means 학습
    # ============================================================
    print(f"\n🔄 K-Means 학습 시작...")
    start_time = datetime.now()
    
    y_pred, kmeans = train_kmeans_baseline(
        X, 
        n_clusters=config['model']['n_clusters'],
        random_state=seed
    )
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ 학습 완료!")
    print(f"   소요 시간: {elapsed_time:.2f}초")
    print(f"   Inertia: {kmeans.inertia_:.2f}")
    
    # ============================================================
    # 4. 결과 저장
    # ============================================================
    results_dir = config['paths']['results_dir']
    
    # 예측 결과 저장
    np.save(
        os.path.join(results_dir, 'baseline_predictions.npy'),
        y_pred
    )
    
    # 클러스터 중심점 저장
    np.save(
        os.path.join(results_dir, 'baseline_centers.npy'),
        kmeans.cluster_centers_
    )
    
    # 메타 정보 저장
    meta_info = {
        'n_clusters': config['model']['n_clusters'],
        'inertia': float(kmeans.inertia_),
        'n_samples': len(y_pred),
        'elapsed_time': elapsed_time,
        'seed': seed
    }
    
    import json
    with open(os.path.join(results_dir, 'baseline_meta.json'), 'w') as f:
        json.dump(meta_info, f, indent=2)
    
    print(f"\n💾 결과 저장 완료:")
    print(f"   {results_dir}/baseline_predictions.npy")
    print(f"   {results_dir}/baseline_centers.npy")
    print(f"   {results_dir}/baseline_meta.json")
    
    # ============================================================
    # 5. 간단한 통계
    # ============================================================
    print(f"\n📊 클러스터 분포:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id}: {count:5d}개 ({count/len(y_pred)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ K-Means 베이스라인 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()