# test_models.py
"""
모델 테스트
Autoencoder, IDEC, K-Means 베이스라인 테스트
"""

import torch
from models import Autoencoder, IDEC, train_kmeans_baseline, print_model_info
from data import get_full_dataset


def test_autoencoder():
    """Autoencoder 테스트"""
    print("\n" + "="*60)
    print("1️⃣  Autoencoder 테스트")
    print("="*60)
    
    # 모델 생성
    model = Autoencoder(input_dim=784, latent_dim=10)
    print_model_info(model)
    
    # 샘플 데이터로 테스트
    x = torch.randn(32, 784)
    x_recon = model(x)
    z = model.encode(x)
    
    print(f"✅ 입력: {x.shape}")
    print(f"✅ 잠재벡터: {z.shape}")
    print(f"✅ 복원: {x_recon.shape}")
    
    return model


def test_idec(autoencoder):
    """IDEC 테스트"""
    print("\n" + "="*60)
    print("2️⃣  IDEC 테스트")
    print("="*60)
    
    # 모델 생성
    model = IDEC(autoencoder, n_clusters=10)
    print_model_info(model)
    
    # 샘플 데이터로 테스트
    x = torch.randn(32, 784)
    x_recon, q, z = model(x)
    pred = model.predict(x)
    
    print(f"✅ 입력: {x.shape}")
    print(f"✅ 복원: {x_recon.shape}")
    print(f"✅ 군집 확률: {q.shape}")
    print(f"✅ 예측 라벨: {pred.shape}")
    print(f"   예측 예시: {pred[:10].numpy()}")
    
    return model


def test_kmeans():
    """K-Means 베이스라인 테스트"""
    print("\n" + "="*60)
    print("3️⃣  K-Means 베이스라인 테스트")
    print("="*60)
    
    # 실제 데이터 로드
    print("데이터 로딩...")
    X, y = get_full_dataset()
    
    # 일부만 사용 (빠른 테스트)
    X_sample = X[:1000]
    
    # K-Means 실행
    y_pred, kmeans = train_kmeans_baseline(X_sample, n_clusters=10)
    
    print(f"✅ 데이터: {X_sample.shape}")
    print(f"✅ 예측 라벨: {y_pred.shape}")
    print(f"   예측 예시: {y_pred[:20]}")
    

def main():
    """전체 테스트 실행"""
    print("\n" + "🧪 " + "="*56 + " 🧪")
    print("   모델 테스트")
    print("🧪 " + "="*56 + " 🧪")
    
    # 1. Autoencoder
    ae = test_autoencoder()
    
    # 2. IDEC
    idec = test_idec(ae)
    
    # 3. K-Means
    test_kmeans()
    
    # 최종
    print("\n" + "="*60)
    print("✅ 모든 모델 테스트 통과!")
    print("="*60)


if __name__ == "__main__":
    main()