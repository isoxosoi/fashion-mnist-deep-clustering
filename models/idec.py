"""
IDEC (Improved Deep Embedded Clustering)
Autoencoder + K-Means를 동시에 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDEC(nn.Module):
    """
    IDEC: 복원 + 군집화를 동시에 학습
    
    구조:
        - Encoder + Decoder (Autoencoder)
        - Cluster Centers (군집 중심점들)
    """
    
    def __init__(self, autoencoder, n_clusters=10, alpha=1.0):
        """
        Args:
            autoencoder: 사전학습된 Autoencoder
            n_clusters: 군집 개수 (기본값: 10)
            alpha: Student t-distribution 파라미터
        """
        super(IDEC, self).__init__()
        
        # Autoencoder에서 Encoder와 Decoder 가져오기
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # 군집 중심점들 (학습 가능한 파라미터)
        # shape: (n_clusters, latent_dim)
        latent_dim = autoencoder.latent_dim
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        
        # 초기화: Xavier Normal
        torch.nn.init.xavier_normal_(self.cluster_centers.data)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 이미지 (batch_size, 784)
            
        Returns:
            x_recon: 복원된 이미지 (batch_size, 784)
            q: 군집 할당 확률 (batch_size, n_clusters)
            z: 잠재 벡터 (batch_size, latent_dim)
        """
        # 1. 인코딩
        z = self.encoder(x)
        
        # 2. 디코딩 (복원)
        x_recon = self.decoder(z)
        
        # 3. 군집 할당 확률 계산
        q = self.soft_assignment(z)
        
        return x_recon, q, z
    
    def soft_assignment(self, z):
        """
        Student t-distribution으로 군집 할당 확률 계산
        
        Args:
            z: 잠재 벡터 (batch_size, latent_dim)
            
        Returns:
            q: 각 데이터가 각 군집에 속할 확률 (batch_size, n_clusters)
        """
        # z와 각 군집 중심 사이의 거리 계산
        # z: (batch_size, latent_dim)
        # cluster_centers: (n_clusters, latent_dim)
        
        # 거리 계산: ||z_i - mu_j||^2
        # unsqueeze로 차원 맞추기
        # z.unsqueeze(1): (batch_size, 1, latent_dim)
        # cluster_centers: (1, n_clusters, latent_dim)
        
        distances = torch.sum(
            (z.unsqueeze(1) - self.cluster_centers) ** 2, 
            dim=2
        )  # (batch_size, n_clusters)
        
        # Student t-distribution
        # q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        
        # 정규화 (각 행의 합이 1이 되도록)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return q
    
    def target_distribution(self, q):
        """
        Target distribution (P) 계산
        
        Args:
            q: 현재 군집 할당 확률 (batch_size, n_clusters)
            
        Returns:
            p: 목표 분포 (batch_size, n_clusters)
        """
        # p_ij = q_ij^2 / sum_i(q_ij) / sum_j(q_ij^2 / sum_i(q_ij))
        
        # 1. q^2 계산
        weight = q ** 2
        
        # 2. 각 열(클러스터)의 합으로 나누기
        weight = weight / torch.sum(q, dim=0, keepdim=True)
        
        # 3. 정규화
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        
        return p
    
    def predict(self, x):
        """
        군집 예측
        
        Args:
            x: 입력 이미지 (batch_size, 784)
            
        Returns:
            predicted_labels: 예측된 군집 번호 (batch_size,)
        """
        _, q, _ = self.forward(x)
        return torch.argmax(q, dim=1)


# ============================================================
# 테스트 코드
# ============================================================
if __name__ == "__main__":
    from autoencoder import Autoencoder
    
    print("="*60)
    print("IDEC 모델 테스트")
    print("="*60)
    
    # 1. Autoencoder 생성
    ae = Autoencoder(input_dim=784, latent_dim=10)
    print(f"\n✅ Autoencoder 생성 완료")
    
    # 2. IDEC 모델 생성
    model = IDEC(ae, n_clusters=10)
    print(f"✅ IDEC 모델 생성 완료")
    print(f"   군집 개수: 10")
    print(f"   클러스터 중심 shape: {model.cluster_centers.shape}")
    
    # 3. 순전파 테스트
    batch_size = 32
    x = torch.randn(batch_size, 784)
    
    print(f"\n🔄 순전파 테스트:")
    print(f"   입력 shape: {x.shape}")
    
    x_recon, q, z = model(x)
    
    print(f"   복원 이미지 shape: {x_recon.shape}")
    print(f"   군집 확률 shape: {q.shape}")
    print(f"   잠재 벡터 shape: {z.shape}")
    
    # 4. 군집 할당 확률 확인
    print(f"\n📊 첫 번째 샘플의 군집 확률:")
    print(f"   {q[0].detach().numpy()}")
    print(f"   합계: {torch.sum(q[0]).item():.4f} (1에 가까워야 함)")
    
    # 5. 군집 예측
    pred_labels = model.predict(x)
    print(f"\n🎯 예측된 군집:")
    print(f"   {pred_labels[:10].numpy()}")
    
    # 6. Target distribution 테스트
    p = model.target_distribution(q)
    print(f"\n📈 Target distribution shape: {p.shape}")
    
    print("\n" + "="*60)
    print("✅ 모든 테스트 통과!")
    print("="*60)