# models/autoencoder.py
"""
Autoencoder 모델
고차원 데이터(784)를 저차원(10)으로 압축했다가 다시 복원
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    간단한 Autoencoder
    
    구조:
        784 → 500 → 500 → 2000 → 10 (Encoder)
        10 → 2000 → 500 → 500 → 784 (Decoder)
    """
    
    def __init__(self, input_dim=784, latent_dim=10, hidden_dims=[500, 500, 2000]):
        """
        Args:
            input_dim: 입력 차원 (28x28 = 784)
            latent_dim: 압축된 차원 (10)
            hidden_dims: 히든 레이어 크기들 [500, 500, 2000]
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # ============================================================
        # Encoder: 784 → 10으로 압축
        # ============================================================
        encoder_layers = []
        
        # 첫 번째 레이어: 784 → 500
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(nn.ReLU())
        
        # 중간 레이어들: 500 → 500 → 2000
        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            encoder_layers.append(nn.ReLU())
        
        # 마지막 레이어: 2000 → 10
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ============================================================
        # Decoder: 10 → 784로 복원
        # ============================================================
        decoder_layers = []
        
        # 첫 번째 레이어: 10 → 2000
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())
        
        # 중간 레이어들: 2000 → 500 → 500
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            decoder_layers.append(nn.ReLU())
        
        # 마지막 레이어: 500 → 784
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # 0~1 사이 값으로
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 이미지 (batch_size, 784)
            
        Returns:
            복원된 이미지 (batch_size, 784)
        """
        z = self.encoder(x)      # 압축
        x_recon = self.decoder(z)  # 복원
        return x_recon
    
    def encode(self, x):
        """
        인코딩만 수행 (압축만)
        
        Args:
            x: 입력 이미지 (batch_size, 784)
            
        Returns:
            압축된 벡터 z (batch_size, 10)
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        디코딩만 수행 (복원만)
        
        Args:
            z: 압축된 벡터 (batch_size, 10)
            
        Returns:
            복원된 이미지 (batch_size, 784)
        """
        return self.decoder(z)


# ============================================================
# 테스트 코드
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Autoencoder 모델 테스트")
    print("="*60)
    
    # 1. 모델 생성
    model = Autoencoder(input_dim=784, latent_dim=10)
    print(f"\n✅ 모델 생성 완료")
    print(f"   입력 차원: 784")
    print(f"   잠재 차원: 10")
    
    # 2. 모델 구조 출력
    print(f"\n📊 모델 파라미터 수:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   총 파라미터: {total_params:,}개")
    
    # 3. 순전파 테스트
    batch_size = 32
    x = torch.randn(batch_size, 784)  # 랜덤 입력
    
    print(f"\n🔄 순전파 테스트:")
    print(f"   입력 shape: {x.shape}")
    
    # 전체 과정 (압축 + 복원)
    x_recon = model(x)
    print(f"   출력 shape: {x_recon.shape}")
    
    # 압축만
    z = model.encode(x)
    print(f"   잠재벡터 shape: {z.shape}")
    
    # 4. 복원 오차 확인
    mse = torch.mean((x - x_recon) ** 2)
    print(f"\n📉 초기 복원 오차 (MSE): {mse.item():.4f}")
    print(f"   (학습 전이라 오차가 큽니다)")
    
    print("\n" + "="*60)
    print("✅ 모든 테스트 통과!")
    print("="*60)