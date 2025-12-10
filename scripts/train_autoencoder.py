# scripts/train_autoencoder.py
"""
Autoencoder 사전학습
이미지 복원 능력 학습
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

from configs import load_config, create_directories
from data import get_fashion_mnist_loaders
from models import create_autoencoder


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    1 에포크 학습
    
    Args:
        model: Autoencoder 모델
        train_loader: 학습 데이터 로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: 디바이스
        
    Returns:
        avg_loss: 평균 손실
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)
        
        # Forward
        x_recon = model(x)
        loss = criterion(x_recon, x)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, test_loader, criterion, device):
    """
    평가
    
    Args:
        model: Autoencoder 모델
        test_loader: 테스트 데이터 로더
        criterion: 손실 함수
        device: 디바이스
        
    Returns:
        avg_loss: 평균 손실
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return avg_loss


def main():
    """Autoencoder 사전학습"""
    
    print("\n" + "="*60)
    print("Autoencoder 사전학습")
    print("="*60)
    
    # ============================================================
    # 1. 설정 로드
    # ============================================================
    config = load_config('configs/config.yaml')
    create_directories(config)
    
    # 디바이스 설정
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
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # ============================================================
    # 3. 모델 생성
    # ============================================================
    print(f"\n🏗️  모델 생성...")
    model = create_autoencoder(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 모델 생성 완료")
    print(f"   Parameters: {total_params:,}개")
    
    # ============================================================
    # 4. 학습 설정
    # ============================================================
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['pretrain_lr']
    )
    
    num_epochs = config['training']['pretrain_epochs']
    
    print(f"\n⚙️  학습 설정:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {config['training']['pretrain_lr']}")
    print(f"   Batch size: {config['data']['batch_size']}")
    
    # ============================================================
    # 5. 학습 루프
    # ============================================================
    print(f"\n🔄 학습 시작...")
    print("-" * 60)
    
    best_loss = float('inf')
    history = {'train_loss': [], 'test_loss': []}
    
    start_time = datetime.now()
    
    for epoch in range(1, num_epochs + 1):
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 평가
        test_loss = evaluate(model, test_loader, criterion, device)
        
        # 기록
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        # 출력
        print(f"Epoch [{epoch:3d}/{num_epochs}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Test Loss: {test_loss:.6f}")
        
        # 베스트 모델 저장
        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint_path = os.path.join(
                config['paths']['checkpoint_dir'],
                'autoencoder_best.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print("-" * 60)
    print(f"✅ 학습 완료!")
    print(f"   소요 시간: {elapsed_time/60:.2f}분")
    print(f"   Best Test Loss: {best_loss:.6f}")
    
    # ============================================================
    # 6. 최종 모델 저장
    # ============================================================
    final_path = os.path.join(
        config['paths']['checkpoint_dir'],
        'autoencoder_final.pth'
    )
    torch.save(model.state_dict(), final_path)
    
    # History 저장
    np.save(
        os.path.join(config['paths']['results_dir'], 'ae_history.npy'),
        history
    )
    
    print(f"\n💾 모델 저장 완료:")
    print(f"   {checkpoint_path}")
    print(f"   {final_path}")
    
    print("\n" + "="*60)
    print("✅ Autoencoder 사전학습 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()