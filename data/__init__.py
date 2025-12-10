# data/__init__.py
"""
Fashion-MNIST 데이터 로더
간단하게 데이터를 불러오는 함수들 모음
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_fashion_mnist_loaders(batch_size=256, data_dir='./data/raw'):
    """
    Fashion-MNIST 데이터를 불러오는 함수
    
    Args:
        batch_size: 한 번에 불러올 이미지 개수 (기본값: 256)
        data_dir: 데이터를 저장할 폴더 (기본값: './data/raw')
    
    Returns:
        train_loader: 학습용 데이터 (60,000개)
        test_loader: 테스트용 데이터 (10,000개)
    """
    
    # 1. 데이터 변환 설정
    # 28x28 이미지를 784차원 벡터로 변환
    transform = transforms.Compose([
        transforms.ToTensor(),              # 이미지를 텐서로 변환 (0~1 사이 값)
        transforms.Lambda(lambda x: x.view(-1))  # 28x28 -> 784로 펼치기
    ])
    
    # 2. Fashion-MNIST 다운로드 및 로드
    print("📥 Fashion-MNIST 데이터 다운로드 중...")
    
    # 학습 데이터 (60,000개)
    train_dataset = datasets.FashionMNIST(
        root=data_dir,           # 저장 위치
        train=True,              # 학습용 데이터
        download=True,           # 없으면 자동 다운로드
        transform=transform      # 위에서 정의한 변환 적용
    )
    
    # 테스트 데이터 (10,000개)
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,             # 테스트용 데이터
        download=True,
        transform=transform
    )
    
    # 3. DataLoader 생성 (배치 단위로 불러오기)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,   # 한 번에 256개씩
        shuffle=True,            # 데이터 섞기 (학습 효과 향상)
        num_workers=0            # Windows에서는 0 권장
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,           # 테스트는 안 섞음
        num_workers=0
    )
    
    print(f"✅ 데이터 로드 완료!")
    print(f"   학습 데이터: {len(train_dataset)}개")
    print(f"   테스트 데이터: {len(test_dataset)}개")
    print(f"   배치 크기: {batch_size}")
    
    return train_loader, test_loader


def get_full_dataset(data_dir='./data/raw'):
    """
    전체 데이터를 한 번에 불러오는 함수
    (K-Means 같은 비딥러닝 알고리즘용)
    
    Returns:
        X: 전체 이미지 데이터 (60,000 x 784)
        y: 전체 라벨 (60,000개)
    """
    
    # 데이터 다운로드
    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True
    )
    
    # 텐서로 변환
    X = dataset.data.float() / 255.0  # 0~255 -> 0~1로 정규화
    X = X.view(-1, 784)               # (60000, 28, 28) -> (60000, 784)
    y = dataset.targets.numpy()       # 라벨
    
    print(f"✅ 전체 데이터 로드 완료!")
    print(f"   데이터 크기: {X.shape}")
    print(f"   라벨 크기: {y.shape}")
    
    return X, y


def get_class_names():
    """
    Fashion-MNIST 클래스 이름 반환
    
    Returns:
        list: 10개 클래스 이름
    """
    class_names = [
        'T-shirt/top',   # 0: 티셔츠
        'Trouser',       # 1: 바지
        'Pullover',      # 2: 풀오버
        'Dress',         # 3: 드레스
        'Coat',          # 4: 코트
        'Sandal',        # 5: 샌들
        'Shirt',         # 6: 셔츠
        'Sneaker',       # 7: 스니커즈
        'Bag',           # 8: 가방
        'Ankle boot'     # 9: 앵클부츠
    ]
    return class_names


# 간단한 사용 예시
if __name__ == "__main__":
    print("="*50)
    print("데이터 로더 테스트")
    print("="*50)
    
    # 1. DataLoader로 불러오기
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=256)
    
    # 첫 번째 배치 확인
    images, labels = next(iter(train_loader))
    print(f"\n첫 번째 배치:")
    print(f"  이미지 shape: {images.shape}")  # (256, 784)
    print(f"  라벨 shape: {labels.shape}")    # (256,)
    
    # 2. 전체 데이터로 불러오기
    print("\n" + "="*50)
    X, y = get_full_dataset()
    
    # 3. 클래스 이름 확인
    print("\n" + "="*50)
    print("클래스 이름:")
    for i, name in enumerate(get_class_names()):
        print(f"  {i}: {name}")
    
    print("\n✅ 모든 테스트 통과!")