# Fashion-MNIST Deep Clustering with IDEC

## 📌 프로젝트 개요

Fashion-MNIST 데이터셋에 대한 비지도 심층 군집화 구현

## 🚀 빠른 시작

### 설치

```bash
git clone https://github.com/yourusername/fashion-mnist-deep-clustering.git
cd fashion-mnist-deep-clustering
pip install -r requirements.txt
```

### 데이터 다운로드

```bash
python data/download.py
```

### 학습 실행

```bash
# 베이스라인
python scripts/train_baseline.py

# Autoencoder
python scripts/train_autoencoder.py

# IDEC
python scripts/train_idec.py
```

## 📊 결과

| 방법         | NMI  | ARI  | 학습시간 |
| ------------ | ---- | ---- | -------- |
| K-Means      | 0.52 | 0.41 | 1분      |
| AE + K-Means | 0.68 | 0.58 | 10분     |
| IDEC         | 0.82 | 0.76 | 20분     |

## 📁 프로젝트 구조

```
(디렉토리 트리)
```

## 🛠️ 기술 스택

- PyTorch
- scikit-learn
- matplotlib

## 📖 참고 문헌

- IDEC Paper: [링크]
- Fashion-MNIST: [링크]

## 👤 작성자

이름 - [GitHub](링크)
