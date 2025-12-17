# Fashion-MNIST Deep Clustering Project

이 프로젝트는 Fashion-MNIST 데이터셋에 대해 3가지 클러스터링 방법(Baseline K-Means, AE + K-Means, IDEC)의 성능을 비교하여 Deep Clustering의 우수성을 검증합니다.

---

## 🎯 프로젝트 개요

### 목적
- Unsupervised Learning을 통한 이미지 클러스터링
- 전통적 방법(K-Means) vs Deep Learning 방법(IDEC) 비교
- End-to-end 학습의 효과 검증

### 데이터셋
- **Fashion-MNIST**: 60,000 train + 10,000 test 이미지
- **이미지 크기**: 28×28 grayscale
- **클래스 수**: 10 (의류 카테고리)

---

## 🏗️ 프로젝트 구조
```
fashion-mnist-clustering/
├── configs/
│   ├── __init__.py
│   └── config.yaml              # 하이퍼파라미터 설정
├── data/
│   └── __init__.py              # 데이터 로더
├── models/
│   ├── __init__.py
│   ├── autoencoder.py           # Autoencoder 정의
│   └── idec.py                  # IDEC 정의
├── scripts/
│   ├── train_baseline.py        # 1. Baseline K-Means
│   ├── train_autoencoder.py     # 2. Autoencoder 사전학습
│   ├── train_ae_kmeans.py       # 3. AE + K-Means
│   ├── train_idec.py            # 4. IDEC 학습
│   ├── evaluate.py              # 5. 평가 (ACC, NMI, ARI)
│   └── visualize.py             # 6. 시각화 (t-SNE, 학습 곡선)
├── results/                     # 결과 저장 (자동 생성)
│   ├── checkpoints/             # 모델 체크포인트
│   ├── figures/                 # 시각화 결과
│   └── *.npy, *.json            # 예측 및 메타 데이터
├── requirements.txt
└── README.md
```

## 📊 주요 결과

### 성능 비교

| Method | Accuracy (ACC) | NMI | ARI | Improvement |
|--------|----------------|-----|-----|-------------|
| Baseline K-Means | 47.38% | 0.5118 | 0.3479 | - |
| AE + K-Means | 56.70% | 0.5670 | 0.3952 | **+9.32%p** |
| **IDEC (Epoch 100)** | **62.87%** ⭐ | **0.6465** | **0.5037** | **+6.17%p** |

**총 개선율: 47.38% → 62.87% (+15.49%p)**

### 계산 효율성 vs 성능 Trade-off

| Method | 학습 시간 | 추론 시간 | Inertia | Accuracy | 비고 |
|--------|----------|----------|---------|----------|------|
| Baseline K-Means | 127.8초 | - | 1,906,648.75 | 47.38% | ⚡ 빠르지만 낮은 성능 |
| AE + K-Means | 9.8초* | ~0.01초/1K | 388,679.625 | 56.70% | ⚖️ 빠르고 준수한 성능 |
| **IDEC** | **~60분** | **~0.01초/1K** | **-** | **62.87%** | **🏆 최고 성능** |

*AE 사전학습 시간(~30분) 제외, K-Means만 측정

**핵심 포인트:**
- **Inertia 감소**: 1,906,648.75 → 388,679.625 (**79.6% ↓**, 차원 축소 효과)
- **계산 속도**: 127.8초 → 9.8초 (**13배 빠름**, AE + K-Means)
- **IDEC의 강점**: 학습은 길지만(**~60분**) 한 번 학습 후 **실시간 추론 가능**

**IDEC의 실용성:**
- ✅ **일회성 학습**: 한 번 학습 후 모델 영구 재사용
- ✅ **빠른 추론**: 새로운 데이터에 대해 실시간 클러스터링 (~0.01초/1K)
- ✅ **프로덕션 준비**: 학습 투자 대비 최고 성능 보장

### 주요 인사이트

1. **IDEC의 압도적 우수성** ⭐
   - 모든 지표(ACC, NMI, ARI)에서 **최고 성능** 달성
   - Baseline 대비 **+15.49%p**, AE + K-Means 대비 **+6.17%p** 향상
   - **End-to-end 최적화**로 clustering에 가장 적합한 representation 학습

2. **차원 축소의 효과**
   - 784차원 → 10차원 축소로 Inertia **79.6% 감소**
   - 계산 속도 **13배 향상** (127.8초 → 9.8초)
   - 고차원의 저주(Curse of Dimensionality) 극복

3. **Joint Optimization의 핵심 가치**
   - Two-stage 방법 (AE + K-Means) 대비 **+6.17%p** 향상
   - 복원과 클러스터링을 **동시 최적화**하여 더 나은 representation 학습
   - Self-training 메커니즘을 통한 지속적 개선

4. **학습 시간 vs 성능 Trade-off**
   - IDEC는 학습 시간이 길지만(**~60분**), **압도적 성능**으로 투자 가치 충분
   - 한 번 학습 후 **영구 재사용** 가능 (추론 시간 ~0.01초/1K)
   - **프로덕션 환경에 최적**: 배포 후 실시간 예측 가능

5. **실험 재현성**
   - 모든 실험은 `seed=42`로 고정
   - 결과는 `results/evaluation_metrics.json`에 저장됨

### 시각화 결과
<img width="5370" height="1674" alt="Image" src="https://github.com/user-attachments/assets/92396e69-5181-40ff-9753-60e161a02423" />

**클러스터 분리도 비교:**
- **Baseline K-Means**: 클러스터가 심하게 겹침 (고차원 공간의 한계)
- **AE + K-Means**: 클러스터가 분리되기 시작 (효과적인 차원 축소)
- **IDEC**: 클러스터가 **명확히 분리됨** (최적화된 representation) ⭐

**IDEC의 시각적 우수성:**
- ✨ 클러스터 간 뚜렷한 경계
- ✨ 클러스터 내부 높은 밀집도
- ✨ Overlapping 거의 제로

---

## 📚 기술 스택

- **Python 3.12**
- **PyTorch 2.0+**: 딥러닝 프레임워크
- **Scikit-learn**: K-Means, 평가 메트릭
- **NumPy, Pandas**: 데이터 처리
- **Matplotlib, Seaborn**: 시각화

---

## 👥 작성자

- **과목**: 인공지능응용
- **프로젝트**: Deep Clustering on Fashion-MNIST
