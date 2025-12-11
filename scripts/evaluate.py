# scripts\evaluate.py
"""
평가 스크립트
ACC, NMI, ARI 계산 및 혼동 행렬 생성
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import numpy as np
import json
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix
)
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns

from configs import load_config
from data import get_full_dataset


def calculate_accuracy(y_true, y_pred):
    """
    Hungarian algorithm으로 최적 매칭 후 accuracy 계산
    
    Args:
        y_true: 실제 라벨 (N,)
        y_pred: 예측 라벨 (N,)
        
    Returns:
        acc: Accuracy
        best_map: 최적 매핑 딕셔너리
    """
    # Confusion matrix 생성
    n_clusters = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # Hungarian algorithm으로 최적 매칭
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # 매칭 딕셔너리 생성
    best_map = {row: col for row, col in zip(row_ind, col_ind)}
    
    # 매칭된 예측값
    y_pred_matched = np.array([best_map[pred] for pred in y_pred])
    
    # Accuracy 계산
    acc = np.sum(y_pred_matched == y_true) / y_true.size
    
    return acc, best_map


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """
    혼동 행렬 시각화
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        title: 그래프 제목
        save_path: 저장 경로
    """
    # Hungarian algorithm으로 최적 매칭
    _, best_map = calculate_accuracy(y_true, y_pred)
    y_pred_matched = np.array([best_map[pred] for pred in y_pred])
    
    # Confusion matrix 계산
    cm = confusion_matrix(y_true, y_pred_matched)
    
    # 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10),
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 저장: {save_path}")


def evaluate_method(method_name, predictions_path, y_true, figures_dir):
    """
    특정 방법의 성능 평가
    
    Args:
        method_name: 방법 이름
        predictions_path: 예측 결과 경로
        y_true: 실제 라벨
        figures_dir: 그림 저장 디렉토리
        
    Returns:
        metrics: 평가 메트릭 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"📊 {method_name} 평가")
    print(f"{'='*60}")
    
    # 예측 결과 로드
    if not os.path.exists(predictions_path):
        print(f"❌ 파일을 찾을 수 없습니다: {predictions_path}")
        return None
    
    y_pred = np.load(predictions_path)
    print(f"✅ 예측 결과 로드: {predictions_path}")
    print(f"   Shape: {y_pred.shape}")
    
    # 메트릭 계산
    acc, best_map = calculate_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"\n📈 성능 메트릭:")
    print(f"   ACC: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   NMI: {nmi:.4f}")
    print(f"   ARI: {ari:.4f}")
    
    # 클러스터 분포
    print(f"\n📊 클러스터 분포:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        percentage = count / len(y_pred) * 100
        print(f"   Cluster {cluster_id}: {count:5d}개 ({percentage:.1f}%)")
    
    # 혼동 행렬 생성
    print(f"\n🎨 혼동 행렬 생성 중...")
    cm_path = os.path.join(figures_dir, f'confusion_matrix_{method_name.lower().replace(" ", "_")}.png')
    plot_confusion_matrix(y_true, y_pred, f'Confusion Matrix - {method_name}', cm_path)
    
    # 메트릭 딕셔너리
    metrics = {
        'method': method_name,
        'acc': float(acc),
        'nmi': float(nmi),
        'ari': float(ari),
        'n_samples': int(len(y_pred)),
        'cluster_distribution': {int(k): int(v) for k, v in zip(unique, counts)},
        'best_mapping': {int(k): int(v) for k, v in best_map.items()}
    }
    
    return metrics


def create_comparison_table(all_metrics, save_path):
    """
    비교표 생성 및 저장
    
    Args:
        all_metrics: 모든 방법의 메트릭 리스트
        save_path: 저장 경로
    """
    print(f"\n{'='*60}")
    print("📊 최종 비교표")
    print(f"{'='*60}")
    
    # 테이블 헤더
    print(f"\n{'방법':<20} {'ACC':<12} {'NMI':<12} {'ARI':<12}")
    print("-" * 60)
    
    # 각 방법 출력
    for metrics in all_metrics:
        if metrics:
            print(f"{metrics['method']:<20} "
                  f"{metrics['acc']:.4f} ({metrics['acc']*100:5.2f}%) "
                  f"{metrics['nmi']:.4f}       "
                  f"{metrics['ari']:.4f}")
    
    print("-" * 60)
    
    # 최고 성능 표시
    if all_metrics:
        valid_metrics = [m for m in all_metrics if m is not None]
        if valid_metrics:
            best_acc = max(valid_metrics, key=lambda x: x['acc'])
            best_nmi = max(valid_metrics, key=lambda x: x['nmi'])
            best_ari = max(valid_metrics, key=lambda x: x['ari'])
            
            print(f"\n🏆 최고 성능:")
            print(f"   ACC: {best_acc['method']} ({best_acc['acc']*100:.2f}%)")
            print(f"   NMI: {best_nmi['method']} ({best_nmi['nmi']:.4f})")
            print(f"   ARI: {best_ari['method']} ({best_ari['ari']:.4f})")
    
    # JSON 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {save_path}")


def main():
    """메인 평가 함수"""
    
    print("\n" + "="*60)
    print("평가 시작")
    print("="*60)
    
    # ============================================================
    # 1. 설정 로드
    # ============================================================
    config = load_config('configs/config.yaml')
    results_dir = config['paths']['results_dir']
    figures_dir = config['paths']['figures_dir']
    
    # ============================================================
    # 2. 실제 라벨 로드
    # ============================================================
    print(f"\n📂 데이터 로딩...")
    _, y_true = get_full_dataset(config['data']['data_dir'])
    print(f"✅ 실제 라벨 로드 완료: {y_true.shape}")
    
    # ============================================================
    # 3. 각 방법 평가
    # ============================================================
    all_metrics = []
    
    # Baseline K-Means
    metrics = evaluate_method(
        "Baseline K-Means",
        os.path.join(results_dir, 'baseline_predictions.npy'),
        y_true,
        figures_dir
    )
    all_metrics.append(metrics)
    
    # AE + K-Means
    metrics = evaluate_method(
        "AE + K-Means",
        os.path.join(results_dir, 'ae_kmeans_predictions.npy'),
        y_true,
        figures_dir
    )
    all_metrics.append(metrics)
    
    # IDEC (마지막 epoch)
    idec_epochs = [100]  # 또는 [10, 50, 100] 등 여러 epoch 확인
    for epoch in idec_epochs:
        idec_path = os.path.join(results_dir, f'idec_predictions_epoch{epoch}.npy')
        if os.path.exists(idec_path):
            metrics = evaluate_method(
                f"IDEC (Epoch {epoch})",
                idec_path,
                y_true,
                figures_dir
            )
            all_metrics.append(metrics)
    
    # ============================================================
    # 4. 비교표 생성
    # ============================================================
    comparison_path = os.path.join(results_dir, 'evaluation_metrics.json')
    create_comparison_table(all_metrics, comparison_path)
    
    print("\n" + "="*60)
    print("✅ 평가 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()