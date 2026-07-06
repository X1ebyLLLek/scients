"""
baseline.py — Сравнительные модели (baselines) для валидации Transformer-подхода.

Реализует классические методы обнаружения аномалий:
  1. Isolation Forest (Liu et al., 2008)
  2. One-Class SVM (Schölkopf et al., 2001)

Используется для экспериментального подтверждения превосходства
Transformer-архитектуры над классическими методами.
"""

import numpy as np
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from config import Config


def build_feature_matrix(session_df, vocab_size: int):
    """
    Преобразует сессии (списки EventCode) в матрицу признаков
    для классических ML-моделей.
    
    Метод: Bag-of-Events — подсчёт частоты каждого типа события в сессии.
    Это стандартный подход для baseline в log-based anomaly detection
    (см. LogCluster, PCA-based detection).
    
    Args:
        session_df: DataFrame с колонкой 'EventCode' (list of int)
        vocab_size: Размер словаря
        
    Returns:
        X: np.ndarray shape (n_sessions, vocab_size) — матрица признаков
        y: np.ndarray shape (n_sessions,) — метки (0=Normal, 1=Anomaly)
    """
    n_sessions = len(session_df)
    X = np.zeros((n_sessions, vocab_size), dtype=np.float32)
    
    for i, codes in enumerate(session_df['EventCode'].values):
        counts = Counter(codes)
        for code, count in counts.items():
            if code < vocab_size:
                X[i, code] = count
        # Нормализация по длине сессии (TF — Term Frequency)
        total = len(codes) if len(codes) > 0 else 1
        X[i] /= total
    
    y = session_df['Label'].values.astype(int)
    
    return X, y


def compute_metrics(y_true, y_pred, y_scores=None):
    """
    Вычисляет стандартный набор метрик для сравнения.
    
    Returns:
        dict с ключами: precision, recall, f1, f2, mcc, roc_auc
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = float('nan')
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        except ValueError:
            metrics['pr_auc'] = float('nan')
    else:
        metrics['roc_auc'] = float('nan')
        metrics['pr_auc'] = float('nan')

    return metrics


def run_isolation_forest(X_train, X_test, y_test):
    """
    Isolation Forest — ансамблевый метод на базе деревьев изоляции.
    
    Принцип: аномалии легче изолировать (требуют меньше разбиений),
    поэтому имеют короткий путь в дереве.
    
    Обучение: только на нормальных данных (novelty detection mode).
    """
    print("  Running Isolation Forest...")
    
    clf = IsolationForest(
        n_estimators=200,
        contamination='auto',
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train)
    
    # score_samples: чем ниже — тем аномальнее
    scores = -clf.score_samples(X_test)  # Инвертируем: выше = аномальнее
    preds = clf.predict(X_test)          # 1=normal, -1=anomaly
    y_pred = np.where(preds == -1, 1, 0)  # Конвертируем в наш формат
    
    metrics = compute_metrics(y_test, y_pred, y_scores=scores)
    print(f"    IF — MCC: {metrics['mcc']:.4f}, F1: {metrics['f1']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")
    
    return metrics


def run_ocsvm(X_train, X_test, y_test):
    """
    One-Class SVM — метод опорных векторов для одноклассовой классификации.
    
    Принцип: строит гиперплоскость, максимально отделяющую данные от origin
    в пространстве признаков. Точки за гиперплоскостью — аномалии.
    
    Обучение: только на нормальных данных.
    
    Примечание: OC-SVM медленный на больших данных, поэтому при N_train > 10000
    мы сэмплируем подмножество для обучения.
    """
    print("  Running One-Class SVM...")
    
    # OC-SVM не масштабируется на большие данные — сэмплируем
    max_train_samples = 10000
    if X_train.shape[0] > max_train_samples:
        indices = np.random.choice(X_train.shape[0], max_train_samples, replace=False)
        X_train_subset = X_train[indices]
        print(f"    Subsampled training set from {X_train.shape[0]} to {max_train_samples}")
    else:
        X_train_subset = X_train
    
    clf = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.05  # ≈ expected fraction of outliers
    )
    clf.fit(X_train_subset)
    
    # decision_function: чем ниже — тем аномальнее
    scores = -clf.decision_function(X_test)  # Инвертируем
    preds = clf.predict(X_test)               # 1=normal, -1=anomaly
    y_pred = np.where(preds == -1, 1, 0)
    
    metrics = compute_metrics(y_test, y_pred, y_scores=scores)
    print(f"    OCSVM — MCC: {metrics['mcc']:.4f}, F1: {metrics['f1']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")
    
    return metrics


def run_baselines(train_df, test_df, vocab_size: int):
    """
    Запускает все baseline-модели и возвращает сводку результатов.
    
    Args:
        train_df: DataFrame с обучающими сессиями (только нормальные для OC detection!)
        test_df: DataFrame с тестовыми сессиями (нормальные + аномальные)
        vocab_size: Размер словаря
        
    Returns:
        results: dict of {model_name: metrics_dict}
    """
    print("\n" + "="*60)
    print("   BASELINE COMPARISON: Classical Anomaly Detection")
    print("="*60)
    
    # Формируем матрицу признаков
    print("\nBuilding feature matrix (Bag-of-Events / TF)...")
    
    # Для baseline: обучение на НОРМАЛЬНЫХ сессиях из train
    train_normal = train_df[train_df['Label'] == 0]
    X_train, _ = build_feature_matrix(train_normal, vocab_size)
    X_test, y_test = build_feature_matrix(test_df, vocab_size)
    
    print(f"  Train (normal only): {X_train.shape[0]} sessions, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} sessions ({(y_test == 1).sum()} anomalies)")
    
    results = {}
    
    # 1. Isolation Forest
    results['Isolation Forest'] = run_isolation_forest(X_train, X_test, y_test)
    
    # 2. One-Class SVM
    results['One-Class SVM'] = run_ocsvm(X_train, X_test, y_test)
    
    # Печатаем сводную таблицу
    print("\n" + "-"*70)
    print(f"{'Model':<20} {'MCC':>8} {'F1':>8} {'F2':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8}")
    print("-"*70)
    for name, m in results.items():
        print(f"{name:<20} {m['mcc']:>8.4f} {m['f1']:>8.4f} {m['f2']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['roc_auc']:>8.4f}")
    print("-"*70)
    
    return results
