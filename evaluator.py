import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, recall_score, matthews_corrcoef, fbeta_score
from config import Config

def evaluate_model(model, test_loader, anomaly_threshold, device, sigma=0.5):
    """
    Evaluates the model on the test set and calculates metrics.
    """
    print("\n--- Test Model and Calculate Performance Metrics (Session-Based) ---")
    model.eval()
    session_losses = defaultdict(list)
    session_true_labels = {}
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    num_passes = Config.NUM_STOCHASTIC_PASSES
    for _ in range(num_passes):
        with torch.no_grad():
            for inputs, targets, labels, mask, session_indices in tqdm(test_loader, desc="Evaluating on Test Set", leave=False):
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                outputs, _ = model(inputs, mask)
                
                loss_per_token = criterion(outputs.permute(0, 2, 1), targets) # (B, L)
                active_mask = targets != -100
                
                for i in range(len(session_indices)):
                    session_id = session_indices[i].item()
                    
                    # Max loss over masked tokens
                    session_mask = active_mask[i]
                    if session_mask.any():
                        max_session_loss = loss_per_token[i][session_mask].max().item()
                        session_losses[session_id].append(max_session_loss)
                    
                    if session_id not in session_true_labels:
                        session_true_labels[session_id] = labels[i].item()

    ordered_session_ids = sorted(session_losses.keys())
    # Use np.mean to be consistent with trainer.py (MLM pseudo-likelihood)
    final_scores = [float(np.mean(session_losses[sid])) if len(session_losses[sid]) > 0 else 0.0 for sid in ordered_session_ids]
    final_true_labels = [int(session_true_labels.get(sid, 0)) for sid in ordered_session_ids]
    
    suspicious_threshold = anomaly_threshold - (Config.SUSPICIOUS_SIGMA_MARGIN * sigma)
    suspicious_threshold = max(0.5, suspicious_threshold)  # Clamp to reasonable minimum

    test_pred_labels_anomaly = [1 if score > anomaly_threshold else 0 for score in final_scores]
    test_pred_labels_detected = [1 if score > suspicious_threshold else 0 for score in final_scores]

    precision, recall, f1, _ = precision_recall_fscore_support(final_true_labels, test_pred_labels_anomaly, average='binary', zero_division=0)
    
    # Scientific metrics
    mcc = matthews_corrcoef(final_true_labels, test_pred_labels_anomaly)
    f2 = fbeta_score(final_true_labels, test_pred_labels_anomaly, beta=2, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(final_true_labels, final_scores)
    except ValueError as e:
        print(f"Could not compute ROC-AUC: {e}")
        roc_auc = np.nan

    detection_rate = recall_score(final_true_labels, test_pred_labels_detected, zero_division=0)

    print("\nPerformance Metrics on Test Set (Session-Based):")
    print(f"  - Anomaly Threshold: {anomaly_threshold:.4f}")
    print(f"  - Suspicious Threshold: {suspicious_threshold:.4f}")
    print("--- Metrics for 'Hard' Anomalies (score > anomaly_threshold) ---")
    print(f"  - Precision: {precision:.4f} (When system says 'anomaly!', it is correct in {precision:.0%})")
    print(f"  - Recall:    {recall:.4f} (System finds {recall:.0%} of explicit anomalies)")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - F2-Score:  {f2:.4f} (Recall-weighted, critical for security)")
    print(f"  - MCC:       {mcc:.4f} (Matthew's Correlation Coefficient, robust to imbalance)")
    print(f"  - ROC-AUC:   {roc_auc:.4f} (Overall discriminative power)")
    print("--- Metrics for 'Any' Detected Threat (score > suspicious_threshold) ---")
    print(f"  - Detection Rate (Recall): {detection_rate:.4f} (System NOTICED {detection_rate:.0%} of ALL real anomalies)")

    return test_pred_labels_anomaly, final_scores, final_true_labels

def visualize_results(test_pred_labels, test_losses, test_true_labels, anomaly_threshold):
    """
    Visualizes confusion matrix and anomaly score distribution.
    """
    print("\n--- Visualize Results ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Model Performance Analysis', fontsize=16)

    # 1. Confusion Matrix
    cm = confusion_matrix(test_true_labels, test_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
                annot_kws={"size": 14})
    ax1.set_title('Confusion Matrix', fontsize=14)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    # 2. Distribution of Anomaly Scores
    normal_scores = [score for score, label in zip(test_losses, test_true_labels) if label == 0]
    abnormal_scores = [score for score, label in zip(test_losses, test_true_labels) if label == 1]

    def safe_plot_scores(ax, data, label, color):
        if len(data) >= 2 and np.var(data) > 1e-8:
            sns.kdeplot(data, label=label, fill=True, ax=ax)
        elif len(data) > 0:
            ax.hist(data, bins=min(20, max(1, len(data))), alpha=0.5, label=label)
            ax.plot(data, [0.0] * len(data), marker='|', linestyle='', color=color, markersize=8)
        else:
            pass

    safe_plot_scores(ax2, normal_scores, 'Normal Sessions', 'blue')
    safe_plot_scores(ax2, abnormal_scores, 'Abnormal Sessions', 'red')

    ax2.axvline(anomaly_threshold, color='green', linestyle='--', label=f'Threshold ({anomaly_threshold:.2f})')
    ax2.set_title('Distribution of Anomaly Scores', fontsize=14)
    ax2.set_xlabel('Anomaly Score (Max CE Loss in Session)')
    ax2.set_ylabel('Density')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: performance.png")


def plot_comparison(all_results: dict):
    """
    Строит сравнительный bar chart: Transformer vs Baselines.
    Сохраняет в comparison.png.
    
    Args:
        all_results: dict {model_name: {metric_name: value}}
    """
    print("\n--- Generating Comparison Chart ---")
    
    metrics_to_plot = ['mcc', 'f1', 'f2', 'precision', 'recall']
    metric_labels = ['MCC', 'F1-Score', 'F2-Score', 'Precision', 'Recall']
    
    models = list(all_results.keys())
    n_models = len(models)
    n_metrics = len(metrics_to_plot)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    # Цвета: Transformer = зелёный/бирюзовый, baselines = оттенки серого/синего
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model_name in enumerate(models):
        values = [all_results[model_name].get(m, 0) for m in metrics_to_plot]
        bars = ax.bar(x + i * width - (n_models - 1) * width / 2, values, 
                      width, label=model_name, color=colors[i % len(colors)],
                      edgecolor='white', linewidth=0.5)
        
        # Значения над столбцами
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison: Transformer (BERT) vs Classical Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: comparison.png")
