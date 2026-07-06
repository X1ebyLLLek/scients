"""
run_ablation.py — Ablation study: вклад каждого научного улучшения в итоговое качество.

Проверяемые компоненты:
  1. Center Loss        — обучение с/без кластеризации нормальных эмбеддингов
  2. Stochastic Scoring — 3 стохастических прохода vs 1 проход
  3. Score Aggregation  — max (точечные аномалии) vs mean (pseudo-log-likelihood)

Протокол (честный, без утечки):
  - Обучаются ДВЕ модели: Full (Center Loss ON) и NoCenterLoss (OFF);
    варианты скоринга (passes, agg) не требуют переобучения — модель Full
    переоценивается с другими параметрами скоринга;
  - для КАЖДОГО варианта порог заново MCC-тюнится на Tune-Val
    (распределение оценок меняется — фиксированный порог был бы нечестен);
  - все варианты оцениваются на одном и том же Test.

Запуск (Colab):
  python run_ablation.py --epochs 15 --seed 42
  python run_ablation.py --epochs 5 --sample_rate 0.2   # быстрая версия

Результат: печатная таблица + ablation_results.csv (таблица для главы 3 диплома).
"""

import argparse
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from seed import set_global_seed
from preprocessing import prepare_data
from utils import make_splits
from dataset import MaskedLogDataset, collate_fn_mlm
from model import TransformerPredictor
from trainer import train_model, compute_session_scores_from_sessions
from baseline import compute_metrics
from baseline_lstm import tune_threshold_mcc_on_scores


def parse_args():
    parser = argparse.ArgumentParser(description="UEBA Ablation Study")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per variant")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Fraction of sessions to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default=Config.URL_STRUCTURED)
    return parser.parse_args()


def train_variant(name, center_loss_enabled, train_loader, hpo_val_loader, vocab_size, device, seed):
    """Обучает модель варианта с чистого листа (свой checkpoint, общий seed)."""
    print("\n" + "=" * 70)
    print(f"   ABLATION TRAINING: {name} (center_loss={center_loss_enabled})")
    print("=" * 70)

    set_global_seed(seed)  # одинаковая инициализация для честного сравнения
    Config.CENTER_LOSS_ENABLED = center_loss_enabled
    Config.CHECKPOINT_PATH = f"checkpoint_ablation_{name}.pth"

    model = TransformerPredictor(
        vocab_size=vocab_size,
        embed_size=Config.EMBED_SIZE,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(device)

    return train_model(model, train_loader, hpo_val_loader, device)


def evaluate_variant(name, model, tune_val_df, test_df, vocab_size, device, passes, agg):
    """MCC-тюнинг порога на Tune-Val + метрики на Test для заданного скоринга."""
    Config.NUM_STOCHASTIC_PASSES = passes
    Config.SCORE_AGG = agg

    print(f"\n--- Ablation variant: {name} (passes={passes}, agg={agg}) ---")

    tune_scores = compute_session_scores_from_sessions(
        tune_val_df['EventCode'].tolist(), model, device, vocab_size)
    threshold, tune_mcc = tune_threshold_mcc_on_scores(tune_scores, tune_val_df['Label'].tolist())
    print(f"    Threshold (MCC-tuned): {threshold:.4f} (val MCC={tune_mcc:.4f})")

    test_scores = compute_session_scores_from_sessions(
        test_df['EventCode'].tolist(), model, device, vocab_size)
    y_true = np.asarray(test_df['Label'].tolist(), dtype=int)
    y_pred = (np.asarray(test_scores) > threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_scores=test_scores)
    metrics['variant'] = name
    metrics['threshold'] = round(threshold, 4)
    print(f"    MCC: {metrics['mcc']:.4f}, F1: {metrics['f1']:.4f}, "
          f"AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")
    return metrics


def main():
    args = parse_args()
    set_global_seed(args.seed)

    Config.NUM_EPOCHS = args.epochs
    Config.DATA_SAMPLE_RATE = args.sample_rate
    Config.URL_STRUCTURED = args.data_path
    Config.RANDOM_STATE = args.seed
    Config.HPO_ENABLED = False  # фиксированная архитектура — сравниваем компоненты, не архитектуры

    print("=" * 70)
    print("   ABLATION STUDY: Center Loss / Stochastic Passes / Score Aggregation")
    print("=" * 70)

    session_df, label_encoder, vocab_size = prepare_data()
    train_df, hpo_val_df, tune_val_df, test_df = make_splits(session_df, random_state=args.seed)

    normal_train_sessions = train_df[train_df['Label'] == 0]['EventCode'].tolist()
    train_dataset = MaskedLogDataset(normal_train_sessions, [0] * len(normal_train_sessions))
    hpo_val_dataset = MaskedLogDataset(hpo_val_df['EventCode'].tolist(), hpo_val_df['Label'].tolist())

    collate = partial(collate_fn_mlm, vocab_size=vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate)
    hpo_val_loader = DataLoader(hpo_val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []

    # === Модель 1: Full (Center Loss ON) + варианты скоринга без переобучения ===
    model_full = train_variant("full", True, train_loader, hpo_val_loader, vocab_size, device, args.seed)
    results.append(evaluate_variant("Full (CL + 3 passes + max)", model_full,
                                    tune_val_df, test_df, vocab_size, device, passes=3, agg="max"))
    results.append(evaluate_variant("- Stochastic (1 pass)", model_full,
                                    tune_val_df, test_df, vocab_size, device, passes=1, agg="max"))
    results.append(evaluate_variant("- Max agg (mean)", model_full,
                                    tune_val_df, test_df, vocab_size, device, passes=3, agg="mean"))

    # === Модель 2: без Center Loss, скоринг по умолчанию ===
    model_nocl = train_variant("nocl", False, train_loader, hpo_val_loader, vocab_size, device, args.seed)
    Config.CENTER_LOSS_ENABLED = True  # восстановить для чистоты
    results.append(evaluate_variant("- Center Loss", model_nocl,
                                    tune_val_df, test_df, vocab_size, device, passes=3, agg="max"))

    # === Итоговая таблица ===
    print("\n" + "=" * 90)
    print("   ABLATION RESULTS (test set)")
    print("=" * 90)
    print(f"{'Variant':<30} {'MCC':>8} {'F1':>8} {'F2':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8} {'PR-AUC':>8}")
    print("-" * 90)
    for m in results:
        print(f"{m['variant']:<30} {m['mcc']:>8.4f} {m['f1']:>8.4f} {m['f2']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")
    print("=" * 90)

    df = pd.DataFrame(results)
    cols = ['variant', 'mcc', 'f1', 'f2', 'precision', 'recall', 'roc_auc', 'pr_auc', 'threshold']
    df[cols].to_csv("ablation_results.csv", index=False)
    print("Saved: ablation_results.csv")


if __name__ == "__main__":
    main()
