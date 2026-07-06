# ==============================================================================
#                      UEBA PROTOTYPE: SCIENTIFIC VERSION
# ------------------------------------------------------------------------------
#         Anomaly Detection in System Logs using a Transformer Model
#         with Baseline Comparison (Isolation Forest, One-Class SVM)
# ==============================================================================

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
import random

from config import Config
from seed import set_global_seed
from utils import make_splits, append_results_log
from preprocessing import prepare_data
from dataset import MaskedLogDataset, collate_fn_mlm
from model import TransformerPredictor
from trainer import train_model, calculate_threshold, tune_threshold_mcc, evaluate_hyperparams
from evaluator import evaluate_model, visualize_results, plot_comparison, report_per_category
from predictor import run_demo
from baseline import run_baselines
from baseline_lstm import run_lstm_baseline

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="UEBA Prototype — Scientific Version")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--data_path", type=str, default=Config.URL_STRUCTURED, help="Path to structured log file")
    parser.add_argument("--no_synthetic", action="store_true", help="Disable synthetic data fallback")
    parser.add_argument("--no_hpo", action="store_true", help="Disable Hyperparameter Optimization")
    parser.add_argument("--no_baselines", action="store_true", help="Skip classical baseline comparison")
    parser.add_argument("--no_lstm", action="store_true", help="Skip LSTM (DeepLog-style) neural baseline")
    parser.add_argument("--no_center_loss", action="store_true", help="Ablation: disable Center Loss")
    parser.add_argument("--passes", type=int, default=Config.NUM_STOCHASTIC_PASSES,
                        help="Ablation: number of stochastic MLM scoring passes")
    parser.add_argument("--score_agg", type=str, choices=["max", "mean"], default=Config.SCORE_AGG,
                        help="Ablation: per-token loss aggregation into session score")
    parser.add_argument("--sample_rate", type=float, default=Config.DATA_SAMPLE_RATE,
                        help="Fraction of sessions to use (speed up experiments)")
    parser.add_argument("--tag", type=str, default="default",
                        help="Experiment tag for results_log.csv (multi-seed aggregation)")
    parser.add_argument("--seed", type=int, default=Config.RANDOM_STATE, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # === ВОСПРОИЗВОДИМОСТЬ ===
    set_global_seed(args.seed)
    
    # Update Config with args
    Config.NUM_EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.URL_STRUCTURED = args.data_path
    Config.RANDOM_STATE = args.seed
    Config.NUM_STOCHASTIC_PASSES = args.passes
    Config.SCORE_AGG = args.score_agg
    Config.DATA_SAMPLE_RATE = args.sample_rate
    if args.no_synthetic:
        Config.USE_SYNTHETIC_FALLBACK = False
    if args.no_hpo:
        Config.HPO_ENABLED = False
    if args.no_center_loss:
        Config.CENTER_LOSS_ENABLED = False
    if args.no_lstm:
        Config.LSTM_ENABLED = False

    # ==========================================================================
    #                     STEP 1-2: DATA LOADING & PREPROCESSING
    # ==========================================================================
    print("=" * 70)
    print("   STEP 1-2: Data Loading and Preprocessing")
    print("=" * 70)
    session_df, label_encoder, vocab_size = prepare_data()

    # ==========================================================================
    #                     DATA SPLITTING (без data leakage!)
    # ==========================================================================
    # Разделение: Train(60%) / HPO-Val(13%) / Tune-Val(13%) / Test(14%)
    # (логика вынесена в utils.make_splits — используется и run_ablation.py)
    # ==========================================================================
    train_df, hpo_val_df, tune_val_df, test_df = make_splits(
        session_df, random_state=Config.RANDOM_STATE
    )

    # Prepare session lists
    normal_train_sessions = train_df[train_df['Label'] == 0]['EventCode'].tolist()
    
    hpo_val_sessions = hpo_val_df['EventCode'].tolist()
    hpo_val_labels = hpo_val_df['Label'].tolist()
    
    tune_val_sessions = tune_val_df['EventCode'].tolist()
    tune_val_labels = tune_val_df['Label'].tolist()
    
    test_sessions = test_df['EventCode'].tolist()
    test_session_labels = test_df['Label'].tolist()

    # DataLoaders
    train_dataset = MaskedLogDataset(normal_train_sessions, [0] * len(normal_train_sessions))
    hpo_val_dataset = MaskedLogDataset(hpo_val_sessions, hpo_val_labels)
    test_dataset = MaskedLogDataset(test_sessions, test_session_labels)

    from functools import partial
    collate = partial(collate_fn_mlm, vocab_size=vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate)
    hpo_val_loader = DataLoader(hpo_val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================================================
    #                     BASELINE COMPARISON (before Transformer)
    # ==========================================================================
    baseline_results = {}
    if not args.no_baselines:
        baseline_results = run_baselines(train_df, test_df, vocab_size)

    # Нейросетевой бейзлайн: LSTM next-event prediction (DeepLog-style).
    # Экспериментально обосновывает выбор Transformer против рекуррентных сетей.
    if Config.LSTM_ENABLED:
        baseline_results['LSTM (DeepLog)'] = run_lstm_baseline(
            train_df, tune_val_df, test_df, vocab_size, device
        )

    # ==========================================================================
    #                     HYPERPARAMETER OPTIMIZATION (AutoML)
    # ==========================================================================
    if Config.HPO_ENABLED:
        print("\n" + "=" * 70)
        print("   HYPERPARAMETER OPTIMIZATION (Random Search)")
        print("=" * 70)
        best_loss = float('inf')
        best_params = {
            'embed_size': Config.EMBED_SIZE,
            'num_heads': Config.NUM_HEADS,
            'num_layers': Config.NUM_LAYERS,
            'dropout': Config.DROPOUT
        }

        for trial in range(Config.HPO_NUM_TRIALS):
            current_params = {
                'embed_size': random.choice(Config.HPO_SEARCH_SPACE['embed_size']),
                'num_heads': random.choice(Config.HPO_SEARCH_SPACE['num_heads']),
                'num_layers': random.choice(Config.HPO_SEARCH_SPACE['num_layers']),
                'dropout': random.choice(Config.HPO_SEARCH_SPACE['dropout'])
            }
            # Constraint: embed_size must be divisible by num_heads
            if current_params['embed_size'] % current_params['num_heads'] != 0:
                valid_heads = [h for h in Config.HPO_SEARCH_SPACE['num_heads'] 
                              if current_params['embed_size'] % h == 0]
                current_params['num_heads'] = valid_heads[0] if valid_heads else 4

            # Используем HPO-Val (а не Tune-Val!) для оценки — без data leakage
            loss = evaluate_hyperparams(train_loader, hpo_val_loader, vocab_size, current_params, device)
            
            if loss < best_loss:
                best_loss = loss
                best_params = current_params
        
        print(f"\n=== AutoML Result ===")
        print(f"Best Configuration: {best_params}")
        print(f"Validation Loss: {best_loss:.4f}")
        
        Config.EMBED_SIZE = best_params['embed_size']
        Config.NUM_HEADS = best_params['num_heads']
        Config.NUM_LAYERS = best_params['num_layers']
        Config.DROPOUT = best_params['dropout']

    # ==========================================================================
    #                     STEP 3: MODEL CREATION
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"   STEP 3: Transformer Model (BERT-style MLM)")
    print(f"{'='*70}")
    model = TransformerPredictor(
        vocab_size=vocab_size,
        embed_size=Config.EMBED_SIZE,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model on {device}. Parameters: {total_params:,}")
    print(f"Architecture: embed={Config.EMBED_SIZE}, heads={Config.NUM_HEADS}, "
          f"layers={Config.NUM_LAYERS}, dropout={Config.DROPOUT}")

    # ==========================================================================
    #                     STEP 4: TRAINING
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"   STEP 4: Training ({Config.NUM_EPOCHS} epochs)")
    print(f"{'='*70}")
    model = train_model(model, train_loader, hpo_val_loader, device)

    # ==========================================================================
    #                     STEP 5: THRESHOLD CALCULATION
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"   STEP 5: Anomaly Threshold (Statistical + MCC)")
    print(f"{'='*70}")
    anomaly_threshold = calculate_threshold(model, train_df, device, vocab_size)
    
    # Тюнинг на Tune-Val (отдельный набор — без leakage!)
    anomaly_threshold, val_sigma = tune_threshold_mcc(
        model, tune_val_sessions, tune_val_labels, 
        anomaly_threshold, device, vocab_size
    )

    # ==========================================================================
    #                     STEP 6: EVALUATION ON TEST SET
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"   STEP 6: Final Evaluation on Test Set")
    print(f"{'='*70}")
    test_pred_labels, test_losses, test_true_labels, test_session_ids, suspicious_threshold = evaluate_model(
        model, test_loader, anomaly_threshold, device, sigma=val_sigma
    )

    # Разбор detection rate по категориям аномалий BGL (KERNDTLB, KERNSTOR, ...)
    report_per_category(
        test_losses, test_true_labels, test_session_ids,
        test_df, anomaly_threshold, suspicious_threshold
    )

    # Собираем метрики Transformer для сравнения
    from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef, roc_auc_score, average_precision_score
    transformer_metrics = {
        'precision': precision_score(test_true_labels, test_pred_labels, zero_division=0),
        'recall': recall_score(test_true_labels, test_pred_labels, zero_division=0),
        'f1': f1_score(test_true_labels, test_pred_labels, zero_division=0),
        'f2': fbeta_score(test_true_labels, test_pred_labels, beta=2, zero_division=0),
        'mcc': matthews_corrcoef(test_true_labels, test_pred_labels),
    }
    try:
        transformer_metrics['roc_auc'] = roc_auc_score(test_true_labels, test_losses)
    except ValueError:
        transformer_metrics['roc_auc'] = float('nan')
    try:
        transformer_metrics['pr_auc'] = average_precision_score(test_true_labels, test_losses)
    except ValueError:
        transformer_metrics['pr_auc'] = float('nan')

    # ==========================================================================
    #                     STEP 7: SAVE ARTIFACTS
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"   STEP 7: Save Model and Artifacts")
    print(f"{'='*70}")
    try:
        torch.save(model.state_dict(), Config.MODEL_PATH)
        with open(Config.ENCODER_PATH, 'wb') as f:
            pickle.dump(label_encoder, f)
        with open(Config.THRESHOLD_PATH, 'w') as f:
            f.write(f"{anomaly_threshold},{val_sigma}")
        print("Model, label encoder, and threshold saved successfully.")
    except Exception as e:
        print(f"Error saving artifacts: {e}")

    # ==========================================================================
    #                     STEP 8: VISUALIZATION & COMPARISON
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"   STEP 8: Visualization")
    print(f"{'='*70}")
    visualize_results(test_pred_labels, test_losses, test_true_labels, anomaly_threshold)
    
    # Сравнительная таблица и график
    all_results = {'Transformer (BERT)': transformer_metrics}
    all_results.update(baseline_results)

    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("   FINAL COMPARISON: Transformer vs Baselines")
        print("=" * 80)
        print(f"{'Model':<20} {'MCC':>8} {'F1':>8} {'F2':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8} {'PR-AUC':>8}")
        print("-" * 80)
        for name, m in all_results.items():
            auc_str = f"{m['roc_auc']:.4f}" if not np.isnan(m.get('roc_auc', float('nan'))) else "   N/A"
            pr_str = f"{m.get('pr_auc', float('nan')):.4f}" if not np.isnan(m.get('pr_auc', float('nan'))) else "   N/A"
            print(f"{name:<20} {m['mcc']:>8.4f} {m['f1']:>8.4f} {m['f2']:>8.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f} {auc_str:>8} {pr_str:>8}")
        print("=" * 80)

        plot_comparison(all_results)

    # Multi-seed протокол: дописываем метрики в results_log.csv,
    # aggregate_results.py посчитает mean±std по сидам
    log_row = {
        'tag': args.tag,
        'seed': args.seed,
        'epochs': Config.NUM_EPOCHS,
        'center_loss': Config.CENTER_LOSS_ENABLED,
        'passes': Config.NUM_STOCHASTIC_PASSES,
        'score_agg': Config.SCORE_AGG,
        'sample_rate': Config.DATA_SAMPLE_RATE,
    }
    for metric_name, value in transformer_metrics.items():
        log_row[metric_name] = round(float(value), 6)
    append_results_log(Config.RESULTS_LOG_PATH, log_row)

    # ==========================================================================
    #                     STEP 9: DEMO
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"   STEP 9: Interactive Demo")
    print(f"{'='*70}")
    run_demo(test_df, device, vocab_size)


if __name__ == "__main__":
    main()