# ==============================================================================
#                      UEBA PROTOTYPE: FINAL DEMO VERSION
# ------------------------------------------------------------------------------
#         Anomaly Detection in System Logs using a Transformer Model
# ==============================================================================
# Refactored Version

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle

from config import Config
from utils import robust_train_test_split
from preprocessing import prepare_data
from dataset import MaskedLogDataset, collate_fn_mlm
from model import TransformerPredictor
from trainer import train_model, calculate_threshold, tune_threshold
from evaluator import evaluate_model, visualize_results
from predictor import run_demo

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="UEBA Prototype")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--data_path", type=str, default=Config.URL_STRUCTURED, help="Path to structured log file")
    parser.add_argument("--no_synthetic", action="store_true", help="Disable synthetic data fallback")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"DEBUG: Default Config.NUM_EPOCHS={Config.NUM_EPOCHS}")
    print(f"DEBUG: Args.epochs={args.epochs}")
    
    # Update Config with args
    Config.NUM_EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.URL_STRUCTURED = args.data_path
    if args.no_synthetic:
        Config.USE_SYNTHETIC_FALLBACK = False

    print("--- Step 1 & 2: Data Loading and Preprocessing ---")
    session_df, label_encoder, vocab_size = prepare_data()

    # 1. Split sessions
    # 40% data with min 100 anomalies for temp_test_df used for val/test
    train_df, temp_test_df = robust_train_test_split(session_df, test_size=0.4, min_anomalies_in_test=100,
                                                     random_state=Config.RANDOM_STATE)

    # 2. Split temp_test_df into Val and Test
    val_df = temp_test_df.sample(frac=0.5, random_state=Config.RANDOM_STATE)
    test_df = temp_test_df.drop(val_df.index)
    val_df = val_df.reset_index(drop=True)

    print(f"Train sessions: {len(train_df)}, Validation sessions: {len(val_df)}, Test sessions: {len(test_df)}")
    print(f"Anomalies in validation set: {val_df['Label'].sum()}")
    print(f"Anomalies in test set: {test_df['Label'].sum()}")

    # 3. Prepare Lists for Datasets
    normal_train_sessions = train_df[train_df['Label'] == 0]['EventCode'].tolist()
    val_sessions = val_df['EventCode'].tolist()
    val_labels = val_df['Label'].tolist()
    test_sessions = test_df['EventCode'].tolist()
    test_session_labels = test_df['Label'].tolist()

    # 4. Create Datasets and DataLoaders
    # Train only on normal sessions
    # Use MaskedLogDataset for training (MLM)
    train_dataset = MaskedLogDataset(normal_train_sessions, [0] * len(normal_train_sessions))
    # For validation and test, we still use MaskedLogDataset but we care about the anomaly score 
    # which is derived from how well we predict the masked tokens (pseudolikelihood)
    val_dataset = MaskedLogDataset(val_sessions, val_labels)
    test_dataset_with_labels = MaskedLogDataset(test_sessions, test_session_labels)

    # We need to pass vocab_size to collate_fn_mlm
    from functools import partial

    
    collate = partial(collate_fn_mlm, vocab_size=vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader_eval = DataLoader(test_dataset_with_labels, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate)

    print(f"Training data samples: {len(train_dataset)}")
    print(f"Validation data samples: {len(val_dataset)}")
    print(f"Test data samples: {len(test_dataset_with_labels)}")

    print("\n--- Step 3: Define Enhanced Transformer Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(
        vocab_size=vocab_size,
        embed_size=Config.EMBED_SIZE,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(device)
    print(f"Model created and moved to {device}.")

    print("\n--- Step 4: Train the Model ---")
    model = train_model(model, train_loader, val_loader, device)

    print("\n--- Step 5: Anomaly Threshold Calculation ---")
    anomaly_threshold = calculate_threshold(model, train_df, device, vocab_size)
    
    # Tune threshold
    anomaly_threshold = tune_threshold(model, val_sessions, val_labels, anomaly_threshold, device, vocab_size)

    print("\n--- Step 6: Model Evaluation ---")
    test_pred_labels, test_losses, test_true_labels = evaluate_model(model, test_loader_eval, anomaly_threshold, device)

    print("\n--- Step 7: Save Model and Artifacts ---")
    try:
        torch.save(model.state_dict(), Config.MODEL_PATH)
        with open(Config.ENCODER_PATH, 'wb') as f:
            pickle.dump(label_encoder, f)
        with open(Config.THRESHOLD_PATH, 'w') as f:
            f.write(str(anomaly_threshold))
        print("Model, label encoder, and threshold saved successfully.")
    except Exception as e:
        print(f"Error saving artifacts: {e}")

    print("\n--- Step 8: Visualize Results ---")
    visualize_results(test_pred_labels, test_losses, test_true_labels, anomaly_threshold)

    print("\n--- Step 9: Demo ---")
    run_demo(test_df, device, vocab_size)

if __name__ == "__main__":
    main()