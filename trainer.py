import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import fbeta_score
from config import Config
from dataset import MaskedLogDataset, collate_fn_mlm
from torch.utils.data import DataLoader

import os

def train_model(model, train_loader, val_loader, device):
    """
    Trains the model and returns the trained model.
    """
    print("\n--- Train the Model on Normal Data (Masked Language Modeling) ---")
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignore unmasked tokens
    # Use standard Adam to avoid potential FSDP/Import hangs with AdamW on some Colab runtimes
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-3)
    # Increase patience to allow model to overcome small fluctuations
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    start_epoch = 0
    patience_counter = 0

    # RESUME LOGIC
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"--> Found checkpoint '{Config.CHECKPOINT_PATH}'. Resuming training...")
        try:
            checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"--> Resumed from Epoch {start_epoch}. Best Val Loss so far: {best_loss:.4f}")
        except Exception as e:
            print(f"--> Error loading checkpoint: {e}. Starting from scratch.")

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        
        # New collate_fn_mlm returns: input_ids, targets, labels, attention_mask, session_indices
        for inputs, targets, labels, mask, session_indices in progress_bar:
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            
            optimizer.zero_grad()
            # BERT forward: output is logits for every token
            outputs = model(inputs, mask) 
            
            # MLM Loss: compare outputs with targets (where targets != -100)
            # outputs: (B, L, V) -> (B*L, V)
            # targets: (B, L) -> (B*L)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            
            # Accuracy on masked tokens only
            active_mask = targets != -100
            if active_mask.any():
                predicted = torch.argmax(outputs, dim=2)
                correct = (predicted[active_mask] == targets[active_mask]).sum().item()
                total = active_mask.sum().item()
                accuracy = correct / total
            else:
                accuracy = 0.0
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_acc += accuracy
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1), acc=total_acc / (progress_bar.n + 1))

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch + 1} average acc: {avg_acc:.4f}")
        scheduler.step(avg_loss)

        # Early stopping based on validation loss
        model.eval()
        val_loss = 0
        val_acc_total = 0
        with torch.no_grad():
            for inputs, targets, labels, mask, session_indices in val_loader:
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                outputs = model(inputs, mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                val_loss += loss.item()
                
                active_mask = targets != -100
                if active_mask.any():
                    predicted = torch.argmax(outputs, dim=2)
                    correct = (predicted[active_mask] == targets[active_mask]).sum().item()
                    total = active_mask.sum().item()
                    val_acc_total += correct / total

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} average validation loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Save best model separately if needed, but we save checkpoint every epoch below
        else:
            patience_counter += 1
        
        # SAVE CHECKPOINT
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss
        }, Config.CHECKPOINT_PATH)
        # print(f"Checkpoint saved: {Config.CHECKPOINT_PATH}")

        if patience_counter >= scheduler.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
    
    # Cleanup checkpoint after full training to save space? 
    # Or keep it? Let's keep it.
    return model

def calculate_threshold(model, train_df, device, vocab_size):
    """
    Calculates the anomaly threshold using the normal data distribution.
    """
    print("\n--- Define Anomaly Threshold using Normal Data Distribution ---")
    model.eval()
    
    normal_train_sessions_for_threshold = train_df[train_df['Label'] == 0]['EventCode'].tolist()
    # MaskedLogDataset expects labels, but here we can pass dummies or None
    threshold_dataset = MaskedLogDataset(normal_train_sessions_for_threshold, [0] * len(normal_train_sessions_for_threshold))
    
    from functools import partial
    from dataset import collate_fn_mlm
    collate = partial(collate_fn_mlm, vocab_size=vocab_size)
    
    threshold_loader = DataLoader(threshold_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                                  collate_fn=collate)

    session_losses = defaultdict(list)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    
    num_passes = 3 # Stabilize by averaging over multiple random masks
    for _ in range(num_passes):
        with torch.no_grad():
            for inputs, targets, labels, mask, session_indices in tqdm(threshold_loader,
                                                                       desc="Calculating threshold passes", leave=False):
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                outputs = model(inputs, mask)
                
                # Loss per token
                # outputs: (B, L, V)
                # targets: (B, L)
                # We want loss per session. Ideally average loss over masked tokens? 
                # Or max loss? 
                # For anomaly detection with BERT, "Pseudo-Log-Likelihood" or "Reconstruction Error" is used.
                # Let's use mean loss of masked tokens per session.
                
                # CrossEntropyLoss with reduction='none' returns (B, L) if input is (B, V, L) or (B, L, V) needs permutation?
                # PyTorch CE expectation: (B, C, d1...) vs (B, d1...). with reduction='none' -> (B, d1...)
                # We need to permute outputs to (B, V, L) for CE loss.
                
                loss_per_token = criterion(outputs.permute(0, 2, 1), targets) # (B, L)
                
                # We only care about masked tokens (targets != -100)
                active_mask = targets != -100
                
                # Calculate mean loss per session for masked tokens
                for i in range(len(session_indices)):
                    session_id = session_indices[i].item()
                    session_mask = active_mask[i]
                    if session_mask.any():
                        # CHANGE: Use MAX to capture sharp point anomalies
                        max_session_loss = loss_per_token[i][session_mask].max().item()
                        session_losses[session_id].append(max_session_loss)

    train_session_max_scores = [np.mean(losses) if losses else 0.0 for losses in session_losses.values()]

    # Use Robust Statistics (Median Absolute Deviation) instead of Percentile
    # This prevents the threshold from being skewed by outliers (dirty normal data)
    scores = np.array(train_session_max_scores)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    
    # Standard robust threshold: Median + 3 * MAD (approx. 3 sigma)
    # If MAD is 0 (very clean data), we fallback to a small epsilon or logic
    if mad == 0:
        mad = np.std(scores) + 1e-6

    anomaly_threshold = median + 3 * mad
    
    print(f"Robust Stats: Median={median:.4f}, MAD={mad:.4f}")

    if len(train_session_max_scores) <= 10:
         print(f"Warning: Not enough normal sessions ({len(train_session_max_scores)}). Using fallback.")
         anomaly_threshold = np.max(train_session_max_scores) * 1.5

    print(f"Calculated anomaly threshold (99th percentile of mean MLM loss per session): {anomaly_threshold:.4f}")
    return anomaly_threshold

def compute_session_scores_from_sessions(sessions_list, model, device, vocab_size, batch_size=Config.BATCH_SIZE):
    dummy_labels = [0] * len(sessions_list)
    temp_dataset = MaskedLogDataset(sessions_list, dummy_labels)
    
    from functools import partial
    from dataset import collate_fn_mlm
    collate = partial(collate_fn_mlm, vocab_size=vocab_size)
    
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.eval()
    session_losses = defaultdict(list)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    num_passes = 3
    for _ in range(num_passes):
        with torch.no_grad():
            for inputs, targets, _, mask, session_indices in temp_loader:
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                outputs = model(inputs, mask)
                
                loss_per_token = criterion(outputs.permute(0, 2, 1), targets)
                active_mask = targets != -100
                
                for i in range(len(session_indices)):
                    sid = session_indices[i].item()
                    # Max loss over masked tokens detects specific "broken" links better than mean
                    session_mask = active_mask[i]
                    if session_mask.any():
                        # CHANGE: Use MAX instead of MEAN to preserve anomaly signal
                        max_session_loss = loss_per_token[i][session_mask].max().item()
                        session_losses[sid].append(max_session_loss)

    # Ensure we return a score for EVERY input session to maintain alignment with labels
    scores = []
    for i in range(len(sessions_list)):
        if i in session_losses and len(session_losses[i]) > 0:
            scores.append(np.mean(session_losses[i]))
        else:
            # Session was filtered out (too short) or had no valid masked tokens (unlikely)
            # Assign 0.0 (Normal) as default since we can't judge it
            scores.append(0.0)
    
    return scores

def tune_threshold(model, val_sessions, val_labels, anomaly_threshold, device, vocab_size):
    """
    Tunes the threshold on the validation set to maximize F1 score (balanced Precision/Recall).
    """
    print("\n-- Tuning threshold on the dedicated validation set --")

    val_scores = compute_session_scores_from_sessions(val_sessions, model, device, vocab_size, batch_size=Config.BATCH_SIZE)

    normal_val_scores = [s for s, l in zip(val_scores, val_labels) if l == 0]
    anomaly_val_scores = [s for s, l in zip(val_scores, val_labels) if l == 1]

    print(f"Validation set scores computed. Normal sessions: {len(normal_val_scores)}, Anomaly sessions: {len(anomaly_val_scores)}")

    if not normal_val_scores or not anomaly_val_scores:
        print("WARNING: Validation set lacks either normal or anomaly samples. Skipping threshold tuning.")
        print(f"Using the threshold calculated from the training set: {anomaly_threshold:.4f}")
        return anomaly_threshold

    # Safe Lower Bound: Don't let threshold drop below 50% of the train set threshold
    # This prevents the "Collapse to Zero" issue seen in validation
    lower_bound = anomaly_threshold * 0.5
    
    candidate_thresholds = np.linspace(max(min(normal_val_scores), lower_bound), max(normal_val_scores) * 2, 100)
    if len(anomaly_val_scores) > 0:
        candidate_thresholds = sorted(list(candidate_thresholds) + list(anomaly_val_scores))

    best_thr = anomaly_threshold
    best_f1 = -1.0

    for thr in candidate_thresholds:
        preds = [1 if s > thr else 0 for s in val_scores]
        # Use F1 score (beta=1.0) for balanced Precision/Recall
        # [SCIENCE MODE] We prioritize overall effectiveness
        f1 = fbeta_score(val_labels, preds, beta=1.0, average='binary', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    if best_f1 <= 0 and anomaly_threshold is not None:
        print("Validation tuning did not find an improved F1. Keeping the train-set threshold.")
    else:
        print(f"Validation tuning result: best_thr={best_thr:.4f} gave best F1-score={best_f1:.4f}")
        print(f"Previous (train 99p) threshold: {anomaly_threshold:.4f}")
        anomaly_threshold = float(best_thr)
        print(f"Threshold updated for evaluation: {anomaly_threshold:.4f}")

    return anomaly_threshold
