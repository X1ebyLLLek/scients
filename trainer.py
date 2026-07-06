import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, fbeta_score
from config import Config
from dataset import MaskedLogDataset, collate_fn_mlm
from torch.utils.data import DataLoader
from model import TransformerPredictor 
from loss import CenterLoss 
import os
from torch.utils.tensorboard import SummaryWriter
import datetime


def aggregate_token_losses(loss_values: torch.Tensor) -> float:
    """
    Агрегация per-token losses в оценку сессии (ablation-параметр SCORE_AGG):
      'max'  — точечные аномалии: одно неожиданное событие поднимает score сессии
      'mean' — усреднённая pseudo-log-likelihood (менее чувствительна к выбросам)
    """
    if Config.SCORE_AGG == "mean":
        return loss_values.mean().item()
    return loss_values.max().item()


def make_scoring_collates(vocab_size):
    """
    Набор collate-функций для проходов скоринга (режим Config.SCORE_MODE):
      'full'       — SCORE_STRIDE детерминированных проходов, каждая позиция
                     маскируется ровно один раз (100% покрытие, LogBERT-style)
      'stochastic' — NUM_STOCHASTIC_PASSES случайных масок 15%
                     (позиция не покрывается с вероятностью 0.85^K)
    """
    from functools import partial
    from dataset import collate_fn_mlm, collate_fn_mlm_strided
    if Config.SCORE_MODE == "full":
        return [partial(collate_fn_mlm_strided, vocab_size=vocab_size,
                        stride=Config.SCORE_STRIDE, offset=k)
                for k in range(Config.SCORE_STRIDE)]
    return [partial(collate_fn_mlm, vocab_size=vocab_size)] * Config.NUM_STOCHASTIC_PASSES


def score_dataset_sessions(dataset, model, device, vocab_size,
                           batch_size=None, desc="Scoring"):
    """
    Единый скоринг сессий (порог, MCC-тюнинг, тест — всё через эту функцию).

    Returns:
        dict: session_idx -> anomaly score
        Семантика:
          full:       агрегация (SCORE_AGG) по ВСЕМ позициям сессии, каждая оценена 1 раз
          stochastic: per-pass агрегация, затем среднее по проходам (историческое поведение)
    """
    batch_size = batch_size or Config.BATCH_SIZE
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    stochastic = Config.SCORE_MODE != "full"

    per_pass_scores = defaultdict(list)   # stochastic: sid -> [score прохода, ...]
    token_stats = {}                      # full: sid -> [max, sum, count]

    model.eval()
    for collate in make_scoring_collates(vocab_size):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
        with torch.no_grad():
            for inputs, targets, labels, mask, session_indices in tqdm(loader, desc=desc, leave=False):
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                outputs, _ = model(inputs, mask)
                loss_per_token = criterion(outputs.permute(0, 2, 1), targets)  # (B, L)
                active_mask = targets != -100

                for i in range(len(session_indices)):
                    sid = session_indices[i].item()
                    session_mask = active_mask[i]
                    if not session_mask.any():
                        continue
                    token_losses = loss_per_token[i][session_mask]
                    if stochastic:
                        per_pass_scores[sid].append(aggregate_token_losses(token_losses))
                    else:
                        mx, sm, ct = token_stats.get(sid, (0.0, 0.0, 0))
                        token_stats[sid] = (max(mx, token_losses.max().item()),
                                            sm + token_losses.sum().item(),
                                            ct + token_losses.numel())

    if stochastic:
        return {sid: float(np.mean(scores)) for sid, scores in per_pass_scores.items()}
    if Config.SCORE_AGG == "mean":
        return {sid: sm / ct for sid, (mx, sm, ct) in token_stats.items()}
    return {sid: mx for sid, (mx, sm, ct) in token_stats.items()}



def evaluate_hyperparams(train_loader, val_loader, vocab_size, params, device):
    """
    Trains a small model for a few epochs to evaluate configuration performance.
    Returns: min_val_loss
    """
    print(f"\n--- HPO Trial: {params} ---")
    
    # Initialize model with specific params
    model = TransformerPredictor(
        vocab_size=vocab_size, 
        embed_size=params['embed_size'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    min_val_loss = float('inf')
    
    # Short Training Loop
    for epoch in range(Config.HPO_TRIAL_EPOCHS):
        model.train()
        for inputs, targets, labels, mask, session_indices in train_loader:
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs, mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets, labels, mask, session_indices in val_loader:
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                outputs, _ = model(inputs, mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            
    print(f"Trial Result: Min Val Loss = {min_val_loss:.4f}")
    return min_val_loss

def train_model(model, train_loader, val_loader, device):
    """
    Trains the model and returns the trained model.
    """
    print("\n--- Train the Model on Normal Data (Masked Language Modeling) ---")
    if not Config.CENTER_LOSS_ENABLED:
        print("    [ABLATION] Center Loss DISABLED — training with pure MLM loss")
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignore unmasked tokens

    # Center Loss Setup
    center_loss_fn = CenterLoss(num_classes=1, feat_dim=Config.EMBED_SIZE, use_gpu=(device.type == 'cuda')).to(device)
    optimizer_center = optim.SGD(center_loss_fn.parameters(), lr=0.5) # Center loss usually needs high lr
    
    # Use standard Adam to avoid potential FSDP/Import hangs with AdamW on some Colab runtimes
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-3)
    # Increase patience to allow model to overcome small fluctuations
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    start_epoch = 0
    patience_counter = 0

    # TensorBoard Setup
    log_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

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
            optimizer_center.zero_grad()
            
            # BERT forward: output is logits for every token AND hidden states
            logits, features = model(inputs, mask) 
            
            # MLM Loss: compare outputs with targets (where targets != -100)
            # logits: (B, L, V) -> (B*L, V)
            # targets: (B, L) -> (B*L)
            loss_mlm = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Center Loss: Force normal session embeddings to cluster
            # We use the MEAN of the sequence features as the session representation
            # mask is (B, L), where False is padding (usually). 
            # Check model definition: mask passed to transformer is usually padding mask
            # If src_key_padding_mask is provided, features at padding are likely noise.
            
            # Simple mean pooling over non-padded tokens
            # mask: True where PADDING (in PyTorch Transformer convention often) or NOT?
            # In dataset.py: padding_mask = (padded_inputs == 0) -> True = Padding
            
            # We want to average features where !padding_mask
            active_features_mask = ~mask # True = Valid Token

            if Config.CENTER_LOSS_ENABLED:
                # features: (B, L, E)
                # Sum valid features
                sum_features = (features * active_features_mask.unsqueeze(-1).float()).sum(dim=1)
                # Count valid tokens
                valid_token_counts = active_features_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)

                session_embeddings = sum_features / valid_token_counts # (B, E)

                loss_center = center_loss_fn(session_embeddings, labels=None) # Assume all normal (0) for clustering

                # Total Loss
                loss = loss_mlm + Config.CENTER_LOSS_WEIGHT * loss_center
            else:
                loss_center = torch.tensor(0.0)
                loss = loss_mlm
            
            loss.backward()
            
            # Accuracy on masked tokens only
            active_mask = targets != -100
            if active_mask.any():
                predicted = torch.argmax(logits, dim=2)
                correct = (predicted[active_mask] == targets[active_mask]).sum().item()
                total = active_mask.sum().item()
                accuracy = correct / total
            else:
                accuracy = 0.0
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_center.step()
            total_loss += loss.item()
            total_acc += accuracy
            
            # Log step-level metrics
            global_step = epoch * len(train_loader) + progress_bar.n
            if global_step % 10 == 0:
                writer.add_scalar('Train/Loss_Total_Step', loss.item(), global_step)
                writer.add_scalar('Train/Loss_MLM_Step', loss_mlm.item(), global_step)
                writer.add_scalar('Train/Loss_Center_Step', loss_center.item(), global_step)
                writer.add_scalar('Train/Accuracy_Step', accuracy, global_step)
                
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1), acc=total_acc / (progress_bar.n + 1))

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch + 1} average acc: {avg_acc:.4f}")
        scheduler.step(avg_loss)
        
        writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)
        writer.add_scalar('Train/Accuracy_Epoch', avg_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Early stopping based on validation loss
        model.eval()
        val_loss = 0
        val_acc_total = 0
        with torch.no_grad():
            for inputs, targets, labels, mask, session_indices in val_loader:
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                logits, _ = model(inputs, mask)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
                
                active_mask = targets != -100
                if active_mask.any():
                    predicted = torch.argmax(logits, dim=2)
                    correct = (predicted[active_mask] == targets[active_mask]).sum().item()
                    total = active_mask.sum().item()
                    val_acc_total += correct / total

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc_total / len(val_loader)
        print(f"Epoch {epoch + 1} average validation loss: {avg_val_loss:.4f}")
        
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', avg_val_acc, epoch)

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
    writer.close()
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

    # Единый скоринг (Config.SCORE_MODE: full-coverage или стохастический)
    scores_by_sid = score_dataset_sessions(threshold_dataset, model, device, vocab_size,
                                           desc="Calculating threshold passes")

    train_session_max_scores = list(scores_by_sid.values()) if scores_by_sid else [0.0]

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

    # Единый скоринг (Config.SCORE_MODE: full-coverage или стохастический)
    scores_by_sid = score_dataset_sessions(temp_dataset, model, device, vocab_size,
                                           batch_size=batch_size, desc="Scoring sessions")

    # Ensure we return a score for EVERY input session to maintain alignment with labels
    # Session was filtered out (too short) → 0.0 (Normal) since we can't judge it
    return [float(scores_by_sid.get(i, 0.0)) for i in range(len(sessions_list))]

def tune_threshold_mcc(model, val_sessions, val_labels, anomaly_threshold, device, vocab_size):
    """
    Scientifically determines the best threshold using Matthew's Correlation Coefficient (MCC).
    MCC is robust to class imbalance.
    Also calculates Sigma (variance) of normal scores for dynamic 'suspicious' intervals.
    """
    print("\n-- Scientific Threshold Tuning (MCC Optimization) --")

    val_scores = compute_session_scores_from_sessions(val_sessions, model, device, vocab_size, batch_size=Config.BATCH_SIZE)

    normal_val_scores = [s for s, l in zip(val_scores, val_labels) if l == 0]
    anomaly_val_scores = [s for s, l in zip(val_scores, val_labels) if l == 1]

    print(f"Validation set stats: {len(normal_val_scores)} Normal, {len(anomaly_val_scores)} Anomalies")

    if not normal_val_scores or not anomaly_val_scores:
        print("WARNING: Validation set incomplete. Using statistical fallback.")
        return anomaly_threshold, 0.5 # Default sigma

    # Calculate Sigma for Dynamic Suspicious Threshold
    val_sigma = np.std(normal_val_scores)
    print(f"Normal Session Sigma (StdDev): {val_sigma:.4f}")

    # Candidate thresholds
    min_score = min(min(normal_val_scores), min(anomaly_val_scores))
    max_score = max(max(normal_val_scores), max(anomaly_val_scores))
    candidate_thresholds = np.linspace(min_score, max_score, 200)

    best_thr = anomaly_threshold
    best_mcc = -1.0
    best_stats = {}

    for thr in candidate_thresholds:
        preds = [1 if s > thr else 0 for s in val_scores]
        
        mcc = matthews_corrcoef(val_labels, preds)
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = thr
            prec = precision_score(val_labels, preds, zero_division=0)
            rec = recall_score(val_labels, preds, zero_division=0)
            best_stats = {'precision': prec, 'recall': rec}

    print(f"Optimization Result: Best MCC={best_mcc:.4f} at Threshold={best_thr:.4f}")
    print(f"Metrics at Best Threshold: Precision={best_stats.get('precision', 0):.4f}, Recall={best_stats.get('recall', 0):.4f}")
    
    # Check if optimized threshold is drastically different
    if best_mcc < 0.2:
        print("Warning: Correlation is weak. Model might be underfitting or data is too noisy.")
        
    return float(best_thr), float(val_sigma)
