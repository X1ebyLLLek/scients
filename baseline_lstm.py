"""
baseline_lstm.py — Нейросетевой бейзлайн: LSTM next-event prediction (DeepLog-style).

Реализует подход Du et al. (2017) "DeepLog: Anomaly Detection and Diagnosis
from System Logs through Deep Learning":
  - LSTM-языковая модель обучается на нормальных сессиях предсказывать
    следующее событие по префиксу;
  - оценка аномальности сессии = агрегированная cross-entropy ошибка
    предсказания следующего события (Config.SCORE_AGG: max | mean);
  - порог подбирается оптимизацией MCC на валидационной выборке
    (тот же протокол, что и у Transformer — честное сравнение).

Назначение: экспериментальное обоснование выбора архитектуры Transformer
против рекуррентных сетей (глава 2 сравнивает их теоретически — этот модуль
закрывает сравнение эмпирически).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

from config import Config
from baseline import compute_metrics


class NextEventDataset(Dataset):
    """Сессия → (input=session[:-1], target=session[1:]) для next-event prediction."""

    def __init__(self, sessions):
        # Нужно минимум 2 события, чтобы был хотя бы один переход
        self.sessions = [s for s in sessions if len(s) >= 2]

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]
        if len(session) > Config.MAX_SEQ_LEN:
            session = session[-Config.MAX_SEQ_LEN:]
        # +1: резервируем 0 под PAD (та же конвенция, что у MaskedLogDataset)
        tokens = torch.tensor([c + 1 for c in session], dtype=torch.long)
        return tokens[:-1], tokens[1:], idx


def collate_next_event(batch):
    inputs, targets, indices = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    # -100 = ignore_index для CrossEntropyLoss на паддинге
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return padded_inputs, padded_targets, torch.tensor(indices, dtype=torch.long)


class LSTMPredictor(nn.Module):
    """LSTM-языковая модель следующего события (DeepLog-style)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2, Config.LSTM_EMBED_SIZE, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=Config.LSTM_EMBED_SIZE,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(Config.LSTM_HIDDEN_SIZE, vocab_size + 2)

    def forward(self, x):
        emb = self.embedding(x)
        output, _ = self.lstm(emb)
        return self.fc(output)  # (B, L, V)


def _score_sessions(model, sessions, device):
    """Оценка аномальности: per-token CE next-event prediction → session score."""
    dataset = NextEventDataset(sessions)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                        collate_fn=collate_next_event)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    model.eval()
    scores_by_idx = {}
    with torch.no_grad():
        for inputs, targets, indices in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_per_token = criterion(logits.permute(0, 2, 1), targets)  # (B, L)
            active = targets != -100
            for i in range(len(indices)):
                if active[i].any():
                    token_losses = loss_per_token[i][active[i]]
                    if Config.SCORE_AGG == "mean":
                        scores_by_idx[indices[i].item()] = token_losses.mean().item()
                    else:
                        scores_by_idx[indices[i].item()] = token_losses.max().item()

    # Выравнивание с исходным списком (короткие сессии получают 0.0 = Normal)
    # ВНИМАНИЕ: индексы dataset — по отфильтрованному списку, мапим обратно
    valid_positions = [i for i, s in enumerate(sessions) if len(s) >= 2]
    scores = [0.0] * len(sessions)
    for ds_idx, orig_idx in enumerate(valid_positions):
        scores[orig_idx] = scores_by_idx.get(ds_idx, 0.0)
    return scores


def tune_threshold_mcc_on_scores(scores, labels):
    """Подбор порога оптимизацией MCC (тот же протокол, что у Transformer)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    candidates = np.linspace(scores.min(), scores.max(), 200)
    best_thr, best_mcc = float(scores.mean()), -1.0
    for thr in candidates:
        preds = (scores > thr).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, float(thr)
    return best_thr, best_mcc


def run_lstm_baseline(train_df, tune_val_df, test_df, vocab_size, device):
    """
    Полный цикл LSTM-бейзлайна: обучение на нормальных сессиях →
    MCC-тюнинг порога на Tune-Val → метрики на Test.

    Returns:
        metrics dict (совместим с run_baselines / plot_comparison)
    """
    print("\n" + "=" * 60)
    print("   NEURAL BASELINE: LSTM Next-Event Prediction (DeepLog-style)")
    print("=" * 60)

    normal_train_sessions = train_df[train_df['Label'] == 0]['EventCode'].tolist()
    dataset = NextEventDataset(normal_train_sessions)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                        collate_fn=collate_next_event)

    model = LSTMPredictor(vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM on {device}. Parameters: {total_params:,} "
          f"(embed={Config.LSTM_EMBED_SIZE}, hidden={Config.LSTM_HIDDEN_SIZE}, "
          f"layers={Config.LSTM_NUM_LAYERS})")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=Config.LSTM_LR)

    model.train()
    for epoch in range(Config.LSTM_EPOCHS):
        total_loss = 0.0
        progress = tqdm(loader, desc=f"LSTM Epoch {epoch + 1}/{Config.LSTM_EPOCHS}", leave=False)
        for inputs, targets, _ in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"  LSTM Epoch {epoch + 1}: avg loss = {total_loss / max(1, len(loader)):.4f}")

    # Тюнинг порога на Tune-Val (без утечки в Test)
    tune_scores = _score_sessions(model, tune_val_df['EventCode'].tolist(), device)
    threshold, tune_mcc = tune_threshold_mcc_on_scores(tune_scores, tune_val_df['Label'].tolist())
    print(f"  LSTM threshold (MCC-tuned on Tune-Val): {threshold:.4f} (val MCC={tune_mcc:.4f})")

    # Финальная оценка на Test
    test_scores = _score_sessions(model, test_df['EventCode'].tolist(), device)
    y_true = np.asarray(test_df['Label'].tolist(), dtype=int)
    y_pred = (np.asarray(test_scores) > threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_scores=test_scores)
    print(f"  LSTM — MCC: {metrics['mcc']:.4f}, F1: {metrics['f1']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}, "
          f"AUC: {metrics['roc_auc']:.4f}")

    return metrics
