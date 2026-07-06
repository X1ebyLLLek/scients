"""
explainer.py — Интерпретация решений UEBA-системы для аналитика ИБ.

Для аномальной сессии отвечает на вопрос "ПОЧЕМУ система считает её аномальной":
находит конкретные события, которые модель не смогла предсказать по контексту
(наибольшая MLM cross-entropy при маскировании).

Метод: многократные стохастические проходы с разными случайными масками.
Для каждой позиции накапливается средняя ошибка предсказания по тем проходам,
где позиция была замаскирована. Позиции с наибольшей средней ошибкой —
наиболее "неожиданные" события, они и предъявляются аналитику.

Это переводит систему из чёрного ящика ("сессия аномальна, score=5.2")
в объяснимый инструмент ("аномальна из-за события KERNDTLB на позиции 17,
которое не встречается в данном контексте").
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from config import Config


def explain_session(model, event_codes, label_encoder, device, vocab_size,
                    top_n=5, num_passes=20):
    """
    Ранжирует события сессии по "неожиданности" для модели.

    Args:
        model: обученный TransformerPredictor
        event_codes: список кодов событий сессии (0-based, как в session_df)
        label_encoder: LabelEncoder для декодирования кодов в имена событий
        device: torch device
        vocab_size: размер словаря
        top_n: сколько самых неожиданных событий вернуть
        num_passes: число стохастических проходов (больше = точнее оценка позиций)

    Returns:
        list of dict: [{position, event_code, event_name, mean_loss, times_masked}, ...]
        отсортирован по убыванию mean_loss
    """
    if len(event_codes) < 2:
        return []

    codes = list(event_codes)
    if len(codes) > Config.MAX_SEQ_LEN:
        codes = codes[-Config.MAX_SEQ_LEN:]

    # Та же конвенция токенов, что и в dataset.py: 0=PAD, события 1..V, MASK=V
    input_ids = torch.tensor([c + 1 for c in codes], dtype=torch.long)
    seq_len = len(codes)
    mask_token_id = vocab_size

    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    position_losses = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for _ in range(num_passes):
            masked_input = input_ids.clone()
            targets = torch.full((seq_len,), -100, dtype=torch.long)

            # Маскируем 15% позиций (минимум одну)
            mask_positions = torch.rand(seq_len) < 0.15
            if not mask_positions.any():
                mask_positions[torch.randint(0, seq_len, (1,))] = True

            targets[mask_positions] = input_ids[mask_positions]
            masked_input[mask_positions] = mask_token_id

            batch_input = masked_input.unsqueeze(0).to(device)
            batch_targets = targets.unsqueeze(0).to(device)
            padding_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)

            logits, _ = model(batch_input, padding_mask)
            loss_per_token = criterion(logits.permute(0, 2, 1), batch_targets)[0]  # (L,)

            for pos in mask_positions.nonzero(as_tuple=True)[0].tolist():
                position_losses[pos].append(loss_per_token[pos].item())

    def decode(code):
        try:
            return str(label_encoder.inverse_transform([code])[0])
        except Exception:
            return f"<code:{code}>"

    ranked = []
    for pos, losses in position_losses.items():
        ranked.append({
            'position': pos,
            'event_code': codes[pos],
            'event_name': decode(codes[pos]),
            'mean_loss': float(np.mean(losses)),
            'times_masked': len(losses),
        })
    ranked.sort(key=lambda r: -r['mean_loss'])
    return ranked[:top_n]


def print_explanation(explanation, session_score=None, threshold=None):
    """Человекочитаемый отчёт для аналитика ИБ."""
    print("\n  --- EXPLAINABILITY REPORT: почему сессия аномальна ---")
    if session_score is not None and threshold is not None:
        print(f"  Session score: {session_score:.4f} (threshold: {threshold:.4f})")
    if not explanation:
        print("  Сессия слишком короткая для анализа.")
        return
    print(f"  {'Pos':>5} {'MeanLoss':>10} {'Masked':>7}  Event")
    for item in explanation:
        print(f"  {item['position']:>5} {item['mean_loss']:>10.4f} "
              f"{item['times_masked']:>7}  {item['event_name'][:80]}")
    print("  Интерпретация: события с наибольшей ошибкой предсказания —")
    print("  наиболее неожиданные в данном контексте (кандидаты на причину алерта).")
