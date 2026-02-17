"""
Модуль для обеспечения воспроизводимости экспериментов.
Фиксирует все источники случайности в единой точке.
"""
import random
import numpy as np
import torch
import os


def set_global_seed(seed: int = 42):
    """
    Фиксирует seed для всех генераторов случайных чисел.
    Гарантирует воспроизводимость результатов при повторных запусках.
    
    Args:
        seed: Значение seed (по умолчанию 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Детерминистичные операции на GPU (может замедлить обучение на ~10%)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Для DataLoader с num_workers > 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Global seed set to {seed} (torch, numpy, random, CUDA)")
