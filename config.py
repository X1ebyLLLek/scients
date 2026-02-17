import os

class Config:
    """Class to hold all hyperparameters and configuration settings."""
    # Scientific Tuning Parameters
    # Вместо фиксированного множителя используем статистический отступ (Sigma)
    # Suspicious = Anomaly_Threshold - (Margin * Sigma)
    SUSPICIOUS_SIGMA_MARGIN: float = 1.0

    # === РЕЖИМ ДАННЫХ ===
    # True = использовать реальные метки BGL ("-" → Normal, всё остальное → Anomaly)
    # False = генерировать синтетические аномалии (как раньше)
    USE_REAL_LABELS: bool = True

    # Data URLs
    # Auto-detect Colab environment or local
    URL_STRUCTURED = "/content/BGL.log_structured.csv" if os.path.exists("/content/BGL.log_structured.csv") else "BGL.log_structured.csv"
    URL_LABELS = None  # Keep None to use synthetic generation
    USE_SYNTHETIC_FALLBACK = True  # If true, generate data when file not found

    # Количество стохастических проходов для MLM scoring (больше = стабильнее, но медленнее)
    NUM_STOCHASTIC_PASSES: int = 3

    # AutoML / Hyperparameter Optimization (HPO) Configuration
    HPO_ENABLED: bool = True
    HPO_NUM_TRIALS: int = 5        # Количество случайных архитектур для проверки
    HPO_TRIAL_EPOCHS: int = 3      # Обучаем каждую только 3 эпохи для быстрой оценки
    
    # Search Space for Random Search
    HPO_SEARCH_SPACE = {
        'embed_size': [64, 128, 256],
        'num_heads': [2, 4, 8],
        'num_layers': [2, 3, 4, 6],
        'dropout': [0.1, 0.2, 0.3, 0.4]
    }

    # Best Model Hyperparameters (will be overwritten by HPO if enabled)
    # Defaults for quick start:
    MAX_SEQ_LEN: int = 256
    EMBED_SIZE: int = 256
    NUM_HEADS: int = 8
    NUM_LAYERS: int = 6
    DROPOUT: float = 0.3

    # Training Hyperparameters
    NUM_EPOCHS = 50 
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.0001
    DATA_SAMPLE_RATE: float = 1.0  # Use full data
    RANDOM_STATE: int = 42

    # Scientific Improvement: Center Loss
    CENTER_LOSS_WEIGHT: float = 0.01  # Weight for the center loss term

    # Rare Event Filtering: events occurring fewer than this many times are mapped to <UNK>
    RARE_EVENT_THRESHOLD: int = 5


    # Synthetic Anomaly Generation
    SYNTHETIC_ANOMALY_MULTIPLIER: float = 15
    SYNTHETIC_QUANTILE_THRESHOLD: float = 0.95

    # File paths for saving artifacts
    MODEL_PATH: str = "transformer_ueba_model.pth"
    ENCODER_PATH: str = "label_encoder.pkl"
    THRESHOLD_PATH: str = "anomaly_threshold.txt"
    CHECKPOINT_PATH: str = "checkpoint.pth"
