import os

class Config:
    """Class to hold all hyperparameters and configuration settings."""
    # --- НОВЫЙ ПАРАМЕТР ---
    # Множитель для порога "подозрительных" сессий. 0.75 означает, что
    # подозрительным будет считаться всё, что выше 75% от основного порога аномальности.
    SUSPICIOUS_THRESHOLD_MULTIPLIER: float = 0.75

    # Data URLs
    # Auto-detect Colab environment or local
    URL_STRUCTURED = "/content/BGL.log_structured.csv" if os.path.exists("/content/BGL.log_structured.csv") else "BGL.log_structured.csv"
    URL_LABELS = None  # Keep None to use synthetic generation
    USE_SYNTHETIC_FALLBACK = True  # If true, generate data when file not found

    # Model Hyperparameters
    MAX_SEQ_LEN: int = 256         # Уменьшаем макс. длину последовательности
    EMBED_SIZE: int = 256          # [SCIENCE MODE] Увеличиваем размер эмбеддинга
    NUM_HEADS: int = 8             # Keep 8 heads
    NUM_LAYERS: int = 6            # [SCIENCE MODE] Увеличиваем глубину
    DROPOUT: float = 0.3           # [SCIENCE MODE] Больше регуляризации

    # Training Hyperparameters
    NUM_EPOCHS = 50 
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.0001
    DATA_SAMPLE_RATE: float = 1.0  # Use full data
    RANDOM_STATE: int = 42

    # Synthetic Anomaly Generation
    SYNTHETIC_ANOMALY_MULTIPLIER: float = 15
    SYNTHETIC_QUANTILE_THRESHOLD: float = 0.95

    # File paths for saving artifacts
    MODEL_PATH: str = "transformer_ueba_model.pth"
    ENCODER_PATH: str = "label_encoder.pkl"
    THRESHOLD_PATH: str = "anomaly_threshold.txt"
    CHECKPOINT_PATH: str = "checkpoint.pth"
