"""
preprocessing.py — Загрузка, парсинг и предобработка данных BGL.

Поддерживает два режима:
  1. USE_REAL_LABELS=True  — используются реальные метки из столбца Label BGL
                             ("-" → Normal=0, всё остальное → Anomaly=1)
  2. USE_REAL_LABELS=False — генерируются синтетические аномалии (injection, deletion, loop)

Если CSV-файл не найден и USE_SYNTHETIC_FALLBACK=True — генерируется полностью синтетический датасет.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from config import Config
from utils import download_and_load_csv
from synthetic_data import generate_synthetic_logs
import random


# ============================================================================
#                  СИНТЕТИЧЕСКАЯ ГЕНЕРАЦИЯ АНОМАЛИЙ (FALLBACK)
# ============================================================================

def generate_anomalies(session_dict, num_anomalies=100):
    """
    Инъекция реалистичных аномалий в подмножество сессий.
    Используется ТОЛЬКО когда USE_REAL_LABELS=False.
    
    Типы аномалий:
    1. Foreign Event Injection (Контекстная аномалия)
    2. Event Deletion (Пропущенный шаг)
    3. Event Repetition (Зацикливание / DoS-паттерн)
    """
    print(f"Generating {num_anomalies} anomalies using Advanced Scientific Patterns...")
    
    session_ids = list(session_dict.keys())
    if len(session_ids) < num_anomalies:
        print(f"Warning: requested {num_anomalies} anomalies but only have {len(session_ids)} sessions.")
        num_anomalies = len(session_ids)
        
    anomalous_session_ids = np.random.choice(session_ids, num_anomalies, replace=False)
    
    # Collect all unique events to use for injection
    all_events = []
    for s_id in session_ids:
        all_events.extend(session_dict[s_id]['EventId'])
    unique_events = list(set(all_events))

    labels = {}
    
    for s_id in anomalous_session_ids:
        events = session_dict[s_id]['EventId']
        if 'EventTemplate' in session_dict[s_id]:
            templates = session_dict[s_id]['EventTemplate']
        else:
            templates = events[:]

        if len(events) < 3: 
            labels[s_id] = 1
            continue

        anomaly_type = np.random.choice(['injection', 'deletion', 'loop'], p=[0.4, 0.3, 0.3])
        
        if anomaly_type == 'injection':
            for _ in range(3): 
                idx = np.random.randint(0, len(events))
                foreign_event = np.random.choice(unique_events)
                events.insert(idx, foreign_event)
                templates.insert(idx, "INJECTED_FOREIGN_EVENT")
            
        elif anomaly_type == 'deletion':
            num_to_delete = max(1, int(len(events) * 0.3))
            for _ in range(num_to_delete):
                if len(events) > 0:
                    idx = np.random.randint(0, len(events))
                    events.pop(idx)
                    templates.pop(idx)
            
        elif anomaly_type == 'loop':
            if len(events) > 1:
                start_idx = np.random.randint(0, len(events) - 1)
                end_idx = min(len(events), start_idx + np.random.randint(2, 5))
                segment = events[start_idx:end_idx]
                seg_templates = templates[start_idx:end_idx]
                for _ in range(10):
                    events[end_idx:end_idx] = segment
                    templates[end_idx:end_idx] = seg_templates

        session_dict[s_id]['EventId'] = events
        if 'EventTemplate' in session_dict[s_id]:
            session_dict[s_id]['EventTemplate'] = templates
        labels[s_id] = 1

    return session_dict, labels


# ============================================================================
#                        ОСНОВНОЙ ПАЙПЛАЙН ДАННЫХ
# ============================================================================

def prepare_data():
    """
    Загружает данные, обрабатывает метки, кодирует события и формирует сессии.
    
    Returns:
        session_df: DataFrame с колонками [BlockId, EventCode (list), Label, SeqLength]
        label_encoder: Обученный LabelEncoder
        vocab_size: Размер словаря (включая PAD)
    """
    # --- ШАГ 1: Загрузка ---
    try:
        df = download_and_load_csv(Config.URL_STRUCTURED, "structured logs")
    except Exception as e:
        if Config.USE_SYNTHETIC_FALLBACK:
            print(f"Could not load real data: {e}")
            print("FALLBACK: Generating synthetic data...")
            df = generate_synthetic_logs()
            # При синтетических данных принудительно включаем синтетические аномалии
            Config.USE_REAL_LABELS = False
        else:
            raise e

    # --- ШАГ 2: Определение столбца сессии (BlockId) ---
    print("Ensuring 'BlockId' column exists for session identification...")
    if 'Node' in df.columns:
        df.rename(columns={'Node': 'BlockId'}, inplace=True)
        print("Renamed 'Node' column to 'BlockId'.")
    elif 'BlockId' not in df.columns:
        df['BlockId'] = df.index.astype(str)
        print("Created 'BlockId' from DataFrame index as a fallback.")

    # --- ШАГ 3: Сэмплирование (если нужно) ---
    if Config.DATA_SAMPLE_RATE < 1.0:
        print(f"Sampling {Config.DATA_SAMPLE_RATE*100}% of sessions for speed...")
        unique_blocks = df['BlockId'].unique()
        selected_blocks = np.random.choice(
            unique_blocks, 
            size=int(len(unique_blocks) * Config.DATA_SAMPLE_RATE), 
            replace=False
        )
        df = df[df['BlockId'].isin(selected_blocks)].copy()
        print(f"Data reduced to {len(df)} rows and {len(selected_blocks)} sessions.")

    # --- ШАГ 4: Обработка меток ---
    if Config.USE_REAL_LABELS and 'Label' in df.columns:
        # ====== РЕЖИМ РЕАЛЬНЫХ МЕТОК BGL ======
        print("\n=== Using REAL BGL Labels ===")
        print(f"Raw label distribution (top 10):")
        print(df['Label'].value_counts().head(10))
        
        # Сохраняем исходный код категории аномалии (KERNDTLB, KERNSTOR, ...)
        # для последующего анализа detection rate по типам аномалий
        raw_labels = df['Label'].astype(str).str.strip()
        df['AnomalyType'] = raw_labels

        # Конвертация: "-" → 0 (Normal), всё остальное → 1 (Anomaly)
        df['Label'] = (raw_labels != '-').astype(int)
        
        normal_count = (df['Label'] == 0).sum()
        anomaly_count = (df['Label'] == 1).sum()
        print(f"After conversion: Normal={normal_count}, Anomaly={anomaly_count} ({anomaly_count/len(df)*100:.2f}%)")
    else:
        # ====== РЕЖИМ СИНТЕТИЧЕСКИХ АНОМАЛИЙ ======
        print("\n=== Using SYNTHETIC Anomaly Generation ===")
        
        if 'EventTemplate' not in df.columns:
            if 'EventId' in df.columns:
                df['EventTemplate'] = df['EventId']
            else:
                df['EventTemplate'] = "Unknown"

        # Группировка в сессии для инъекции аномалий
        print("Converting DataFrame to session dictionary...")
        session_groups = df.groupby('BlockId')
        session_dict = {}
        for block_id, group in tqdm(session_groups, desc="Grouping sessions"):
            session_dict[block_id] = {
                'EventId': group['EventId'].tolist(),
                'EventTemplate': group['EventTemplate'].tolist()
            }
        
        # 5% аномалий
        num_anomalies = int(len(session_dict) * 0.05) 
        session_dict, labels = generate_anomalies(session_dict, num_anomalies)

        # Реконструкция DataFrame
        print("Reconstructing DataFrame from modified sessions...")
        rows = []
        for block_id, data in session_dict.items():
            label = labels.get(block_id, 0)
            events = data['EventId']
            templates = data['EventTemplate']
            for i in range(len(events)):
                rows.append({
                    'BlockId': block_id,
                    'EventId': events[i],
                    'EventTemplate': templates[i],
                    'Label': label
                })
        
        df = pd.DataFrame(rows)
        df['Label'] = df['Label'].astype(int)
        # В синтетическом режиме реальных категорий нет — помечаем единым кодом
        df['AnomalyType'] = np.where(df['Label'] == 1, 'SYNTHETIC', '-')
        print(f"Synthetic labels: {df.groupby('Label').size().to_dict()}")

    # --- ШАГ 5: Определение столбца событий ---
    event_col = None
    for c in ['EventId', 'EventTemplate', 'Event']:
        if c in df.columns:
            event_col = c
            break

    if event_col is None:
        raise KeyError("No event ID column found (expected 'EventId' or 'EventTemplate').")

    # --- ШАГ 6: Фильтрация редких событий + кодирование ---
    print(f"\nEncoding events from column '{event_col}' with <UNK> for rare tokens...")
    
    event_counts = df[event_col].astype(str).value_counts()
    rare_events = event_counts[event_counts < Config.RARE_EVENT_THRESHOLD].index
    if len(rare_events) > 0:
        print(f"Cleaning vocabulary: {len(rare_events)} rare event types mapped to <UNK>.")
        df.loc[df[event_col].astype(str).isin(rare_events), event_col] = '<UNK>'
    else:
        print("No rare events found — vocabulary is clean.")

    label_encoder = LabelEncoder()
    unique_events = df[event_col].astype(str).unique()
    if '<UNK>' not in unique_events:
        all_known_events = np.append(unique_events, '<UNK>')
    else:
        all_known_events = unique_events
        
    label_encoder.fit(all_known_events)
    mapping = {label: i for i, label in enumerate(label_encoder.classes_)}
    unknown_code = mapping['<UNK>']
    df['EventCode'] = df[event_col].astype(str).map(mapping).fillna(unknown_code).astype(int)

    vocab_size = len(label_encoder.classes_) + 1  # +1 для PAD (index 0)
    print(f"Vocabulary size (unique event types + PAD): {vocab_size}")

    # --- ШАГ 7: Группировка в сессии ---
    print(f"Sessionizing by 'BlockId' (unique nodes: {df['BlockId'].nunique()})...")

    def session_category(types):
        """Доминирующая категория аномалии в сессии ('-' если нормальная)."""
        from collections import Counter
        bad = [t for t in types if t != '-']
        return Counter(bad).most_common(1)[0][0] if bad else '-'

    agg_spec = {
        'EventCode': list,
        'Label': 'max'  # Если хоть одна строка аномальная → вся сессия аномальная
    }
    if 'AnomalyType' in df.columns:
        agg_spec['AnomalyType'] = session_category

    session_df = df.groupby('BlockId').agg(agg_spec).reset_index()
    if 'AnomalyType' in session_df.columns:
        session_df.rename(columns={'AnomalyType': 'AnomalyCategory'}, inplace=True)
    else:
        session_df['AnomalyCategory'] = np.where(session_df['Label'] == 1, 'UNKNOWN', '-')

    print(f"Total sessions before filtering: {len(session_df)}")

    # Фильтрация слишком коротких сессий
    session_df['SeqLength'] = session_df['EventCode'].apply(len)
    session_df = session_df[session_df['SeqLength'] >= 2].copy()
    print(f"After length filter (>=2): {len(session_df)} sessions")

    # --- ШАГ 8: Балансировка представления ---
    anomalous_sessions = session_df[session_df['Label'] == 1]
    normal_sessions = session_df[session_df['Label'] == 0]

    print(f"Final dataset: {len(normal_sessions)} Normal, {len(anomalous_sessions)} Anomalous sessions")
    print(f"Anomaly rate: {len(anomalous_sessions)/len(session_df)*100:.2f}%")

    # Используем все данные (Science Mode)
    session_df = pd.concat([normal_sessions, anomalous_sessions]) \
        .sample(frac=1, random_state=Config.RANDOM_STATE) \
        .reset_index(drop=True)

    print(f"Final session count for modeling: {len(session_df)}")

    return session_df, label_encoder, vocab_size
