import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from config import Config
from utils import download_and_load_csv

from synthetic_data import generate_synthetic_logs
import random

def generate_anomalies(session_dict, num_anomalies=100):
    """
    Injects realistic anomalies into a subset of sessions.
    Types of anomalies:
    1. Foreign Event Injection (Contextual Anomaly)
    2. Event Deletion (Missing Step)
    3. Event Repetition (Loop/DoS)
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
        # Handle case where EventTemplate might not exist in synthetic data or needs to be managed
        if 'EventTemplate' in session_dict[s_id]:
            templates = session_dict[s_id]['EventTemplate']
        else:
            templates = events[:] # Fallback

        if len(events) < 3: 
            labels[s_id] = 1
            continue

        anomaly_type = np.random.choice(['injection', 'deletion', 'loop'], p=[0.4, 0.3, 0.3])
        
        # 1. Foreign Event Injection (Aggressive)
        if anomaly_type == 'injection':
            # Inject a BURST of foreign events to make it clear
            for _ in range(3): 
                idx = np.random.randint(0, len(events))
                foreign_event = np.random.choice(unique_events)
                events.insert(idx, foreign_event)
                templates.insert(idx, "INJECTED_FOREIGN_EVENT")
            
        # 2. Event Deletion (Aggressive)
        elif anomaly_type == 'deletion':
            # Delete 30% of the session
            num_to_delete = max(1, int(len(events) * 0.3))
            for _ in range(num_to_delete):
                if len(events) > 0:
                    idx = np.random.randint(0, len(events))
                    events.pop(idx)
                    templates.pop(idx)
            
        # 3. Event Repetition (Loop) (Aggressive)
        elif anomaly_type == 'loop':
            if len(events) > 1:
                start_idx = np.random.randint(0, len(events) - 1)
                end_idx = min(len(events), start_idx + np.random.randint(2, 5))
                segment = events[start_idx:end_idx]
                seg_templates = templates[start_idx:end_idx]
                # Repeat the segment 10 times (clear DoS pattern)
                for _ in range(10):
                    events[end_idx:end_idx] = segment
                    templates[end_idx:end_idx] = seg_templates

        session_dict[s_id]['EventId'] = events
        if 'EventTemplate' in session_dict[s_id]:
            session_dict[s_id]['EventTemplate'] = templates
        labels[s_id] = 1 # Mark as anomaly

    return session_dict, labels

def prepare_data():
    """
    Loads data, generates synthetic anomalies (if needed), processes labels, 
    encodes events, and sessions.
    Returns:
        session_df: DataFrame with session data
        label_encoder: Fitted LabelEncoder
        vocab_size: Size of vocabulary
    """
    # Загружаем данные
    try:
        df = download_and_load_csv(Config.URL_STRUCTURED, "structured logs")
    except Exception as e:
        if Config.USE_SYNTHETIC_FALLBACK:
            print(f"could not load real data: {e}")
            print("FALLBACK: Generating synthetic data...")
            df = generate_synthetic_logs()
        else:
            raise e

    # СНАЧАЛА создаем 'BlockId' из 'Node'
    print("Ensuring 'BlockId' column exists for session identification...")
    if 'Node' in df.columns:
        df.rename(columns={'Node': 'BlockId'}, inplace=True)
        print("Renamed 'Node' column to 'BlockId'.")
    elif 'BlockId' not in df.columns:
        # Fallback, если нет ни 'Node', ни 'BlockId'
        df['BlockId'] = df.index.astype(str)
        print("Created 'BlockId' from DataFrame index as a fallback.")
    
        df['BlockId'] = df.index.astype(str)
        print("Created 'BlockId' from DataFrame index as a fallback.")
    
    # --- ОПТИМИЗАЦИЯ: Сэмплируем данные ДО генерации аномалий ---
    if Config.DATA_SAMPLE_RATE < 1.0:
        print(f"Sampling {Config.DATA_SAMPLE_RATE*100}% of sessions for speed...")
        unique_blocks = df['BlockId'].unique()
        selected_blocks = np.random.choice(unique_blocks, size=int(len(unique_blocks) * Config.DATA_SAMPLE_RATE), replace=False)
        df = df[df['BlockId'].isin(selected_blocks)].copy()
        print(f"Data reduced to {len(df)} rows and {len(selected_blocks)} sessions.")

    # Теперь, когда 'BlockId' точно существует, генерируем аномалии
    if Config.URL_LABELS:
        df_labels = download_and_load_csv(Config.URL_LABELS, "anomaly labels")
    else:
        print("No label file provided. Generating realistic synthetic anomalies...")
        
        # Ensure we have EventTemplate
        if 'EventTemplate' not in df.columns:
            if 'EventId' in df.columns:
                df['EventTemplate'] = df['EventId']
            else:
                # Fallback if both missing (unlikely if data loaded correctly)
                df['EventTemplate'] = "Unknown"

        # Convert to dictionary for processing
        print("Converting DataFrame to session dictionary...")
        session_groups = df.groupby('BlockId')
        session_dict = {}
        # Using tqdm for progress
        for block_id, group in tqdm(session_groups, desc="Grouping sessions"):
             session_dict[block_id] = {
                 'EventId': group['EventId'].tolist(),
                 'EventTemplate': group['EventTemplate'].tolist()
             }
        
        # Inject anomalies
        # 5% anomalies by default
        num_anomalies = int(len(session_dict) * 0.05) 
        session_dict, labels = generate_anomalies(session_dict, num_anomalies)

        # Reconstruct DataFrame
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
        
        # Replace df with new data
        df = pd.DataFrame(rows)
        # Ensure Label is int
        df['Label'] = df['Label'].astype(int)
        
        # Create df_labels for compatibility with downstream logic
        df_labels = df[['BlockId', 'Label']].drop_duplicates().reset_index(drop=True)
        print(f"Synthetic labels created. Anomalies injected into {df_labels['Label'].sum()} sessions out of {len(df_labels)} total.")

    print("\n--- Preprocess Data with Sessionization and Label Merging ---")

    if df_labels['Label'].dtype == 'object':
        print("Converting text labels ('Anomaly'/'Normal') to numeric (1/0)...")
        df_labels['Label'] = df_labels['Label'].apply(lambda x: 1 if str(x).lower() == 'anomaly' else 0)
    else:
        print("Labels are already numeric. Skipping text-to-numeric conversion.")

    print("Sessionizing BGL data using the 'Node' column...")
    if 'Label' not in df.columns:
        print("Merging structured logs with labels...")
        df = df.merge(df_labels, on='BlockId', how='left')
        df['Label'] = df['Label'].fillna(0)
    else:
        print("Labels already exist in the main dataframe, skipping merge.")

    event_col = None
    for c in ['EventId', 'EventTemplate', 'Event']:
        if c in df.columns:
            event_col = c
            break

    if event_col is None:
        raise KeyError("No event ID column found (expected 'EventId' or 'EventTemplate').")

    print("Encoding event IDs/templates with <UNK> token...")
    
    # --- RARE EVENT FILTERING ---
    # Events occurring less than 5 times are mapped to <UNK> to reduce noise (dirty data cleanup)
    event_counts = df[event_col].astype(str).value_counts()
    rare_events = event_counts[event_counts < 5].index
    print(f"Cleaning vocabulary: {len(rare_events)} rare event types mapped to <UNK>.")
    
    df.loc[df[event_col].isin(rare_events), event_col] = '<UNK>'

    label_encoder = LabelEncoder()
    unique_events = df[event_col].astype(str).unique()
    # Ensure <UNK> is in the vocabulary
    if '<UNK>' not in unique_events:
        all_known_events = np.append(unique_events, '<UNK>')
    else:
        all_known_events = unique_events
        
    label_encoder.fit(all_known_events)
    mapping = {label: i for i, label in enumerate(label_encoder.classes_)}
    unknown_code = mapping['<UNK>']
    df['EventCode'] = df[event_col].astype(str).map(mapping).fillna(unknown_code).astype(int)

    vocab_size = len(label_encoder.classes_) + 1
    print(f"Vocabulary size (unique event templates): {vocab_size - 1}")

    if 'BlockId' not in df.columns:
        if 'Node' in df.columns:
            df['BlockId'] = df['Node']
        else:
            df['BlockId'] = df.index.astype(str)

    print(f"Using session identifier: 'BlockId' (unique: {df['BlockId'].nunique()})")

    session_df = df.groupby('BlockId').agg({
        'EventCode': list,
        'Label': 'max'
    }).reset_index()

    print(f"Total sessions before any filtering: {len(session_df)}")

    session_df['SeqLength'] = session_df['EventCode'].apply(len)
    session_df = session_df[session_df['SeqLength'] >= 2].copy()
    print(f"Filtered out short sessions. Remaining: {len(session_df)}")
    print(f"Anomalies remaining after length filter: {session_df['Label'].sum()}")

    # --- БЕЗОПАСНОЕ СЭМПЛИРОВАНИЕ ---
    anomalous_sessions = session_df[session_df['Label'] == 1]
    normal_sessions = session_df[session_df['Label'] == 0]

    print(f"Found {len(anomalous_sessions)} anomalous and {len(normal_sessions)} normal sessions.")

    # [SCIENCE MODE] Use 100% of data for maximum result fidelity
    sample_fraction = 1.0 
    target_total_count = int(len(session_df) * sample_fraction)
    target_normal_count = max(0, target_total_count - len(anomalous_sessions))

    if len(normal_sessions) > target_normal_count and sample_fraction < 1.0:
        print(f"Sampling normal sessions down to {target_normal_count}...")
        normal_sessions_sampled = normal_sessions.sample(n=target_normal_count, random_state=Config.RANDOM_STATE)
    else:
        print("Using all available normal sessions (Scientific Mode).")
        normal_sessions_sampled = normal_sessions

    session_df = pd.concat([normal_sessions_sampled, anomalous_sessions]) \
        .sample(frac=1, random_state=Config.RANDOM_STATE) \
        .reset_index(drop=True)

    print(f"Final session count for modeling: {len(session_df)}")
    print(f"Anomalies in the final set: {session_df['Label'].sum()}")

    return session_df, label_encoder, vocab_size
