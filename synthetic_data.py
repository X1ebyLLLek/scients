import pandas as pd
import numpy as np
import random
from typing import List

def generate_synthetic_logs(num_sessions: int = 1000, max_seq_len: int = 50) -> pd.DataFrame:
    """
    Generates a DataFrame mimicking BGL structured logs.
    """
    print(f"Generating synthetic data: {num_sessions} sessions...")
    
    # Define some mock EventTemplates
    templates = [
        "Instruction cache parity error corrected",
        "Midplane switch packet error caused by other port",
        "Fan spinning too fast",
        "Fan spinning too slow",
        "Power supply failure detected",
        "RAS kernel info: ciod: login",
        "RAS kernel info: ciod: logout",
        "data TLB error interrupt",
        "instruction TLB error interrupt",
        "program interrupt: illegal instruction",
        "program interrupt: privileged instruction",
        "program interrupt: trap",
        "external input interrupt (FIC)",
        "external input interrupt (JTAG)",
        "external input interrupt (DMA)",
        "floating point unavailable interrupt",
        "decrementer interrupt"
    ]
    
    event_ids = [f"E{i}" for i in range(len(templates))]
    
    data = []
    
    for i in range(num_sessions):
        block_id = f"Blk_{i}"
        # Random sequence length
        seq_len = random.randint(5, max_seq_len)
        
        # Create a coherent sequence (mimic a repeated pattern)
        
        # Create a coherent sequence (mimic a repeated pattern)
        # Sequence A: Login (5) -> Instruction (0) -> Data (7) -> Logout (6)
        # Sequence B: Login (5) -> Error (2) -> Recovery (3) -> Logout (6)
        
        pattern_A = [5, 0, 7, 6]
        pattern_B = [5, 2, 3, 6]
        
        session_events = []
        current_len = 0
        
        while current_len < seq_len:
            # Choose a pattern
            if random.random() > 0.3:
                pat = pattern_A
            else:
                pat = pattern_B
            
            # Append pattern to session
            for event_idx in pat:
                if current_len >= seq_len:
                    break
                
                # Removed random noise to ensure "Normal" data is truly normal.
                # Anomalies should only come from the injection process in preprocessing.py
                pass

                session_events.append({
                    'BlockId': block_id,
                    'EventId': event_ids[event_idx],
                    'EventTemplate': templates[event_idx],
                    'Content': f"Mock content for {event_ids[event_idx]}"
                })
                current_len += 1
                
        data.extend(session_events)
        
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} log lines.")
    return df
