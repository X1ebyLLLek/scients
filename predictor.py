import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from config import Config
from utils import apply_contextual_weighting
from model import TransformerPredictor

from dataset import collate_fn_mlm
from functools import partial

def analyze_log_sequence(
        log_events: List[str],
        predictor_model: nn.Module,
        le: Any,
        threshold: float,
        vocab_size: int,
        context_user: str = 'user',
        context_resource: str = 'standard',
        sigma: float = 0.5 # Default sigma if not provided
) -> Dict[str, Any]:
    """
    Analyzes a log session using the BERT model.
    Since BERT is trained with random masking (MLM), to get a robust anomaly score,
    we replicate the session N times, apply random masks, and average the reconstruction loss.
    """
    if len(log_events) < 2:
        return {"error": "Sequence is too short.", "overall_verdict": "Insufficient Data"}

    try:
        known_classes = set(le.classes_)
        # Use 1-based indexing for known classes, 0 is padding.
        # But le.transform returns 0-based index. So we add 1.
        # Unknowns? 
        # If <UNK> in classes, use it.
        # Otherwise, maybe hash or skip?
        # The training data used <UNK>.
        unknown_token_code = le.transform(['<UNK>'])[0] if '<UNK>' in known_classes else 0 
        
        event_codes = []
        for ev in log_events:
            if ev in known_classes:
                event_codes.append(le.transform([ev])[0])
            else:
                event_codes.append(unknown_token_code)
                 
    except Exception as e:
        return {"error": f"Encoding failed: {e}", "overall_verdict": "Processing Failed"}

    dev = next(predictor_model.parameters()).device
    predictor_model.eval()
    
    # BERT Input Prep
    # We want to score this session.
    # Create a batch of N copies to average out the random masking variance.
    num_copies = Config.NUM_STOCHASTIC_PASSES
    
    # Prepare batch for collate_fn_mlm
    # batch list of tuples: (tensor_input, label, session_idx)
    # tensor_input should be 1-based (dataset does +1).
    # Here we have 0-based codes.
    input_tensor = torch.tensor([c + 1 for c in event_codes], dtype=torch.long)
    
    batch = [(input_tensor, 0, 0) for _ in range(num_copies)]
    
    # Use collate_fn_mlm
    padded_inputs, targets, labels, mask, indices = collate_fn_mlm(batch, vocab_size=vocab_size)
    
    padded_inputs = padded_inputs.to(dev)
    targets = targets.to(dev)
    mask = mask.to(dev)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    
    scores = []
    with torch.no_grad():
        outputs, _ = predictor_model(padded_inputs, mask)
        # outputs: (B, L, V)
        # targets: (B, L)
        
        loss_per_token = criterion(outputs.permute(0, 2, 1), targets) # (B, L)
        active_mask = targets != -100
        
        for i in range(num_copies):
            # Max loss over masked tokens for this copy
            if active_mask[i].any():
                score = loss_per_token[i][active_mask[i]].max().item()
                scores.append(score)
    
    if not scores:
        session_score = 0.0
    else:
        session_score = float(np.mean(scores))

    # Dynamic Stat Threshold
    suspicious_thr = threshold - (Config.SUSPICIOUS_SIGMA_MARGIN * sigma)
    # Clamp to reasonable bounds (can't be negative, but loss is >0 anyway)
    suspicious_thr = max(0.5, suspicious_thr)

    final_score = apply_contextual_weighting(session_score, context_user, context_resource)

    if final_score >= threshold:
        verdict = "ANOMALY DETECTED"
        risk_level = "High"
    elif final_score >= suspicious_thr:
        verdict = "SUSPICIOUS BEHAVIOR"
        risk_level = "Medium"
    else:
        verdict = "Normal Behavior"
        risk_level = "Low"

    return {
        "overall_verdict": verdict,
        "risk_level": risk_level,
        "max_anomaly_score": round(session_score, 4),
        "context": f"User: '{context_user}', Resource: '{context_resource}'",
        "final_weighted_score": round(final_score, 4),
        "anomaly_threshold": round(threshold, 4),
        "suspicious_threshold": round(suspicious_thr, 4),
        "sigma": round(sigma, 4),
        "num_windows": num_copies # actually num_stochastic_passes
    }

def run_demo(test_df, device, vocab_size):
    print("\n--- Step 9: Interactive Predictor for Demonstration ---")
    predictor_model = None
    le_predictor = None
    loaded_threshold = None
    loaded_sigma = 0.5 # default
    artifacts_loaded = False

    try:
        print("Loading artifacts for predictor demo...")
        if not os.path.exists(Config.MODEL_PATH): raise FileNotFoundError(f"Model file not found: {Config.MODEL_PATH}")
        if not os.path.exists(Config.ENCODER_PATH): raise FileNotFoundError(f"Encoder file not found: {Config.ENCODER_PATH}")
        if not os.path.exists(Config.THRESHOLD_PATH): raise FileNotFoundError(f"Threshold file not found: {Config.THRESHOLD_PATH}")

        predictor_model = TransformerPredictor(
            vocab_size=vocab_size,
            embed_size=Config.EMBED_SIZE, num_heads=Config.NUM_HEADS,
            num_layers=Config.NUM_LAYERS, dropout=Config.DROPOUT
        ).to(device)
        predictor_model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        predictor_model.eval()

        with open(Config.ENCODER_PATH, 'rb') as f:
            le_predictor = pickle.load(f)
        with open(Config.THRESHOLD_PATH, 'r') as f:
            # Format: 'THRESHOLD,SIGMA' or just 'THRESHOLD' for backward compat
            content = f.read().strip()
            if ',' in content:
                loaded_threshold, loaded_sigma = map(float, content.split(','))
            else:
                loaded_threshold = float(content)
                
        print("Artifacts loaded successfully.")
        artifacts_loaded = True
    except Exception as e:
        print(f"\n--- PREDICTOR DEMO SKIPPED: Error loading artifacts: {e}. ---")
        artifacts_loaded = False

    if artifacts_loaded:
        print("\n--- PREDICTOR DEMO ---")
        normal_test_sessions_df = test_df[test_df['Label'] == 0]
        abnormal_test_sessions_df = test_df[test_df['Label'] == 1]
        
        long_normal_sessions = normal_test_sessions_df[normal_test_sessions_df['SeqLength'] > 5]
        long_abnormal_sessions = abnormal_test_sessions_df[abnormal_test_sessions_df['SeqLength'] > 5]

        if not long_normal_sessions.empty and not long_abnormal_sessions.empty:
            normal_codes_example = long_normal_sessions.iloc[0]['EventCode']
            abnormal_codes_example = long_abnormal_sessions.iloc[0]['EventCode']

            def safe_inverse_transform(le, codes):
                try:
                    return list(le.inverse_transform(codes))
                except Exception:
                    return [str(c) for c in codes]

            normal_sequence_example = safe_inverse_transform(le_predictor, normal_codes_example)
            abnormal_sequence_example = safe_inverse_transform(le_predictor, abnormal_codes_example)

            print(f"1. Analyzing a real normal session (length: {len(normal_sequence_example)})...")
            result_normal = analyze_log_sequence(normal_sequence_example, predictor_model, le_predictor, loaded_threshold, vocab_size=vocab_size, sigma=loaded_sigma)
            print(f"   Verdict: {result_normal.get('overall_verdict', 'Error')} (Risk: {result_normal.get('risk_level', 'N/A')})")
            print(f"   Score: {result_normal.get('max_anomaly_score')} vs Thresholds (Suspicious: {result_normal.get('suspicious_threshold')}, Anomaly: {result_normal.get('anomaly_threshold')})\n")

            print(f"2. Analyzing a real abnormal session (length: {len(abnormal_sequence_example)})...")
            result_abnormal = analyze_log_sequence(abnormal_sequence_example, predictor_model, le_predictor, loaded_threshold, vocab_size=vocab_size, sigma=loaded_sigma)
            print(f"   Verdict: {result_abnormal.get('overall_verdict', 'Error')} (Risk: {result_abnormal.get('risk_level', 'N/A')})")
            print(f"   Score: {result_abnormal.get('max_anomaly_score')} vs Thresholds (Suspicious: {result_abnormal.get('suspicious_threshold')}, Anomaly: {result_abnormal.get('anomaly_threshold')})")

            # Explainability: какие именно события сделали сессию аномальной
            try:
                from explainer import explain_session, print_explanation
                explanation = explain_session(
                    predictor_model, abnormal_codes_example, le_predictor,
                    device, vocab_size, top_n=5
                )
                print_explanation(
                    explanation,
                    session_score=result_abnormal.get('max_anomaly_score'),
                    threshold=result_abnormal.get('anomaly_threshold')
                )
            except Exception as e:
                print(f"   (Explainability report failed: {e})")
            print()

            print("3. Analyzing the same abnormal session from an 'admin' on a 'critical' resource...")
            result_critical = analyze_log_sequence(abnormal_sequence_example, predictor_model, le_predictor, loaded_threshold, vocab_size=vocab_size, context_user='admin', context_resource='critical', sigma=loaded_sigma)
            print(f"   Verdict: {result_critical.get('overall_verdict', 'Error')} (Risk: {result_critical.get('risk_level', 'N/A')})")
            print(f"   Context: {result_critical.get('context')}")
            print(f"   Original Score: {result_critical.get('max_anomaly_score')} -> Weighted Score: {result_critical.get('final_weighted_score')}")
        else:
            print("Could not find suitable sessions for demo.")
    print("\n--- Prototype Execution Finished ---")
