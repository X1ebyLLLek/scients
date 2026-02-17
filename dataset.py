import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Any
import random
from config import Config

class MaskedLogDataset(Dataset):
    """
    Dataset for BERT-style Masked Language Modeling (MLM).
    Takes a whole session as input.
    """
    def __init__(self, sessions: List[List[int]], labels: List[int] = None, mask_prob: float = 0.15):
        self.sessions = sessions
        self.labels = labels
        self.mask_prob = mask_prob
        # Filter too short sessions
        self.valid_indices = [i for i,	session in enumerate(sessions) if len(session) >= 2]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        session_idx = self.valid_indices[idx]
        session = self.sessions[session_idx]
        
        # Truncate if too long
        if len(session) > Config.MAX_SEQ_LEN:
             # For anomaly detection, recent events matter, but for general LM training, 
             # random crops might be better. Let's stick to recent for now.
             session = session[-Config.MAX_SEQ_LEN:]
             
        # Add 1 to reserve 0 for padding. Now [1, vocab_size]
        # We need a special token for MASK. Let's say vocab_size + 1 is MASK.
        # But we don't know vocab_size here easily without passing it.
        # Check Config? No.
        # Convention: Labels are already 0-indexed.
        # Input to model: 0=PAD, 1...N=Events, N+1=MASK.
        # Let's assume the caller handles vocab size or we use robust encoding.
        
        # Actually, let's keep it simple:
        # Input tokens: 1..N
        # 0 is PAD
        # We need a MASK token. Let's use a fixed high ID or pass it?
        # Better: The collator should handle masking to be dynamic!
        # Dataset just returns the sessions.
        
        input_tensor = torch.tensor([s + 1 for s in session], dtype=torch.long)
        
        label = 0
        if self.labels:
             label = self.labels[session_idx]
             
        return input_tensor, label, session_idx

def collate_fn_mlm(batch: List[Tuple[torch.Tensor, int, int]], vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dynamic masking for MLM.
    Returns:
        masked_inputs: Tensor (B, L) with some tokens replaced by MASK_ID
        targets: Tensor (B, L) with original tokens at masked positions, -100 elsewhere
        labels: Tensor (B,) session anomaly labels
        padding_mask: Tensor (B, L)
        session_indices: Tensor (B,)
    """
    inputs_list, labels_list, session_indices_list = zip(*batch)
    
    padded_inputs = pad_sequence(inputs_list, batch_first=True, padding_value=0)
    labels = torch.tensor(labels_list, dtype=torch.long)
    session_indices = torch.tensor(session_indices_list, dtype=torch.long)
    padding_mask = (padded_inputs == 0)
    
    # Create MLM targets
    targets = padded_inputs.clone()
    probability_matrix = torch.full(targets.shape, 0.15)
    probability_matrix.masked_fill_(padding_mask, value=0.0) # Don't mask padding
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Set targets to -100 (ignore index) for unmasked tokens
    targets[~masked_indices] = -100
    
    # Replace masked input tokens with MASK token
    # MASK token ID = vocab_size + 1 (since 0 is PAD, original vocab is 1..V)
    # Be careful with indices. 
    # Original events: 0..V-1.
    # Dataset outputs: 1..V.
    # MASK ID should be V + 1.
    
    # We need to know MASK_ID. 
    # Let's define MASK_ID as a constant or derived.
    # Since we shift by 1, max index in padded_inputs is vocab_size. 
    # So MASK_ID can be vocab_size + 1.
    
    # MASK token ID. 
    # vocab_size includes PAD (0) and events (1..len).
    # So max event index is vocab_size - 1. 
    # We can use vocab_size as the MASK ID.
    mask_token_id = vocab_size
    
    # 80% replace with MASK
    indices_replaced = torch.bernoulli(torch.full(targets.shape, 0.8)).bool() & masked_indices
    padded_inputs[indices_replaced] = mask_token_id
    
    # 10% replace with random token (1..vocab_size)
    indices_random = torch.bernoulli(torch.full(targets.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(1, vocab_size + 1, targets.shape, dtype=torch.long)
    padded_inputs[indices_random] = random_words[indices_random]
    
    # 10% keep original (already in padded_inputs)
    
    return padded_inputs, targets, labels, padding_mask, session_indices
