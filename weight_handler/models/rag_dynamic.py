# models/rag_dynamic.py
import os, json
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

FEATURE_DIM = 6  # [news, general, focus, flow, mentions, twitter]

class SimpleWeightMLP(nn.Module):
    """
    Baseline: map masked features -> scalar score.
    Optionally learns attention-like weights internally.
    """
    def __init__(self, in_dim=FEATURE_DIM*2):  # feats + mask
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: [B, 12] for 6 feats + 6 mask
        return self.net(x).squeeze(-1)

class LSTMWeighting(nn.Module):
    """
    Sequence model: (T, feat+mask) -> scalar score at T.
    """
    def __init__(self, feat_dim=FEATURE_DIM*2, hidden=48, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq):  # [B, T, 12]
        h, _ = self.lstm(seq)
        last = h[:, -1, :]     # use last step representation
        return self.head(last).squeeze(-1)

def load_dynamic_model(model_dir: str) -> Optional[nn.Module]:
    """
    Try to load a trained PyTorch model (MLP or LSTM). Returns eval() or None.
    """
    try:
        paths = [os.path.join(model_dir, "rag_mlp.pt"),
                 os.path.join(model_dir, "rag_lstm.pt")]
        for p in paths:
            if os.path.exists(p):
                bundle = torch.load(p, map_location="cpu")
                kind = bundle.get("kind", "mlp")
                if kind == "mlp":
                    model = SimpleWeightMLP()
                else:
                    model = LSTMWeighting()
                model.load_state_dict(bundle["state_dict"])
                model.eval()
                return model
    except Exception:
        return None
    return None

@torch.no_grad()
def predict_score(model: nn.Module, feats: List[float], mask: List[int],
                  recent_seq: Optional[List[Tuple[List[float], List[int]]]] = None) -> float:
    """
    If model is LSTM and 'recent_seq' provided: use sequence.
    Else: use single-step MLP.
    """
    if isinstance(model, LSTMWeighting) and recent_seq:
        import numpy as np
        seq = []
        for f, m in recent_seq:
            seq.append((f + m))
        x = torch.tensor([seq], dtype=torch.float32)  # [1, T, 12]
        y = model(x).item()
        return float(y)
    else:
        x = torch.tensor([feats + mask], dtype=torch.float32)  # [1, 12]
        y = model(x).item()
        return float(y)
