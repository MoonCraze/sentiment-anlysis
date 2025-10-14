# train/train_mlp.py
import os, json, math, random
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from weight_handler.models.rag_dynamic import SimpleWeightMLP, FEATURE_DIM

#from models.rag_dynamic import SimpleWeightMLP, FEATURE_DIM

SNAP_PATH = "test data/rag_snapshots_labeled.jsonl"  # same rows but 'label' filled
SAVE_DIR  = "ml_models"; os.makedirs(SAVE_DIR, exist_ok=True)

class SnapDS(Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("label") is None:  # skip unlabeled
                    continue
                feats = row["feats"]; mask = row["mask"]; y = float(row["label"])
                self.rows.append((feats, mask, y))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        f, m, y = self.rows[i]
        x = torch.tensor(f + m, dtype=torch.float32)  # [12]
        return x, torch.tensor([y], dtype=torch.float32)

def train():
    ds = SnapDS(SNAP_PATH)
    n = len(ds); assert n > 100, "Not enough labeled samples"
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    model = SimpleWeightMLP(in_dim=FEATURE_DIM*2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.SmoothL1Loss()  # robust regression

    for epoch in range(10):
        model.train()
        total = 0.0
        for x, y in dl:
            opt.zero_grad()
            yp = model(x).unsqueeze(-1)  # [B,1]
            loss = loss_fn(yp, y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"epoch {epoch} loss={total/len(ds):.4f}")

    torch.save({"kind": "mlp", "state_dict": model.state_dict()},
               os.path.join(SAVE_DIR, "rag_mlp.pt"))

if __name__ == "__main__":
    train()
