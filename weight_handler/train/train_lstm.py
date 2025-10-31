# train/train_lstm.py
import os, json, math, collections
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.rag_dynamic import LSTMWeighting, FEATURE_DIM

SNAP_PATH = "test_data/rag_snapshots_labeled.jsonl"
SAVE_DIR  = "ml_models"; os.makedirs(SAVE_DIR, exist_ok=True)
T = 7  # sequence length

class SeqDS(Dataset):
    def __init__(self, path, T):
        by_coin = collections.defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("label") is None: continue
                by_coin[r["coin"]].append(r)
        # sort by ts
        for c in by_coin:
            by_coin[c].sort(key=lambda r: r["ts"])

        self.samples = []
        for c, rows in by_coin.items():
            if len(rows) <= T: continue
            # build rolling windows
            for i in range(len(rows) - T):
                seq = rows[i:i+T]
                y   = rows[i+T]["label"]  # predict next-step return
                x = [s["feats"] + s["mask"] for s in seq]  # [T, 12]
                self.samples.append((x, y))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        x = torch.tensor(x, dtype=torch.float32)  # [T, 12]
        return x, torch.tensor([y], dtype=torch.float32)

def train():
    ds = SeqDS(SNAP_PATH, T=T)
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    model = LSTMWeighting(feat_dim=FEATURE_DIM*2, hidden=48)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(12):
        model.train()
        tot = 0.0
        for x, y in dl:
            opt.zero_grad()
            yp = model(x).unsqueeze(-1)
            loss = loss_fn(yp, y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        print(f"epoch {epoch} loss={tot/len(ds):.4f}")

    torch.save({"kind": "lstm", "state_dict": model.state_dict()},
               os.path.join(SAVE_DIR, "rag_lstm.pt"))

if __name__ == "__main__":
    train()
