# models/rag_seqcache.py
import json, os
from typing import Dict, List, Tuple

def get_recent_sequence(path: str, coin: str, T: int = 7) -> List[Tuple[List[float], List[int]]]:
    seq = []
    if not os.path.exists(path): return seq
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("coin") == coin and r.get("feats") and r.get("mask"):
                seq.append((r["feats"], r["mask"]))
    return seq[-T:]  # last T entries if available
