# models/rag_logging.py
import os, json, time
from typing import Dict, Any, List

def append_snapshot(path: str, ts: float, profiles: Dict[str, Dict[str, Any]]):
    """Append one JSON line per coin: {ts, coin, feats, mask, label?} (label filled later)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for coin, v in profiles.items():
            feats = [
                v["score_breakdown"].get("news_sent", 0.0),
                v["score_breakdown"].get("general_sent", 0.0),
                v["score_breakdown"].get("focus_sent", 0.0),
                v["score_breakdown"].get("flow_z", 0.0),
                v["score_breakdown"].get("mentions_z", 0.0),
                v["score_breakdown"].get("twitter_sent", 0.0),
            ]
            mask = [1 if v.get("news_sent") is not None else 0,
                    1 if v.get("general_sent") is not None else 0,
                    1 if v.get("focus_sent") is not None else 0,
                    1 if v["score_breakdown"].get("flow_z") else 0,
                    1 if v["score_breakdown"].get("mentions_z") else 0,
                    1 if v.get("twitter_sent") is not None else 0]
            row = {"ts": ts, "coin": coin, "feats": feats, "mask": mask, "label": None}
            f.write(json.dumps(row) + "\n")
