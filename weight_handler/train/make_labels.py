# train/make_labels.py
import json, os
from collections import defaultdict

SNAP_IN   = "test_data/rag_snapshots.jsonl"
SNAP_OUT  = "test_data/rag_snapshots_labeled.jsonl"
PRICES    = "test_data/ohlc.json"  # {"BTC": [{"ts":unix, "close":...}, ...], ...}

def nearest_future(ts_list, ts):
    # find next timestamp > ts; assume ts_list sorted
    import bisect
    i = bisect.bisect_right(ts_list, ts)
    return ts_list[i] if i < len(ts_list) else None

def run():
    with open(PRICES, "r", encoding="utf-8") as f:
        ohlc = json.load(f)
    # build per-coin ts->price map
    pmap = {}
    for c, rows in ohlc.items():
        rows.sort(key=lambda r: r["ts"])
        pmap[c] = ( [r["ts"] for r in rows], {r["ts"]: r["close"] for r in rows} )

    out = open(SNAP_OUT, "w", encoding="utf-8")
    with open(SNAP_IN, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            coin = r["coin"]; ts = int(r["ts"])
            if coin not in pmap:
                r["label"] = None
            else:
                ts_list, price_map = pmap[coin]
                nxt = nearest_future(ts_list, ts)
                if nxt is None:
                    r["label"] = None
                else:
                    p0 = price_map.get(ts_list[max(0, ts_list.index(nxt)-1)], None)
                    p1 = price_map[nxt]
                    if p0 is None or p0 <= 0:
                        r["label"] = None
                    else:
                        r["label"] = float((p1 - p0) / p0)  # simple return; or log-return
            out.write(json.dumps(r) + "\n")
    out.close()

if __name__ == "__main__":
    run()
