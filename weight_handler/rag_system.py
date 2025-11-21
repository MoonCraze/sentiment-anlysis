# models/rag_system.py
import os, json, math, time, re, datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from threading import Lock

# ----------------------------
# Paths & constants
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "test data"))
VIS_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "visualizations"))
EVAL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "evaluation", "results"))

PATHS = {
    "news_sentiment":    os.path.join(DATA_DIR, "sentiment_output_for_news.json"),
    "general_sentiment": os.path.join(DATA_DIR, "sentiment_output_general.json"),
    "focus_sentiment":   os.path.join(DATA_DIR, "sentiment_output_for_coin_finder.json"),
    "coin_flow":         os.path.join(DATA_DIR, "Analysis_output_for_coin_flow.json"),
    "coin_finder":       os.path.join(DATA_DIR, "coin_keywords_extracted.json"),
    "verified_focus":    os.path.join(DATA_DIR, "verified_sentiment_output_focus_group.json"),
    "twitter_cache":     os.path.join(DATA_DIR, "twitter_sentiment_cache.json"),
    "snapshots":         os.path.join(DATA_DIR, "rag_snapshots.jsonl"),
    "ml_models":         os.path.abspath(os.path.join(BASE_DIR, "..", "ml_models")),
}

# Best-effort plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# ----------------------------
# Canonicalization & tagging
# ----------------------------
CANON: Dict[str, set] = {
    "BTC": {"$BTC", "BTC", "BITCOIN"},
    "ETH": {"$ETH", "ETH", "ETHEREUM"},
    "XRP": {"$XRP", "XRP"},
    "SOL": {"$SOL", "SOL", "SOLANA"},
    "ADA": {"$ADA", "ADA", "CARDANO"},
    "BNB": {"$BNB", "BNB", "BINANCE"},
    "LTC": {"$LTC", "LTC", "LITECOIN"},
    "CRO": {"$CRO", "CRO", "CRONOS"},
    "XVG": {"$XVG", "XVG", "VERGE"},
    "ONDO": {"$ONDO", "ONDO"},
}
TOKEN2TICKER: Dict[str, str] = {tok: t for t, toks in CANON.items() for tok in toks}

R_DOLLAR = re.compile(r'(?<!\w)\$([A-Za-z][A-Za-z0-9.\-]{1,9})(?!\w)')
NAME_TOKENS = sorted(
    {n for toks in CANON.values() for n in toks if not n.startswith("$") and n.isalpha()},
    key=len, reverse=True
)
R_NAMES = re.compile(r'(?<!\w)(' + "|".join(map(re.escape, NAME_TOKENS)) + r')(?!\w)', flags=re.IGNORECASE)

def canon_coin(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    u = token.upper().strip()
    if u in TOKEN2TICKER:
        return TOKEN2TICKER[u]
    if u.startswith("$") and u[1:] in TOKEN2TICKER:
        return TOKEN2TICKER[u[1:]]
    if u in CANON:
        return u
    return None

def extract_coins(text: str) -> set:
    coins = set()
    if not text:
        return coins
    for m in R_DOLLAR.findall(text):
        c = canon_coin(m)
        if c:
            coins.add(c)
    for m in R_NAMES.findall(text):
        c = canon_coin(m)
        if c:
            coins.add(c)
    return coins

# ----------------------------
# Scaling & helpers
# ----------------------------
LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}

def _safe_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _winsorize(values: Dict[str, float], p: float = 0.02) -> Dict[str, float]:
    if not values:
        return {}
    xs = sorted(values.values())
    lo = xs[int(p * len(xs))]
    hi = xs[int(max(0, (1 - p) * len(xs) - 1))]
    return {k: min(max(v, lo), hi) for k, v in values.items()}

def _robust_scale(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    xs = sorted(values.values())
    n = len(xs)
    med = xs[n // 2]
    mad = sorted(abs(x - med) for x in xs)[n // 2]
    mad = mad if mad > 0 else 1.0
    return {k: (v - med) / (1.4826 * mad) for k, v in values.items()}

def _shrinkage_ratio(pos: int, neg: int, prior: int = 2) -> Optional[float]:
    total = pos + neg
    if total <= 0:
        return None
    p_hat = (pos + prior) / (pos + neg + 2 * prior)
    return 2 * p_hat - 1  # [-1,1]

def _to_pct(pos: int, neg: int) -> Tuple[float, float]:
    total = max(1, pos + neg)  # ignore neutrals for pct
    return round(100 * pos / total, 2), round(100 * neg / total, 2)

def _ts_str(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H-%M-%S")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
def _norm_query_key(q: str) -> Optional[str]:
    if not q:
        return None
    uq = q.strip().upper()
    # keep it reasonably bounded; accept letters, digits, symbols common in pump/tickers
    if 1 <= len(uq) <= 128:
        return uq
    return None

# ----------------------------
# Available-coin → twitter cache ingest (optional helper)
# ----------------------------
def ingest_twitter_sentiment_cache(rows):
    """
    Accept a dict (single available-coin result) or list of dicts:
      {"query": "...", "positive": int, "negative": int, "total_mentions": int, ...}
    Merge into test data/twitter_sentiment_cache.json.
    """
    cache_path = PATHS["twitter_cache"]
    _ensure_dir(os.path.dirname(cache_path))
    try:
        existing = json.load(open(cache_path, "r", encoding="utf-8"))
        if not isinstance(existing, list):
            existing = []
    except Exception:
        existing = []

    def normalize(one):
        q = (one.get("query") or "").strip()
        pos = int(one.get("positive", 0))
        neg = int(one.get("negative", 0))
        tot = int(one.get("total_mentions", pos + neg))
        return {"query": q, "positive": pos, "negative": neg, "total_mentions": tot}

    if isinstance(rows, dict):
        rows = [rows]
    normalized = [normalize(r) for r in rows if isinstance(r, dict)]

    by_q = {(r.get("query") or "").upper(): r for r in existing if isinstance(r, dict) and r.get("query")}
    for r in normalized:
        by_q[(r["query"] or "").upper()] = r

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(list(by_q.values()), f, ensure_ascii=False, indent=2)

# ----------------------------
# Dynamic model loading (optional)
# ----------------------------
try:
    from models.rag_dynamic import load_dynamic_model, predict_score  # type: ignore
    _DYN_MODEL = None
    _DYN_MODEL_TRIED = False
except Exception:
    _DYN_MODEL = None
    _DYN_MODEL_TRIED = True

# Optional recent-sequence provider for LSTM
try:
    from models.rag_seqcache import get_recent_sequence  # type: ignore
except Exception:
    def get_recent_sequence(path: str, coin: str, T: int = 7):
        return []

# Optional snapshot appender for training
try:
    from models.rag_logging import append_snapshot  # type: ignore
except Exception:
    def append_snapshot(path: str, ts: float, profiles: Dict[str, Dict[str, Any]]):
        pass

# ----------------------------
# Visualization savers
# ----------------------------
def _save_run_artifacts(run_dir: str, profiles_sorted: Dict[str, Dict[str, Any]]):
    try:
        with open(os.path.join(run_dir, "profiles.json"), "w", encoding="utf-8") as f:
            json.dump(profiles_sorted, f, ensure_ascii=False, indent=2)

        rows = []
        for coin, v in profiles_sorted.items():
            sb = v.get("score_breakdown", {})
            rows.append({
                "coin": coin,
                "score": v.get("score", 0.0),
                "news_sent": sb.get("news_sent", 0.0),
                "general_sent": sb.get("general_sent", 0.0),
                "focus_sent": sb.get("focus_sent", 0.0),
                "flow_z": sb.get("flow_z", 0.0),
                "mentions_z": sb.get("mentions_z", 0.0),
                "twitter_sent": sb.get("twitter_sent", 0.0),
                "confidence": v.get("confidence", 0.0),
                "evidence": v.get("evidence", 0),
                "mode": sb.get("_mode", "static-weights"),
            })
        if rows:
            headers = list(rows[0].keys())
            with open(os.path.join(run_dir, "scores.csv"), "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for r in rows:
                    f.write(",".join(str(r[h]) for h in headers) + "\n")
    except Exception:
        pass

def _save_visualizations(run_dir: str, profiles_sorted: Dict[str, Dict[str, Any]], top_n: int = 10):
    if not _HAS_MPL:
        return
    try:
        import numpy as np
        coins = list(profiles_sorted.keys())[:max(1, top_n)]
        if not coins:
            return

        scores = [profiles_sorted[c].get("score", 0.0) for c in coins]
        sb = [profiles_sorted[c].get("score_breakdown", {}) for c in coins]
        news = [x.get("news_sent", 0.0) for x in sb]
        gen  = [x.get("general_sent", 0.0) for x in sb]
        focus= [x.get("focus_sent", 0.0) for x in sb]
        flow = [x.get("flow_z", 0.0) for x in sb]
        ment = [x.get("mentions_z", 0.0) for x in sb]
        twit = [x.get("twitter_sent", 0.0) for x in sb]

        # Leaderboard
        plt.figure(figsize=(12, 6))
        x = range(len(coins))
        plt.bar(x, scores)
        plt.xticks(x, coins, rotation=45, ha="right")
        plt.ylabel("Score")
        plt.title(f"Top {len(coins)} Coins — Composite Score")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "leaderboard_top10.png"), dpi=180)
        plt.close()

        # Stacked breakdown
        plt.figure(figsize=(12, 7))
        x = np.arange(len(coins))
        b1 = np.array(news); b2 = np.array(gen); b3 = np.array(focus)
        b4 = np.array(flow); b5 = np.array(ment); b6 = np.array(twit)
        plt.bar(x, b1, label="news_sent")
        plt.bar(x, b2, bottom=b1, label="general_sent")
        plt.bar(x, b3, bottom=b1+b2, label="focus_sent")
        plt.bar(x, b4, bottom=b1+b2+b3, label="flow_z")
        plt.bar(x, b5, bottom=b1+b2+b3+b4, label="mentions_z")
        plt.bar(x, b6, bottom=b1+b2+b3+b4+b5, label="twitter_sent")
        plt.xticks(x, coins, rotation=45, ha="right")
        plt.ylabel("Component Value (signed)")
        plt.title(f"Score Breakdown (Stacked) — Top {len(coins)}")
        plt.legend(ncol=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "breakdown_top10.png"), dpi=180)
        plt.close()

        # Heatmap
        feat_mat = np.vstack([news, gen, focus, flow, ment, twit])
        plt.figure(figsize=(12, 4.5))
        im = plt.imshow(feat_mat, aspect="auto")
        plt.colorbar(im, fraction=0.022, pad=0.04)
        plt.yticks(range(6), ["news", "general", "focus", "flow_z", "mentions_z", "twitter"])
        plt.xticks(range(len(coins)), coins, rotation=45, ha="right")
        plt.title(f"Feature Heatmap — Top {len(coins)}")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "heatmap_features.png"), dpi=180)
        plt.close()
    except Exception:
        pass

# ----------------------------
# Evaluation artifacts → evaluation/results
# ----------------------------
def _save_evaluation(eval_dir: str, profiles_sorted: Dict[str, Dict[str, Any]]):
    try:
        _ensure_dir(eval_dir)
        coins = list(profiles_sorted.keys())
        if not coins:
            return

        feats = ["news_sent", "general_sent", "focus_sent", "flow_z", "mentions_z", "twitter_sent"]
        mat = []
        for c in coins:
            sb = profiles_sorted[c].get("score_breakdown", {})
            mat.append([float(sb.get(f, 0.0)) for f in feats])

        # coverage
        cover = {f: 0 for f in feats}
        for c in coins:
            sb = profiles_sorted[c].get("score_breakdown", {})
            for f in feats:
                v = sb.get(f, None)
                if f in ("flow_z", "mentions_z"):
                    if v is not None:
                        cover[f] += 1
                else:
                    if v not in (None, 0.0):
                        cover[f] += 1

        metrics = {
            "coins_indexed": len(coins),
            "feature_coverage": {f: round(cover[f] / max(1, len(coins)), 4) for f in feats},
        }
        with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # CSV table
        headers = ["coin", "score"] + feats + ["confidence", "evidence"]
        with open(os.path.join(eval_dir, "top_table.csv"), "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for coin, v in profiles_sorted.items():
                sb = v.get("score_breakdown", {})
                row = [coin, v.get("score", 0.0)] + [sb.get(f, 0.0) for f in feats] + [v.get("confidence", 0.0), v.get("evidence", 0)]
                f.write(",".join(str(x) for x in row) + "\n")

        # plots
        if _HAS_MPL:
            import numpy as np
            scores = [profiles_sorted[c].get("score", 0.0) for c in coins]
            plt.figure(figsize=(8,4))
            plt.hist(scores, bins=20)
            plt.title("Score Distribution")
            plt.xlabel("Score"); plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(eval_dir, "eval_hist_scores.png"), dpi=160)
            plt.close()

            try:
                M = np.array(mat, dtype=float)  # [N,6]
                C = np.corrcoef(M.T)
                plt.figure(figsize=(6,5))
                im = plt.imshow(C, vmin=-1, vmax=1, aspect="equal")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xticks(range(len(feats)), feats, rotation=45, ha="right")
                plt.yticks(range(len(feats)), feats)
                plt.title("Feature Correlation")
                plt.tight_layout()
                plt.savefig(os.path.join(eval_dir, "correlation_matrix.png"), dpi=180)
                plt.close()
            except Exception:
                pass

            try:
                M = np.array(mat, dtype=float)
                for j, ftr in enumerate(feats):
                    plt.figure(figsize=(6,3))
                    plt.hist(M[:, j], bins=20)
                    plt.title(f"Histogram — {ftr}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(eval_dir, f"feature_hist_{ftr}.png"), dpi=160)
                    plt.close()
            except Exception:
                pass
    except Exception:
        pass

# ----------------------------
# In-memory index & lock
# ----------------------------
_RAG_INDEX: Dict[str, Dict[str, Any]] = {}
_RAG_TS: float = 0.0
_BUILD_LOCK: Lock = Lock()

# ----------------------------
# Core builder
# ----------------------------
def build_rag_index(weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Build per-coin profiles from pipeline outputs, normalize, score
    (optionally with a dynamic model), cache, and write artifacts.
    """
    global _RAG_INDEX, _RAG_TS, _DYN_MODEL, _DYN_MODEL_TRIED
    with _BUILD_LOCK:
        weights = weights or {
            "news_sent": 0.25,
            "general_sent": 0.15,
            "focus_sent": 0.20,
            "flow": 0.25,
            "mentions": 0.10,
            "twitter_sent": 0.05,
        }

        # load dynamic model once
        if not _DYN_MODEL_TRIED:
            try:
                _DYN_MODEL = load_dynamic_model(PATHS["ml_models"])  # may be None
            except Exception:
                _DYN_MODEL = None
            _DYN_MODEL_TRIED = True

        profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "news_sent": None,
            "general_sent": None,
            "focus_sent": None,
            "flow": 0.0,
            "mentions": 0.0,
            "twitter_pos": 0,
            "twitter_neg": 0,
            "twitter_total": 0,
            "twitter_pos_pct": 0.0,
            "twitter_neg_pct": 0.0,
            "twitter_sent": None,
            "sources": [],
        })

        # 1) Coin Flow
        coin_flow = _safe_load(PATHS["coin_flow"])
        if isinstance(coin_flow, dict):
            agg = coin_flow.get("aggregated_flows", {})
            for coin_raw, val in agg.items():
                coin = canon_coin(coin_raw)
                if not coin:
                    continue
                profiles[coin]["flow"] = profiles[coin].get("flow", 0.0) + float(val)
                profiles[coin]["sources"].append("coin_flow")

        # 2) Focus Sentiment
        focus_sent = _safe_load(PATHS["focus_sentiment"])
        if isinstance(focus_sent, dict):
            avg = focus_sent.get("average_sentiment", {})
            for coin_raw, s in avg.items():
                coin = canon_coin(coin_raw)
                if not coin:
                    continue
                profiles[coin]["focus_sent"] = float(s)
                profiles[coin]["sources"].append("focus_sentiment")

        # 3) Coin Finder Mentions
        cf = _safe_load(PATHS["coin_finder"])
        if isinstance(cf, dict):
            ckf = cf.get("coin_keywords_filtered", {})
            mention_scores: Dict[str, float] = defaultdict(float)
            for _, kw_counts in ckf.items():
                if not isinstance(kw_counts, dict):
                    continue
                for word, cnt in kw_counts.items():
                    coin = canon_coin((word or "").strip())
                    if coin:
                        mention_scores[coin] += float(cnt)
            for coin, m in mention_scores.items():
                profiles[coin]["mentions"] = profiles[coin].get("mentions", 0.0) + m
                profiles[coin]["sources"].append("coin_finder")

        # 4) General Sentiment
        gen = _safe_load(PATHS["general_sentiment"])
        if isinstance(gen, list):
            per_coin_scores = defaultdict(list)
            for item in gen:
                text = item.get("text") or ""
                label = (item.get("sentiment") or "NEUTRAL").upper()
                score = LABEL_MAP.get(label, 0)
                for coin in extract_coins(text):
                    per_coin_scores[coin].append(score)
            for coin, ss in per_coin_scores.items():
                if ss:
                    profiles[coin]["general_sent"] = sum(ss) / len(ss)
                    profiles[coin]["sources"].append("general_sentiment")

        # 5) News Sentiment
        news = _safe_load(PATHS["news_sentiment"])
        if isinstance(news, list):
            per_coin_scores = defaultdict(list)
            for item in news:
                text = item.get("text") or ""
                label = (item.get("dominant_sentiment") or "NEUTRAL").upper()
                score = LABEL_MAP.get(label, 0)
                for coin in extract_coins(text):
                    per_coin_scores[coin].append(score)
            for coin, ss in per_coin_scores.items():
                if ss:
                    profiles[coin]["news_sent"] = sum(ss) / len(ss)
                    profiles[coin]["sources"].append("news_sentiment")

        # 6) Twitter cache (from Available-coin runs)
        tw = _safe_load(PATHS["twitter_cache"])
        if isinstance(tw, list):
            for row in tw:
                q_raw = row.get("query") or ""
                # Try canonical first (BTC/ETH/…)
                coin = canon_coin(q_raw)
                if not coin:
                    # Try extracting $TICKER from the query text
                    ex = extract_coins(q_raw)
                    coin = next(iter(ex)) if ex else None
                # If still unknown, index under the raw query key
                if not coin:
                    coin = _norm_query_key(q_raw)
                if not coin:
                    continue

                pos = int(row.get("positive", 0))
                neg = int(row.get("negative", 0))
                total = int(row.get("total")) if "total" in row else int(row.get("total_mentions", pos + neg))
                profiles[coin]["twitter_pos"] += pos
                profiles[coin]["twitter_neg"] += neg
                profiles[coin]["twitter_total"] += total
                profiles[coin]["sources"].append("twitter_sentiment")

        # Normalize numeric
        flow_map = {c: v["flow"] for c, v in profiles.items()}
        mentions_map = {c: v["mentions"] for c, v in profiles.items()}
        flow_z = _robust_scale(_winsorize(flow_map))
        mentions_z = _robust_scale(_winsorize(mentions_map))

        # Twitter pct + shrunk ratio
        for coin, v in profiles.items():
            pos, neg = int(v["twitter_pos"]), int(v["twitter_neg"])
            pos_pct, neg_pct = _to_pct(pos, neg)
            v["twitter_pos_pct"] = pos_pct
            v["twitter_neg_pct"] = neg_pct
            v["twitter_sent"] = _shrinkage_ratio(pos, neg, prior=2)

        # Final scoring
        for coin, v in profiles.items():
            detail: Dict[str, Any] = {}
            ns = v["news_sent"] if v["news_sent"] is not None else 0.0
            gs = v["general_sent"] if v["general_sent"] is not None else 0.0
            fs = v["focus_sent"] if v["focus_sent"] is not None else 0.0
            tws = v["twitter_sent"] if v["twitter_sent"] is not None else 0.0
            fl = flow_z.get(coin, 0.0)
            mn = mentions_z.get(coin, 0.0)

            detail["news_sent"] = ns
            detail["general_sent"] = gs
            detail["focus_sent"] = fs
            detail["flow_z"] = fl
            detail["mentions_z"] = mn
            detail["twitter_sent"] = tws

            evidence = 0
            evidence += 1 if v["news_sent"] is not None else 0
            evidence += 1 if v["general_sent"] is not None else 0
            evidence += 1 if v["focus_sent"] is not None else 0
            evidence += 1 if coin in flow_z else 0
            evidence += 1 if coin in mentions_z else 0
            evidence += 1 if v["twitter_sent"] is not None else 0

            feats = [ns, gs, fs, fl, mn, tws]
            mask = [
                int(v["news_sent"] is not None),
                int(v["general_sent"] is not None),
                int(v["focus_sent"] is not None),
                int(coin in flow_z),
                int(coin in mentions_z),
                int(v["twitter_sent"] is not None),
            ]

            if '_DYN_MODEL' in globals() and _DYN_MODEL is not None:
                try:
                    recent_seq = get_recent_sequence(PATHS["snapshots"], coin, T=7)
                except Exception:
                    recent_seq = []
                try:
                    dyn_score = predict_score(_DYN_MODEL, feats, mask, recent_seq=recent_seq)
                    v["score"] = round(float(dyn_score), 4)
                    detail["_mode"] = "dynamic-model"
                except Exception:
                    score = (weights["news_sent"] * ns +
                             weights["general_sent"] * gs +
                             weights["focus_sent"] * fs +
                             weights["flow"] * fl +
                             weights["mentions"] * mn +
                             weights["twitter_sent"] * tws)
                    v["score"] = round(score, 4)
                    detail["_mode"] = "static-weights"
            else:
                score = (weights["news_sent"] * ns +
                         weights["general_sent"] * gs +
                         weights["focus_sent"] * fs +
                         weights["flow"] * fl +
                         weights["mentions"] * mn +
                         weights["twitter_sent"] * tws)
                v["score"] = round(score, 4)
                detail["_mode"] = "static-weights"

            agree = 0
            for k in ["news_sent", "general_sent", "focus_sent", "twitter_sent"]:
                val = detail.get(k, 0.0)
                if val and math.copysign(1, val) == math.copysign(1, v["score"]):
                    agree += 1
            v["evidence"] = evidence
            v["confidence"] = round(min(1.0, 0.15 * evidence + 0.05 * agree), 3)
            v["score_breakdown"] = detail

        _RAG_INDEX = dict(sorted(
            profiles.items(),
            key=lambda kv: (kv[1]["score"], kv[1].get("confidence", 0.0)),
            reverse=True
        ))
        _RAG_TS = time.time()

        try:
            append_snapshot(PATHS["snapshots"], _RAG_TS, _RAG_INDEX)
        except Exception:
            pass

        try:
            _ensure_dir(VIS_DIR)
            run_dir = os.path.join(VIS_DIR, _ts_str(_RAG_TS))
            _ensure_dir(run_dir)
            _save_run_artifacts(run_dir, _RAG_INDEX)
            _save_visualizations(run_dir, _RAG_INDEX, top_n=10)
        except Exception:
            pass

        try:
            _ensure_dir(EVAL_DIR)
            _save_evaluation(EVAL_DIR, _RAG_INDEX)
        except Exception:
            pass

        return {"coins_indexed": len(_RAG_INDEX), "updated_at": _RAG_TS}

# ----------------------------
# Query helpers
# ----------------------------
def rag_top(top_k: int = 10) -> List[Dict[str, Any]]:
    items = list(_RAG_INDEX.items())[:max(1, top_k)]
    out = []
    for coin, v in items:
        out.append({"coin": coin, "score": v["score"], **v})
    return out

def rag_explain(coin: str) -> Dict[str, Any]:
    c_input = (coin or "")
    c = c_input.upper()

    v = _RAG_INDEX.get(c)
    if not v:
        # try canonical mapping ($BTC -> BTC, 'Bitcoin' -> BTC)
        c_can = canon_coin(c_input)
        if c_can and c_can in _RAG_INDEX:
            c, v = c_can, _RAG_INDEX[c_can]

    if not v:
        # fuzzy fallback: substring match
        for k in _RAG_INDEX.keys():
            if c in k or k in c:
                c = k
                v = _RAG_INDEX[k]
                break

    if not v:
        return {"coin": coin, "found": False}

    sb = v.get("score_breakdown", {})
    ns = sb.get("news_sent", 0.0)
    gs = sb.get("general_sent", 0.0)
    fs = sb.get("focus_sent", 0.0)
    fl = sb.get("flow_z", 0.0)
    mn = sb.get("mentions_z", 0.0)
    tw = sb.get("twitter_sent", 0.0)

    why = []
    if ns: why.append(f"news_sent={ns:+.2f}")
    if gs: why.append(f"general_sent={gs:+.2f}")
    if fs: why.append(f"focus_sent={fs:+.2f}")
    if fl: why.append(f"flow_z={fl:+.2f}")
    if mn: why.append(f"mentions_z={mn:+.2f}")
    if tw: why.append(f"twitter_sent={tw:+.2f}")

    pos = int(v.get("twitter_pos", 0))
    neg = int(v.get("twitter_neg", 0))
    tot = int(v.get("twitter_total", pos + neg))
    pos_pct = float(v.get("twitter_pos_pct", 0.0))
    neg_pct = float(v.get("twitter_neg_pct", 0.0))
    mode = sb.get("_mode", "static-weights")

    raw_score = float(v.get("score", 0.0))
    scaled_score = round(raw_score * 100.0, 4)

    twf= round(tw * 100.0, 4)

    return {
        "coin": c,
        "found": True,
        "score": scaled_score ,
        "why": ", ".join(why) or "no strong signals",
        "mode": mode,
        "evidence": v.get("evidence", 0),
        "confidence": v.get("confidence", 0.0),
        "sources": sorted(set(v.get("sources", []))),
        # <- EXACT available-coin tallies + the derived RAG feature
        "twitter_details": {
            "pos": pos,
            "neg": neg,
            "total": tot,
            "pos_pct": pos_pct,
            "neg_pct": neg_pct,
            "shrunk_sentiment": twf
        },
        "raw": v
    }
