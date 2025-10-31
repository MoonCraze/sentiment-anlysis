# models/Available_coin_analysis_scraper.py
import os, io, base64, json, re
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aiohttp
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from configurations import config

# ===================== CONFIG =====================
# Base URL of your FastAPI wrapper around twitter-cli-scraper.js
LOCAL_API_BASE = getattr(config, "LOCAL_TWITTER_API_BASE", "http://127.0.0.3:8000")

# Sentiment thresholds
POS_THRESH = float(getattr(config, "VADER_POS_THRESH", 0.25))
NEG_THRESH = float(getattr(config, "VADER_NEG_THRESH", -0.25))

# ===================== VADER & PLOTTING =====================
def _ensure_vader() -> bool:
    try:
        SentimentIntensityAnalyzer()
        return True
    except Exception:
        try:
            nltk.download("vader_lexicon", quiet=True)
            SentimentIntensityAnalyzer()
            return True
        except Exception:
            return False

def _mk_bar_b64(pos: int, neg: int, total: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 0.9), dpi=100)
    ax.axis("off")
    fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
    if total <= 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        buf = io.BytesIO(); plt.tight_layout(pad=1.0)
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    pos_w = pos / total
    neg_w = neg / total
    ax.barh([0], [pos_w], left=0, height=0.5)
    ax.barh([0], [neg_w], left=pos_w, height=0.5)
    ax.text(0, 0.85, f"{pos_w*100:.2f}%", va="center", ha="left", fontsize=14)
    ax.text(1, 0.85, f"{neg_w*100:.2f}%", va="center", ha="right", fontsize=14)
    ax.text(0, 1.35, "Community Mentions", va="center", ha="left", fontsize=16)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.8)
    buf = io.BytesIO(); plt.tight_layout(pad=1.0)
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# ===================== HELPERS =====================
def _normalize_text_for_dedup(s: str) -> str:
    s = s or ""
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# ===================== LOCAL FASTAPI BACKEND =====================
async def _fetch_texts_via_local_fastapi(session: aiohttp.ClientSession, query: str, cap: int) -> Tuple[List[str], Dict[str, Any]]:
    """
    Calls your FastAPI endpoint:
      POST {LOCAL_API_BASE}/scrape  with {query,maxTweets}
    Returns (texts, diagnostics).
    """
    url = f"{LOCAL_API_BASE.rstrip('/')}/scrape"
    payload = {"query": query, "maxTweets": cap}
    diag: Dict[str, Any] = {"source": "local_fastapi", "endpoint": url, "saved": True, "notes": []}

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            text = await resp.text()
            if resp.status != 200:
                # bubble an error line as faux “text”; we’ll separate it later
                return [f"[error] local_fastapi http {resp.status}: {text[:200]}"], {**diag, "error": f"http {resp.status}"}
            data = json.loads(text)
    except Exception as e:
        return [f"[error] local_fastapi network: {e}"], {**diag, "error": f"network: {e}"}

    # Expected shape from your app:
    # {
    #   ok: True, file: "twitter_...json", file_url: "/files/...",
    #   summary: {...},
    #   result: { query, timestamp, totalTweets, tweets: [...] }
    # }
    result = data.get("result") or {}
    tweets = result.get("tweets") or []

    # Extract texts with dedup
    out: List[str] = []
    seen = set()
    for t in tweets:
        s = t.get("text", "") or ""
        key = _normalize_text_for_dedup(s)
        if s and key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) >= cap:
            break

    diag.update({
        "file": data.get("file"),
        "file_url": data.get("file_url"),
        "total_returned": len(out),
        "raw_count": len(tweets),
    })
    return out, diag

# ===================== PUBLIC ENTRYPOINT =====================
async def available_coin_search(query: str, max_results: int = 300) -> Dict[str, Any]:
    """
    Fetch tweets via your local FastAPI (/scrape), run VADER sentiment,
    and return summary + base64 bar image.
    """
    vader_ok = _ensure_vader()
    errors: List[str] = []
    diagnostics: Dict[str, Any] = {}

    async with aiohttp.ClientSession() as session:
        texts, diagnostics = await _fetch_texts_via_local_fastapi(session, query, max_results)

    # Separate backend error lines from real tweets
    real_texts = [t for t in texts if not t.startswith("[error] ")]
    for t in texts:
        if t.startswith("[error] "):
            errors.append(t)

    total = len(real_texts)
    pos = neg = neu = 0
    if vader_ok:
        sia = SentimentIntensityAnalyzer()
        for s in real_texts:
            c = sia.polarity_scores(s)["compound"]
            if c > POS_THRESH: pos += 1
            elif c < NEG_THRESH: neg += 1
            else: neu += 1
    else:
        neu = total
        errors.append("VADER lexicon unavailable; marked all neutral.")
    total = pos + neg
    bar_b64 = _mk_bar_b64(pos, neg, total)
    pos_pct = round(100 * pos / total, 2) if total else 0.0
    neg_pct = round(100 * neg / total, 2) if total else 0.0

    return {
        "query": query,
        "total_mentions": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "bar_image_base64": bar_b64,
        "sample_texts": real_texts[:10],
        "errors": errors,
        "diagnostics": diagnostics,
        "empty": total == 0,
    }
