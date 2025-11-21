# models/Available_coin_analysis.py
import os, io, base64, json, asyncio, inspect
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import aiohttp
from configurations import config

# ------------------------ utils ------------------------
def _ensure_vader():
    try:
        SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", quiet=True)

async def _maybe_await(x):
    return await x if inspect.isawaitable(x) else x

def _mk_bar_b64(pos: int, neg: int, total: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 0.9), dpi=100)
    ax.axis("off")
    fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
    pos_w = (pos / total) if total else 0.0
    neg_w = (neg / total) if total else 0.0
    pos_pct = round(100 * pos_w, 2)
    neg_pct = round(100 * neg_w, 2)
    ax.barh([0], [pos_w], left=0, height=0.5)
    ax.barh([0], [neg_w], left=pos_w, height=0.5)
    ax.text(0, 0.85, f"{pos_pct:.2f}%", va="center", ha="left", fontsize=14)
    ax.text(1, 0.85, f"{neg_pct:.2f}%", va="center", ha="right", fontsize=14)
    ax.text(0, 1.35, "Community Mentions", va="center", ha="left", fontsize=16)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.8)
    buf = io.BytesIO(); plt.tight_layout(pad=1.0)
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# --------------------  SocialData Tools client --------------------
SOCIALDATA_BASE = "https://api.socialdata.tools"

def _get_socialdata_key() -> str:
    # Prefer env or config; fallback to the provided key
    return (
        os.getenv("SOCIALDATA_API_KEY")
        or getattr(config, "SOCIALDATA_API_KEY", None)
        or "3627|tQe9teqve9V3bpR7lmqe48dPLGeqMHnSN5egMk7S0dbe417f"
    )

# Optional defaults (override in configurations/config.py if you want)
DEFAULT_SEARCH_TYPE = getattr(config, "SOCIALDATA_DEFAULT_TYPE", "Latest")  # "Latest" or "Top"
DEFAULT_LANG = getattr(config, "SOCIALDATA_LANG", None)  # e.g., "en" or None

async def _socialdata_fetch_tweet_by_id(session: aiohttp.ClientSession, tweet_id: str) -> Optional[dict]:
    """
    GET /twitter/tweets/{id} → single tweet details (contains full_text/text, counts, user, etc).
    """
    url = f"{SOCIALDATA_BASE}/twitter/tweets/{tweet_id}"
    headers = {"Authorization": f"Bearer {_get_socialdata_key()}", "Accept": "application/json"}
    timeout = aiohttp.ClientTimeout(total=30)
    async with session.get(url, headers=headers, timeout=timeout) as resp:
        if resp.status != 200:
            _ = await resp.text()
            return None
        try:
            return await resp.json()
        except Exception:
            return None

async def _socialdata_search_texts(
    session: aiohttp.ClientSession,
    query: str,
    cap: int,
    search_type: str = "Latest",
    lang: Optional[str] = None,
) -> List[str]:
    """
    GET /twitter/search?query=...&type=Latest|Top with pagination via next_cursor until `cap` is reached.
    If `lang` is set and not already present as an operator in the query (lang:xx), pass it as a param.
    """
    headers = {"Authorization": f"Bearer {_get_socialdata_key()}", "Accept": "application/json"}
    endpoint = f"{SOCIALDATA_BASE}/twitter/search"

    texts: List[str] = []
    seen: set = set()
    cursor: Optional[str] = None

    # Whether the query already constrains language (operator like "lang:en")
    q_has_lang = " lang:" in (" " + (query or "").lower())

    while len(texts) < cap:
        params = {"query": query, "type": search_type}
        if cursor:
            params["cursor"] = cursor
        if lang and not q_has_lang:
            params["lang"] = lang

        timeout = aiohttp.ClientTimeout(total=40)
        async with session.get(endpoint, headers=headers, params=params, timeout=timeout) as resp:
            body = await resp.text()
            if resp.status != 200:
                # stop on error; return what we have so far
                break
            try:
                data = json.loads(body)
            except Exception:
                break

            # Be defensive about possible payload shapes
            page = data.get("tweets") or data.get("data") or data.get("statuses") or []
            if not isinstance(page, list):
                break

            for t in page:
                s = (t.get("full_text") or t.get("text") or "").strip()
                if not s:
                    continue
                k = s.lower()
                if k not in seen:
                    seen.add(k)
                    texts.append(s)
                if len(texts) >= cap:
                    break

            cursor = data.get("next_cursor")
            if not cursor or not page:
                break

    return texts[:cap]

def _parse_ids_query(query: str) -> List[str]:
    """
    Accept patterns like:
      ids:1549281861687451648
      ids:1549281861687451648,1729591119699124560
      ids:1549281861687451648 1729591119699124560
    Returns a list of digit-only strings.
    """
    q = (query or "").strip()
    if not q.lower().startswith("ids:"):
        return []
    raw = q.split(":", 1)[1].strip()
    parts = [p.strip() for p in raw.replace(",", " ").split()]
    return [p for p in parts if p.isdigit()]

async def _socialdata_get_texts_from_ids(session: aiohttp.ClientSession, ids: List[str], cap: int) -> List[str]:
    texts: List[str] = []
    seen: set = set()
    for tid in ids:
        if len(texts) >= cap:
            break
        data = await _socialdata_fetch_tweet_by_id(session, tid)
        if not data:
            continue
        s = (data.get("full_text") or data.get("text") or "").strip()
        if not s:
            continue
        k = s.lower()
        if k not in seen:
            seen.add(k)
            texts.append(s)
    return texts[:cap]

# ---- fallback orchestrator for keyword search ----
async def _search_with_fallbacks(
    session: aiohttp.ClientSession,
    query: str,
    cap: int,
    preferred_type: str,
    preferred_lang: Optional[str],
) -> (List[str], Optional[str]):
    """
    Returns (texts, debug_note). Tries several strategies:
      1) preferred_type + preferred_lang
      2) opposite_type + preferred_lang
      3) preferred_type + no lang
      4) opposite_type + no lang
    """
    debug_note = None
    types = [preferred_type, "Top" if preferred_type == "Latest" else "Latest"]

    # first try with language (if any)
    for t in types:
        texts = await _socialdata_search_texts(session, query, cap, search_type=t, lang=preferred_lang)
        if texts:
            return texts, None

    # then try without language
    for t in types:
        texts = await _socialdata_search_texts(session, query, cap, search_type=t, lang=None)
        if texts:
            return texts, None

    debug_note = f"no results for query='{query}' with fallbacks (types tried: {types}, lang tried: {preferred_lang} and None)"
    return [], debug_note

# --------------------  Resolver (keeps original function name) --------------------
async def _twitterapi_advanced_search(
    session: aiohttp.ClientSession,
    query: str,
    max_results: int,
    since: Optional[str] = None,
    until: Optional[str] = None,
    lang: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[str]:
    """
    Re-purposed to SocialData:
      - If query starts with 'ids:', fetch those tweets by ID via SocialData Tools.
      - Else, treat 'query' as a keyword/X-operator search and call /twitter/search
        with robust fallbacks across Latest/Top and with/without lang.
    """
    # 1) Explicit IDs
    ids = _parse_ids_query(query)
    if ids:
        return await _socialdata_get_texts_from_ids(session, ids, max_results)

    # 2) Keyword / operator search with fallbacks
    preferred_type = DEFAULT_SEARCH_TYPE if DEFAULT_SEARCH_TYPE in ("Latest", "Top") else "Latest"
    preferred_lang = lang if lang else DEFAULT_LANG
    texts, note = await _search_with_fallbacks(
        session=session,
        query=query,
        cap=max_results,
        preferred_type=preferred_type,
        preferred_lang=preferred_lang,
    )
    if not texts and note:
        # Return a helpful message as a single "text" so upstream UI isn't blank.
        return [f"[debug] {note}"]
    return texts

# ------------------------ public entrypoint ------------------------
async def available_coin_search(query: str, max_results: int = 300) -> Dict[str, Any]:
    """
    Fetch tweets via SocialData Tools (by tweet IDs or keyword search), run VADER sentiment,
    and return summary + bar image (base64).
    """
    _ensure_vader()

    async with aiohttp.ClientSession() as session:
        try:
            texts = await _twitterapi_advanced_search(
                session=session,
                query=query,
                max_results=max_results,
                lang=None,   # or pass DEFAULT_LANG if you want to lock language globally
                user_id=None,
            )
        except Exception as e:
            return {
                "query": query, "total_mentions": 0, "positive": 0, "negative": 0,
                "positive_pct": 0.0, "negative_pct": 0.0, "bar_image_base64": "",
                "sample_texts": [f"error: {e}"]
            }
    clean_texts = [t for t in texts if not str(t).startswith("[debug]")]
    # --- sentiment ---
    sia = SentimentIntensityAnalyzer()
    pos = neg = 0

    positive_texts: List[str] = []
    negative_texts: List[str] = []

    for s in texts:
        c = sia.polarity_scores(s)["compound"]
        if c > 0.25:
            pos += 1
            positive_texts.append(s)
        elif c < -0.25:
            neg += 1
            negative_texts.append(s)

    total = pos + neg
    pos_pct = round(100 * pos / total, 2) if total else 0.0
    neg_pct = round(100 * neg / total, 2) if total else 0.0
    bar_b64 = _mk_bar_b64(pos, neg, total)

    max_samples_per_class = 5

    sample_items: List[Dict[str, Any]] = []

    # positive samples (green)
    for t in positive_texts[:max_samples_per_class]:
        sample_items.append({
            "text": t,                 # full_text already handled upstream
            "sentiment": "positive",
            "color": "green",          # frontend can map this to style / class
        })

    # negative samples (red)
    for t in negative_texts[:max_samples_per_class]:
        sample_items.append({
            "text": t,
            "sentiment": "negative",
            "color": "red",
        })

    # If absolutely nothing was found, still return a small debug note
    if not sample_items and not clean_texts:
        sample_items.append({
            "text": "No tweets found for this query.",
            "sentiment": "none",
            "color": "gray",
        })

    return {
        "query": query,
        "total_mentions": total,
        "positive": pos,
        "negative": neg,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "bar_image_base64": bar_b64,
        "sample_texts": sample_items,
    }
