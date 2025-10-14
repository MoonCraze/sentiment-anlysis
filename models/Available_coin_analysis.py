# models/Availble_coin_analysis.py
import os, io, base64, json, inspect, asyncio
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from twikit import Client
from configurations import config

base_dir = os.path.dirname(__file__)
cookies_path = os.path.abspath(os.path.join(base_dir, "..", "configurations", "cookies.json"))

def _ensure_vader():
    try:
        SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", quiet=True)

async def _maybe_await(x):
    if inspect.isawaitable(x):
        return await x
    return x

def _get_creds():
    email = os.getenv("TWIKIT_EMAIL", getattr(config, "TWITTER_EMAIL", None))
    username = os.getenv("TWIKIT_USERNAME", getattr(config, "TWITTER_USERNAME", None))
    password = os.getenv("TWIKIT_PASSWORD", getattr(config, "TWITTER_PASSWORD", None))
    if username and username.startswith("@"):
        username = username[1:]
    return email, username, password

async def _twikit_login_and_save(client: Client) -> None:
    email, username, password = _get_creds()
    if not all([email, username, password]):
        raise RuntimeError("Twikit auth missing: set TWIKIT_EMAIL, TWIKIT_USERNAME, TWIKIT_PASSWORD or config.*")
    os.makedirs(os.path.dirname(cookies_path), exist_ok=True)
    await _maybe_await(client.login(auth_info_1=email, auth_info_2=username, password=password))
    await _maybe_await(client.save_cookies(cookies_path))

async def _validate_login(client: Client) -> bool:
    try:
        me = await _maybe_await(client.get_me())
        return bool(me and getattr(me, "id", None))
    except Exception:
        return False

async def _load_or_login(client: Client, force_fresh: bool = False) -> None:
    if force_fresh:
        try:
            if os.path.exists(cookies_path): os.remove(cookies_path)
        except Exception:
            pass

    # Try load cookies if present & valid JSON
    if os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0:
        try:
            with open(cookies_path, "r", encoding="utf-8") as f:
                json.load(f)
            await _maybe_await(client.load_cookies(cookies_path))
            if await _validate_login(client):
                return
        except Exception:
            pass  # fall through to fresh login

    # Fresh login
    await _twikit_login_and_save(client)
    if not await _validate_login(client):
        raise RuntimeError("Twikit login failed: credentials may be wrong or 2FA challenge unresolved.")

async def available_coin_search(query: str, max_results: int = 300) -> Dict[str, Any]:
    _ensure_vader()

    client = Client("en-US")
    await _load_or_login(client)

    async def _fetch(q: str):
        return await _maybe_await(client.search_tweet(query=q, product="Latest"))

    # Attempt search; if unauthorized, re-login once and retry
    try:
        res = await _fetch(query)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "unauthorized" in msg or "code: 32" in msg:
            await _load_or_login(client, force_fresh=True)
            res = await _fetch(query)
        else:
            # hard failure
            return {
                "query": query, "total_mentions": 0, "positive": 0, "negative": 0,
                "positive_pct": 0.0, "negative_pct": 0.0, "bar_image_base64": "",
                "sample_texts": [f"error: {e}"]
            }

    texts, seen = [], set()

    async def handle_async_iter(it):
        async for t in it:
            s = (getattr(t, "text", "") or "").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(s)
            if len(texts) >= max_results:
                break

    def handle_sync_iter(it):
        for t in (it or []):
            s = (getattr(t, "text", "") or "").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(s)
            if len(texts) >= max_results:
                break

    # Twikit can return sync/async iterables
    if hasattr(res, "__aiter__"):
        await handle_async_iter(res)
    else:
        handle_sync_iter(res)

    # If nothing came back, avoid showing a scary auth blob if the library puts it in 'text'
    texts = [t for t in texts if "Could not authenticate you" not in t]

    # --- sentiment ---
    sia = SentimentIntensityAnalyzer()
    pos = neg = 0
    for s in texts:
        c = sia.polarity_scores(s)["compound"]
        if c > 0.25: pos += 1
        elif c < -0.25: neg += 1

    total = len(texts)
    pos_pct = round(100 * pos / total, 2) if total else 0.0
    neg_pct = round(100 * neg / total, 2) if total else 0.0

    # --- chart ---
    fig, ax = plt.subplots(figsize=(8, 0.9), dpi=100)
    ax.axis("off")
    fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
    pos_w = (pos / total) if total else 0
    neg_w = (neg / total) if total else 0
    ax.barh([0], [pos_w], left=0, height=0.5)
    ax.barh([0], [neg_w], left=pos_w, height=0.5)
    ax.text(0, 0.85, f"{pos_pct:.2f}%", va="center", ha="left", fontsize=14)
    ax.text(1, 0.85, f"{neg_pct:.2f}%", va="center", ha="right", fontsize=14)
    ax.text(0, 1.35, "Community Mentions", va="center", ha="left", fontsize=16)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.8)
    buf = io.BytesIO(); plt.tight_layout(pad=1.0)
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    bar_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "query": query,
        "total_mentions": total,
        "positive": pos,
        "negative": neg,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "bar_image_base64": bar_b64,
        "sample_texts": texts[:10],
    }
