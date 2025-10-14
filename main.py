#import json
import os

from fastapi import FastAPI,Query
import uvicorn

from evaluation.general import general_router
from models.Available_coin_analysis import available_coin_search
# --- Analysis Functions ---
from models.News_handler import analyze_discord_news_sentiment
from evaluation.news import router as eval_news_router
from models.coin_find_and_sentiment import analyze_verified_coin_sentiment_flow
from models.coin_finder import extract_coin_keywords_from_ner
from models.coinflow_Analysis import analyze_coin_flow_analysis

from models.coinflow_With_sentiment import analyze_coin_flow_and_sentiment
from models.general_handler import analyze_general_tweet_sentiment
from reponse_handler.focus_sentiment_response import get_focus_sentiment_summary

# --- Response Handlers (Summaries) ---
from reponse_handler.general_response import get_general_sentiment_summary
from reponse_handler.news_response import get_news_sentiment_summary
from weight_handler.rag_system import build_rag_index, rag_top, rag_explain

app = FastAPI(
    title="Crypto Sentiment Analysis API",
    description="Run various types of sentiment analyses and view summaries.",
    version="1.0.0"
)

APP_ROOT = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(APP_ROOT, "test data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)
# ============================
# Sentiment Analysis Endpoints
# ============================

# @app.get("/run-sentiment-analysis-news", tags=["Sentiment Analysis"])
# def run_sentiment_analysis():
#     analyze_discord_news_sentiment()
#     return {"message": "Sentiment analysis for news completed and results saved."}
#
# @app.get("/run-sentiment-general", tags=["Sentiment Analysis"])
# def run_sentiment_analysis_g():
#     analyze_general_tweet_sentiment()
#     return {"message": "Sentiment analysis for general tweets completed and results saved."}
#
# @app.get("/run-sentiment-focused", tags=["Sentiment Analysis"])
# def run_sentiment_analysis_focus():
#     analyze_coin_flow_and_sentiment()
#     return {"message": "Sentiment analysis for focused tweets completed and results saved."}

# =======================
# coin Endpoints
# =======================
@app.get("/run-coin-finder", tags=["Coin Analysis"])
def run_coin_finder():
    result = extract_coin_keywords_from_ner()  # already returns a dict
    return {
        "message": result.get("message", "Coin keyword extraction completed."),
        "top_new_coins": result.get("top_new_coins", []),
        "top_keywords_clean": result.get("top_keywords_clean", [])
    }

@app.get("/run-coin-finder-and-evaluate", tags=["Coin Analysis"])
def run_coin_finder_and_evaluate():
    result = analyze_verified_coin_sentiment_flow()
    return {
        "message": "Coin evaluation and sentiment flow analysis completed and results saved.",
        "top_positive_potential_coins": result
    }

@app.get("/run-coin-flow-and-evaluate", tags=["Coin Flow Analysis"])
def run_coin_finder_and_flow_evaluate():
    analyze_coin_flow_analysis()
    return {
        "message": "Coin flow analysis completed. Charts and results saved."
    }
@app.get("/coin-sentiment", tags=["Search Sentiment"])
async def coin_sentiment(
    coin: str = Query(..., description="Coin or keyword to search, e.g., 'Bitcoin' or '$SOL'"),
    max_results: int = Query(300, ge=20, le=800)
):
    try:
        return await available_coin_search(coin.strip(), max_results=max_results)
    except Exception as e:
        return {
            "query": coin, "total_mentions": 0, "positive": 0, "negative": 0,
            "positive_pct": 0.0, "negative_pct": 0.0,
            "bar_image_base64": "", "sample_texts": [f"error: {type(e).__name__}: {e}"]
        }

# =======================
# Summary Endpoints
# =======================

@app.get("/news-sentiment-summary", tags=["Summary"])
def get_news_summary():
    analyze_discord_news_sentiment()
    return get_news_sentiment_summary()
app.include_router(eval_news_router)   # this adds POST /evaluate/news to Swagger
app.include_router(general_router)
@app.get("/general-sentiment-summary", tags=["Summary"])
def get_general_summary():
    analyze_general_tweet_sentiment()
    return get_general_sentiment_summary()
@app.get("/focus-sentiment-summary", tags=["Summary"])
def get_focus_summary():
    analyze_coin_flow_and_sentiment()
    return get_focus_sentiment_summary()

# =======================
# RAG Endpoints
# =======================

@app.post("/rag/refresh", tags=["RAG"])
def rag_refresh():
    meta = build_rag_index()
    return {"message": "RAG index refreshed", **meta}

@app.get("/rag/top", tags=["RAG"])
def rag_get_top(k: int = Query(10, ge=1, le=50)):
    if not rag_top(1):  # empty -> build once
        build_rag_index()
    return {"top": rag_top(k)}

@app.get("/rag/explain", tags=["RAG"])
def rag_get_explain(coin: str = Query(..., description="Coin or ticker e.g. BTC, $BTC, Bitcoin")):
    if not rag_top(1):
        build_rag_index()
    return rag_explain(coin)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
