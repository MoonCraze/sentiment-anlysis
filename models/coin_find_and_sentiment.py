import re
import json
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict
from transformers import pipeline

from extrctor.tweets_extractor import fetch_discord_messages
from models.coin_finder import extract_coin_keywords_from_ner
from services.tweet_converter import run_preprocessing_focus

base_dir = os.path.dirname(__file__)
nltk.download('vader_lexicon')

# === FinBERT Setup ===
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
known_coin_names = {
    "ethereum", "bitcoin", "xrp", "solana", "ondo", "cronos",
    "binance", "picoin", "cardano", "cryptonews", "crypto",
    "litecoin", "whale", "finance", "transfer"
}

def analyze_verified_coin_sentiment_flow():
    # === Step 1: Run coin extraction and get the path ===
    extract_coin_keywords_from_ner()

    coin_list_file = os.path.join(base_dir, "..", "test_data", "coin_keywords_extracted.json")

    with open(coin_list_file, 'r', encoding='utf-8') as f:
        ner_data = json.load(f)
        raw_data = ner_data.get("coin_keywords_filtered", {})

    # === Extract all coin-like names (excluding numbers and known coins) ===
    potential_names = set()
    for section in raw_data.values():
        for name in section:
            name_clean = name.strip().lower()
            if name_clean not in known_coin_names and not name_clean.isdigit():
                potential_names.add(name.strip())

    # === Step 2: Fetch and preprocess focused messages ===
    channel_type = "focus_based"
    raw_json_file = os.path.join(base_dir, "..", "test_data", "raw_focus_messages.json")
    output_json_file = os.path.join(base_dir, "..", "test_data", "verified_sentiment_output_focus_group.json")

    fetch_discord_messages(channel_type, raw_json_file)
    preprocessed_path = run_preprocessing_focus(input_path=raw_json_file)

    with open(preprocessed_path, 'r', encoding='utf-8') as f:
        tweets = json.load(f)

    # === Step 3: Initialize tools ===
    flow_pattern = r'\$(\w+)\s*([+-])\$(\d+(?:\.\d+)?[KM]?)'
    analyzer = SentimentIntensityAnalyzer()

    coin_flows = defaultdict(list)
    coin_sentiments = defaultdict(list)
    coin_sentiment_scores = {}

    # === Step 4: Analyze tweets ===
    for tweet in tweets:
        tweet_lower = tweet.lower()

        # --- Check if tweet contains any known or potential coin name ---
        matched_coins = [name for name in potential_names.union(known_coin_names) if name.lower() in tweet_lower]
        if not matched_coins:
            continue

        # --- Flow Extraction ---
        for coin, sign, value_str in re.findall(flow_pattern, tweet):
            if coin.lower() not in known_coin_names:
                continue

            if value_str.endswith('K'):
                multiplier, numeric = 1_000, value_str[:-1]
            elif value_str.endswith('M'):
                multiplier, numeric = 1_000_000, value_str[:-1]
            else:
                multiplier, numeric = 1, value_str

            try:
                val = float(numeric) * multiplier
                net = val if sign == '+' else -val
                coin_flows[coin].append(net)
            except ValueError:
                continue

        # --- Sentiment Analysis ---
        sentiment_score = analyzer.polarity_scores(tweet)['compound']
        for coin in matched_coins:
            coin_sentiments[coin].append(sentiment_score)

    # === Step 5: Aggregate Results ===
    aggregated_flows = {coin: sum(vals) for coin, vals in coin_flows.items()}
    averaged_sentiments = {coin: sum(scores) / len(scores) for coin, scores in coin_sentiments.items() if scores}

    # === Step 6: Evaluate potential coin names using FinBERT ===
    finbert_potentials = {}
    for name in potential_names:
        try:
            result = finbert(name)[0]
            if result['label'].lower() == 'positive' and result['score'] > 0.85:
                finbert_potentials[name] = result['score']
        except Exception as e:
            continue

    # === Step 7: Save Output ===
    output = {
        "detailed_flows": dict(coin_flows),
        "aggregated_flows": aggregated_flows,
        "average_sentiment": averaged_sentiments,
        "potential_positive_coin_names": finbert_potentials
    }

    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)

    print(f"Verified coin flow and sentiment saved to {output_json_file}")
    print("\nTop potential coins from FinBERT:")
    for coin, score in sorted(finbert_potentials.items(), key=lambda x: x[1], reverse=True):
        print(f"{coin}: {score:.4f}")
