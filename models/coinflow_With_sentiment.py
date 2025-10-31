import re
import json
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

from extrctor.tweets_extractor import fetch_discord_messages
from services.tweet_converter import run_preprocessing_general, run_preprocessing_focus

base_dir = os.path.dirname(__file__)

def analyze_coin_flow_and_sentiment():
    # === Step 1: Ensure VADER is available ===
    nltk.download('vader_lexicon')

    # === Step 2: Define file paths and fetch messages ===
    channel_type = "focus_based"
    raw_json_file = os.path.join(base_dir, "..", "test_data", "raw_focus_messages.json")
    output_json_file = os.path.join(base_dir, "..", "test_data", "sentiment_output_for_coin_finder.json")

    fetch_discord_messages(channel_type, raw_json_file)
    preprocessed_path = run_preprocessing_focus(input_path=raw_json_file)

    # === Step 3: Load preprocessed tweets ===
    with open(preprocessed_path, 'r', encoding='utf-8') as f:
        tweets = json.load(f)

    # === Step 4: Initialize tools ===
    flow_pattern = r'\$(\w+)\s*([+-])\$(\d+(?:\.\d+)?[KM]?)'
    coin_pattern = r'\$(\w+)'
    analyzer = SentimentIntensityAnalyzer()

    coin_data = {}
    sentiment_data = {}

    # === Step 5: Process each tweet ===
    for tweet in tweets:
        # --- Coin Flow Extraction ---
        for coin, sign, value_str in re.findall(flow_pattern, tweet):
            if value_str.endswith('K'):
                multiplier, numeric = 1_000, value_str[:-1]
            elif value_str.endswith('M'):
                multiplier, numeric = 1_000_000, value_str[:-1]
            else:
                multiplier, numeric = 1, value_str

            try:
                val = float(numeric) * multiplier
            except ValueError:
                continue

            net = val if sign == '+' else -val
            coin_data.setdefault(coin, []).append(net)

        # --- Sentiment Analysis ---
        coins = re.findall(coin_pattern, tweet)
        if coins:
            score = analyzer.polarity_scores(tweet)['compound']
            for coin in coins:
                sentiment_data.setdefault(coin, []).append(score)

    # === Step 6: Aggregate results ===
    aggregated_flows = {coin: sum(vals) for coin, vals in coin_data.items()}
    averaged_sentiment = {
        coin: sum(scores) / len(scores)
        for coin, scores in sentiment_data.items()
    }

    # === Step 7: Save results ===
    output = {
        "detailed_flows": coin_data,
        "aggregated_flows": aggregated_flows,
        "average_sentiment": averaged_sentiment
    }

    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)

    print(f"Flow and sentiment data saved to {output_json_file}")
