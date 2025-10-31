import re
import json
import os

from extrctor.tweets_extractor import fetch_discord_messages
from services.chart_plotiing import save_coin_chart
from services.tweet_converter import run_coinflow_focus

base_dir = os.path.dirname(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))

def analyze_coin_flow_analysis():
    # === Step 1: Define file paths and fetch messages ===
    #channel_type = "focus_based"
    #raw_json_file = os.path.join(base_dir, "..", "test_data", "preprocessed_data1.json")
    output_json_file = os.path.join(base_dir, "..", "test_data", "Analysis_output_for_coin_flow_a.json")

    #fetch_discord_messages(channel_type, raw_json_file)
    #preprocessed_path = run_coinflow_focus(input_path=raw_json_file)
    preprocessed_path =os.path.join(base_dir, "..", "data", "preprocessed_data1.json")
    # === Step 2: Load preprocessed tweets ===
    with open(preprocessed_path, 'r', encoding='utf-8') as f:
        tweets = json.load(f)
    print("Data loaded successfully!")
    # === Step 3: Define regex pattern and data structure ===
    flow_pattern = r'\$(\w+)\s*([+-])\$(\d+(?:\.\d+)?[KM]?)'
    coin_data = {}

    # === Step 4: Extract and process coin flows ===
    for tweet in tweets:
        matches = re.findall(flow_pattern, tweet)
        for coin, sign, value_str in matches:
            multiplier = 1
            if value_str.endswith('K'):
                multiplier = 1_000
                numeric_part = value_str[:-1]
            elif value_str.endswith('M'):
                multiplier = 1_000_000
                numeric_part = value_str[:-1]
            else:
                numeric_part = value_str

            try:
                numeric_value = float(numeric_part) * multiplier
            except ValueError:
                continue

            numeric_value = numeric_value if sign == '+' else -numeric_value

            if coin not in coin_data:
                coin_data[coin] = []
            coin_data[coin].append(numeric_value)

    # === Step 5: Aggregate net flows ===
    aggregated_flows = {coin: sum(values) for coin, values in coin_data.items()}

    # === Step 6: Save results ===
    output_data = {
        "detailed_flows": coin_data,
        "aggregated_flows": aggregated_flows
    }

    with open(output_json_file, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"Coin flow analysis saved to {output_json_file}")

    # === Step 6: Generate charts per coin ===
    chart_dir = os.path.join(base_dir, "..", "test_data", "charts1")
    for coin, flows in coin_data.items():
        save_coin_chart(coin, flows, chart_dir)

    print(f"Charts saved to {chart_dir}")