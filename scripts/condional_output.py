import re
import json
import os

from services.chart_plotiing import save_coin_chart
base_dir = os.path.dirname(__file__)
# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the JSON file.
file_path = os.path.join(script_dir, '..', 'data', 'preprocessed_data1.json')

with open(file_path, 'r') as f:
    tweets = json.load(f)

print("Data loaded successfully!")
# Define a regex pattern to extract coin symbols and net flow values.
pattern = r'\$(\w+)\s*([+-])\$(\d+(?:\.\d+)?[KM]?)'

# Create a dictionary to store detailed net flow values for each coin.
coin_data = {}

for tweet in tweets:
    # Find all matches for coin symbols and their associated net flow values.
    matches = re.findall(pattern, tweet)
    for coin, sign, value_str in matches:
        # Check for suffixes indicating thousands (K) or millions (M).
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
            continue  # Skip if conversion fails.

        # Apply the sign to the numeric value.
        numeric_value = numeric_value if sign == '+' else -numeric_value

        # Add the value to the list for the corresponding coin.
        if coin not in coin_data:
            coin_data[coin] = []
        coin_data[coin].append(numeric_value)

# Aggregate net flow values by summing them for each coin.
aggregated = {coin: sum(values) for coin, values in coin_data.items()}

# Combine the detailed flows and aggregated results into one dictionary.
output_data = {
    "detailed_flows": coin_data,
    "aggregated_flows": aggregated
}

# Save the processed data to a JSON file.
with open("../raw_focus_messages1/coin_flow_data.json", "w") as outfile:
    json.dump(output_data, outfile, indent=4)

print("Data has been processed and saved to coin_flow_data.json")

chart_dir = os.path.join(base_dir, "..", "test_data", "charts")
for coin, flows in coin_data.items():
    save_coin_chart(coin, flows, chart_dir)

print(f"Charts saved to {chart_dir}")