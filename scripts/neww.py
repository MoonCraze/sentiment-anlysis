import json
import re
import html
import os

# === Helper Function ===
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# === Paths ===
base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "..", "test_data", "preprocessed_data_run_coinfinder_focus.json")

# === Load data ===
with open(input_file_path, "r", encoding="utf-8") as f:
    raw_lines = json.load(f)

output_lines = []

# === Process each tweet ===
for line in raw_lines:
    # FIX: Extract actual tweet string from dict
    if isinstance(line, dict):
        tweet = line.get("text") or line.get("content") or next(iter(line.values()))
    elif isinstance(line, str):
        tweet = line
    else:
        continue

    tweet = html.unescape(tweet)
    tweet = remove_emojis(tweet)  # ✅ Emoji removal

    # Extract author name and handle
    author_match = re.search(r'—\s+(.*?)\s+\(@([^)]*)\)', tweet)
    if not author_match:
        continue

    author_name = author_match.group(1).strip()
    handle = author_match.group(2).strip()

    # Extract tweet content
    content = re.sub(r'^\*\*.*?\*\*\s*', '', tweet)
    content = re.sub(r'\s*—\s+.*?\(@.*?\).*$', '', content).strip()

    # Extract readable date and ISO timestamp
    date_match = re.search(r'(\w+\s\d{1,2},\s\d{4})\s+(20\d{2}-\d{2}-\d{2}T[\d:+]+)', tweet)
    readable_date = date_match.group(1) if date_match else "Unknown"
    iso_timestamp = date_match.group(2) if date_match else "Unknown"

    # Final formatted line
    formatted = f"**{author_name} (@{handle}) / Twitter** {content} {content} — {author_name} (@{handle}) {readable_date} {iso_timestamp} @{handle}"
    output_lines.append(formatted)

# === Save output ===
output_file_path = os.path.join(base_dir, "..", "test_data", "Analysis_output_for_coin_flow_asqww.json")
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(output_lines, f, ensure_ascii=False, indent=2)

print(f"✅ Conversion complete. Output saved to '{output_file_path}'")
