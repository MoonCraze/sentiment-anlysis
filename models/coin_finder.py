import json
import os
import re
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

from extrctor.tweets_extractor import fetch_discord_messages
from model_loader.berta_models import load_deberta_ner_model
from services.tweet_converter import run_preprocessing_news

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

base_dir = os.path.dirname(__file__)

# Define known coins (normalized to lowercase for easier matching)
known_coin_names = {
    "Ethereum", "Bitcoin", "XRP", "Solana", "Ondo", "Cronos", "Binance","Picoin", "Cardano",
    "Cryptonews", "Crypto", "Litecoin", "Whale","Finance", "Transfer",
    "Tether", "usdt","USDT", "usdc","USDC", "bnb","BNB", "Matic", "Avalanche",
    "Tron", "Polkadot", "Dai", "Chainlink", "Uniswap", "Aptos", "Arbitrum", "Near", "Render",
    "Kaspa", "Stellar", "Vechain", "Filecoin", "Aave", "Lido", "Maker", "Sui", "Algorand",
    "Theta", "Elrond", "Mina", "Iota", "Chiliz", "Ocean", "Optimism", "Injective", "Blur",
    "Flow", "Gala", "Sandbox", "Axie", "enj", "loopring", "decentraland", "1inch", "bittorrent",
    "zksync", "mantle", "flare","JONDONI_CRYPTO"
}


# Normalize known names once (case-insensitive match)
KNOWN_COIN_NAMES_LOWER = {n.lower() for n in known_coin_names}

def extract_coin_keywords_from_ner():
    # === Load tools ===
    ner_pipeline = load_deberta_ner_model()
    stop_words = set(stopwords.words('english'))

    # === File paths ===
    channel_type = "finder"
    raw_json_file = os.path.join(base_dir, "..", "test_data", "raw_news_messages.json")
    output_json_file = os.path.join(base_dir, "..", "test_data", "coin_keywords_extracted.json")

    # === Fetch and preprocess ===
    fetch_discord_messages(channel_type, raw_json_file)
    preprocessed_path = run_preprocessing_news(input_path=raw_json_file)

    with open(preprocessed_path, 'r', encoding='utf-8') as f:
        tweets = [line.strip() for line in f if line.strip()]

    # === Analyze per tweet ===
    coin_keyword_collector = defaultdict(list)  # coin -> [keywords...]
    coin_counts = Counter()                     # coin -> mention count

    GENERIC_BADWORDS = {
        "news","twitter","https","http","billion","million","globalgoals",
        "finance","radar","economist","union_build","us","u","the","new","rt"
    }

    def clean_kw(w: str) -> str:
        w = w.strip().replace("##", "")
        if w.startswith("@") or w.startswith("http"):
            return ""
        # keep simple tokens only
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,19}", w):
            return ""
        return w

    for tweet in tweets:
        # capture $TICKER-like tokens (alnum/underscore)
        coins_found = re.findall(r'\$(\w+)', tweet)
        if not coins_found:
            continue

        # count coins (normalized to UPPER for consistency)
        for c in coins_found:
            coin_counts[c.upper()] += 1

        # NER keywords
        ner_results = ner_pipeline(tweet)
        ner_keywords = [
            ent['word']
            for ent in ner_results
            if ent.get('entity_group') in ['ORG', 'PRODUCT', 'PER', 'MISC']
        ]

        # POS keywords (only noun-ish, not stopwords)
        words = word_tokenize(tweet)
        pos_tags = pos_tag(words)
        pos_keywords = [
            w for w, tag in pos_tags
            if tag in ('NN', 'NNS', 'NNP', 'NNPS')
            and w.lower() not in stop_words
        ]

        # clean + filter noise
        raw_keywords = ner_keywords + pos_keywords
        filtered_keywords = []
        for w in raw_keywords:
            w = clean_kw(w)
            if not w:
                continue
            if w.lower() in GENERIC_BADWORDS:
                continue
            filtered_keywords.append(w)

        # attach cleaned keywords to each coin in this tweet
        for coin in coins_found:
            coin_keyword_collector[coin.upper()].extend(filtered_keywords)

    # === Aggregate + Filter ===
    coin_keywords_filtered = {
        coin: {kw: cnt for kw, cnt in Counter(kws).items() if cnt > 5}
        for coin, kws in coin_keyword_collector.items()
    }

    # === Save per-coin keywords ===
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)

    # === Top coins (exclude known coins) ===
    # Option A: allow any shape (default)
    top_new_coins = [
        c for c, _ in coin_counts.most_common(50)
        if c.lower() not in KNOWN_COIN_NAMES_LOWER
    ]

    # Option B (stricter): only ticker-ish shapes (2–6 alnum, uppercase)
    # top_new_coins = [
    #     c for c, _ in coin_counts.most_common(50)
    #     if re.fullmatch(r"[A-Z0-9]{2,6}", c)
    #     and c.lower() not in KNOWN_COIN_NAMES_LOWER
    # ]

    # === Global top keywords (clean), excluding known coins & badwords ===
    keyword_counter = Counter()
    for kwdict in coin_keywords_filtered.values():
        keyword_counter.update(kwdict)

    top_keywords_clean = [
        kw for kw, _ in keyword_counter.most_common(50)
        if kw.lower() not in KNOWN_COIN_NAMES_LOWER
        and kw.lower() not in GENERIC_BADWORDS
    ]

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "coin_keywords_filtered": coin_keywords_filtered,
            "top_new_coins": top_new_coins,
            "top_keywords_clean": top_keywords_clean
        }, f, indent=4, ensure_ascii=False)

    print("Filtered coin keyword subjects saved to:", output_json_file)
    print("Potential new coins:", top_new_coins[:15])
    print("Clean keywords (context):", top_keywords_clean[:20])

    # return a structured object (easier for API)
    return {
        "message": "Coin keyword extraction completed.",
        "top_new_coins": top_new_coins,
        "top_keywords_clean": top_keywords_clean
    }
