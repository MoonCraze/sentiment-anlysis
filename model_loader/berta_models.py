# models/model_loader.py
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)

# =============================
# 🔹 NER MODELS
# =============================

def load_deberta_ner_model():
    """Generic English NER (Roberta-large fine-tuned for entities)"""
    model_name = "Jean-Baptiste/roberta-large-ner-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def load_finetuned_crypto_ner():
    """Crypto-specific NER model (detects coins, tickers, hashtags)"""
    model_name = "birgermoell/bert-base-cased-finetuned-financial-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# =============================
# 🔹 SENTIMENT MODELS
# =============================

def load_finbert_sentiment_model():
    """Financial sentiment (FinBERT: Positive, Negative, Neutral)"""
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


def load_deberta_sentiment_model():
    """General sentiment model (trained on tweets)"""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def load_multilingual_sentiment_model():
    """Handles multilingual crypto tweets / Telegram posts"""
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# =============================
# 🔹 ADVANCED FINANCIAL + SOCIAL
# =============================

def load_lm_financial_roberta():
    """Financial document sentiment (reports, articles)"""
    model_name = "mrm8488/bert-small-finetuned-financial-news-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


def load_crypto_tweet_sentiment():
    """Crypto-specific transformer trained on tweets"""
    model_name = "ahmedrachid/FinancialBERT-Sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

# =============================
# 🔹 OPTIONAL CUSTOM LOCAL MODEL
# =============================

def load_local_custom_model(local_path: str):
    """
    Load your fine-tuned local model (from Colab or local directory).
    Example: 'models/deberta_sentiment_ckpt/best'
    """
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_path)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

