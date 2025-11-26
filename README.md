# Sentiment Analysis with Bi-LSTM + Attention

> A simple Bi-LSTM + Attention model for binary and multi-class sentiment analysis, with example data loading and training scripts.

---

## 📋 Description

This repository provides:

- **Data loaders** for common sentiment datasets (IMDB, SST-2, Twitter US Airline Sentiment).
- **Model code** implementing a Bi-LSTM encoder + attention layer.
- **Training scripts** with configurable hyperparameters.
- **Evaluation notebooks** for accuracy, F1, and confusion matrices.

---

# 🧠 AI-Assisted Twitter Scraper & Sentiment Analysis

This project combines:

1. **Twitter Scraper** – to collect tweets for specific topics/coins  
2. **Sentiment Analysis Engine** – to analyze scraped tweets using  
   **VADER**, **FinBERT**, and **DeBERTa-NER**

---

## 📂 Project Structure

```
fyp_sent_test/
│
├── scrapper/
│   └── twitter-scrapper-main/
│       └── app.py                # Twitter scraper entrypoint
│
└── Sentimnet_analysis/
    └── main.py                   # Sentiment analysis pipeline entrypoint
```

---

## 🔧 Prerequisites

- Python 3.10+
- pip installed
- Google Chrome/Chromium installed (for browser-based scraping, if required)
- (Optional but recommended) Python virtual environment

---

## 🚀 1. Run the Twitter Scraper

**Step 1 — Navigate to the scraper folder**

From the project root:

```bash
cd scrapper/twitter-scrapper-main
```

**Step 2 — Install dependencies**

Install Puppeteer (Python package used in this project):

```bash
pip install pupperteer
```

If you have a `requirements.txt` file, you can instead run:

```bash
pip install -r requirements.txt
```

**Step 3 — Run the scraper**

```bash
python app.py
```

This will:

- Log in to Twitter/X if needed
- Scrape tweets based on your configured query/logic
- Save the results into an output folder or a JSON file to be used by the sentiment analysis module

---

## 🤖 2. Run Sentiment Analysis

After scraping, go back to the main project folder and then into the sentiment analysis module.

**Step 4 — Navigate to the sentiment analysis module**

From the scraper folder:

```bash
cd ../../Sentimnet_analysis
```

Or, if you are already in the project root (`fyp_sent_test`):

```bash
cd Sentimnet_analysis
```

**Step 5 — Run the pipeline**

```bash
python main.py
```

This script will:

- Load the previously scraped tweet data
- Process NER (Named Entity Recognition) to detect coins/tokens
- Run VADER and FinBERT sentiment analysis
- Combine results into a final sentiment/scoring output

---

## 📦 Output

Typical outputs include:

- Processed sentiment JSON files
- Coin-wise sentiment scores
- Log files for debugging and tracing
- Trend/signal outputs for further analysis or trading logic

*(The exact output paths and formats depend on your internal config in `main.py`.)*

---

## 📝 Notes

- Make sure the Twitter Scraper saves its JSON/output in the same path that `main.py` expects.
- If you change folder names or paths, update the configuration in:
  - `twitter-scrapper-main/app.py`
  - `Sentimnet_analysis/main.py`
- You can extend this project by:
  - Adding new sentiment models
  - Connecting to your trading engine
  - Logging results into a database/dashboard

---

## 👨‍💻 Author

Developed by Visal Sandeep Adikari

---

## 🔬 Model Architecture Overview

```
Input Text
   ↓
Tokenizer
   ↓
Embedding Layer (GloVe, FastText, or trainable)
   ↓
Bi-LSTM Encoder
   ↓
Attention Layer
   ↓
Dropout → Dense → Softmax
   ↓
Predicted Label
```

- **Embedding**: 100–300 dimensions (pre-trained or trainable)
- **Bi-LSTM**: Hidden size configurable (e.g., 128 units each direction)
- **Attention**: Weighted sum of hidden states
- **Output**: Softmax over 2–5 classes

---
