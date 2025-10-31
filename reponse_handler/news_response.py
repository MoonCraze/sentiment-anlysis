import os
import json

base_dir = os.path.dirname(__file__)
news_sentiment_file = os.path.join(base_dir, "..", "test_data", "sentiment_output_for_news.json")

def get_news_sentiment_summary():
    try:
        with open(news_sentiment_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract last 8 messages
        last_msgs = data[:20] if len(data) >= 8 else data

        # Map sentiments to numerical scores
        sentiment_scores = {
            "POSITIVE": 1,
            "NEUTRAL": 0,
            "NEGATIVE": -1
        }

        scores = []
        for item in last_msgs:
            label = item.get("dominant_sentiment", "NEUTRAL").upper()
            scores.append(sentiment_scores.get(label, 0))

        avg_score = round(sum(scores) / len(scores), 3) if scores else 0

        if avg_score > 0.25:
            status = "GOOD"
        elif avg_score < -0.25:
            status = "BAD"
        else:
            status = "NEUTRAL"

        return {
            "status": status,
            "average_score": avg_score,
            "analyzed_messages": len(last_msgs)
        }

    except Exception as e:
        return {"error": str(e)}
