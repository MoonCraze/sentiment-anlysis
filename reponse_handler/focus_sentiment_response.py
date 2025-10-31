import os
import json

base_dir = os.path.dirname(__file__)
focus_sentiment_file = os.path.join(base_dir, "..", "test_data", "sentiment_output_for_coin_finder.json")

def get_focus_sentiment_summary():
    try:
        with open(focus_sentiment_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        avg_sentiment_dict = data.get("average_sentiment", {})
        if not avg_sentiment_dict:
            return {"status": "NO DATA", "average_sentiment": {}, "message": "No sentiment data found."}

        coin_scores = list(avg_sentiment_dict.values())
        if not coin_scores:
            return {"status": "NO DATA", "average_sentiment": {}, "message": "No sentiment scores available."}

        avg_score_all = round(sum(coin_scores) / len(coin_scores), 3)

        if avg_score_all > 0.25:
            status = "GOOD"
        elif avg_score_all < -0.25:
            status = "BAD"
        else:
            status = "NEUTRAL"

        return {
            "status": status,
            "average_sentiment_across_coins": avg_score_all,
            "coin_count": len(avg_sentiment_dict)
        }

    except Exception as e:
        return {"error": str(e)}
