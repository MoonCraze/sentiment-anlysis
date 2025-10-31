# evaluation/general.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
import os

from models.general_handler import analyze_general_tweet_sentiment

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DATA_DIR = os.path.join(APP_ROOT, "test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# ---------- Schemas ----------
class GeneralEvalRequest(BaseModel):
    gt_path: str = Field(
        default=r"C:\Users\visal\PycharmProjects\Sentimnet_analysis\test_data\sentiment_output_general.json",
        description=("Ground-truth file (.json/.csv). Accepts {text:label}, "
                     "[{text,label}], [{text,predicted_label}], or index-aligned list."),
        examples=[r"C:\Users\visal\PycharmProjects\Sentimnet_analysis\test_data\sentiment_output_general.json"],
    )
    use_preprocessed_json_path: Optional[str] = Field(
        default=r"C:\Users\visal\PycharmProjects\Sentimnet_analysis\test_data\preprocessed_data_general.json",
        description="Preprocessed JSON array of texts for the 'general' channel.",
        examples=[r"C:\Users\visal\PycharmProjects\Sentimnet_analysis\test_data\preprocessed_data_general.json"],
    )
    channel_type: Optional[str] = Field(
        default="general",
        examples=["general"],
    )

class GeneralEvalResponse(BaseModel):
    message: str
    predictions_path: str
    metrics_path: str
    metrics: Dict[str, Any]

# ---------- Router ----------
general_router = APIRouter(prefix="/evaluate", tags=["Sentiment Analysis"])

@general_router.post(
    "/general",
    response_model=GeneralEvalResponse,
    summary="Run DeBERTa/Twitter-RoBERTa predictions + metrics on general channel",
)
def evaluate_general(req: GeneralEvalRequest):
    """
    Click Execute in Swagger to:
      1) Read texts (from req.use_preprocessed_json_path; else fetch+preprocess)
      2) Run DeBERTa/Twitter-RoBERTa sentiment → save predictions JSON
      3) Compute metrics using req.gt_path → save metrics JSON
      4) Return metrics JSON + file locations (also saves HTML/CSV/PNG visuals)
    """
    try:
        metrics = analyze_general_tweet_sentiment(
            ground_truth_path=req.gt_path,
            channel_type=req.channel_type,
            use_preprocessed_json_path=req.use_preprocessed_json_path,
            output_dir=TEST_DATA_DIR,
        )
        if metrics is None:
            raise HTTPException(status_code=400, detail="Ground-truth unusable; metrics not computed.")
        return GeneralEvalResponse(
            message="✅ General predictions and metrics completed.",
            predictions_path=os.path.join(TEST_DATA_DIR, "sentiment_output_general.json"),
            metrics_path=os.path.join(TEST_DATA_DIR, "sentiment_metrics_general.json"),
            metrics=metrics,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluate_general error: {type(e).__name__}: {e}")
