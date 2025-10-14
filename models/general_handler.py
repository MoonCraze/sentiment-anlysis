# models/general_handler.py
import os
import json
import datetime
from typing import List, Dict, Tuple, Optional, Union

import pandas as pd
from extrctor.tweets_extractor import fetch_discord_messages
from model_loader.berta_models import load_deberta_sentiment_model
from services.tweet_converter import run_preprocessing_general

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
TEST_DATA_DIR = os.path.join(ROOT_DIR, "test data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Canonical label order (twitter-roberta-base-sentiment)
LABELS = ["negative", "neutral", "positive"]


# -------------------------------
# Visualization & summary outputs
# -------------------------------
def _save_visual_evaluation_general(metrics: dict, output_dir: str) -> Dict[str, str]:
    """
    Save visual evaluation files (CSV, HTML, confusion matrix chart).
    Files are created under: {output_dir}/evaluation_general/results/
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, "evaluation_general", "results")
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, f"metrics_{timestamp}.csv")
    html_path = os.path.join(result_dir, f"metrics_{timestamp}.html")
    cm_path = os.path.join(result_dir, f"confusion_matrix_{timestamp}.png")

    # --- CSV (top-level metrics) ---
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    # --- Confusion Matrix Visualization ---
    cm = metrics.get("confusion_matrix")
    if cm:
        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (General)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

    # --- HTML summary ---
    html = f"""
    <html><head><title>General Sentiment Evaluation Results</title></head>
    <body style="font-family: Arial, sans-serif; margin: 24px;">
      <h2>DeBERTa (twitter-roberta) – Evaluation Summary</h2>
      <p><b>Samples Evaluated:</b> {metrics.get('num_samples_evaluated', 0)}</p>
      <p><b>Accuracy:</b> {metrics.get('accuracy', 0):.4f}</p>
      <p><b>Macro F1:</b> {metrics.get('f1_macro', 0):.4f}</p>
      <p><b>Weighted F1:</b> {metrics.get('f1_weighted', 0):.4f}</p>
      <hr />
      <h3>Per-Class Report</h3>
      <pre style="background:#f6f8fa;padding:12px;border-radius:6px;">{json.dumps(metrics.get("per_class_report", {}), indent=2)}</pre>
      <hr />
      <h3>Confusion Matrix</h3>
      <img src="confusion_matrix_{timestamp}.png" width="520" />
    </body></html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return {"csv_path": csv_path, "html_path": html_path, "cm_path": cm_path}


# -----------------
# Label normalizer
# -----------------
def _normalize_label(x: Union[str, None]) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s in ("neg", "negative", "bear", "bearish"):   return "negative"
    if s in ("neu", "neutral", "mixed", "none"):      return "neutral"
    if s in ("pos", "positive", "bull", "bullish"):   return "positive"
    return s


# -------------------------
# Ground-truth loader (GT)
# -------------------------
def _load_ground_truth_labels(gt_path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Supports:
      • CSV: columns text,label
      • JSON dict: {text: label}
      • JSON list of dicts: [{text, label|predicted_label|dominant_sentiment}, ...]
      • JSON list of labels (index-aligned)
    Returns:
      (map_by_text, list_by_index)
    """
    ext = os.path.splitext(gt_path)[1].lower()
    map_by_text: Dict[str, str] = {}
    list_by_index: List[str] = []

    if ext == ".csv":
        df = pd.read_csv(gt_path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not {"text", "label"}.issubset(set(df.columns)):
            raise ValueError("CSV ground-truth must have columns: text,label")
        for _, r in df.iterrows():
            map_by_text[str(r["text"]).strip()] = _normalize_label(r["label"])

    elif ext == ".json":
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for k, v in data.items():
                map_by_text[str(k).strip()] = _normalize_label(v)

        elif isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Ground-truth list is empty.")
            first = data[0]
            if isinstance(first, dict):
                any_text = any(str(it.get("text", "")).strip() for it in data)
                if any_text:
                    for it in data:
                        txt = str(it.get("text", "")).strip()
                        if not txt:
                            continue
                        lab = (it.get("label")
                               or it.get("predicted_label")
                               or it.get("dominant_sentiment"))
                        map_by_text[txt] = _normalize_label(lab)
                else:
                    for it in data:
                        lab = (it.get("label")
                               or it.get("predicted_label")
                               or it.get("dominant_sentiment"))
                        list_by_index.append(_normalize_label(lab))
            else:
                list_by_index = [_normalize_label(x) for x in data]
        else:
            raise ValueError("Unsupported JSON ground-truth structure.")
    else:
        raise ValueError("Ground-truth file must be .csv or .json")

    return map_by_text, list_by_index


# -------------------------------------
# Predictions with per-class probabilities
# -------------------------------------
def _predict_with_probs(hf_pipeline, texts: List[str]) -> Tuple[List[str], List[List[float]]]:
    """
    Returns:
      y_pred: list[str] in LABELS space
      probs:  list[[neg, neu, pos]]
    Note: modern HF pipelines support per-class scores by using top_k=None.
    """
    preds: List[str] = []
    probs_all: List[List[float]] = []

    outputs = hf_pipeline(
        texts,
        truncation=True,
        max_length=512,
        batch_size=16,
        top_k=None  # replaces deprecated return_all_scores=True
    )
    for out in outputs:
        lab2score = {str(d["label"]).lower(): float(d["score"]) for d in out}
        probs = [lab2score.get(l, 0.0) for l in LABELS]
        probs_all.append(probs)
        preds.append(LABELS[max(range(len(probs)), key=lambda i: probs[i])])

    return preds, probs_all


# -----------------------
# Evaluation (classification)
# -----------------------
def _evaluate_classification(y_true, y_pred, probs) -> Dict[str, Union[float, dict, list, None]]:
    acc = accuracy_score(y_true, y_pred)
    prec_mi, rec_mi, f1_mi, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    prec_ma, rec_ma, f1_ma, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    y_true_bin = label_binarize(y_true, classes=LABELS)
    probs_df = pd.DataFrame(probs, columns=LABELS)

    auc_ovr, ap_per_class = {}, {}
    for lab in LABELS:
        try:
            auc_ovr[lab] = roc_auc_score(y_true_bin[:, LABELS.index(lab)], probs_df[lab])
        except Exception:
            auc_ovr[lab] = None
        try:
            ap_per_class[lab] = average_precision_score(y_true_bin[:, LABELS.index(lab)], probs_df[lab])
        except Exception:
            ap_per_class[lab] = None

    valid_aucs = [v for v in auc_ovr.values() if v is not None]
    valid_aps  = [v for v in ap_per_class.values() if v is not None]
    macro_auc = sum(valid_aucs)/len(valid_aucs) if valid_aucs else None
    macro_ap  = sum(valid_aps)/len(valid_aps) if valid_aps else None

    return {
        "accuracy": acc,
        "precision_micro": prec_mi, "recall_micro": rec_mi, "f1_micro": f1_mi,
        "precision_macro": prec_ma, "recall_macro": rec_ma, "f1_macro": f1_ma,
        "precision_weighted": prec_wt, "recall_weighted": rec_wt, "f1_weighted": f1_wt,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "roc_auc_ovr_per_class": auc_ovr,
        "macro_auc_ovr": macro_auc,
        "average_precision_per_class": ap_per_class,
        "macro_average_precision": macro_ap
    }


# -----------------------------
# Public APIs (no GT + with GT)
# -----------------------------
def analyze_general_tweet_sentiment(
    ground_truth_path: Optional[str] = None,
    channel_type: str = "general",
    use_preprocessed_json_path: Optional[str] = None,
    output_dir: Optional[str] = TEST_DATA_DIR
) -> Optional[dict]:
    """
    If ground_truth_path is provided, returns metrics dict.
    Always writes:
      - {output_dir}/sentiment_output_general.json  (per-text predictions)
      - {output_dir}/sentiment_metrics_general.json (when GT provided)
      - {output_dir}/evaluation_general/results/*   (CSV, HTML, Confusion Matrix PNG)
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_json_file = os.path.join(output_dir, "raw_general_messages.json")
    preds_json_file = os.path.join(output_dir, "sentiment_output_general.json")
    metrics_json_file = os.path.join(output_dir, "sentiment_metrics_general.json")

    # === Load preprocessed texts ===
    if use_preprocessed_json_path:
        with open(use_preprocessed_json_path, "r", encoding="utf-8") as f:
            tweets = json.load(f)
    else:
        fetch_discord_messages(channel_type, raw_json_file)
        preprocessed_path = run_preprocessing_general(input_path=raw_json_file)
        with open(preprocessed_path, "r", encoding="utf-8") as f:
            tweets = json.load(f)

    # Extract plain texts (handle dict or str)
    tweet_texts: List[str] = []
    for item in tweets:
        if isinstance(item, dict) and "text" in item:
            tweet_texts.append(str(item["text"]))
        elif isinstance(item, str):
            tweet_texts.append(item)

    if not tweet_texts:
        raise ValueError("No texts found to analyze for general channel.")

    # === Load sentiment model (DeBERTa/Twitter-RoBERTa sentiment) ===
    sentiment_pipeline = load_deberta_sentiment_model()  # should create pipeline(..., top_k=None) internally if you updated it
    # If not, we still pass top_k=None in _predict_with_probs when calling

    # === Predict labels + probs ===
    y_pred, probs = _predict_with_probs(sentiment_pipeline, tweet_texts)

    # === Save predictions ===
    pred_rows = []
    for text, lab, pr in zip(tweet_texts, y_pred, probs):
        pred_rows.append({
            "text": text,
            "predicted_label": lab.upper(),
            "prob_negative": pr[0],
            "prob_neutral": pr[1],
            "prob_positive": pr[2],
        })
    with open(preds_json_file, "w", encoding="utf-8") as f:
        json.dump(pred_rows, f, indent=2, ensure_ascii=False)

    # === If no ground-truth → stop here ===
    if not ground_truth_path:
        return None

    # === Load ground truth & align ===
    gt_map, gt_list = _load_ground_truth_labels(ground_truth_path)

    y_true, aligned_pred, aligned_probs = [], [], []
    missing = 0

    if gt_list:
        if len(gt_list) != len(tweet_texts):
            raise ValueError(
                f"Index-aligned GT length ({len(gt_list)}) does not match number of texts ({len(tweet_texts)})."
            )
        for lab, pred, pr in zip(gt_list, y_pred, probs):
            y_true.append(_normalize_label(lab))
            aligned_pred.append(pred)
            aligned_probs.append(pr)
    else:
        for text, pred, pr in zip(tweet_texts, y_pred, probs):
            key = text.strip()
            if key in gt_map:
                y_true.append(_normalize_label(gt_map[key]))
                aligned_pred.append(pred)
                aligned_probs.append(pr)
            else:
                missing += 1

    if not y_true:
        raise ValueError(
            "No ground-truth labels matched. Provide JSON as {text:label}, "
            "[{text,label}] or [{text,predicted_label}] or an index-aligned list."
        )

    # === Metrics ===
    metrics = _evaluate_classification(y_true, aligned_pred, aligned_probs)
    metrics["num_samples_evaluated"] = len(y_true)
    metrics["num_samples_missing_gt"] = missing

    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # === Visualizations / results bundle ===
    vis_paths = _save_visual_evaluation_general(metrics, output_dir)
    metrics["visual_results"] = vis_paths

    return metrics
