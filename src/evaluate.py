import json
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
COMPARISON_PATH = BASE_DIR / "model" / "model_comparison.json"
RESULT_COLUMNS = [
    "model_name",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "cv_mean",
    "cv_std",
]


def load_comparison_results():
    logger.info("Loading model comparison results from %s", COMPARISON_PATH)
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    numeric_cols = [col for col in RESULT_COLUMNS if col != "model_name"]
    df[numeric_cols] = df[numeric_cols].round(4)
    logger.info("Loaded %s model comparison rows", len(df))
    return df


def get_best_model_name(df):
    logger.info("Selecting best model by highest f1")
    best_model_name = str(df.loc[df["f1"].idxmax(), "model_name"])
    logger.info("Best model selected: %s", best_model_name)
    return best_model_name
