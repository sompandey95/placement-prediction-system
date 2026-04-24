import json
import logging
from pathlib import Path

import joblib
import pandas as pd


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_placement_model.joblib"
FEATURE_PATH = BASE_DIR / "model" / "feature_columns.json"


def load_artifacts():
    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    logger.info("Loading feature columns from %s", FEATURE_PATH)
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return model, feature_cols


def predict_student(student_dict, model, feature_cols):
    logger.info("Preparing student input for prediction")
    student_df = pd.DataFrame([student_dict])
    student_df = pd.get_dummies(student_df, drop_first=True)

    logger.info("Aligning input columns with trained feature columns")
    for col in feature_cols:
        if col not in student_df.columns:
            student_df[col] = 0
    student_df = student_df[feature_cols]

    logger.info("Running placement prediction")
    prob = float(model.predict_proba(student_df)[0][1])
    pred = int(model.predict(student_df)[0])

    logger.info("Prediction complete: pred=%s, prob=%.4f", pred, prob)
    return prob, pred
