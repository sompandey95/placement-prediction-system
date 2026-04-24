import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.preprocess import load_and_preprocess
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "placement_cleaned.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_placement_model.joblib"
FEATURE_PATH = MODEL_DIR / "feature_columns.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.json"

TARGET_COLUMN = "Placement_Status"
DROP_COLUMNS = ["Student_ID", "Name"]
RANDOM_STATE = 42


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            min_samples_split=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def evaluate_all_models(models, X_train, X_test, y_train, y_test):
    results = []
    fitted_models = {}

    print("\n[Evaluation] Training and evaluating models...")
    for model_name, model in models.items():
        print(f"\n[Evaluation] Model: {model_name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        results.append(
            {
                "model_name": model_name,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
            }
        )
        fitted_models[model_name] = model

        print("[Evaluation] Classification report:")
        print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

    results_df = pd.DataFrame(results)
    print("\n[Evaluation] Model comparison:")
    print(results_df.to_string(index=False))

    return results, fitted_models


def tune_best_model(fitted_models, results, X_train, y_train):
    best_result = max(results, key=lambda item: item["f1"])
    best_model_name = best_result["model_name"]
    best_model = fitted_models[best_model_name]

    print(f"\n[Tuning] Best model by weighted F1: {best_model_name}")

    if best_model_name == "Random Forest":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    elif best_model_name == "XGBoost":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
        }
    else:
        print(f"[Tuning] Tuning skipped for {best_model_name}; returning fitted model as-is.")
        return best_model

    print(f"[Tuning] Running RandomizedSearchCV for {best_model_name}...")
    search = RandomizedSearchCV(
        estimator=best_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)

    print(f"[Tuning] Best params found: {search.best_params_}")
    print(f"[Tuning] Best CV F1 score: {search.best_score_:.4f}")
    return search.best_estimator_


def save_artifacts(model, feature_cols, results):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    with open(FEATURE_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[Save] Training artifacts saved successfully.")
    print(f"[Save] Model: {MODEL_PATH}")
    print(f"[Save] Feature columns: {FEATURE_PATH}")
    print(f"[Save] Model comparison: {COMPARISON_PATH}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("[Pipeline] Starting placement prediction training pipeline...")
    X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess()

    models = get_models()
    results, fitted_models = evaluate_all_models(models, X_train, X_test, y_train, y_test)

    tuned_model = tune_best_model(fitted_models, results, X_train, y_train)
    save_artifacts(tuned_model, feature_cols, results)

    print("\n[Pipeline] Done. You can now run: streamlit run app.py")
