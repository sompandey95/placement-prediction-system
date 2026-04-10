import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "placement_cleaned.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "rf_placement_model.joblib"
FEATURE_PATH = MODEL_DIR / "feature_columns.json"

TARGET_COLUMN = "Placement_Status"
DROP_COLUMNS = ["Student_ID", "Name"]


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=DROP_COLUMNS, inplace=True)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model


def save_artifacts(model, feature_cols):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(FEATURE_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nModel saved     : {MODEL_PATH}")
    print(f"Features saved  : {FEATURE_PATH}")
    print(f"Feature columns : {feature_cols}")


if __name__ == "__main__":
    print("Loading and preprocessing dataset...")
    X, y = load_and_preprocess()
    print(f"Dataset shape: {X.shape} | Class distribution:\n{y.value_counts().to_string()}\n")

    print("Training RandomForestClassifier...\n")
    model = train_model(X, y)

    save_artifacts(model, list(X.columns))
    print("\nDone. You can now run: streamlit run app.py")
