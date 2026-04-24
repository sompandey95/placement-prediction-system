import logging
from pathlib import Path

import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config():
    logger.info("Loading config from %s", CONFIG_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_preprocess():
    config = load_config()
    data_path = BASE_DIR / config["paths"]["data"]
    target_column = config["model"]["target_column"]
    drop_columns = config["model"]["drop_columns"]
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]
    smote_random_state = config["model"]["smote_random_state"]

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Raw dataset shape: %s", df.shape)

    logger.info("Dropping columns: %s", drop_columns)
    df = df.drop(columns=drop_columns)

    logger.info("Splitting features and target column: %s", target_column)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info("Applying one-hot encoding")
    X = pd.get_dummies(X, drop_first=True)
    feature_cols = list(X.columns)
    logger.info("Encoded feature shape: %s", X.shape)

    logger.info("Running train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Applying SMOTE to training data only")
    smote = SMOTE(random_state=smote_random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info("Training shape before SMOTE: %s", X_train.shape)
    logger.info("Training shape after SMOTE: %s", X_train_resampled.shape)

    return X_train_resampled, X_test, y_train_resampled, y_test, feature_cols
