import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


TITANIC_URL = ("https://raw.githubusercontent.com/datasciencedojo/" +
               "datasets/master/titanic.csv")
LOCAL_PATHS = ["data/titanic.csv", "/data/titanic.csv"]

TARGET = "Survived"
NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Pclass"]
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]


def load_raw_data() -> pd.DataFrame:
    for path in LOCAL_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.read_csv(TITANIC_URL)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype("category")
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def get_train_test_data(
    test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    raw = load_raw_data()
    cleaned = clean_data(raw)
    X, y = prepare_dataset(cleaned)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
