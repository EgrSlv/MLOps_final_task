import pandas as pd
import pytest

from src.data.clean import clean_data, prepare_dataset, load_raw_data
from src.data.features import FeatureEngineer


def test_load_raw_data():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert "Survived" in df.columns
    assert len(df) > 0

def test_clean_data():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    assert cleaned.isnull().sum().sum() == 0
    assert "PassengerId" not in cleaned.columns
    assert "Name" not in cleaned.columns
    assert "Survived" in cleaned.columns

def test_prepare_dataset():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    X, y = prepare_dataset(cleaned)
    assert "Survived" not in X.columns
    assert len(X) == len(y)

def test_feature_engineer():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    X, y = prepare_dataset(cleaned)
    fe = FeatureEngineer()
    X_fe = fe.fit_transform(X)
    assert X_fe.select_dtypes(include=["category", "object"]).empty
    assert not X_fe.isnull().any().any()
    X_fe2 = fe.transform(X.head())
    assert len(X_fe2) == len(X.head())
